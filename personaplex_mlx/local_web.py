# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import asyncio
import os
import queue
import tarfile
from pathlib import Path
from typing import Optional

import aiohttp
from aiohttp import web
import mlx.core as mx
import numpy as np
import rustymimi
import sentencepiece
import sphn
import webbrowser

from . import models, utils
from .persona_utils import (
    DEFAULT_HF_REPO,
    get_lm_config,
    get_or_download_mimi,
    get_or_download_model_file,
    get_or_download_tokenizer,
    get_voice_prompt_dir,
    hf_hub_download,
    load_lm_weights,
    resolve_voice_prompt,
    seed_all,
    wrap_with_system_tags,
)

SAMPLE_RATE = 24000
FRAME_SIZE = 1920


def colorize(text: str, color: str) -> str:
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def log(level: str, msg: str):
    if level == "warning":
        prefix = colorize("[Warn]", "1;31")
    elif level == "info":
        prefix = colorize("[Info]", "1;34")
    elif level == "error":
        prefix = colorize("[Err ]", "1;31")
    else:
        raise ValueError(f"Unknown level {level}")
    print(prefix + " " + msg)


class ServerState:
    def __init__(
        self,
        model: models.Lm,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        mimi_file: str,
        args,
    ):
        self.model = model
        self.text_tokenizer = text_tokenizer
        self.mimi_file = mimi_file
        self.args = args
        self.lock = asyncio.Lock()
        self.gen = models.LmGen(
            model=model,
            max_steps=args.steps + 1024,
            text_sampler=utils.Sampler(temp=args.text_temp, top_k=args.text_topk),
            audio_sampler=utils.Sampler(temp=args.audio_temp, top_k=args.audio_topk),
            check=False,
            audio_silence_frame_cnt=int(0.5 * 12.5),
        )

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        voice_name = request.query.get("voice_prompt", "")
        text_prompt = request.query.get("text_prompt", "")
        seed = int(request.query.get("seed", self.args.seed))
        if voice_name == "":
            voice_name = self.args.voice_prompt or self.args.voice
        if text_prompt == "":
            text_prompt = self.args.text_prompt

        voice_path = resolve_voice_prompt(
            voice=None,
            voice_prompt=voice_name,
            voice_prompt_dir=self.args.voice_prompt_dir,
        )

        input_queue: queue.Queue[np.ndarray] = queue.Queue()
        text_queue: queue.Queue[str] = queue.Queue()
        audio_tokenizer = rustymimi.StreamTokenizer(self.mimi_file, num_codebooks=8)  # type: ignore
        opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
        opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
        close = False

        async def recv_loop():
            nonlocal close
            all_pcm_data = None
            try:
                async for message in ws:
                    if message.type in (
                        aiohttp.WSMsgType.ERROR,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                    ):
                        break
                    if message.type != aiohttp.WSMsgType.BINARY:
                        continue
                    payload = message.data
                    if not isinstance(payload, bytes) or len(payload) == 0:
                        continue
                    if payload[0] != 1:
                        continue
                    pcm = opus_reader.append_bytes(payload[1:])
                    if pcm.shape[-1] == 0:
                        continue
                    if all_pcm_data is None:
                        all_pcm_data = pcm
                    else:
                        all_pcm_data = np.concatenate((all_pcm_data, pcm))
                    while all_pcm_data.shape[-1] >= FRAME_SIZE:
                        chunk = all_pcm_data[:FRAME_SIZE]
                        all_pcm_data = all_pcm_data[FRAME_SIZE:]
                        input_queue.put_nowait(chunk.astype(np.float32))
            finally:
                close = True

        async def encode_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                try:
                    pcm = input_queue.get(block=False)
                except queue.Empty:
                    continue
                audio_tokenizer.encode(pcm)

        async def model_loop():
            while True:
                if close:
                    return
                tokens = audio_tokenizer.get_encoded()
                if tokens is None:
                    await asyncio.sleep(0.001)
                    continue
                model_input = mx.array(tokens).transpose(1, 0)[:, : self.gen.user_codebooks]
                model_input = model_input[:, :, None]
                text_token = self.gen.step(input_tokens=model_input)
                if text_token is not None:
                    text_value = int(text_token[0].item())
                    if text_value not in (0, 3):
                        piece = self.text_tokenizer.id_to_piece(text_value)  # type: ignore
                        text_queue.put_nowait(piece.replace("▁", " "))
                audio_tokens = self.gen.last_audio_tokens()
                if audio_tokens is not None:
                    audio_tokenizer.decode(np.array(audio_tokens).astype(np.uint32))

        async def send_loop():
            nonlocal close
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                if ws.closed:
                    close = True
                    return
                decoded = audio_tokenizer.get_decoded()
                if decoded is not None:
                    msg = opus_writer.append_pcm(decoded)
                    if len(msg) > 0:
                        try:
                            await ws.send_bytes(b"\x01" + msg)
                        except aiohttp.ClientConnectionResetError:
                            close = True
                            return
                try:
                    text = text_queue.get(block=False)
                    try:
                        await ws.send_bytes(b"\x02" + text.encode("utf-8"))
                    except aiohttp.ClientConnectionResetError:
                        close = True
                        return
                except queue.Empty:
                    pass

        async with self.lock:
            seed_all(seed)
            self.gen.reset_streaming()
            self.gen.load_voice_prompt_embeddings(voice_path)
            if text_prompt:
                self.gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(text_prompt))  # type: ignore
            else:
                self.gen.text_prompt_tokens = None
            self.gen.step_system_prompts()
            await ws.send_bytes(b"\x00")
            await asyncio.gather(recv_loop(), encode_loop(), model_loop(), send_loop())
        return ws


def get_static_path(static: Optional[str]) -> Optional[str]:
    if static is None:
        log("info", "retrieving static content")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz_path = Path(dist_tgz)
        dist = dist_tgz_path.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz_path, "r:gz") as tar:
                tar.extractall(path=dist_tgz_path.parent)
        return str(dist)
    if static == "none":
        return None
    return static


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--moshi-weight", type=str)
    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("-q", "--quantized", type=int, choices=[4, 8])
    parser.add_argument("--steps", default=4000, type=int)
    parser.add_argument("--hf-repo", type=str, default=DEFAULT_HF_REPO)
    parser.add_argument("--lm-config", type=str)
    parser.add_argument("--voice", type=str, default="NATF2")
    parser.add_argument("--voice-prompt", type=str)
    parser.add_argument("--voice-prompt-dir", type=str)
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
    )
    parser.add_argument("--seed", type=int, default=42424242)
    parser.add_argument("--audio-temp", type=float, default=0.8)
    parser.add_argument("--text-temp", type=float, default=0.7)
    parser.add_argument("--audio-topk", type=int, default=250)
    parser.add_argument("--text-topk", type=int, default=25)
    parser.add_argument("--static", type=str)
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        ),
    )
    args = parser.parse_args()

    log(
        "warning",
        "Echo cancellation is not implemented in this barebone web client. Use headphones to avoid feedback.",
    )

    args.voice_prompt_dir = get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    static_path = get_static_path(args.static)
    lm_config = get_lm_config(args.lm_config, args.hf_repo)
    tokenizer_file = get_or_download_tokenizer(args.hf_repo, args.tokenizer)
    model_file, _ = get_or_download_model_file(
        args.hf_repo, args.quantized, args.moshi_weight
    )
    mimi_file = get_or_download_mimi(args.hf_repo, args.mimi_weight)

    seed_all(args.seed)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)  # type: ignore
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    load_lm_weights(model, lm_config, model_file, args.quantized)
    state = ServerState(model, text_tokenizer, mimi_file, args)

    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, name="static")

    ssl_context = None
    protocol = "http"
    if args.ssl is not None:
        import ssl

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert_file = os.path.join(args.ssl, "cert.pem")
        key_file = os.path.join(args.ssl, "key.pem")
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        protocol = "https"

    log("info", f"listening on {protocol}://{args.host}:{args.port}")
    if not args.no_browser:
        webbrowser.open(f"{protocol}://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
