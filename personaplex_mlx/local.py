# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import queue
import sys
import time
from enum import Enum

import mlx.core as mx
import numpy as np
import rustymimi
import sentencepiece
import sounddevice as sd

from . import models, utils
from .client_utils import AnyPrinter, Printer, RawPrinter
from .persona_utils import (
    DEFAULT_HF_REPO,
    get_lm_config,
    get_or_download_mimi,
    get_or_download_model_file,
    get_or_download_tokenizer,
    get_voice_prompt_dir,
    load_lm_weights,
    resolve_voice_prompt,
    seed_all,
    wrap_with_system_tags,
)

SAMPLE_RATE = 24000
CHANNELS = 1


class PrinterType(Enum):
    TOKEN = 1
    PENDING = 2
    INFO = 3
    WARNING = 4
    ERROR = 5
    LAG = 6
    HEADER = 7
    EVENT = 8
    QSIZE = 9


def full_warmup(audio_tokenizer, client_to_server, server_to_client, rounds: int = 8):
    for _ in range(rounds):
        pcm_data = np.zeros(1920, dtype=np.float32)
        audio_tokenizer.encode(pcm_data)
        while True:
            time.sleep(0.005)
            data = audio_tokenizer.get_encoded()
            if data is not None:
                client_to_server.put_nowait(data)
                break
        try:
            audio_tokens = server_to_client.get(timeout=0.05)
        except queue.Empty:
            continue
        audio_tokenizer.decode(audio_tokens)
        while True:
            time.sleep(0.005)
            if audio_tokenizer.get_decoded() is not None:
                break


def server(printer_q, client_to_server, server_to_client, args):
    def log(msg: str):
        printer_q.put_nowait((PrinterType.INFO, msg))

    lm_config = get_lm_config(args.lm_config, args.hf_repo)
    model_file, _ = get_or_download_model_file(
        hf_repo=args.hf_repo,
        quantized=args.quantized,
        explicit_model_file=args.moshi_weight,
    )
    tokenizer_file = get_or_download_tokenizer(args.hf_repo, args.tokenizer)

    log(f"[SERVER] loading text tokenizer {tokenizer_file}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)  # type: ignore
    seed_all(args.seed)

    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    log(f"[SERVER] loading weights {model_file}")
    load_lm_weights(model, lm_config, model_file, args.quantized)
    log("[SERVER] weights loaded")

    gen = models.LmGen(
        model=model,
        max_steps=args.steps + 1024,
        text_sampler=utils.Sampler(temp=args.text_temp, top_k=args.text_topk),
        audio_sampler=utils.Sampler(temp=args.audio_temp, top_k=args.audio_topk),
        check=False,
        audio_silence_frame_cnt=int(0.5 * 12.5),
    )

    voice_prompt_dir = get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    voice_prompt_path = resolve_voice_prompt(
        voice=args.voice,
        voice_prompt=args.voice_prompt,
        voice_prompt_dir=voice_prompt_dir,
    )
    gen.load_voice_prompt_embeddings(voice_prompt_path)
    if args.text_prompt:
        gen.text_prompt_tokens = text_tokenizer.encode(wrap_with_system_tags(args.text_prompt))  # type: ignore
    else:
        gen.text_prompt_tokens = None

    gen.reset_streaming()
    log("[SERVER] stepping system prompts")
    gen.step_system_prompts()
    log("[SERVER] system prompts loaded")

    server_to_client.put("start")
    log("[SERVER] connected")
    printed_header = False
    try:
        while True:
            data = client_to_server.get()
            printer_q.put_nowait((PrinterType.EVENT, "s_get"))
            if not printed_header:
                printed_header = True
                printer_q.put_nowait((PrinterType.HEADER, ""))
            data = mx.array(data).transpose(1, 0)[:, : gen.user_codebooks]
            data = data[:, :, None]
            text_token = gen.step(input_tokens=data)
            if text_token is not None:
                text_value = int(text_token[0].item())
                if text_value not in (0, 3):
                    piece = text_tokenizer.id_to_piece(text_value)  # type: ignore
                    piece = piece.replace("▁", " ")
                    printer_q.put_nowait((PrinterType.TOKEN, piece))
                else:
                    printer_q.put_nowait((PrinterType.PENDING, ""))
            audio_tokens = gen.last_audio_tokens()
            if audio_tokens is not None:
                server_to_client.put_nowait(np.array(audio_tokens).astype(np.uint32))
            printer_q.put_nowait((PrinterType.EVENT, "s_put"))
    except KeyboardInterrupt:
        pass


def client(printer_q, client_to_server, server_to_client, args):
    mimi_file = get_or_download_mimi(args.hf_repo, args.mimi_weight)
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    audio_tokenizer = rustymimi.StreamTokenizer(mimi_file, num_codebooks=8)  # type: ignore
    start = server_to_client.get()
    printer_q.put_nowait(
        (PrinterType.INFO, f"[CLIENT] received '{start}' from server, starting")
    )

    full_warmup(audio_tokenizer, client_to_server, server_to_client)

    async def send_loop():
        while True:
            await asyncio.sleep(0.001)
            try:
                pcm_data = input_queue.get(block=False)
            except queue.Empty:
                continue
            printer_q.put_nowait((PrinterType.EVENT, "encode"))
            audio_tokenizer.encode(pcm_data)

    async def recv_loop():
        while True:
            data = audio_tokenizer.get_decoded()
            if data is None:
                await asyncio.sleep(0.001)
                continue
            printer_q.put_nowait((PrinterType.EVENT, "decoded"))
            output_queue.put_nowait(data)

    async def send_loop2():
        while True:
            data = audio_tokenizer.get_encoded()
            if data is None:
                await asyncio.sleep(0.001)
                continue
            printer_q.put_nowait((PrinterType.EVENT, "encoded"))
            client_to_server.put_nowait(data)

    async def recv_loop2():
        while True:
            try:
                audio_tokens = server_to_client.get(block=False)
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue
            printer_q.put_nowait((PrinterType.EVENT, "decode"))
            audio_tokenizer.decode(audio_tokens)

    def on_input(in_data, frames, timing, status):
        _ = frames, timing, status
        input_queue.put_nowait(in_data[:, 0].astype(np.float32))

    in_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=1920, callback=on_input
    )

    cnt_output = 0
    last_qsize = 0

    def on_output(out_data, frames, timing, status):
        _ = frames, timing, status
        nonlocal cnt_output, last_qsize
        cnt_output += 1
        qsize = output_queue.qsize()
        if qsize != last_qsize:
            last_qsize = qsize
            printer_q.put_nowait((PrinterType.QSIZE, qsize))
        try:
            pcm_data = output_queue.get(block=False)
            out_data[:, 0] = pcm_data
        except queue.Empty:
            if cnt_output > 3:
                printer_q.put_nowait((PrinterType.LAG, ""))
            out_data.fill(0)

    out_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=1920,
        callback=on_output,
    )

    async def go():
        with in_stream, out_stream:
            await asyncio.gather(recv_loop(), send_loop(), recv_loop2(), send_loop2())

    try:
        asyncio.run(go())
    except KeyboardInterrupt:
        pass


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
    args = parser.parse_args()

    client_to_server = multiprocessing.Queue()
    server_to_client = multiprocessing.Queue()
    printer_q = multiprocessing.Queue()

    printer: AnyPrinter
    if sys.stdout.isatty():
        printer = Printer()
    else:
        printer = RawPrinter()
    printer.log(
        "warning",
        "Echo cancellation is not implemented in this barebone local client. Use headphones to avoid feedback.",
    )

    subprocess_args = printer_q, client_to_server, server_to_client, args
    p1 = multiprocessing.Process(target=client, args=subprocess_args)
    p2 = multiprocessing.Process(target=server, args=subprocess_args)
    p1.start()
    p2.start()

    events = []
    try:
        while p1.is_alive() and p2.is_alive():
            time.sleep(0.001)
            try:
                ty, value = printer_q.get_nowait()
            except queue.Empty:
                continue
            if ty == PrinterType.TOKEN:
                printer.print_token(value)
            elif ty == PrinterType.PENDING:
                printer.print_pending()
            elif ty == PrinterType.INFO:
                printer.log("info", value)
            elif ty == PrinterType.WARNING:
                printer.log("warning", value)
            elif ty == PrinterType.ERROR:
                printer.log("error", value)
            elif ty == PrinterType.LAG:
                printer.print_lag()
                events.append({"event": "lag", "time": time.time()})
            elif ty == PrinterType.HEADER:
                printer.print_header()
            elif ty == PrinterType.EVENT:
                events.append({"event": value, "time": time.time()})
            elif ty == PrinterType.QSIZE:
                events.append({"event": "qsize", "qsize": value, "time": time.time()})
    except KeyboardInterrupt:
        printer.log("warning", "Interrupting, exiting connection")
        p1.terminate()
        p2.terminate()

    chrome_events = []
    for e in events:
        name, ph, tid, args_dict = "unk", "X", 1, {}
        event = e["event"]
        if event == "s_get":
            name, ph, tid = "model", "B", 3
        elif event == "s_put":
            name, ph, tid = "model", "E", 3
        elif event == "encode":
            name, ph, tid = "encode", "B", 1
        elif event == "encoded":
            name, ph, tid = "encode", "E", 1
        elif event == "decode":
            name, ph, tid = "decode", "B", 2
        elif event == "decoded":
            name, ph, tid = "decode", "E", 2
        elif event == "lag":
            name, ph, tid = "lag", "i", 2
        elif event == "qsize":
            name, ph, tid = "qsize", "C", 4
            args_dict["qsize"] = e["qsize"]
        chrome_events.append(
            {
                "name": name,
                "cat": "",
                "ph": ph,
                "ts": e["time"] * 1e6,
                "pid": 1,
                "tid": tid,
                "args": args_dict,
            }
        )
    with open("mlx-trace.json", "w", encoding="utf-8") as fobj:
        json.dump(chrome_events, fobj)

    p1.join()
    p2.join()
    printer.log("info", "All done")


if __name__ == "__main__":
    main()
