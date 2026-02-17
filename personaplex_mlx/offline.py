# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json

import mlx.core as mx
import numpy as np
import rustymimi
import sentencepiece
import sphn

from . import models, utils
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


def log(level: str, msg: str):
    print(f"[{level}] {msg}")


def _reshape_input_tokens(encoded: np.ndarray, user_codebooks: int) -> mx.array:
    tokens = mx.array(encoded).transpose(0, 2, 1)[:, :, :user_codebooks]
    if tokens.shape[1] == user_codebooks and tokens.shape[2] == 1:
        return tokens
    if tokens.shape[1] == 1 and tokens.shape[2] == user_codebooks:
        return tokens.transpose(0, 2, 1)
    raise ValueError(f"unexpected encoded shape {tokens.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-wav", required=True, type=str)
    parser.add_argument("--output-wav", required=True, type=str)
    parser.add_argument("--output-text", required=True, type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--moshi-weight", type=str)
    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("-q", "--quantized", type=int, choices=[4, 8])
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

    seed_all(args.seed)

    lm_config = get_lm_config(args.lm_config, args.hf_repo)
    tokenizer_file = get_or_download_tokenizer(args.hf_repo, args.tokenizer)
    model_file, _ = get_or_download_model_file(
        args.hf_repo, args.quantized, args.moshi_weight
    )
    mimi_file = get_or_download_mimi(args.hf_repo, args.mimi_weight)

    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)  # type: ignore
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    log("info", f"loading weights {model_file}")
    load_lm_weights(model, lm_config, model_file, args.quantized)

    gen = models.LmGen(
        model=model,
        max_steps=100000,
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
    gen.step_system_prompts()
    log("info", "system prompts loaded")

    audio_tokenizer = rustymimi.Tokenizer(mimi_file, num_codebooks=8)  # type: ignore
    in_pcms, _ = sphn.read(args.input_wav, sample_rate=24000)
    total_samples = in_pcms.shape[-1]
    steps = (total_samples + 1919) // 1920
    all_out_pcm = []
    generated_text_tokens: list[str] = []
    text_token_map = ["EPAD", "BOS", "EOS", "PAD"]

    for idx in range(steps):
        start = idx * 1920
        end = min((idx + 1) * 1920, total_samples)
        pcm_data = in_pcms[:, start:end]
        if pcm_data.shape[-1] < 1920:
            pad = 1920 - pcm_data.shape[-1]
            pcm_data = np.pad(pcm_data, ((0, 0), (0, pad)), mode="constant")
        encoded = audio_tokenizer.encode_step(pcm_data[None, 0:1])
        model_input = _reshape_input_tokens(encoded, gen.user_codebooks)
        text_token = gen.step(input_tokens=model_input)
        if text_token is not None:
            token_id = int(text_token[0].item())
            if token_id in (0, 1, 2, 3):
                generated_text_tokens.append(text_token_map[token_id])
            else:
                piece = text_tokenizer.id_to_piece(token_id)  # type: ignore
                generated_text_tokens.append(piece.replace("▁", " "))
        audio_tokens = gen.last_audio_tokens()
        if audio_tokens is not None:
            decode_tokens = np.array(audio_tokens[:, :, None]).astype(np.uint32)
            out_pcm = audio_tokenizer.decode_step(decode_tokens)
            all_out_pcm.append(out_pcm)

    if not all_out_pcm:
        raise RuntimeError("no output audio generated")

    all_out_pcm_np = np.concatenate(all_out_pcm, axis=-1)
    all_out_pcm_np = all_out_pcm_np[:, :, :total_samples]
    rustymimi.write_wav(args.output_wav, all_out_pcm_np[0, 0], sample_rate=24000)
    with open(args.output_text, "w", encoding="utf-8") as fobj:
        json.dump(generated_text_tokens, fobj, ensure_ascii=False)
    log("info", f"wrote {args.output_wav}")
    log("info", f"wrote {args.output_text}")


if __name__ == "__main__":
    main()
