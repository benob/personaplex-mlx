# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np
import rustymimi
import sentencepiece
import sphn

from . import models, utils
from .client_utils import make_log
from .persona_utils import (
    DEFAULT_HF_REPO,
    get_lm_config,
    get_or_download_mimi,
    get_or_download_model_file,
    get_or_download_tokenizer,
    load_lm_weights,
    seed_all,
)


def log(level: str, msg: str):
    print(make_log(level, msg))


def _reshape_input_tokens(encoded: np.ndarray, user_codebooks: int) -> mx.array:
    tokens = mx.array(encoded).transpose(0, 2, 1)[:, :, :user_codebooks]
    if tokens.shape[1] == user_codebooks and tokens.shape[2] == 1:
        return tokens
    if tokens.shape[1] == 1 and tokens.shape[2] == user_codebooks:
        return tokens.transpose(0, 2, 1)
    raise ValueError(f"unexpected encoded shape {tokens.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--moshi-weights", type=str)
    parser.add_argument("--mimi-weights", type=str)
    parser.add_argument("--hf-repo", type=str, default=DEFAULT_HF_REPO)
    parser.add_argument("--lm-config", type=str)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("infile", type=str)
    parser.add_argument("outfile", type=str, nargs="?", default="")
    args = parser.parse_args()

    seed_all(299792458)
    lm_config = get_lm_config(args.lm_config, args.hf_repo)
    moshi_weights, _ = get_or_download_model_file(args.hf_repo, None, args.moshi_weights)
    mimi_weights = get_or_download_mimi(args.hf_repo, args.mimi_weights)
    tokenizer = get_or_download_tokenizer(args.hf_repo, args.tokenizer)

    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    load_lm_weights(model, lm_config, moshi_weights, quantized=None)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)  # type: ignore

    in_pcms, _ = sphn.read(args.infile, sample_rate=24000)
    steps = in_pcms.shape[-1] // 1920
    audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=8)  # type: ignore
    gen = models.LmGen(
        model=model,
        max_steps=steps + 128,
        text_sampler=utils.Sampler(top_k=25, temp=args.temp),
        audio_sampler=utils.Sampler(top_k=250, temp=args.temp),
        check=False,
    )

    all_out_pcm = []
    start_time = time.time()
    for idx in range(steps):
        pcm_data = in_pcms[:, idx * 1920 : (idx + 1) * 1920]
        encoded = audio_tokenizer.encode_step(pcm_data[None, 0:1])
        model_input = _reshape_input_tokens(encoded, gen.user_codebooks)
        text_token = gen.step(input_tokens=model_input)
        if text_token is not None:
            text_value = int(text_token[0].item())
            if text_value not in (0, 3):
                piece = text_tokenizer.id_to_piece(text_value)  # type: ignore
                print(piece.replace("▁", " "), end="", flush=True)
        audio_tokens = gen.last_audio_tokens()
        if audio_tokens is not None:
            out_pcm = audio_tokenizer.decode_step(
                np.array(audio_tokens[:, :, None]).astype(np.uint32)
            )
            all_out_pcm.append(out_pcm)
    print()
    token_per_second = steps / (time.time() - start_time)
    log("info", f"steps={steps} token_per_second={token_per_second}")

    if args.outfile and all_out_pcm:
        out_pcm = np.concatenate(all_out_pcm, axis=-1)
        rustymimi.write_wav(args.outfile, out_pcm[0, 0], sample_rate=24000)
        log("info", f"wrote {args.outfile}")


if __name__ == "__main__":
    main()
