#!/usr/bin/env -S uv run python

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


def load_tokens(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fobj:
        data = json.load(fobj)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON list")
    return [str(x) for x in data]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--hypothesis", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()

    ref_tokens = load_tokens(args.reference)
    hyp_tokens = load_tokens(args.hypothesis)
    ref_text = normalize(" ".join(ref_tokens))
    hyp_text = normalize(" ".join(hyp_tokens))
    score = SequenceMatcher(None, ref_text, hyp_text).ratio()
    print(f"similarity={score:.4f}")
    if score < args.threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
