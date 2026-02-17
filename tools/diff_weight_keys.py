#!/usr/bin/env -S uv run python

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi


def _prefix(key: str) -> str:
    parts = key.split(".")
    if len(parts) < 2:
        return key
    return ".".join(parts[:2])


def _load_tensors(api: HfApi, repo_id: str) -> dict[str, object]:
    meta = api.get_safetensors_metadata(repo_id)
    file_meta = meta.files_metadata["model.safetensors"]
    return file_meta.tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--target", default="nvidia/personaplex-7b-v1")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/weight_diff.md"),
        help="Write report to this markdown file.",
    )
    args = parser.parse_args()

    api = HfApi()
    base = _load_tensors(api, args.base)
    target = _load_tensors(api, args.target)

    base_keys = set(base.keys())
    target_keys = set(target.keys())
    only_target = sorted(target_keys - base_keys)
    only_base = sorted(base_keys - target_keys)

    shape_changed: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key in sorted(base_keys & target_keys):
        base_shape = tuple(base[key].shape)
        target_shape = tuple(target[key].shape)
        if base_shape != target_shape:
            shape_changed.append((key, base_shape, target_shape))

    prefix_counts = Counter(_prefix(k) for k in only_target)

    lines: list[str] = []
    lines.append("# PersonaPlex vs Base Moshiko Weight Diff")
    lines.append("")
    lines.append(f"- Base repo: `{args.base}`")
    lines.append(f"- Target repo: `{args.target}`")
    lines.append(f"- Base tensor keys: `{len(base_keys)}`")
    lines.append(f"- Target tensor keys: `{len(target_keys)}`")
    lines.append(f"- Added keys in target: `{len(only_target)}`")
    lines.append(f"- Removed keys in target: `{len(only_base)}`")
    lines.append(f"- Shape-changed shared keys: `{len(shape_changed)}`")
    lines.append("")
    lines.append("## Added key prefixes")
    lines.append("")
    for prefix, count in sorted(prefix_counts.items()):
        lines.append(f"- `{prefix}`: {count}")
    lines.append("")
    lines.append("## Shape changes")
    lines.append("")
    for key, base_shape, target_shape in shape_changed:
        lines.append(f"- `{key}`: `{base_shape}` -> `{target_shape}`")
    lines.append("")
    lines.append("## Added keys (full list)")
    lines.append("")
    for key in only_target:
        lines.append(f"- `{key}`")
    lines.append("")
    lines.append("## Removed keys (full list)")
    lines.append("")
    for key in only_base:
        lines.append(f"- `{key}`")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
