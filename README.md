# personaplex_mlx

PersonaPlex inference on Apple Silicon via MLX.

## Install

```bash
pip install -e .
```

Python 3.12 is recommended.

You must accept the PersonaPlex model license on Hugging Face:
`https://huggingface.co/nvidia/personaplex-7b-v1`

If needed, export your token:

```bash
export HF_TOKEN=<your_token>
```

## Local realtime mode

```bash
python -m personaplex_mlx.local -q 4 --voice NATF2 --text-prompt "You are a senior staff engineer conducting a behavioral interview."
```

## Web mode

```bash
python -m personaplex_mlx.local_web -q 4 --voice NATF2 --text-prompt "You enjoy having a good conversation."
```

## Offline mode

```bash
python -m personaplex_mlx.offline \
  --voice NATF2 \
  --text-prompt "$(cat /tmp/personaplex_plan/nvidia/assets/test/prompt_service.txt)" \
  --input-wav /tmp/personaplex_plan/nvidia/assets/test/input_service.wav \
  --output-wav output.wav \
  --output-text output.json \
  --seed 42424242
```

## Notes

- Voice prompts are loaded from PersonaPlex `voices.tgz` by default.
- `--voice NATF2` resolves to `NATF2.pt`.
- Local and web clients are barebone and do not implement echo cancellation. Use headphones to avoid feedback loops.
- Use `tools/diff_weight_keys.py` to regenerate the checkpoint key diff report in `docs/weight_diff.md`.
- Use `tools/compare_transcripts.py` to compute normalized text similarity between PyTorch and MLX offline outputs.
