# PersonaPlex-MLX

MLX inference for NVIDIA PersonaPlex on Apple Silicon.

This package supports:

- Realtime local mode (`personaplex_mlx.local`)
- Realtime web mode (`personaplex_mlx.local_web`)
- Offline WAV-to-WAV mode (`personaplex_mlx.offline`)

Console entrypoints are also installed: `personaplex-local`, `personaplex-local-web`, `personaplex-offline`.

## Requirements

- Apple Silicon Mac
- Python 3.12
- Hugging Face access to `nvidia/personaplex-7b-v1`

Install:

```bash
pip install -e .
```

## Model Access

Accept the model license:
`https://huggingface.co/nvidia/personaplex-7b-v1`

Set your token:

```bash
export HF_TOKEN=<your_token>
```

## Quickstart

Launch realtime web mode (recommended first):

```bash
python -m personaplex_mlx.local_web \
  -q 4 \
  --voice NATF2 \
  --text-prompt "You enjoy having a good conversation."
```

Open `http://localhost:8998` in your browser.

Realtime local terminal mode:

```bash
python -m personaplex_mlx.local \
  -q 4 \
  --voice NATF2 \
  --text-prompt "You enjoy having a good conversation."
```

Offline inference:

```bash
python -m personaplex_mlx.offline \
  --voice NATF2 \
  --text-prompt "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way." \
  --input-wav input.wav \
  --output-wav output.wav \
  --output-text output.json \
  --seed 42424242
```

## Voices

Built-in voice IDs:

- `NATF0` `NATF1` `NATF2` `NATF3`
- `NATM0` `NATM1` `NATM2` `NATM3`
- `VARF0` `VARF1` `VARF2` `VARF3` `VARF4`
- `VARM0` `VARM1` `VARM2` `VARM3` `VARM4`

`--voice NATF2` resolves to `NATF2.pt` from the downloaded `voices/` bundle.

## Notes

- First run downloads model assets from Hugging Face.
- Local and web clients are barebone and do not include echo cancellation. Use headphones to avoid feedback.

## Attribution

This project is an MLX port of NVIDIA PersonaPlex for Apple Silicon.

- NVIDIA PersonaPlex repo: `https://github.com/NVIDIA/personaplex`
- PersonaPlex model card: `https://huggingface.co/nvidia/personaplex-7b-v1`

## Citation

If you use PersonaPlex in research, cite:

```bibtex
@misc{roy2026personaplexvoicerolecontrol,
  title={PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models},
  author={Rajarshi Roy and Jonathan Raiman and Sang-gil Lee and Teodor-Dumitru Ene and Robert Kirby and Sungwon Kim and Jaehyeon Kim and Bryan Catanzaro},
  year={2026},
  eprint={2602.06053},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2602.06053}
}
```

## License

- Code: MIT (`LICENSE`)
- Model weights: NVIDIA Open Model License (via Hugging Face model card)
