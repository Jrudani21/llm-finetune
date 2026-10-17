# AGENTS.md - llm-finetune

Fine-tunes a Qwen2.5 statistics specialist locally (RTX 4060, 8 GB VRAM) for the personal assistant. See `README.md` for usage and `PRD.md` for the purpose.

## Run

- `python train.py [--base BASE] [--steps N] [--epochs N] [--out DIR]`, then `python export_to_ollama.py --adapter lora_model --name <name>`.
- Dataset: `python generate_stats_dataset.py` (needs a local Ollama model) and `python review_dataset.py`.

## Test

- No automated tests. See `TESTING.md` for how each script is checked by hand.

## Conventions

- Training uses the local GPU; do not start it without the owner's say-so.
- Follow `CODE_STYLE.md`; plain hyphens in text, no em-dashes.
- Read `ARCHITECTURE.md` and `SECURITY.md` before changing how data flows or what is stored.
- Never commit secrets, API keys or real personal data.

## Verified against

`README.md`, `train.py`, `generate_stats_dataset.py`, `review_dataset.py`, `export_to_ollama.py`.
