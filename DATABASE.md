# DATABASE.md - llm-finetune

The repository does not use a traditional database. It processes data through file-based workflows for dataset generation, model training, and export.

## Data Files

### Inputs
- `stats_dataset.jsonl` (generated, gitignored)
  - JSON Lines file containing instruction/output pairs for fine-tuning.
  - Each line is a JSON object with keys: `instruction`, `output`, `topic`.
  - Produced by `generate_stats_dataset.py`.
  - Consumed by `review_dataset.py` and `train.py`.

### Outputs
- `lora_model/` (directory)
  - Contains LoRA adapter weights after training via `train.py`.
  - Used as input to `export_to_ollama.py`.
- `<name>-gguf/` (directory)
  - Contains GGUF files after conversion via `export_to_ollama.py`.
  - Intermediate step for Ollama deployment.
- Ollama model (e.g., `qwen25-stats:latest`)
  - Created by `export_to_ollama.py` via `ollama create`.
  - Stored and served by the local Ollama instance (outside the repo).

## Data Flow
1. Generation: `generate_stats_dataset.py` → `stats_dataset.jsonl`
2. Review: `review_dataset.py` validates `stats_dataset.jsonl` (no modification if clean)
3. Training: `train.py` reads `stats_dataset.jsonl`, writes LoRA adapter to `lora_model/`
4. Export: `export_to_ollama.py` reads `lora_model/`, writes GGUF to `<name>-gguf/`, creates Ollama model

## Notes
- All data files are stored locally; no external databases or cloud services are used.
- The Ollama model is managed by the Ollama application, not the repository.
- The `stats_dataset.jsonl` file is large enough to be gitignored (240 lines) but small enough to review manually.

## Verified against
README.md
train.py
export_to_ollama.py
generate_stats_dataset.py
review_dataset.py