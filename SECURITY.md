# SECURITY.md - llm-finetune

The repository contains no network-exposed services and processes only local data. Security considerations are limited to local file handling and secrets management.

## Trust Boundaries
- **Local file system**: All scripts read and write files within the repository or local directories (e.g., `stats_dataset.jsonl`, `lora_model/`).
- **Ollama interaction**: The `export_to_ollama.py` script invokes the `ollama` CLI tool locally; no network exposure is introduced by the repository itself.
- **Model generation**: The dataset generation step uses a local Ollama instance (via `http://localhost:11434/api/chat`) with the `qwen3-coder:30b` model. This assumes the Ollama service is running locally and is not exposed externally.

## Controls
- **Secrets handling**: The repository does not require API keys, tokens, or passwords for its core functionality. All operations (dataset generation, training, export) are performed locally without external API calls.
- **Input handling**: Scripts that read `stats_dataset.jsonl` (e.g., `review_dataset.py`, `train.py`) parse JSON lines and validate structure. Malformed lines are flagged or cause errors, but no injection vulnerabilities are present as the data is not executed.
- **Data sanitization**: The dataset generation script (`generate_stats_dataset.py`) extracts JSON arrays from model output and writes them as-is. The review script checks for contamination (markdown, JSON remnants, mojibake) but does not modify the data.
- **No privilege escalation**: The scripts run with the user's privileges and do not attempt to modify system files or install software beyond Python package installation (via `pip`/`uv`).

## Notes
- The repository is intended for personal, local use. If deployed in a shared environment, consider securing the Ollama service and restricting file permissions on generated data and models.
- The GitHub repository does not contain any secrets; the `.gitignore` file excludes `stats_dataset.jsonl` and model directories.

## Verified against
README.md
train.py
export_to_ollama.py
generate_stats_dataset.py
review_dataset.py