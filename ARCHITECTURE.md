# ARCHITECTURE.md - llm-finetune

The repository follows a linear pipeline for creating a domain-specific LLM via instruction fine-tuning.

## Components
1. **Dataset Generation** (`generate_stats_dataset.py`)
   - Uses local Ollama with qwen3-coder:30b to produce instruction/output pairs.
   - Output: `stats_dataset.jsonl` (gitignored).

2. **Dataset Review** (`review_dataset.py`)
   - Automated gate checking for structural issues, contamination, self-contradiction, topic drift, and boilerplate.
   - Prints a report; exits with code 0 if clean, 1 if flags.

3. **Training** (`train.py`)
   - QLoRA fine-tune via Unsloth + trl `SFTTrainer` on the reviewed dataset.
   - Workarounds for trl EOS_TOKEN bug and fp16/bf16 dtype issues.
   - Saves LoRA adapter to `lora_model/`.

4. **Export** (`export_to_ollama.py`)
   - Merges LoRA into 16-bit base weights, saves as GGUF via `save_pretrained_gguf` with `save_method="merged_16bit"`.
   - Converts to quantized GGUF (default q4_k_m) using llama.cpp tools (assumed installed).
   - Creates Ollama model via a Modelfile (reuses Unsloth's or generates via `create_ollama_modelfile`).
   - Registers with Ollama (`ollama create`).

## Data Flow
- Raw generation → `stats_dataset.jsonl`
- Review → same file (if clean) or manual correction
- Training → `lora_model/` (LoRA weights)
- Export → `<name>-gguf/` (GGUF files) → Ollama model

## Dependencies
- Python packages: unsloth, trl, torch, transformers, datasets
- Local Ollama server with qwen3-coder:30b for generation
- llama.cpp tooling (`unsloth_convert_hf_to_gguf.py`, `llama-quantize.exe`) for GGUF conversion

## Verified against
README.md
train.py
export_to_ollama.py
generate_stats_dataset.py
review_dataset.py