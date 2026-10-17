# TESTING.md - llm-finetune

The repository does not contain an automated test suite (e.g., pytest, unittest). Verification is performed by running the provided scripts and checking their outputs.

## Verification Steps

Each step in the pipeline can be run independently to verify correctness.

### 1. Dataset Generation
Run: `python generate_stats_dataset.py`
- Expected output: Prints progress per topic and writes `stats_dataset.jsonl`.
- Verify: File exists and contains 240 JSON lines (20 per topic × 12 topics).
- Note: The script requires a local Ollama instance with `qwen3-coder:30b` model running.

### 2. Dataset Review
Run: `python review_dataset.py [path/to/dataset]`
- Expected output: Reviews the dataset and prints a report.
- Verify: Exit code 0 indicates no flags; exit code 1 indicates issues found.
- The script checks for structural issues, contamination, self-contradiction, topic drift, and boilerplate.

### 3. Training
Run: `python train.py [--base BASE_MODEL] [--steps STEPS] [--epochs EPOCHS] [--out OUTPUT_DIR]`
- Expected output: Trains a LoRA adapter and saves it to the specified output directory (default `lora_model/`).
- Verify: Training loss decreases; adapter files are present in the output directory.
- Note: The script includes workarounds for the trl SFTTrainer EOS_TOKEN bug and fp16/bf16 dtype issues.

### 4. Export to Ollama
Run: `python export_to_ollama.py --adapter ADAPTER_PATH --name MODEL_NAME`
- Expected output: Merges LoRA, saves GGUF, creates Ollama model.
- Verify: 
  - GGUF files are generated in `<model_name>-gguf/`.
  - Ollama model is created (check with `ollama list`).
  - Model responds correctly to a test prompt (e.g., `ollama run <model_name> "What is overdispersion in Poisson regression?"`).

### 5. Probe Script (Optional)
Run: `python probe_sfttrainer.py`
- Expected output: Prints versions and verifies that trl's SFTTrainer works on the current stack.
- Verify: Prints ">> TRAINER WORKED. loss = <value>" and ">> VERDICT: trl SFTTrainer is usable on this stack."

## What Is Not Run
- No automated test suite is present; the above steps are manual verification.
- The scripts are not run in a CI environment in this repository.
- Performance benchmarks (e.g., training speed, VRAM usage) are not automated.

## Verified against
README.md
train.py
export_to_ollama.py
generate_stats_dataset.py
review_dataset.py
probe_sfttrainer.py