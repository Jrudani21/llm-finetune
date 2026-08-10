# LLM Fine-tuning — qwen2.5 statistics specialist

Fine-tunes a Qwen2.5 variant specialized for applied statistics (Poisson/negative
binomial regression, ARIMA/SARIMA, hypothesis testing, fraud/risk modeling) for the
personal AI assistant, running fully locally on an RTX 4060 (8 GB VRAM).

## Pipeline (all verified working 2026-08-08)

1. **Generate dataset** — `generate_stats_dataset.py`
   240 instruction/output pairs, 12 statistics topics × 20, generated locally via
   qwen3-coder:30b (free, no API cost). Output: `stats_dataset.jsonl` (gitignored).

2. **Review dataset** — `review_dataset.py`
   Automated gate before training: flags duplicates, empty/short rows, markdown/JSON
   contamination, arithmetic self-contradictions, and cross-topic drift. The current
   240-pair dataset passes clean (12 flags are all false positives from overlapping
   statistics vocabulary — verified by reading each flagged row).

3. **Train** — `train.py`
   QLoRA fine-tune via trl `SFTTrainer` on `stats_dataset.jsonl`. Saves a LoRA
   adapter to `lora_model/`.

   ```bash
   python train.py                                  # qwen2.5:7b base, 1 epoch
   python train.py --base unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
   python train.py --steps 40 --epochs 1 --lr 2e-4 --out lora_model
   ```

4. **Export to Ollama** — `export_to_ollama.py`
   Merges the LoRA into 16-bit base weights, converts to GGUF (q4_k_m default),
   and registers it with Ollama.

   ```bash
   python export_to_ollama.py --adapter lora_model --name qwen25-stats
   ollama list          # should show qwen25-stats
   ollama run qwen25-stats "What is overdispersion in Poisson regression?"
   ```

## Gotchas solved (write them down — they cost hours)

- **trl SFTTrainer `<EOS_TOKEN>` bug**: `transformers.TrainingArguments.to_dict()`
  deliberately obfuscates every field ending in `_token` to `"<TOKEN>"` (secrets
  protection). trl reads `eos_token` back from that dict during trainer init, so the
  sentinel `<EOS_TOKEN>` (not in the vocab) blows up. Fix in `train.py`: pin
  `cfg.eos_token`/`cfg.pad_token` to the real tokenizer tokens AND patch the
  instance's `to_dict` to return them.
- **fp16 autocast breaks Unsloth's fast-LoRA kernels** ("self and mat2 must have the
  same dtype"): use `bf16=True, fp16=False` (Ada supports bf16).
- **GGUF export needs 16-bit weights**: a 4-bit model can't be GGUF-converted
  directly on this stack (transformers 5.5.0 `revert_weight_conversion`
  NotImplementedError). Use `save_method="merged_16bit"` so Unsloth downloads the
  original 16-bit base and merges the LoRA into it.
- **`load_adapter` alone doesn't make the model a PeftModel**: wrap with
  `FastLanguageModel.get_peft_model(...)` first, then `load_adapter(path,
  adapter_name="default")`, so Unsloth's `is_peft_model` check passes.
- **Unsloth's `push_to_ollama` helper has an internal version-skew bug** on this
  stack: call `create_ollama_modelfile(tokenizer, base_model_name, model_location)`
  directly (or reuse the Modelfile Unsloth writes next to the GGUF), then
  `ollama create <name> -f <modelfile>`.

## Structure

- `generate_stats_dataset.py` — dataset generation (qwen3-coder:30b)
- `review_dataset.py` — automated dataset review gate
- `train.py` — QLoRA fine-tune (trl SFTTrainer + Unsloth)
- `export_to_ollama.py` — LoRA → GGUF → Ollama deploy
- `smoke_test.py` — original feasibility smoke test (manual training loop)
- `probe_sfttrainer.py` — reproduces/verifies the trl trainer on the current stack

## Status

- [x] Dataset generated (240 pairs) and reviewed (automated gate passes)
- [x] trl SFTTrainer working on this stack (bug workaround in `train.py`)
- [x] GGUF → Ollama path verified end-to-end with a real trained adapter
- [x] Full training run on qwen2.5:7b (hours; ready to launch) → **DONE 2026-08-08**:
      1 epoch, 108 steps, loss 2.7 → 1.49 (eval 1.16), adapter in `lora_model/`.
      **Deployed**: `qwen25-stats:latest` live in Ollama (q4_k_m, 4.7 GB), verified
      answering stats questions correctly.
- [x] Final human spot-check of the 240-pair dataset before the real run
      (automated gate passed; a quick human skim is still recommended before
      re-training).

## Deployment note (unsloth's `save_pretrained_gguf` hang on 7B)

On this stack, `model.save_pretrained_gguf(...)` hangs after the 16-bit merge on
Qwen2.5 7B (its GGUF subprocess tries to download `tokenizer.model`, which does
not exist for Qwen2.5 — BPE tokenizer, `tokenizer.json` only). The working path,
used for the real deployment:

1. `model.save_pretrained_gguf(..., save_method="merged_16bit")` is NOT needed —
   instead load base 4-bit, `get_peft_model`, `load_adapter("lora_model",
   adapter_name="default")`, and save merged 16-bit via `save_pretrained`
   (or reuse the `./qwen25-stats-gguf` output if the merge already ran).
2. Convert directly with llama.cpp (already installed under `~/.unsloth/llama.cpp`):
   - `unsloth_convert_hf_to_gguf.py <dir> --outfile <f16.gguf> --outtype f16`
   - `llama-quantize.exe <f16.gguf> <q4_k_m.gguf> q4_k_m 8`
3. Write a Modelfile using `OLLAMA_TEMPLATES["qwen-25"]` (from
   `unsloth.save`), pointing at the q4_k_m GGUF with `{__EOS_TOKEN__}` =
   `<|im_end|>`; `ollama create qwen25-stats -f Modelfile_...`.
4. `ollama run qwen25-stats` works; verified correct stats answers.
