# PRD.md - llm-finetune

Fine-tune a Qwen2.5 variant specialized for applied statistics (Poisson/negative binomial regression, ARIMA/SARIMA, hypothesis testing, fraud/risk modeling) for the personal AI assistant, running fully locally on an RTX 4060 (8 GB VRAM).

## Problem
The base LLM lacks depth in niche statistics topics needed for accurate reasoning in data science tasks. Public APIs are costly or unavailable for private data.

## Solution
Create a small, high-quality instruction dataset focused on statistics, fine-tune via QLoRA, and deploy as a local Ollama model.

## Key Features
- Dataset generation via local qwen3-coder:30b (free, no API cost)
- Automated review gate to detect dataset issues before training
- QLoRA fine-tuning using Unsloth and trl SFTTrainer with bug workarounds
- Export pipeline: LoRA -> merged 16-bit -> GGUF -> Ollama
- Verified end-to-end on qwen2.5:7b base model

## User Stories
- As a developer, I can generate a statistics dataset locally without API fees.
- As a data scientist, I can review the dataset for correctness before training.
- As a machine learning engineer, I can fine-tune a 7B model on consumer GPU.
- As an end-user, I can chat with the fine-tuned model via Ollama for stats queries.

## Out of Scope
- Multi-GPU training
- Online learning or continual updates
- Integration with external vector databases
- Public model hosting

## Verified against
README.md
train.py
export_to_ollama.py
generate_stats_dataset.py
review_dataset.py