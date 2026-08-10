"""Train the stats specialist: QLoRA fine-tune of a Qwen2.5 model on
stats_dataset.jsonl, then save the LoRA adapter for GGUF/Ollama export.

Addresses the project's open items:
- trl SFTTrainer bug: worked around by pinning eos_token/pad_token on the
  SFTConfig instance AND patching its to_dict() so transformers'
  secrets-obfuscation ("<EOS_TOKEN>") can't corrupt the token.
- Real training run on the 240-pair stats dataset (was: only a smoke test).

Usage:
    python train.py                     # train qwen2.5:7b base (default)
    python train.py --base unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
    python train.py --steps 40 --epochs 1 --lr 2e-4 --out lora_model

After training, deploy with:
    python export_to_ollama.py --adapter lora_model --name qwen25-stats
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from datasets import Dataset
import torch
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

DEFAULT_BASE = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
DATASET = Path(__file__).parent / "stats_dataset.jsonl"
MAX_SEQ = 1024  # 240 short examples; keep VRAM headroom on 8 GB


def format_row(row: dict) -> str:
    return (
        f"### Question:\n{row['instruction']}\n"
        f"### Answer:\n{row['output']}"
    )


def load_dataset() -> Dataset:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")
    rows = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    texts = [format_row(r) for r in rows]
    ds = Dataset.from_dict({"text": texts})
    print(f"Loaded {len(texts)} training rows from {DATASET.name}.")
    return ds


def main() -> int:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune the stats specialist.")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--out", default="lora_model")
    parser.add_argument("--steps", type=int, default=0, help="max steps; 0 = full epochs")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--max-seq", type=int, default=MAX_SEQ)
    parser.add_argument("--eval-frac", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Base model: {args.base}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    ds = load_dataset()
    if args.eval_frac:
        split = ds.train_test_split(test_size=args.eval_frac, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None

    # --- trl bug workaround (see module docstring) ---
    cfg = SFTConfig(
        dataset_text_field="text",
        max_length=args.max_seq,
        output_dir=args.out + "_runs",
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        max_steps=args.steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="steps" if args.eval_frac else "no",        eval_steps=20,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        report_to=[],
        fp16=False,
        bf16=True,  # Ada supports bf16; unsloth's fast-LoRA kernels need it over fp16 autocast
        remove_unused_columns=False,
    )
    # Pin the REAL tokens: SFTConfig defaults to None, and transformers'
    # TrainingArguments.to_dict() obfuscates *_token fields into "<TOKEN>"
    # placeholders (secrets protection). trl reads eos_token back from that
    # dict during trainer init, so the sentinel "<EOS_TOKEN>" must never
    # flow through. Patch this instance's to_dict to return the real values.
    cfg.eos_token = tokenizer.eos_token
    cfg.pad_token = tokenizer.pad_token or tokenizer.eos_token
    _orig_to_dict = cfg.to_dict

    def _safe_to_dict(self=None):
        d = _orig_to_dict()
        d["eos_token"] = cfg.eos_token
        d["pad_token"] = cfg.pad_token
        return d

    cfg.to_dict = _safe_to_dict

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    print("Starting training...")
    result = trainer.train()
    print(f"Training done. loss = {result.training_loss:.4f}")

    print(f"Saving LoRA adapter to {args.out}/ ...")
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Adapter saved. Deploy with:")
    print(f"  python export_to_ollama.py --adapter {args.out} --name qwen25-stats")
    return 0


if __name__ == "__main__":
    sys.exit(main())
