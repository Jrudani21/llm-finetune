"""Probe: does trl's SFTTrainer work with the current unsloth/trl/transformers
stack, or is the <EOS_TOKEN> sentinel bug from the project note still present?

Runs ONE real training step on the 0.5B 4-bit model with a 2-row toy dataset
so we know the answer without committing hours. Prints a clear verdict.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from datasets import Dataset
import torch
from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

TOY = [
    {"instruction": "What is the mean of 2, 4, 6?", "output": "4"},
    {"instruction": "What is overdispersion in Poisson regression?", "output": "When variance exceeds the mean."},
]


def format_row(row):
    return f"### Question:\n{row['instruction']}\n### Answer:\n{row['output']}"


def main():
    print(f"trl={__import__('trl').__version__} transformers={__import__('transformers').__version__}")
    # Instrument to_dict to detect the obfuscation path (trace-style)
    from transformers import TrainingArguments as _TA
    _orig = _TA.to_dict

    def _traced(self):
        res = _orig(self)
        if res.get("eos_token") == "<EOS_TOKEN>":
            import traceback
            print(f"[to_dict] {type(self).__name__} obfuscated eos_token -> <EOS_TOKEN> (real was {getattr(self, 'eos_token', None)!r})")
            traceback.print_stack(limit=6)
        return res

    _TA.to_dict = _traced
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        max_seq_length=256,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
    )

    # sanity: what does SFTConfig resolve eos/pad to on this stack?
    cfg = SFTConfig(dataset_text_field="text", max_length=256,
                    output_dir="_probe_out", report_to=[])
    print("SFTConfig.eos_token =", repr(cfg.eos_token))
    print("SFTConfig.pad_token =", repr(cfg.pad_token))
    print("tokenizer.eos_token =", repr(tokenizer.eos_token))
    # Force the REAL tokens explicitly (the <EOS_TOKEN> sentinel bug fix).
    cfg.eos_token = tokenizer.eos_token
    cfg.pad_token = tokenizer.pad_token or tokenizer.eos_token
    # transformers.TrainingArguments.to_dict() obfuscates *_token fields to
    # "<TOKEN>" (secrets protection). That turns eos_token into <EOS_TOKEN>
    # and breaks trl's check. Patch this instance's to_dict to keep the real
    # values.
    _orig_cfg_to_dict = cfg.to_dict
    def _cfg_to_dict(self=None):
        d = _orig_cfg_to_dict()
        d["eos_token"] = cfg.eos_token
        d["pad_token"] = cfg.pad_token
        return d
    cfg.to_dict = _cfg_to_dict

    texts = [format_row(r) for r in TOY]
    ds = Dataset.from_dict({"text": texts})

    args = TrainingArguments(
        output_dir="_probe_out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        report_to=[],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, args=cfg,
        train_dataset=ds,
    )
    print("cfg.eos_token right before trainer:", repr(cfg.eos_token))
    print("cfg.pad_token right before trainer:", repr(cfg.pad_token))
    result = trainer.train()
    print(">> TRAINER WORKED. loss =", result.training_loss)
    print(">> VERDICT: trl SFTTrainer is usable on this stack.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(">> VERDICT: SFTTrainer FAILED:", type(e).__name__, e)
        sys.exit(1)
