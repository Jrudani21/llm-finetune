"""Feasibility smoke test only: proves 4-bit load + LoRA + forward/backward
+ optimizer step actually works end-to-end on this machine (native Windows,
RTX 4060), via a manual training loop -- deliberately bypassing trl's
SFTTrainer, which hit a version-skew bug in this unsloth/trl combo
(SFTConfig.eos_token resolving to a literal "<EOS_TOKEN>" sentinel that
isn't a real vocab token). Manual loop tests the actual GPU/quant/LoRA
mechanics a real training run depends on; the trainer-wrapper choice for
the real run can be revisited separately (plain transformers.Trainer,
an older/newer trl pin, etc.) without blocking on this."""
import torch
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    max_seq_length=512,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,
)
FastLanguageModel.for_training(model)
model.gradient_checkpointing_disable()

toy_examples = [
    "### Question:\nWhat is the mean of 2, 4, 6?\n### Answer:\n4",
    "### Question:\nWhat is overdispersion in Poisson regression?\n### Answer:\nWhen variance exceeds the mean.",
] * 4

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

model.train()
losses = []
for step, text in enumerate(toy_examples):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    print(f"step {step}: loss={loss.item():.4f}")

print("\n=== SMOKE TEST PASSED ===")
print(f"first loss: {losses[0]:.4f}, last loss: {losses[-1]:.4f}")
print(f"peak VRAM used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
