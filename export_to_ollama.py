"""Export a fine-tuned model to Ollama: merge LoRA (if any) -> save GGUF ->
create an Ollama model from it, ready to chat with.

This is the deployment path the project note requires verifying BEFORE
spending hours training (open item: "Verify the GGUF export path reaches
Ollama before training for hours"). Run it now with the tiny smoke model:

    python export_to_ollama.py --base unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit --name stats-smoke

and after a real run:

    python export_to_ollama.py --adapter lora_model --name qwen25-stats

Then:  ollama list   -> should show the new model
       ollama run <name> "What is overdispersion in Poisson regression?"
"""
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a model/adapter to Ollama via GGUF.")
    parser.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
                        help="base model (unsloth bnb-4bit id)")
    parser.add_argument("--adapter", default=None,
                        help="path to a trained LoRA adapter dir; merges it into the base")
    parser.add_argument("--name", required=True,
                        help="name for the Ollama model, e.g. qwen25-stats")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--quant", default="q4_k_m",
                        help="GGUF quantization (default q4_k_m; q8_0 for higher quality)")
    parser.add_argument("--max-seq", type=int, default=2048)
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    print(f"Loading base: {args.base} (4-bit)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
    )
    if args.adapter:
        print(f"Loading LoRA adapter: {args.adapter}")
        # Wrap in a PEFT model FIRST (load_adapter alone doesn't make the
        # model a PeftModel instance, so unsloth's is_peft_model check fails
        # and it refuses to GGUF-convert 4-bit weights).
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=False,
        )
        model.load_adapter(args.adapter, adapter_name="default")
        print("Adapter loaded; will merge during GGUF export.")

    gguf_dir = f"./{args.name}-gguf"
    print(f"Saving GGUF ({args.quant}) to {gguf_dir}/ ...")
    # save_method="merged_16bit" downloads the original 16-bit base weights
    # and merges the LoRA into them, so llama.cpp gets dequantized weights
    # (a 4-bit model can't be GGUF-converted directly on this stack).
    model.save_pretrained_gguf(
        gguf_dir,
        tokenizer=tokenizer,
        quantization_method=args.quant,
        save_method="merged_16bit",
    )
    print("GGUF saved.")

    # --- Create the Ollama model directly (unsloth's push_to_ollama helper
    # has an internal version-skew bug on this stack) ---
    import glob
    from pathlib import Path

    gguf_dirs = sorted(glob.glob(str(Path(gguf_dir).parent / (Path(gguf_dir).name + "_gguf"))))
    gguf_files = sorted(glob.glob(str(Path(gguf_dir) / "*.gguf")))
    if not gguf_files and gguf_dirs:
        gguf_files = sorted(glob.glob(str(Path(gguf_dirs[0]) / "*.gguf")))
    if not gguf_files:
        print("ERROR: no .gguf file found under", gguf_dir)
        return 1
    gguf_path = gguf_files[0]
    print(f"GGUF file: {gguf_path}")

    # Unsloth already wrote a Modelfile next to the GGUF; reuse it if present
    modelfile_dir = Path(gguf_path).parent
    modelfile_candidates = [
        modelfile_dir / "Modelfile",
        Path(gguf_dir) / "Modelfile",
    ]
    modelfile_path = next((p for p in modelfile_candidates if p.exists()), None)
    if modelfile_path is None:
        from unsloth.save import create_ollama_modelfile
        modelfile = create_ollama_modelfile(
            tokenizer=tokenizer,
            base_model_name=args.base,
            model_location=str(gguf_path),
        )
        if not modelfile:
            print("ERROR: could not build an Ollama Modelfile (no template mapping).")
            return 1
        modelfile_path = Path(f"Modelfile_{args.name}")
        modelfile_path.write_text(modelfile, encoding="utf-8")
    print(f"Using Modelfile: {modelfile_path}")

    import subprocess
    full_name = f"{args.name}:{args.tag}"
    print(f"Running: ollama create {full_name}")
    r = subprocess.run(
        ["ollama", "create", full_name, "-f", str(modelfile_path)],
        capture_output=True, text=True, timeout=600,
    )
    print(r.stdout.strip())
    if r.returncode != 0:
        print("ollama create stderr:", r.stderr.strip())
        return 1

    print(f"\nDone. Verify with:  ollama list")
    print(f"Chat with:          ollama run {full_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
