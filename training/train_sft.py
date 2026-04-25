"""
training/train_sft.py

SFT warmup on the combined AdaptAssist dataset.
Saves loss log to artifacts/sft_loss_log.json for offline plotting.

Usage:
    python training/train_sft.py \
        --model unsloth/Llama-3.2-3B-Instruct \
        --dataset data/combined_sft.json \
        --output checkpoints/sft \
        --epochs 2
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def train(args):
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset}.\n"
            "Run: python data/generate_sft_data.py --n 80 --output data/sft_dataset.json\n"
            "Then: python data/dataset_loader.py --output data/combined_sft.json"
        )

    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0.05, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )
    model.print_trainable_parameters()

    print(f"Loading dataset: {args.dataset}")
    with open(args.dataset) as f:
        raw = json.load(f)

    from collections import Counter
    sources = Counter(s.get("source", "?") for s in raw)
    print(f"Total samples: {len(raw)}")
    for src, cnt in sources.most_common():
        print(f"  {src}: {cnt}")

    hf_ds = Dataset.from_list([{"messages": s["messages"]} for s in raw])
    split = hf_ds.train_test_split(test_size=0.05, seed=42)
    print(f"Train: {len(split['train'])} | Eval: {len(split['test'])}")

    sft_cfg = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        max_seq_length=4096,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=split["train"], eval_dataset=split["test"],
        args=sft_cfg,
    )

    print(f"Starting SFT training ({args.epochs} epochs)...")
    trainer.train()

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Model saved to {args.output}")

    # Save loss log
    log = trainer.state.log_history
    log_path = "artifacts/sft_loss_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Loss log saved to {log_path}")

    # Print summary
    train_losses = [l["loss"] for l in log if "loss" in l and "eval_loss" not in l]
    eval_losses  = [l["eval_loss"] for l in log if "eval_loss" in l]
    if train_losses:
        print(f"\nSFT Summary:")
        print(f"  Initial train loss: {train_losses[0]:.4f}")
        print(f"  Final train loss:   {train_losses[-1]:.4f}")
        print(f"  Loss reduction:     {train_losses[0] - train_losses[-1]:+.4f}")
    if eval_losses:
        print(f"  Best eval loss:     {min(eval_losses):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset", default="data/combined_sft.json")
    parser.add_argument("--output",  default="checkpoints/sft")
    parser.add_argument("--epochs",  type=int, default=2)
    args = parser.parse_args()
    train(args)