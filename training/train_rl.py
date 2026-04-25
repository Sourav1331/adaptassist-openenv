"""
training/train_rl.py

Full GRPO RL training script against the live AdaptAssist HF Space.
Saves reward logs to artifacts/grpo_reward_log.json after each checkpoint.

Usage (after SFT):
    python training/train_rl.py \
        --sft_checkpoint checkpoints/sft \
        --env_url https://souravdanyal-adaptassist-env.hf.space \
        --steps 300 \
        --curriculum
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

# ── Reward functions ──────────────────────────────────────────────────────────

def format_reward(completions, **kwargs):
    """Reward any completion that is valid JSON with 'tool' and 'params'."""
    scores = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        try:
            t = text.strip()
            if t.startswith("```"):
                t = t.split("\n", 1)[1].rsplit("```", 1)[0]
            d = json.loads(t)
            valid = isinstance(d, dict) and "tool" in d and "params" in d
            scores.append(1.0 if valid else 0.3)
        except Exception:
            scores.append(0.0)
    return scores


def thought_reward(completions, **kwargs):
    """Small bonus for including a 'thought' key — encourages chain-of-thought."""
    scores = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        try:
            t = text.strip().lstrip("```json").rstrip("```")
            d = json.loads(t)
            scores.append(0.15 if isinstance(d.get("thought"), str) and d["thought"].strip() else 0.0)
        except Exception:
            scores.append(0.0)
    return scores


def drift_detection_reward(completions, **kwargs):
    """Bonus when model correctly calls detect_schema_change with a valid module."""
    scores = []
    valid_modules = {"calendar", "email", "restaurant"}
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        try:
            t = text.strip().lstrip("```json").rstrip("```")
            d = json.loads(t)
            is_detect = d.get("tool") == "detect_schema_change"
            module_ok = d.get("params", {}).get("module") in valid_modules
            scores.append(0.35 if is_detect and module_ok else 0.0)
        except Exception:
            scores.append(0.0)
    return scores


def preference_first_reward(completions, **kwargs):
    """Bonus for calling get_user_prefs — signals the model learned to read context."""
    scores = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        try:
            t = text.strip().lstrip("```json").rstrip("```")
            d = json.loads(t)
            scores.append(0.15 if d.get("tool") == "get_user_prefs" else 0.0)
        except Exception:
            scores.append(0.0)
    return scores


def env_reward(completions, prompts, **kwargs):
    """Main reward: calls the live AdaptAssist HF Space for each completion."""
    env_url = os.environ.get("ADAPTASSIST_ENV_URL", "https://souravdanyal-adaptassist-env.hf.space")
    scores = []
    for completion, prompt in zip(completions, prompts):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        try:
            t = text.strip()
            if t.startswith("```"):
                t = t.split("\n", 1)[1].rsplit("```", 1)[0]
            action = json.loads(t)

            # Infer difficulty from prompt text
            prompt_str = str(prompt).lower()
            difficulty = "easy"
            for d in ["hard", "medium"]:
                if d in prompt_str:
                    difficulty = d
                    break

            requests.post(f"{env_url}/reset", json={"difficulty": difficulty}, timeout=15)
            result = requests.post(
                f"{env_url}/step",
                json={"tool": action.get("tool", "finish"),
                      "params": action.get("params", {}),
                      "thought": action.get("thought", "")},
                timeout=15,
            ).json()
            scores.append(float(result.get("reward", 0.0)))
        except Exception:
            scores.append(0.0)
        time.sleep(0.05)
    return scores


# ── Curriculum helper ─────────────────────────────────────────────────────────

def build_grpo_dataset(sft_json_path: str, difficulty_filter: str | None = None) -> Dataset:
    """Build prompt dataset from SFT data, optionally filtering by difficulty tag."""
    with open(sft_json_path) as f:
        raw = json.load(f)

    samples = []
    for s in raw:
        msgs = s.get("messages", [])
        if len(msgs) < 2:
            continue
        if difficulty_filter:
            content = " ".join(m.get("content", "") for m in msgs).lower()
            if difficulty_filter not in content:
                continue
        samples.append({"prompt": msgs[:2]})

    if not samples:
        # fallback: use all data
        samples = [{"prompt": s["messages"][:2]} for s in raw if len(s.get("messages", [])) >= 2]

    ds = Dataset.from_list(samples)
    return ds.train_test_split(test_size=0.05, seed=42)["train"]


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("checkpoints/grpo", exist_ok=True)

    if args.env_url:
        os.environ["ADAPTASSIST_ENV_URL"] = args.env_url

    print(f"Loading model from {args.sft_checkpoint} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.sft_checkpoint,
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

    sft_data = "data/combined_sft.json"
    if not os.path.exists(sft_data):
        raise FileNotFoundError(f"SFT dataset not found at {sft_data}. Run data/generate_sft_data.py first.")

    reward_log = []

    # ── Curriculum: train easy → medium → hard ────────────────────────────────
    stages = [("easy", args.steps // 3), ("medium", args.steps // 3), ("hard", args.steps - 2 * (args.steps // 3))] \
        if args.curriculum else [("mixed", args.steps)]

    for stage_name, stage_steps in stages:
        print(f"\n{'='*60}")
        print(f"Stage: {stage_name} ({stage_steps} steps)")
        print("=" * 60)

        diff_filter = None if stage_name == "mixed" else stage_name
        train_ds = build_grpo_dataset(sft_data, diff_filter)
        print(f"  Dataset size: {len(train_ds)} prompts")

        grpo_cfg = GRPOConfig(
            learning_rate=args.lr,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=4,
            max_prompt_length=1024,
            max_completion_length=512,
            max_steps=stage_steps,
            save_steps=max(10, stage_steps // 3),
            logging_steps=5,
            output_dir=f"checkpoints/grpo/{stage_name}",
            report_to="none",
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[format_reward, env_reward, drift_detection_reward,
                          preference_first_reward, thought_reward],
            args=grpo_cfg,
            train_dataset=train_ds,
        )

        trainer.train()

        # Save reward log for this stage
        log = trainer.state.log_history
        for entry in log:
            entry["stage"] = stage_name
        reward_log.extend(log)

        log_path = "artifacts/grpo_reward_log.json"
        with open(log_path, "w") as f:
            json.dump(reward_log, f, indent=2)
        print(f"  Reward log saved to {log_path}")

    # Save final model
    final_path = "checkpoints/grpo/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nFinal model saved to {final_path}")

    # Print summary
    print("\n=== TRAINING SUMMARY ===")
    for key in ["rewards/env_reward", "rewards/format_reward",
                "rewards/drift_detection_reward"]:
        vals = [l[key] for l in reward_log if key in l]
        if vals:
            n = len(vals)
            init = sum(vals[:max(1, n//10)]) / max(1, n//10)
            final = sum(vals[-(n//10):]) / max(1, n//10)
            print(f"  {key:<40s}  init={init:.4f}  final={final:.4f}  delta={final-init:+.4f}")

    return reward_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_checkpoint", default="checkpoints/sft")
    parser.add_argument("--env_url", default="https://souravdanyal-adaptassist-env.hf.space")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--curriculum", action="store_true", default=True)
    args = parser.parse_args()
    train(args)