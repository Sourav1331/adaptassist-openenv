"""
scripts/plot_training_curves.py

Reads artifacts/sft_loss_log.json and artifacts/grpo_reward_log.json
and produces two committed plot files:
  - plots/sft_loss.png
  - plots/reward_curves.png   ← the main one judges look at

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --sft artifacts/sft_loss_log.json \
                                           --grpo artifacts/grpo_reward_log.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rolling(vals, w=20):
    return [sum(vals[max(0, i - w):i + 1]) / len(vals[max(0, i - w):i + 1])
            for i in range(len(vals))]


def plot_sft(log_path: str, out_path: str):
    if not os.path.exists(log_path):
        print(f"[skip] SFT log not found at {log_path}")
        return

    with open(log_path) as f:
        log = json.load(f)

    train_steps = [l["step"] for l in log if "loss" in l and "eval_loss" not in l]
    train_loss  = [l["loss"] for l in log if "loss" in l and "eval_loss" not in l]
    eval_steps  = [l["step"] for l in log if "eval_loss" in l]
    eval_loss   = [l["eval_loss"] for l in log if "eval_loss" in l]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_steps, train_loss, color="#4C72B0", linewidth=1.5,
            alpha=0.5, label="Train loss (per step)")
    if len(train_loss) > 5:
        ax.plot(train_steps, rolling(train_loss, 30), color="#4C72B0",
                linewidth=2.5, label="Train loss (rolling avg)")
    if eval_loss:
        ax.plot(eval_steps, eval_loss, color="#C44E52", linewidth=2,
                linestyle="--", marker="o", markersize=5, label="Eval loss")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("AdaptAssist — SFT Phase: Llama 3.2 3B on ~7500 samples")
    ax.legend()
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_grpo(log_path: str, out_path: str):
    if not os.path.exists(log_path):
        print(f"[skip] GRPO log not found at {log_path}")
        return

    with open(log_path) as f:
        log = json.load(f)

    def extract(key):
        pts = [(l["step"], l[key]) for l in log if key in l]
        if not pts:
            return [], []
        steps, vals = zip(*pts)
        return list(steps), list(vals)

    env_steps,   env_vals   = extract("rewards/env_reward")
    fmt_steps,   fmt_vals   = extract("rewards/format_reward")
    drift_steps, drift_vals = extract("rewards/drift_detection_reward")
    pref_steps,  pref_vals  = extract("rewards/preference_first_reward")

    # Stage boundaries for background shading
    stages = {}
    for l in log:
        stage = l.get("stage", "mixed")
        step  = l.get("step", 0)
        if stage not in stages:
            stages[stage] = [step, step]
        else:
            stages[stage][1] = max(stages[stage][1], step)

    stage_colors = {"easy": "#E8F5E9", "medium": "#FFF3E0", "hard": "#FFEBEE", "mixed": "#FFFFFF"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("AdaptAssist v2 — GRPO RL Training Reward Curves\n"
                 "(Llama 3.2 3B, SFT → GRPO, Curriculum: easy→medium→hard)",
                 fontsize=13, fontweight="bold")

    panels = [
        (axes[0, 0], env_steps,   env_vals,   "Environment Reward\n(task completion + drift + preference)", "#1D9E75"),
        (axes[0, 1], fmt_steps,   fmt_vals,   "Format Reward\n(valid JSON with tool+params keys)",          "#378ADD"),
        (axes[1, 0], drift_steps, drift_vals, "Drift Detection Reward\n(detect_schema_change with module)", "#D85A30"),
        (axes[1, 1], pref_steps,  pref_vals,  "Preference Alignment Reward\n(get_user_prefs called first)",  "#8172B2"),
    ]

    for ax, steps, vals, label, color in panels:
        # Stage background shading
        for stage, (s0, s1) in stages.items():
            ax.axvspan(s0, s1, alpha=0.12,
                       color=stage_colors.get(stage, "#FFFFFF"), label=stage)

        if steps:
            ax.scatter(steps, vals, alpha=0.2, s=10, color=color)
            if len(vals) > 5:
                ax.plot(steps, rolling(vals, 20), color=color, linewidth=2.5,
                        label="rolling avg (w=20)")

            # Annotate start and end values
            ax.annotate(f"start: {vals[0]:.3f}", xy=(steps[0], vals[0]),
                        fontsize=8, color=color, xytext=(5, 5),
                        textcoords="offset points")
            ax.annotate(f"end: {vals[-1]:.3f}", xy=(steps[-1], vals[-1]),
                        fontsize=8, color=color, xytext=(-40, 5),
                        textcoords="offset points")
        else:
            ax.text(0.5, 0.5, "No data\n(run training first)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="gray")

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Reward")
        ax.set_title(label, fontsize=10)
        ax.set_ylim(-0.1, 1.15)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft",  default="artifacts/sft_loss_log.json")
    parser.add_argument("--grpo", default="artifacts/grpo_reward_log.json")
    parser.add_argument("--sft_out",  default="plots/sft_loss.png")
    parser.add_argument("--grpo_out", default="plots/reward_curves.png")
    args = parser.parse_args()

    plot_sft(args.sft, args.sft_out)
    plot_grpo(args.grpo, args.grpo_out)