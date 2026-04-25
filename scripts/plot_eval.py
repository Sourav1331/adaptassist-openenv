"""
scripts/plot_eval.py

Reads artifacts/baseline_eval.json and plots:
  1. Mean reward per policy per difficulty (grouped bar chart)
  2. Drift adaptation rate per policy (line chart)

Usage:
    python scripts/plot_eval.py --input artifacts/baseline_eval.json --output plots/baseline_rewards.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(input_path: str, output_path: str):
    with open(input_path) as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    policies = list(data.keys())
    difficulties = ["easy", "medium", "hard"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    rewards = {p: [data[p][d]["mean_reward"] for d in difficulties] for p in policies}
    drift_adapt = {p: [data[p][d]["mean_drift_adaptation"] for d in difficulties] for p in policies}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("AdaptAssist — Baseline Policy Comparison", fontsize=14, fontweight="bold")

    # ── Plot 1: Mean reward grouped bar ───────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(difficulties))
    width = 0.18
    for i, (policy, color) in enumerate(zip(policies, colors)):
        bars = ax.bar(x + i * width, rewards[policy], width,
                      label=policy.replace("_", " ").title(), color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:+.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Difficulty", fontsize=11)
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_title("Mean Reward by Difficulty", fontsize=12)
    ax.set_xticks(x + width * (len(policies) - 1) / 2)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(-0.5, 1.0)

    # ── Plot 2: Drift adaptation rate line chart ───────────────────────────────
    ax2 = axes[1]
    for policy, color in zip(policies, colors):
        ax2.plot(difficulties, drift_adapt[policy],
                 marker="o", label=policy.replace("_", " ").title(),
                 color=color, linewidth=2, markersize=7)

    ax2.set_xlabel("Difficulty", fontsize=11)
    ax2.set_ylabel("Drift Adaptation Rate", fontsize=11)
    ax2.set_title("Schema Drift Detection Rate by Difficulty", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_ylim(-0.05, 1.1)
    ax2.axhline(1.0, color="gray", linewidth=0.7, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="artifacts/baseline_eval.json")
    parser.add_argument("--output", default="plots/baseline_rewards.png")
    args = parser.parse_args()
    plot(args.input, args.output)