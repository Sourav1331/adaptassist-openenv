"""
scripts/generate_demo_transcript.py

Generates a markdown transcript showing a full episode with the
drift_aware_expert policy. Used as the "after training" comparison target.

Usage:
    python scripts/generate_demo_transcript.py \
        --task-type book_restaurant \
        --difficulty medium \
        --seed 7 \
        --output artifacts/demo_transcript.md
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.env import AdaptAssistEnv


def _drift_aware_expert(obs: dict, rng: random.Random) -> dict:
    history_tools = [h["tool"] for h in obs.get("history", [])]

    if "get_user_prefs" not in history_tools:
        return {"tool": "get_user_prefs", "params": {},
                "thought": "I should first read user preferences before acting."}

    if obs.get("drift_warning") and obs.get("detected_drift_count", 0) < obs.get("active_drift_count", 0):
        return {"tool": "detect_schema_change", "params": {"module": "restaurant"},
                "thought": "Drift warning is active. Let me detect which module changed."}

    incomplete = [t for t in obs["tasks"] if not t["completed"]]
    if not incomplete:
        return {"tool": "finish", "params": {}, "thought": "All tasks completed."}

    task_type = incomplete[0]["type"]

    if task_type == "book_restaurant":
        if "search_restaurants" not in history_tools:
            return {"tool": "search_restaurants",
                    "params": {"cuisine": "Italian"},
                    "thought": "User prefers Italian. Searching for restaurants."}
        return {"tool": "book_restaurant",
                "params": {"restaurant_id": "rst_0", "time_slot": "19:00"},
                "thought": "Found a matching restaurant. Booking."}

    if task_type == "reply_email":
        if "get_inbox" not in history_tools:
            return {"tool": "get_inbox", "params": {},
                    "thought": "Need to see inbox first."}
        return {"tool": "send_reply",
                "params": {"email_id": "eml_0", "body": "Thank you for reaching out. I'll review and follow up.",
                           "tone": "professional"},
                "thought": "Replying with professional tone as per user preference."}

    return {"tool": "finish", "params": {}, "thought": "Episode complete."}


def generate(difficulty: int, seed: int, output: str):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    rng = random.Random(seed)

    env = AdaptAssistEnv(difficulty=difficulty, max_steps=20, n_tasks=2, seed=seed)
    obs = env.reset()

    diff_names = {1: "easy", 2: "medium", 3: "hard"}
    lines = [
        "# AdaptAssist — Episode Demo Transcript",
        "",
        f"**Difficulty**: {diff_names[difficulty]}  ",
        f"**Seed**: {seed}  ",
        f"**Active drifts**: {env._active_drifts}  ",
        "",
        "## Tasks",
        "",
    ]
    for t in obs["tasks"]:
        lines.append(f"- `{t['id']}` {t['description']}")
    lines += ["", "---", ""]

    total_reward = 0.0

    for step_num in range(20):
        action = _drift_aware_expert(obs, rng)
        thought = action.pop("thought", "")

        lines += [
            f"### Step {step_num + 1} — `{action['tool']}`",
            "",
        ]
        if thought:
            lines += [f"**🧠 Thought**: {thought}", ""]

        lines += [
            f"**Action**: `{action['tool']}`  ",
            f"**Params**: `{json.dumps(action['params'])}`  ",
            "",
        ]

        obs, reward, done, info = env.step(action)
        total_reward += reward
        tool_result = info.get("tool_result", {})

        lines += [
            f"**Result**: `{json.dumps(tool_result)[:200]}`  ",
            f"**Reward**: `{reward:+.4f}`  ",
            f"**Tasks completed**: {info.get('tasks_completed', 0)}/{info.get('total_tasks', 0)}  ",
            "",
        ]

        if done:
            break

    summary = env.get_episode_summary()
    lines += [
        "---",
        "",
        "## Episode Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total reward | `{total_reward:+.4f}` |",
        f"| Task completion rate | `{summary['task_completion_rate']:.2f}` |",
        f"| Schema adaptation rate | `{summary['schema_adaptation_rate']:.2f}` |",
        f"| Preference alignment | `{summary['preference_alignment_rate']:.2f}` |",
        f"| Steps used | `{summary['steps_used']}` |",
        "",
    ]

    with open(output, "w") as f:
        f.write("\n".join(lines))
    print(f"Transcript saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts/demo_transcript.md")
    args = parser.parse_args()
    generate(args.difficulty, args.seed, args.output)