"""
scripts/evaluate_baselines.py

Evaluates 4 baseline policies against AdaptAssist across all difficulties.
Produces artifacts/baseline_eval.json for plotting.

Usage:
    python scripts/evaluate_baselines.py --seeds 10 --output artifacts/baseline_eval.json
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.env import AdaptAssistEnv


# ── Baseline policies ─────────────────────────────────────────────────────────

def policy_random(obs: dict, rng: random.Random) -> dict:
    tool = rng.choice(AdaptAssistEnv.TOOLS)
    params: dict = {}
    if tool == "read_calendar":
        params = {"date": "2026-05-01"}
    elif tool == "detect_schema_change":
        params = {"module": rng.choice(["calendar", "email", "restaurant"])}
    elif tool == "send_reply":
        params = {"email_id": "eml_0", "body": "Noted.", "tone": "casual"}
    elif tool == "book_restaurant":
        params = {"restaurant_id": "rst_0", "time_slot": "19:00"}
    return {"tool": tool, "params": params}


def policy_always_finish(obs: dict, rng: random.Random) -> dict:
    return {"tool": "finish", "params": {}}


def policy_greedy_complete(obs: dict, rng: random.Random) -> dict:
    """Simple rule-based: tries to complete tasks in order without drift awareness."""
    incomplete = [t for t in obs["tasks"] if not t["completed"]]
    if not incomplete:
        return {"tool": "finish", "params": {}}

    task_type = incomplete[0]["type"]
    history_tools = [h["tool"] for h in obs.get("history", [])]

    sequences = {
        "resolve_conflict": ["read_calendar", "delete_event", "create_event", "finish"],
        "reply_email":      ["get_inbox", "send_reply", "finish"],
        "book_restaurant":  ["get_user_prefs", "search_restaurants", "book_restaurant", "finish"],
        "reschedule":       ["read_calendar", "delete_event", "create_event", "finish"],
        "prioritise_day":   ["read_calendar", "finish"],
    }
    seq = sequences.get(task_type, ["finish"])
    for s in seq:
        if s not in history_tools:
            params: dict = {}
            if s == "read_calendar":
                params = {"date": "2026-05-01"}
            elif s == "send_reply":
                params = {"email_id": "eml_0", "body": "Understood, thanks.", "tone": "professional"}
            elif s == "search_restaurants":
                params = {"cuisine": "Italian"}
            elif s == "book_restaurant":
                params = {"restaurant_id": "rst_0", "time_slot": "19:00"}
            elif s == "delete_event":
                params = {"event_id": "evt_0"}
            elif s == "create_event":
                params = {"title": "Rescheduled Meeting", "date": "2026-05-10",
                          "begins_at": "10:00", "ends_at": "11:00"}
            return {"tool": s, "params": params}

    return {"tool": "finish", "params": {}}


def policy_drift_aware_expert(obs: dict, rng: random.Random) -> dict:
    """Expert: checks prefs first, detects drift when warned, completes tasks."""
    history_tools = [h["tool"] for h in obs.get("history", [])]

    # Step 1: always read prefs first
    if "get_user_prefs" not in history_tools:
        return {"tool": "get_user_prefs", "params": {}}

    # Step 2: detect drift if warned and not yet detected
    if obs.get("drift_warning") and obs.get("detected_drift_count", 0) < obs.get("active_drift_count", 0):
        for module in ["calendar", "email", "restaurant"]:
            return {"tool": "detect_schema_change", "params": {"module": module}}

    # Step 3: work on incomplete tasks
    incomplete = [t for t in obs["tasks"] if not t["completed"]]
    if not incomplete:
        return {"tool": "finish", "params": {}}

    task_type = incomplete[0]["type"]

    # Task-specific optimal sequences (drift-aware field names)
    if task_type == "reply_email":
        if "get_inbox" not in history_tools:
            return {"tool": "get_inbox", "params": {}}
        return {"tool": "send_reply",
                "params": {"email_id": "eml_0", "body": "Thank you for reaching out.",
                           "tone": "professional"}}

    if task_type == "book_restaurant":
        if "search_restaurants" not in history_tools:
            return {"tool": "search_restaurants", "params": {"cuisine": "Italian"}}
        return {"tool": "book_restaurant",
                "params": {"restaurant_id": "rst_0", "time_slot": "19:00"}}

    if task_type == "resolve_conflict":
        if "read_calendar" not in history_tools:
            return {"tool": "read_calendar", "params": {"date": "2026-05-01"}}
        if "delete_event" not in history_tools:
            return {"tool": "delete_event", "params": {"event_id": "evt_0"}}
        return {"tool": "create_event",
                "params": {"title": "Rescheduled", "date": "2026-05-02",
                           "begins_at": "14:00", "ends_at": "15:00"}}

    return {"tool": "finish", "params": {}}


POLICIES = {
    "random":              policy_random,
    "always_finish":       policy_always_finish,
    "greedy_complete":     policy_greedy_complete,
    "drift_aware_expert":  policy_drift_aware_expert,
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_episode(policy_fn, difficulty: int, seed: int) -> dict:
    rng = random.Random(seed + difficulty * 1000)
    env = AdaptAssistEnv(difficulty=difficulty, max_steps=30, n_tasks=3, seed=seed)
    obs = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action = policy_fn(obs, rng)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    summary = env.get_episode_summary()
    return {
        "seed": seed,
        "difficulty": difficulty,
        "total_reward": round(total_reward, 4),
        "tasks_completed": summary["task_completion_rate"],
        "drift_adaptation": summary["schema_adaptation_rate"],
        "steps": steps,
    }


def evaluate_all(n_seeds: int, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results = {}

    for policy_name, policy_fn in POLICIES.items():
        results[policy_name] = {}
        for diff_name, diff_int in [("easy", 1), ("medium", 2), ("hard", 3)]:
            episodes = []
            for s in range(n_seeds):
                ep = run_episode(policy_fn, diff_int, seed=s)
                episodes.append(ep)
                print(f"  {policy_name:22s} | {diff_name:6s} | seed {s:2d} | "
                      f"reward={ep['total_reward']:+.3f} | "
                      f"tasks={ep['tasks_completed']:.2f} | "
                      f"drift={ep['drift_adaptation']:.2f}")

            rewards = [e["total_reward"] for e in episodes]
            results[policy_name][diff_name] = {
                "mean_reward": round(sum(rewards) / len(rewards), 4),
                "min_reward":  round(min(rewards), 4),
                "max_reward":  round(max(rewards), 4),
                "mean_tasks_completed": round(
                    sum(e["tasks_completed"] for e in episodes) / len(episodes), 4),
                "mean_drift_adaptation": round(
                    sum(e["drift_adaptation"] for e in episodes) / len(episodes), 4),
                "episodes": episodes,
            }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
    _print_table(results)
    return results


def _print_table(results: dict):
    print("\n" + "=" * 70)
    print(f"{'Policy':22s} | {'easy':>8s} | {'medium':>8s} | {'hard':>8s}")
    print("-" * 70)
    for policy_name, diffs in results.items():
        row = f"{policy_name:22s}"
        for diff in ["easy", "medium", "hard"]:
            mean = diffs.get(diff, {}).get("mean_reward", float("nan"))
            row += f" | {mean:+8.4f}"
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", default="artifacts/baseline_eval.json")
    args = parser.parse_args()

    print(f"Evaluating {len(POLICIES)} policies x 3 difficulties x {args.seeds} seeds ...\n")
    evaluate_all(args.seeds, args.output)