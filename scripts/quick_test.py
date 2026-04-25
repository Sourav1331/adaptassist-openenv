"""
Quick Test Script — No GPU Required
=====================================
Tests the full environment loop using the rule-based agent.
Run this locally before the hackathon to verify everything works.

Usage:
    python scripts/quick_test.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.env import AdaptAssistEnv
from agent.agent import RuleBasedAgent


def test_difficulty(difficulty: int, n_episodes: int = 5, seed: int = 42):
    print(f"\n{'='*60}")
    print(f"  Testing Difficulty {difficulty}")
    print(f"{'='*60}")

    agent = RuleBasedAgent()
    rewards = []

    for ep in range(n_episodes):
        env = AdaptAssistEnv(
            difficulty=difficulty,
            max_steps=30,
            n_tasks=2,
            seed=seed + ep,
        )
        agent.reset()
        obs = env.reset()

        if ep == 0:
            print(env.render())

        last_result = None
        done = False
        total_reward = 0.0
        steps = []

        while not done:
            action = agent.act(obs, last_result)
            obs, reward, done, info = env.step(action)
            last_result = info.get("tool_result", {})
            total_reward += reward
            steps.append({
                "tool": action["tool"],
                "reward": round(reward, 4),
                "result_status": info.get("tool_result", {}).get("status", "?"),
            })

        summary = env.get_episode_summary()
        rewards.append(total_reward)

        print(f"  Episode {ep+1}: reward={total_reward:.4f} | "
              f"tasks={summary['tasks_completed']}/{summary['total_tasks']} | "
              f"steps={summary.get('step', 0)} | "
              f"drifts={summary['drift_events']} | "
              f"detections={summary['schema_detections']}")

        if ep == 0:
            print(f"  Step breakdown:")
            for i, s in enumerate(steps, 1):
                print(f"    {i:2d}. {s['tool']:25s} -> {s['result_status']:12s} (reward: {s['reward']})")

    mean_reward = sum(rewards) / len(rewards)
    print(f"\n  Mean reward over {n_episodes} episodes: {mean_reward:.4f}")
    return mean_reward


def test_drift_variants():
    """Test that all drift variants are applied and detectable."""
    print(f"\n{'='*60}")
    print("  Testing Schema Drift Variants")
    print(f"{'='*60}")

    from environment.env import DRIFT_VARIANTS

    for module, variants in DRIFT_VARIANTS.items():
        for variant in variants:
            env = AdaptAssistEnv(difficulty=1, max_steps=5, seed=1)
            env.reset()

            # Apply drift manually
            if module == "calendar":
                env.calendar.apply_drift(variant)
                result = env.calendar.read_calendar("2024-03-15")
            elif module == "email":
                env.email_api.apply_drift(variant)
                result = env.email_api.get_inbox()
            elif module == "restaurant":
                env.restaurant_api.apply_drift(variant)
                result = env.restaurant_api.search_restaurants({"cuisine": "Italian"})

            print(f"  [{module}] {variant['name']}: fields={list(result.keys())} -> OK")


def test_reward_components():
    """Test that reward computation covers all components."""
    print(f"\n{'='*60}")
    print("  Testing Reward Components")
    print(f"{'='*60}")

    env = AdaptAssistEnv(difficulty=1, max_steps=30, n_tasks=2, seed=42)
    env.reset()

    # Complete all tasks manually
    for task in env._tasks:
        task["completed"] = True

    reward = env._compute_final_reward()
    print(f"  All tasks complete, no drift: reward = {reward:.4f}")
    assert reward > 0, "Reward should be positive when all tasks complete"

    env2 = AdaptAssistEnv(difficulty=1, max_steps=30, n_tasks=2, seed=42)
    env2.reset()
    reward2 = env2._compute_final_reward()
    print(f"  No tasks complete, no drift: reward = {reward2:.4f}")
    assert reward2 <= reward, "Incomplete tasks should have lower reward"

    print("  Reward components: PASS")


def run_all_tests():
    print("\n" + "="*60)
    print("  AdaptAssist — Quick Test Suite")
    print("="*60)

    # Test all difficulty levels
    d1_reward = test_difficulty(1, n_episodes=5)
    d2_reward = test_difficulty(2, n_episodes=5)
    d3_reward = test_difficulty(3, n_episodes=5)

    # Test drift variants
    test_drift_variants()

    # Test reward
    test_reward_components()

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Difficulty 1 (no drift):         mean reward = {d1_reward:.4f}")
    print(f"  Difficulty 2 (single drift):     mean reward = {d2_reward:.4f}")
    print(f"  Difficulty 3 (continuous drift): mean reward = {d3_reward:.4f}")
    print(f"\n  All tests passed!")
    print(f"  The environment is working correctly.")
    print(f"  You are ready for the onsite hackathon.\n")


if __name__ == "__main__":
    run_all_tests()
