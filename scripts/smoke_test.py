"""
scripts/smoke_test.py

Fast no-GPU check: imports env, runs 3 episodes across all difficulties,
checks rewards are sane. Use before every commit.

Usage:
    python scripts/smoke_test.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.env import AdaptAssistEnv
import random

PASS = "✅"
FAIL = "❌"


def check(cond: bool, msg: str):
    print(f"  {PASS if cond else FAIL}  {msg}")
    if not cond:
        raise AssertionError(msg)


def run_smoke():
    print("AdaptAssist Smoke Test\n" + "=" * 40)

    # 1. Import check
    try:
        env = AdaptAssistEnv(difficulty=1, seed=0)
        print(f"{PASS}  Import OK")
    except Exception as e:
        print(f"{FAIL}  Import FAILED: {e}")
        sys.exit(1)

    # 2. Reset check
    obs = env.reset()
    check("tasks" in obs, "obs has 'tasks'")
    check("available_tools" in obs, "obs has 'available_tools'")
    check("drift_warning" in obs, "obs has 'drift_warning'")
    check(obs["step"] == 0, "step is 0 after reset")
    check(len(obs["tasks"]) == 3, "3 tasks generated")
    print()

    # 3. Step check
    obs, reward, done, info = env.step({"tool": "get_user_prefs", "params": {}})
    check(isinstance(reward, float), "reward is float")
    check(-1.0 <= reward <= 1.0, f"reward {reward} in [-1, 1]")
    check(not done, "not done after first step")
    check("tasks_completed" in info, "info has tasks_completed")
    print()

    # 4. All difficulties
    for diff_name, diff_int in [("easy", 1), ("medium", 2), ("hard", 3)]:
        env2 = AdaptAssistEnv(difficulty=diff_int, max_steps=15, seed=42)
        obs2 = env2.reset()
        total_r = 0.0
        for _ in range(10):
            act = {"tool": random.choice(AdaptAssistEnv.TOOLS[:5]), "params": {}}
            obs2, r, done2, _ = env2.step(act)
            total_r += r
            if done2:
                break
        check(-10 < total_r < 10, f"{diff_name}: cumulative reward {total_r:.3f} is sane")

    print()

    # 5. Drift detection
    env3 = AdaptAssistEnv(difficulty=2, seed=1)
    env3.reset()
    env3._active_drifts = ["email_field_rename"]
    _, r, _, _ = env3.step({"tool": "detect_schema_change", "params": {"module": "email"}})
    check(r > 0, f"correct drift detection rewarded (got {r:.3f})")

    # 6. Anti-gaming
    env4 = AdaptAssistEnv(difficulty=1, seed=3)
    env4.reset()
    _, r1, _, _ = env4.step({"tool": "get_user_prefs", "params": {}})
    _, r2, _, _ = env4.step({"tool": "get_user_prefs", "params": {}})
    check(r2 <= r1, f"repeated call penalised (r1={r1:.3f} r2={r2:.3f})")

    # 7. Episode summary
    env5 = AdaptAssistEnv(difficulty=1, seed=5)
    env5.reset()
    env5.step({"tool": "finish", "params": {}})
    summary = env5.get_episode_summary()
    check("task_completion_rate" in summary, "summary has task_completion_rate")
    check("schema_adaptation_rate" in summary, "summary has schema_adaptation_rate")

    print()
    print("=" * 40)
    print(f"{PASS}  All smoke tests passed.")


if __name__ == "__main__":
    run_smoke()