"""tests/test_reward.py — Reward bounds, anti-gaming, episode summary."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.env import AdaptAssistEnv


def run_episode(difficulty=1, seed=0, n_steps=5):
    env = AdaptAssistEnv(difficulty=difficulty, max_steps=20, n_tasks=3, seed=seed)
    env.reset()
    rewards = []
    for _ in range(n_steps):
        _, r, done, _ = env.step({"tool": "get_user_prefs", "params": {}})
        rewards.append(r)
        if done:
            break
    return env, rewards


class TestRewardBounds:
    def test_all_rewards_in_range(self):
        _, rewards = run_episode(n_steps=15)
        for r in rewards:
            assert -1.0 <= r <= 1.0, f"Reward {r} out of [-1, 1]"

    def test_finish_reward_in_range(self):
        env = AdaptAssistEnv(difficulty=2, seed=7)
        env.reset()
        _, reward, _, _ = env.step({"tool": "finish", "params": {}})
        assert -1.0 <= reward <= 1.0


class TestAntiGaming:
    def test_repeated_same_call_decreasing_reward(self):
        env, _ = run_episode(n_steps=1)
        rewards = []
        for _ in range(5):
            _, r, done, _ = env.step({"tool": "get_user_prefs", "params": {}})
            rewards.append(r)
            if done:
                break
        # rewards should not increase with repetition
        assert rewards[-1] <= rewards[0]

    def test_varied_calls_better_than_spam(self):
        env1 = AdaptAssistEnv(difficulty=1, seed=5)
        env1.reset()
        spam_rewards = []
        for _ in range(6):
            _, r, done, _ = env1.step({"tool": "get_user_prefs", "params": {}})
            spam_rewards.append(r)
            if done:
                break

        env2 = AdaptAssistEnv(difficulty=1, seed=5)
        env2.reset()
        varied_rewards = []
        for tool in ["get_user_prefs", "get_inbox", "read_calendar",
                     "get_user_prefs", "get_inbox", "read_calendar"]:
            _, r, done, _ = env2.step({"tool": tool, "params": {}})
            varied_rewards.append(r)
            if done:
                break

        assert sum(varied_rewards) >= sum(spam_rewards)


class TestEpisodeSummary:
    def test_summary_keys_present(self):
        env = AdaptAssistEnv(difficulty=1, seed=3)
        env.reset()
        env.step({"tool": "finish", "params": {}})
        summary = env.get_episode_summary()
        for key in ["task_completion_rate", "schema_adaptation_rate",
                    "preference_alignment_rate", "steps_used",
                    "episode_reward", "total_episodes"]:
            assert key in summary, f"Missing key: {key}"

    def test_completion_rate_between_0_and_1(self):
        env = AdaptAssistEnv(difficulty=1, seed=3)
        env.reset()
        env.step({"tool": "finish", "params": {}})
        summary = env.get_episode_summary()
        assert 0.0 <= summary["task_completion_rate"] <= 1.0

    def test_total_episodes_increments(self):
        env = AdaptAssistEnv(difficulty=1, seed=3)
        for i in range(3):
            env.reset()
            env.step({"tool": "finish", "params": {}})
        assert env._total_episodes == 3