"""tests/test_env_basic.py — Basic reset / step / done tests."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.env import AdaptAssistEnv


def make_env(difficulty=1, seed=42):
    env = AdaptAssistEnv(difficulty=difficulty, max_steps=10, n_tasks=3, seed=seed)
    return env


class TestReset:
    def test_reset_returns_observation(self):
        env = make_env()
        obs = env.reset()
        assert isinstance(obs, dict)
        assert "tasks" in obs
        assert "available_tools" in obs
        assert "drift_warning" in obs
        assert "step" in obs

    def test_reset_step_is_zero(self):
        env = make_env()
        obs = env.reset()
        assert obs["step"] == 0

    def test_reset_has_tasks(self):
        env = make_env()
        obs = env.reset()
        assert len(obs["tasks"]) == 3

    def test_reset_clears_history(self):
        env = make_env()
        env.reset()
        env.step({"tool": "get_user_prefs", "params": {}})
        env.reset()
        obs = env.observation()
        assert obs["step"] == 0


class TestStep:
    def test_step_returns_tuple(self):
        env = make_env()
        env.reset()
        obs, reward, done, info = env.step({"tool": "get_user_prefs", "params": {}})
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_unknown_tool_negative_reward(self):
        env = make_env()
        env.reset()
        _, reward, _, _ = env.step({"tool": "fly_to_moon", "params": {}})
        assert reward < 0

    def test_finish_ends_episode(self):
        env = make_env()
        env.reset()
        _, _, done, _ = env.step({"tool": "finish", "params": {}})
        assert done is True

    def test_step_after_done_no_crash(self):
        env = make_env()
        env.reset()
        env.step({"tool": "finish", "params": {}})
        obs, reward, done, info = env.step({"tool": "get_user_prefs", "params": {}})
        assert done is True

    def test_max_steps_ends_episode(self):
        env = make_env()
        env.reset()
        done = False
        for _ in range(15):  # more than max_steps=10
            _, _, done, _ = env.step({"tool": "get_user_prefs", "params": {}})
        assert done is True

    def test_repeated_calls_penalised(self):
        env = make_env()
        env.reset()
        _, r1, _, _ = env.step({"tool": "get_user_prefs", "params": {}})
        _, r2, _, _ = env.step({"tool": "get_user_prefs", "params": {}})
        # second identical call should have lower reward
        assert r2 < r1


class TestState:
    def test_state_returns_dict(self):
        env = make_env()
        env.reset()
        s = env.state()
        assert "difficulty" in s
        assert "step" in s
        assert "tasks_completed" in s
        assert "done" in s

    def test_observation_matches_reset(self):
        env = make_env()
        obs_reset = env.reset()
        obs_get = env.observation()
        assert obs_reset["step"] == obs_get["step"]
        assert len(obs_reset["tasks"]) == len(obs_get["tasks"])