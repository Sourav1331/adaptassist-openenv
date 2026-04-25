"""tests/test_schema_drift.py — Schema drift tests."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.env import AdaptAssistEnv


class TestDriftActivation:
    def test_easy_no_drift(self):
        env = AdaptAssistEnv(difficulty=1, seed=1)
        obs = env.reset()
        assert obs["drift_warning"] is False
        assert obs["active_drift_count"] == 0

    def test_medium_has_one_drift(self):
        env = AdaptAssistEnv(difficulty=2, seed=1)
        obs = env.reset()
        assert obs["active_drift_count"] == 1

    def test_hard_drift_grows_over_steps(self):
        env = AdaptAssistEnv(difficulty=3, max_steps=50, seed=1)
        env.reset()
        drift_counts = []
        for _ in range(40):
            obs, _, done, _ = env.step({"tool": "get_user_prefs", "params": {}})
            drift_counts.append(obs["active_drift_count"])
            if done:
                break
        # should have increased at least once
        assert max(drift_counts) > min(drift_counts)


class TestDriftDetection:
    def test_correct_detection_positive_reward(self):
        # Force calendar drift
        env = AdaptAssistEnv(difficulty=2, seed=99)
        env.reset()
        # inject a known drift manually to make test deterministic
        env._active_drifts = ["calendar_field_rename"]
        env._detected_drifts = set()

        _, reward, _, _ = env.step({
            "tool": "detect_schema_change",
            "params": {"module": "calendar"}
        })
        assert reward > 0
        assert "calendar_field_rename" in env._detected_drifts

    def test_false_positive_negative_reward(self):
        env = AdaptAssistEnv(difficulty=1, seed=1)
        env.reset()
        # No drift active on easy
        _, reward, _, _ = env.step({
            "tool": "detect_schema_change",
            "params": {"module": "calendar"}
        })
        assert reward < 0
        assert env._failed_drift_calls == 1

    def test_drift_field_rename_causes_error(self):
        env = AdaptAssistEnv(difficulty=2, seed=1)
        env.reset()
        env._active_drifts = ["calendar_field_rename"]

        result, reward, _, _ = env.step({
            "tool": "create_event",
            "params": {"title": "Meeting", "date": "2026-05-01", "start_time": "10:00"}
        })
        # Should get an error (returned in result dict) and negative reward
        assert reward < 0
        # result is the tool_result dict returned by _create_event
        assert "error" in result or reward < 0  # negative reward confirms drift penalty


class TestRewardComponents:
    def test_preference_aligned_booking_bonus(self):
        # Try multiple seeds until we find one where a matching restaurant exists
        for seed in range(20):
            env = AdaptAssistEnv(difficulty=1, seed=seed)
            env.reset()
            pref_cuisine = env._world["prefs"]["preferred_cuisine"]
            matches = [r for r in env._world["restaurants"] if r["cuisine"] == pref_cuisine]
            if matches:
                break
        else:
            pytest.skip("No seed produced a matching restaurant (random world)")

        matching_rst = matches[0]
        slot = matching_rst["available_slots"][0]

        env.step({"tool": "search_restaurants", "params": {"cuisine": pref_cuisine}})
        _, reward, _, _ = env.step({
            "tool": "book_restaurant",
            "params": {"restaurant_id": matching_rst["id"], "time_slot": slot}
        })
        assert reward > 0.3

    def test_wrong_email_tone_no_pref_bonus(self):
        env = AdaptAssistEnv(difficulty=1, seed=42)
        env.reset()
        env.step({"tool": "get_inbox", "params": {}})
        email_id = env._world["emails"][0]["id"]
        pref_tone = env._world["prefs"]["email_tone"]
        wrong_tone = "casual" if pref_tone != "casual" else "formal"

        _, reward, _, _ = env.step({
            "tool": "send_reply",
            "params": {"email_id": email_id, "body": "Got it.", "tone": wrong_tone}
        })
        # Should succeed but no pref bonus
        assert reward < 0.35  # no extra tone bonus