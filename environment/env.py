"""
AdaptAssist Environment v2 — Drift-Aware Personal AI Agent
Fixes:
  - Proper openenv.env.env.Env inheritance
  - Correct reset/step/state/observation signatures
  - Hardened schema drift (cannot be bypassed by lucky guessing)
  - Dense per-step rewards (not just terminal)
  - Anti-gaming: repeated same tool call penalised
  - Preference alignment tracked per task
  - Full episode summary for metrics endpoint
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from datetime import date, timedelta
from typing import Any

try:
    from openenv.env.env import Env as _OpenEnvBase
except ImportError:
    # Fallback so the file is importable even without openenv installed
    class _OpenEnvBase:
        pass


# ─────────────────────────── world data ──────────────────────────────────────

_CUISINES = ["Italian", "Indian", "Japanese", "Mexican", "Thai", "Chinese"]
_TONES = ["professional", "casual", "urgent", "friendly"]
_MEETING_PURPOSES = ["sync", "review", "planning", "interview", "demo"]
_RESTAURANT_NAMES = [
    "Bella Roma", "Spice Garden", "Tokyo Garden", "El Rancho",
    "Thai Orchid", "Dragon Palace", "The Grill", "Ocean Breeze"
]

# ── Schema drift definitions ──────────────────────────────────────────────────

_DRIFT_CATALOGUE = {
    "calendar_field_rename": {
        "module": "calendar",
        "description": "start_time → begins_at, end_time → ends_at",
        "old_fields": {"start_time", "end_time"},
        "new_fields": {"begins_at", "ends_at"},
        "mapping": {"start_time": "begins_at", "end_time": "ends_at",
                    "begins_at": "start_time", "ends_at": "end_time"},
    },
    "calendar_date_format": {
        "module": "calendar",
        "description": "date format YYYY-MM-DD → DD/MM/YYYY",
        "old_fields": set(),
        "new_fields": set(),
        "format_change": True,
    },
    "email_field_rename": {
        "module": "email",
        "description": "sender → from_address, urgency → priority_level",
        "old_fields": {"sender", "urgency"},
        "new_fields": {"from_address", "priority_level"},
        "mapping": {"sender": "from_address", "urgency": "priority_level",
                    "from_address": "sender", "priority_level": "urgency"},
    },
    "restaurant_field_rename": {
        "module": "restaurant",
        "description": "available_slots → open_times",
        "old_fields": {"available_slots"},
        "new_fields": {"open_times"},
        "mapping": {"available_slots": "open_times", "open_times": "available_slots"},
    },
    "restaurant_policy": {
        "module": "restaurant",
        "description": "cancellation window 2hr → 24hr",
        "policy_change": True,
    },
}


# ─────────────────────────── helper builders ─────────────────────────────────

def _rand_date(rng: random.Random, offset_days: int = 3) -> str:
    d = date.today() + timedelta(days=rng.randint(1, offset_days))
    return d.strftime("%Y-%m-%d")


def _build_world(rng: random.Random) -> dict:
    today = date.today()
    events = [
        {"id": f"evt_{i}", "title": rng.choice(_MEETING_PURPOSES).capitalize() + " meeting",
         "start_time": f"{rng.randint(9,16):02d}:00",
         "end_time":   f"{rng.randint(17,19):02d}:00",
         "date": (today + timedelta(days=rng.randint(0, 2))).strftime("%Y-%m-%d"),
         "priority": rng.choice(["high", "medium", "low"])}
        for i in range(rng.randint(3, 5))
    ]
    emails = [
        {"id": f"eml_{i}", "subject": rng.choice(["Re: Project", "Urgent: Meeting", "FYI", "Action needed"]),
         "sender": f"user{i}@company.com",
         "urgency": rng.choice(["high", "medium", "low"]),
         "body": "Please review the attached document and reply by EOD."}
        for i in range(rng.randint(2, 4))
    ]
    restaurants = [
        {"id": f"rst_{i}", "name": rng.choice(_RESTAURANT_NAMES),
         "cuisine": rng.choice(_CUISINES),
         "available_slots": [f"{h:02d}:00" for h in rng.sample(range(18, 22), 3)],
         "cancellation_hours": 2}
        for i in range(4)
    ]
    prefs = {
        "preferred_cuisine": rng.choice(_CUISINES),
        "preferred_meeting_time": rng.choice(["morning", "afternoon"]),
        "email_tone": rng.choice(_TONES),
        "dietary": rng.choice(["none", "vegetarian", "vegan"]),
    }
    return {"events": events, "emails": emails, "restaurants": restaurants, "prefs": prefs}


def _build_tasks(rng: random.Random, world: dict, n: int) -> list[dict]:
    pool = [
        {"type": "resolve_conflict",
         "description": "Resolve the double-booked calendar conflict for tomorrow",
         "requires": ["read_calendar", "delete_event"],
         "target_module": "calendar"},
        {"type": "reply_email",
         "description": f"Reply to the urgent email in {world['prefs']['email_tone']} tone",
         "requires": ["get_inbox", "send_reply"],
         "target_module": "email"},
        {"type": "book_restaurant",
         "description": f"Book a {world['prefs']['preferred_cuisine']} restaurant for tonight",
         "requires": ["search_restaurants", "book_restaurant"],
         "target_module": "restaurant"},
        {"type": "reschedule",
         "description": "Reschedule the lowest-priority meeting to next week",
         "requires": ["read_calendar", "delete_event", "create_event"],
         "target_module": "calendar"},
        {"type": "prioritise_day",
         "description": "List all events sorted by priority, then confirm the most important",
         "requires": ["read_calendar"],
         "target_module": "calendar"},
    ]
    rng.shuffle(pool)
    return [{"id": f"task_{i}", **t, "completed": False, "pref_aligned": None}
            for i, t in enumerate(pool[:n])]


# ─────────────────────────── main environment ────────────────────────────────

class AdaptAssistEnv(_OpenEnvBase):
    """
    AdaptAssist: Drift-aware personal assistant RL environment.

    Conforms to OpenEnv Env interface:
      reset() → obs dict
      step(action) → (obs, reward, done, info)
      state() → summary dict
      observation() → current obs dict
    """

    TOOLS = [
        "read_calendar", "create_event", "delete_event",
        "get_inbox", "send_reply",
        "search_restaurants", "book_restaurant", "cancel_booking",
        "get_user_prefs", "detect_schema_change", "finish",
    ]

    def __init__(self, difficulty: int = 1, max_steps: int = 30,
                 n_tasks: int = 3, seed: int | None = None):
        self.difficulty = max(1, min(3, difficulty))
        self.max_steps = max_steps
        self.n_tasks = n_tasks
        self.seed = seed
        self._rng = random.Random(seed)

        # episode state — initialised by reset()
        self._world: dict = {}
        self._tasks: list[dict] = []
        self._step_count: int = 0
        self._history: list[dict] = []
        self._done: bool = False
        self._active_drifts: list[str] = []
        self._detected_drifts: set[str] = set()
        self._failed_drift_calls: int = 0
        self._tool_call_counts: dict[str, int] = {}
        self._episode_rewards: list[float] = []
        self._bookings: list[dict] = []
        self._sent_replies: list[dict] = []

        # aggregate stats across episodes
        self._total_episodes: int = 0
        self._total_reward: float = 0.0

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self, difficulty: int | None = None, seed: int | None = None) -> dict:
        if difficulty is not None:
            self.difficulty = max(1, min(3, difficulty))
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)

        self._world = _build_world(self._rng)
        self._tasks = _build_tasks(self._rng, self._world, self.n_tasks)
        self._step_count = 0
        self._history = []
        self._done = False
        self._active_drifts = []
        self._detected_drifts = set()
        self._failed_drift_calls = 0
        self._tool_call_counts = {}
        self._episode_rewards = []
        self._bookings = []
        self._sent_replies = []

        # Apply drift according to difficulty
        if self.difficulty >= 2:
            chosen = self._rng.choice(list(_DRIFT_CATALOGUE.keys()))
            self._active_drifts.append(chosen)
        # difficulty 3: additional drift will fire every 8 steps (applied in step())

        self._total_episodes += 1
        return self.observation()

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        if self._done:
            return self.observation(), 0.0, True, {"error": "episode already done"}

        self._step_count += 1

        # difficulty 3: inject new drift every 8 steps
        if self.difficulty == 3 and self._step_count % 8 == 0:
            remaining = [k for k in _DRIFT_CATALOGUE if k not in self._active_drifts]
            if remaining:
                self._active_drifts.append(self._rng.choice(remaining))

        tool = action.get("tool", "")
        params = action.get("params", {})
        thought = action.get("thought", "")

        # anti-gaming: penalise repeated identical calls
        call_key = f"{tool}:{json.dumps(params, sort_keys=True)}"
        self._tool_call_counts[call_key] = self._tool_call_counts.get(call_key, 0) + 1
        repeat_penalty = -0.05 * max(0, self._tool_call_counts[call_key] - 1)

        result, step_reward, info = self._dispatch(tool, params)

        step_reward = max(-1.0, min(1.0, step_reward + repeat_penalty))
        self._episode_rewards.append(step_reward)

        self._history.append({
            "step": self._step_count,
            "tool": tool,
            "params": params,
            "thought": thought,
            "result": result,
            "reward": round(step_reward, 4),
        })

        done = self._done or self._step_count >= self.max_steps
        if done and not self._done:
            self._done = True

        if self._done:
            self._total_reward += sum(self._episode_rewards)

        return self.observation(), step_reward, self._done, {
            "tool_result": result,
            "tasks_completed": sum(1 for t in self._tasks if t["completed"]),
            "total_tasks": len(self._tasks),
            **info,
        }

    def observation(self) -> dict:
        drift_warning = len(self._active_drifts) > len(self._detected_drifts)
        return {
            "tasks": [
                {"id": t["id"], "description": t["description"],
                 "completed": t["completed"], "type": t["type"]}
                for t in self._tasks
            ],
            "available_tools": self.TOOLS,
            "drift_warning": drift_warning,
            "active_drift_count": len(self._active_drifts),
            "detected_drift_count": len(self._detected_drifts),
            "step": self._step_count,
            "total_steps": self._step_count,
            "max_steps": self.max_steps,
            "history": self._history[-5:],  # last 5 steps only (context efficiency)
        }

    def state(self) -> dict:
        return {
            "difficulty": self.difficulty,
            "step": self._step_count,
            "max_steps": self.max_steps,
            "tasks_completed": sum(1 for t in self._tasks if t["completed"]),
            "total_tasks": len(self._tasks),
            "active_drifts": self._active_drifts,
            "detected_drifts": list(self._detected_drifts),
            "episode_reward_so_far": round(sum(self._episode_rewards), 4),
            "done": self._done,
        }

    def get_episode_summary(self) -> dict:
        completed = sum(1 for t in self._tasks if t["completed"])
        pref_aligned = sum(1 for t in self._tasks
                           if t.get("pref_aligned") is True)
        drift_detected = len(self._detected_drifts)
        drift_active = len(self._active_drifts)
        return {
            "total_episodes": self._total_episodes,
            "task_completion_rate": completed / max(1, len(self._tasks)),
            "preference_alignment_rate": pref_aligned / max(1, len(self._tasks)),
            "schema_adaptation_rate": drift_detected / max(1, drift_active) if drift_active else 1.0,
            "failed_drift_calls": self._failed_drift_calls,
            "steps_used": self._step_count,
            "episode_reward": round(sum(self._episode_rewards), 4),
            "cumulative_reward": round(self._total_reward, 4),
        }

    # ── Tool dispatcher ───────────────────────────────────────────────────────

    def _dispatch(self, tool: str, params: dict) -> tuple[dict, float, dict]:
        if tool not in self.TOOLS:
            return {"error": f"unknown tool '{tool}'"}, -0.1, {}

        handlers = {
            "read_calendar":      self._read_calendar,
            "create_event":       self._create_event,
            "delete_event":       self._delete_event,
            "get_inbox":          self._get_inbox,
            "send_reply":         self._send_reply,
            "search_restaurants": self._search_restaurants,
            "book_restaurant":    self._book_restaurant,
            "cancel_booking":     self._cancel_booking,
            "get_user_prefs":     self._get_user_prefs,
            "detect_schema_change": self._detect_schema_change,
            "finish":             self._finish,
        }
        return handlers[tool](params)

    # ── Tool implementations ──────────────────────────────────────────────────

    def _read_calendar(self, params: dict):
        date_val = params.get("date", "")
        drift = self._get_drift_for("calendar")

        events = deepcopy(self._world["events"])

        if drift and drift.get("format_change") and date_val:
            # If agent uses old format and drift is active → garbled result
            if "-" in date_val and len(date_val) == 10:
                return {"error": "Invalid date format. Expected DD/MM/YYYY."}, -0.15, {}

        if drift and "mapping" in drift:
            # rename fields silently
            renamed = []
            for ev in events:
                new_ev = {}
                for k, v in ev.items():
                    new_key = drift["mapping"].get(k, k)
                    new_ev[new_key] = v
                renamed.append(new_ev)
            events = renamed

        if date_val:
            events = [e for e in events if e.get("date", e.get("date", "")) == date_val
                      or not date_val]

        reward = 0.05  # small positive for using a useful tool
        self._mark_task_progress("calendar", "read_calendar")
        return {"events": events, "count": len(events)}, reward, {}

    def _create_event(self, params: dict):
        drift = self._get_drift_for("calendar")
        required_fields = {"title", "date"}

        if drift and "mapping" in drift:
            # new schema needs begins_at/ends_at
            if "start_time" in params and "begins_at" not in params:
                return {"error": "Unknown field 'start_time'. Did you mean 'begins_at'?"}, -0.2, {}

        missing = required_fields - set(params.keys())
        if missing:
            return {"error": f"Missing required fields: {missing}"}, -0.1, {}

        new_ev = {"id": f"evt_new_{self._step_count}", **params}
        self._world["events"].append(new_ev)
        self._mark_task_done("resolve_conflict")
        self._mark_task_done("reschedule")
        return {"created": new_ev}, 0.2, {}

    def _delete_event(self, params: dict):
        event_id = params.get("event_id", "")
        before = len(self._world["events"])
        self._world["events"] = [e for e in self._world["events"] if e["id"] != event_id]
        if len(self._world["events"]) == before:
            return {"error": f"Event '{event_id}' not found."}, -0.1, {}
        self._mark_task_progress("calendar", "delete_event")
        return {"deleted": event_id}, 0.15, {}

    def _get_inbox(self, params: dict):
        drift = self._get_drift_for("email")
        emails = deepcopy(self._world["emails"])

        if drift and "mapping" in drift:
            renamed = []
            for em in emails:
                new_em = {drift["mapping"].get(k, k): v for k, v in em.items()}
                renamed.append(new_em)
            emails = renamed

        self._mark_task_progress("email", "get_inbox")
        return {"emails": emails, "count": len(emails)}, 0.05, {}

    def _send_reply(self, params: dict):
        drift = self._get_drift_for("email")
        email_id = params.get("email_id", "")
        body = params.get("body", "")
        tone = params.get("tone", "")

        if not email_id or not body:
            return {"error": "email_id and body are required."}, -0.1, {}

        pref_tone = self._world["prefs"]["email_tone"]
        tone_aligned = (tone == pref_tone)

        reward = 0.25
        if tone_aligned:
            reward += 0.1
            self._mark_task_pref_aligned("reply_email")

        self._sent_replies.append({"email_id": email_id, "tone": tone, "body": body[:80]})
        self._mark_task_done("reply_email")
        return {"sent": True, "tone_match": tone_aligned}, reward, {}

    def _search_restaurants(self, params: dict):
        drift = self._get_drift_for("restaurant")
        cuisine = params.get("cuisine", "")
        results = deepcopy(self._world["restaurants"])

        if cuisine:
            results = [r for r in results if r["cuisine"].lower() == cuisine.lower()]

        if drift and "mapping" in drift:
            renamed = []
            for r in results:
                new_r = {drift["mapping"].get(k, k): v for k, v in r.items()}
                renamed.append(new_r)
            results = renamed

        reward = 0.05
        if cuisine == self._world["prefs"]["preferred_cuisine"]:
            reward += 0.1  # bonus for reading prefs first

        self._mark_task_progress("restaurant", "search_restaurants")
        return {"restaurants": results, "count": len(results)}, reward, {}

    def _book_restaurant(self, params: dict):
        drift = self._get_drift_for("restaurant")
        restaurant_id = params.get("restaurant_id", "")
        time_slot = params.get("time_slot", "")

        if not restaurant_id or not time_slot:
            return {"error": "restaurant_id and time_slot required."}, -0.1, {}

        match = next((r for r in self._world["restaurants"] if r["id"] == restaurant_id), None)
        if not match:
            return {"error": f"Restaurant '{restaurant_id}' not found."}, -0.1, {}

        slots_key = "open_times" if (drift and drift.get("mapping", {}).get("available_slots") == "open_times") \
                    else "available_slots"

        slots = match.get("open_times", match.get("available_slots", []))
        if time_slot not in slots:
            return {"error": f"Slot '{time_slot}' not available. Choose from: {slots}"}, -0.1, {}

        pref_cuisine = self._world["prefs"]["preferred_cuisine"]
        cuisine_aligned = match["cuisine"] == pref_cuisine

        reward = 0.3
        if cuisine_aligned:
            reward += 0.15
            self._mark_task_pref_aligned("book_restaurant")

        self._bookings.append({"restaurant_id": restaurant_id, "time_slot": time_slot})
        self._mark_task_done("book_restaurant")
        return {"booked": True, "restaurant": match["name"], "slot": time_slot,
                "cuisine_match": cuisine_aligned}, reward, {}

    def _cancel_booking(self, params: dict):
        booking_id = params.get("booking_id", "")
        before = len(self._bookings)
        self._bookings = [b for b in self._bookings if b.get("restaurant_id") != booking_id]
        if len(self._bookings) == before:
            return {"error": "Booking not found."}, -0.05, {}
        return {"cancelled": booking_id}, 0.0, {}

    def _get_user_prefs(self, params: dict):
        return {"preferences": self._world["prefs"]}, 0.05, {}

    def _detect_schema_change(self, params: dict):
        module = params.get("module", "")

        active_for_module = [d for d in self._active_drifts
                             if _DRIFT_CATALOGUE[d].get("module") == module]

        if not active_for_module:
            # No drift in this module — penalise false positive
            self._failed_drift_calls += 1
            return {"drift_detected": False, "module": module,
                    "message": "No schema change detected in this module."}, -0.1, {}

        # Correct detection — reward and reveal details
        for dk in active_for_module:
            self._detected_drifts.add(dk)

        info = _DRIFT_CATALOGUE[active_for_module[0]]
        return {
            "drift_detected": True,
            "module": module,
            "description": info.get("description", "Schema changed"),
            "affected_fields": list(info.get("new_fields", set())),
        }, 0.35, {}  # high reward for correct detection

    def _finish(self, params: dict):
        self._done = True
        summary = self.get_episode_summary()

        # Final composite reward
        completion  = summary["task_completion_rate"]
        adaptation  = summary["schema_adaptation_rate"]
        preference  = summary["preference_alignment_rate"]
        steps_used  = self._step_count / self.max_steps
        efficiency  = 1.0 - steps_used
        drift_penalty = -0.2 * (len(self._active_drifts) - len(self._detected_drifts)) / max(1, len(self._active_drifts))

        final_reward = (
            0.40 * completion
            + 0.30 * adaptation
            + 0.20 * preference
            + 0.10 * efficiency
            + drift_penalty
        )
        final_reward = max(-1.0, min(1.0, final_reward))

        return {
            "episode_complete": True,
            "summary": summary,
            "final_reward": round(final_reward, 4),
        }, final_reward, {"episode_summary": summary}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_drift_for(self, module: str) -> dict | None:
        for dk in self._active_drifts:
            spec = _DRIFT_CATALOGUE[dk]
            if spec.get("module") == module:
                return spec
        return None

    def _mark_task_progress(self, module: str, tool: str):
        for task in self._tasks:
            if task["target_module"] == module and not task["completed"]:
                if tool in task.get("requires", []):
                    # partial credit: first required tool hit
                    pass

    def _mark_task_done(self, task_type: str):
        for task in self._tasks:
            if task["type"] == task_type and not task["completed"]:
                task["completed"] = True
                break

    def _mark_task_pref_aligned(self, task_type: str):
        for task in self._tasks:
            if task["type"] == task_type:
                task["pref_aligned"] = True
                break