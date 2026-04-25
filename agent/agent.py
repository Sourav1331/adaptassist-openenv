"""
Agent Module
============
Handles prompt construction, tool call parsing, and the
inference loop for the AdaptAssist environment.

Works with any HuggingFace causal LM (Llama, Mistral, etc.)
"""

import json
import re
from typing import Optional


# ─────────────────────────────────────────────
#  System Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are AdaptAssist, an expert personal executive assistant AI agent.

You manage calendars, emails, and restaurant bookings on behalf of a busy professional.

## Your available tools:

1. read_calendar(date) — List all events on a given date
2. create_event(title, date, start_time, end_time, attendees, priority) — Book a new event
3. delete_event(event_id) — Remove an event
4. get_inbox() — Read unread emails requiring replies
5. send_reply(email_id, body, tone) — Reply to an email
6. search_restaurants(cuisine, date, time_slot) — Find available restaurants
7. book_restaurant(restaurant_id, date, time_slot, party_size) — Make a reservation
8. cancel_booking(booking_id, hours_before) — Cancel a reservation
9. get_user_prefs() — Get user preferences (preferred times, cuisines, tone)
10. detect_schema_change(module) — Check if an API module has changed schema [calendar/email/restaurant]
11. finish() — Signal all tasks are complete

## IMPORTANT RULES:
- Always get_user_prefs FIRST before booking or scheduling
- If a tool returns an error with unexpected field names, IMMEDIATELY call detect_schema_change for that module
- After detecting drift, adjust your tool parameters accordingly
- Resolve conflicts before creating new events
- Match email reply tone to user preferences
- Prefer user's preferred cuisines when booking restaurants
- Call finish() only when ALL tasks are completed

## Response format:
You MUST respond with EXACTLY this JSON structure — no other text:

```json
{
  "thought": "Brief reasoning about what to do next",
  "tool": "tool_name",
  "params": {
    "key": "value"
  }
}
```
"""


# ─────────────────────────────────────────────
#  Prompt Builder
# ─────────────────────────────────────────────

class PromptBuilder:
    """Builds conversation history for multi-turn agent inference."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build_initial_prompt(self, observation: dict) -> str:
        """Build the first user message from the initial observation."""
        tasks_text = "\n".join(
            f"  {i+1}. [{t['id']}] {t['description']}"
            for i, t in enumerate(observation["tasks"])
        )
        hint = observation.get("hint", "")
        drift_warning = observation.get("drift_warning", False)

        user_msg = f"""You have {len(observation['tasks'])} tasks to complete.

## Your tasks:
{tasks_text}

## Status:
- Step: {observation['step']}/{observation['max_steps']}
- Schema drift warning: {'YES — some APIs may have changed' if drift_warning else 'No drift detected'}

{hint}

Begin by checking user preferences, then tackle each task systematically.
Respond with your first tool call in JSON format."""
        return user_msg

    def build_tool_result_message(self, tool: str, params: dict, result: dict, info: dict) -> str:
        """Build the message after a tool call returns."""
        tasks_status = "\n".join(
            f"  [{t['id']}] {'DONE' if t['completed'] else 'pending'}: {t['description'][:60]}..."
            for t in info.get("tasks_obs", [])
        )

        drift_note = ""
        if info.get("drift_event"):
            drift_note = f"\n[SYSTEM] Schema drift detected in {info['drift_event']['module']} module!"

        return f"""Tool: {tool}
Params: {json.dumps(params, indent=2)}
Result: {json.dumps(result, indent=2)}

Tasks progress ({info.get('tasks_completed', 0)}/{info.get('total_tasks', 0)} done):
{tasks_status}
Step: {info.get('step', 0)}/{info.get('max_steps', 30)}{drift_note}

What is your next action?"""

    def build_messages(self, history: list[dict]) -> list[dict]:
        """Convert history list to HuggingFace messages format."""
        messages = [{"role": "system", "content": self.system_prompt}]
        for entry in history:
            messages.append({"role": entry["role"], "content": entry["content"]})
        return messages

    def messages_to_text(self, messages: list[dict], tokenizer) -> str:
        """Apply chat template from tokenizer."""
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# ─────────────────────────────────────────────
#  Action Parser
# ─────────────────────────────────────────────

class ActionParser:
    """Parse LLM text output into structured tool calls."""

    JSON_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    BARE_JSON_PATTERN = re.compile(r"\{[^{}]*\"tool\"[^{}]*\}", re.DOTALL)

    def parse(self, text: str) -> Optional[dict]:
        """
        Extract action dict from model output.
        Returns {"tool": str, "params": dict, "thought": str} or None.
        """
        text = text.strip()

        # Try fenced JSON block first
        match = self.JSON_PATTERN.search(text)
        if match:
            try:
                action = json.loads(match.group(1))
                return self._validate(action)
            except json.JSONDecodeError:
                pass

        # Try bare JSON
        match = self.BARE_JSON_PATTERN.search(text)
        if match:
            try:
                action = json.loads(match.group(0))
                return self._validate(action)
            except json.JSONDecodeError:
                pass

        # Try parsing entire text as JSON
        try:
            action = json.loads(text)
            return self._validate(action)
        except json.JSONDecodeError:
            pass

        # Fallback: extract tool name heuristically
        for tool in ["read_calendar", "create_event", "delete_event", "get_inbox",
                     "send_reply", "search_restaurants", "book_restaurant",
                     "cancel_booking", "get_user_prefs", "detect_schema_change", "finish"]:
            if tool in text:
                return {"tool": tool, "params": {}, "thought": "Parsed heuristically"}

        return None

    def _validate(self, action: dict) -> Optional[dict]:
        if not isinstance(action, dict):
            return None
        if "tool" not in action:
            return None
        action.setdefault("params", {})
        action.setdefault("thought", "")
        return action


# ─────────────────────────────────────────────
#  Agent Loop
# ─────────────────────────────────────────────

class AdaptAssistAgent:
    """
    Full inference agent loop.
    Works with any HuggingFace model + tokenizer.

    Usage:
        agent = AdaptAssistAgent(model, tokenizer)
        trajectory = agent.run_episode(env)
    """

    def __init__(self, model, tokenizer, max_new_tokens: int = 512,
                 temperature: float = 0.7, verbose: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.verbose = verbose
        self.prompt_builder = PromptBuilder()
        self.parser = ActionParser()

    def _generate(self, messages: list[dict]) -> str:
        """Run inference and return generated text."""
        import torch

        text = self.prompt_builder.messages_to_text(messages, self.tokenizer)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def run_episode(self, env) -> dict:
        """
        Run a full episode in the environment.

        Returns:
            trajectory: {
                "observations": list,
                "actions": list,
                "rewards": list,
                "infos": list,
                "total_reward": float,
                "messages": list,  # full conversation for SFT
            }
        """
        obs = env.reset()
        history = []
        trajectory = {
            "observations": [obs],
            "actions": [],
            "rewards": [],
            "infos": [],
            "total_reward": 0.0,
            "messages": [],
        }

        # Initial user message
        initial_msg = self.prompt_builder.build_initial_prompt(obs)
        history.append({"role": "user", "content": initial_msg})

        if self.verbose:
            print(env.render())

        done = False
        while not done:
            # Build messages and generate
            messages = self.prompt_builder.build_messages(history)
            raw_output = self._generate(messages)

            if self.verbose:
                print(f"\n[Agent] {raw_output[:200]}...")

            # Parse action
            action = self.parser.parse(raw_output)
            if action is None:
                action = {"tool": "finish", "params": {}, "thought": "Could not parse output"}

            # Add assistant turn to history
            history.append({"role": "assistant", "content": raw_output})

            # Step environment
            next_obs, reward, done, info = env.step(action)

            # Add tasks_obs for the message builder
            info["tasks_obs"] = next_obs["tasks"]
            info["max_steps"] = getattr(env, "episode_max_length", getattr(env, "max_steps", 0))

            # Build and add result message
            result_msg = self.prompt_builder.build_tool_result_message(
                action["tool"], action["params"],
                info["tool_result"], info,
            )
            if not done:
                history.append({"role": "user", "content": result_msg})

            # Record trajectory
            trajectory["observations"].append(next_obs)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            trajectory["infos"].append(info)
            trajectory["total_reward"] += reward

            if self.verbose:
                print(f"  Tool: {action['tool']} | Reward: {reward:.3f} | Done: {done}")

        trajectory["messages"] = self.prompt_builder.build_messages(history)
        trajectory["summary"] = env.get_episode_summary()

        if self.verbose:
            print(f"\nEpisode done. Total reward: {trajectory['total_reward']:.4f}")
            print(f"Summary: {json.dumps(trajectory['summary'], indent=2)}")

        return trajectory


# ─────────────────────────────────────────────
#  Rule-Based Reference Agent (for SFT data gen)
# ─────────────────────────────────────────────

class RuleBasedAgent:
    """
    Deterministic expert agent for generating SFT training data.
    Plays the environment correctly without a language model.
    """

    def __init__(self):
        self.got_prefs = False
        self.inbox_checked = False
        self.drift_checked = set()
        self.state = {}

    def reset(self):
        self.got_prefs = False
        self.inbox_checked = False
        self.drift_checked = set()
        self.state = {}

    def act(self, obs: dict, last_result: Optional[dict] = None) -> dict:
        """Return next action based on observation and last result."""

        # Step 1: always get prefs first
        if not self.got_prefs:
            self.got_prefs = True
            return {
                "tool": "get_user_prefs",
                "params": {},
                "thought": "Always start by understanding user preferences",
            }

        # Step 2: handle errors by detecting drift
        if last_result and last_result.get("status") == "error":
            for module in ["calendar", "email", "restaurant"]:
                if module not in self.drift_checked:
                    self.drift_checked.add(module)
                    return {
                        "tool": "detect_schema_change",
                        "params": {"module": module},
                        "thought": f"Got an error, checking if {module} schema has drifted",
                    }

        # Step 3: work through tasks in sequence
        for task in obs["tasks"]:
            if task["completed"]:
                continue
            t = task["type"]

            # --- resolve_conflict / reschedule_travel ---
            if t in ("resolve_conflict", "reschedule_travel"):
                task_id = task["id"]
                state_key = f"calendar_read_{task_id}"
                if not self.state.get(state_key):
                    self.state[state_key] = True
                    return {
                        "tool": "read_calendar",
                        "params": {"date": "2024-03-15"},
                        "thought": "Read calendar to find conflicts",
                    }
                # After reading, try to create a new event in a free slot
                return {
                    "tool": "create_event",
                    "params": {
                        "title": "Rescheduled Meeting",
                        "date": "2024-03-15",
                        "start_time": "15:00",
                        "end_time": "16:00",
                        "priority": "medium",
                        "attendees": [],
                    },
                    "thought": "Create event in a free time slot to resolve conflict",
                }

            # --- reply_urgent_email ---
            if t == "reply_urgent_email":
                task_id = task["id"]
                if not self.state.get(f"inbox_read_{task_id}"):
                    self.state[f"inbox_read_{task_id}"] = True
                    return {
                        "tool": "get_inbox",
                        "params": {},
                        "thought": "Check inbox for urgent emails",
                    }
                if last_result and "emails" in last_result:
                    emails = last_result.get("emails", [])
                    urgent = next(
                        (e for e in emails
                         if e.get("urgency") == "urgent" or e.get("priority_level") == "urgent"),
                        emails[0] if emails else None,
                    )
                    if urgent:
                        return {
                            "tool": "send_reply",
                            "params": {
                                "email_id": urgent.get("id"),
                                "body": "Thank you for your email. I will review this and provide the requested information by end of day. Best regards.",
                                "tone": "professional",
                            },
                            "thought": "Reply to urgent email professionally",
                        }
                # Fallback: reply to first inbox item
                return {
                    "tool": "send_reply",
                    "params": {
                        "email_id": "m1",
                        "body": "Thank you for reaching out. I will respond with the necessary details by end of day.",
                        "tone": "professional",
                    },
                    "thought": "Send professional reply",
                }

            # --- book_dinner ---
            if t == "book_dinner":
                task_id = task["id"]
                if not self.state.get(f"restaurant_searched_{task_id}"):
                    self.state[f"restaurant_searched_{task_id}"] = True
                    return {
                        "tool": "search_restaurants",
                        "params": {"cuisine": "Italian", "date": "2024-03-15", "time_slot": "19:00"},
                        "thought": "Search for Italian restaurants (user preference)",
                    }
                if last_result and "results" in last_result:
                    results = last_result.get("results", [])
                    if results:
                        r = results[0]
                        slots_key = "open_times" if "open_times" in r else "available_slots"
                        slots = r.get(slots_key, [])
                        slot = slots[0] if slots else "19:00"
                        return {
                            "tool": "book_restaurant",
                            "params": {
                                "restaurant_id": r["id"],
                                "date": "2024-03-15",
                                "time_slot": slot,
                                "party_size": 2,
                            },
                            "thought": "Book the first available Italian restaurant",
                        }
                # Fallback direct book
                return {
                    "tool": "book_restaurant",
                    "params": {
                        "restaurant_id": "r1",
                        "date": "2024-03-15",
                        "time_slot": "19:00",
                        "party_size": 2,
                    },
                    "thought": "Book Bella Italia at 19:00",
                }

            # --- prioritize_day ---
            if t == "prioritize_day":
                return {
                    "tool": "read_calendar",
                    "params": {"date": "2024-03-15"},
                    "thought": "Read calendar to prioritize events",
                }

            # --- reply_and_book ---
            if t == "reply_and_book":
                task_id = task["id"]
                if not task.get("sub_reply"):
                    if not self.state.get(f"inbox_read_{task_id}"):
                        self.state[f"inbox_read_{task_id}"] = True
                        return {"tool": "get_inbox", "params": {},
                                "thought": "Get inbox for reply_and_book task"}
                    return {
                        "tool": "send_reply",
                        "params": {"email_id": "m3", "body": "Happy to confirm lunch! I have booked a table for us.", "tone": "professional"},
                        "thought": "Reply confirming lunch",
                    }
                if not task.get("sub_book"):
                    return {
                        "tool": "book_restaurant",
                        "params": {"restaurant_id": "r1", "date": "2024-03-15", "time_slot": "12:00", "party_size": 2},
                        "thought": "Book restaurant for lunch",
                    }

        return {"tool": "finish", "params": {}, "thought": "All tasks appear complete"}

    def reset(self):
        self.got_prefs = False
        self.inbox_checked = False
        self.drift_checked = set()
        self.state = {}

    def run_episode(self, env) -> dict:
        """Generate an expert trajectory."""
        self.reset()
        obs = env.reset()
        trajectory = {
            "observations": [obs],
            "actions": [],
            "rewards": [],
            "infos": [],
            "total_reward": 0.0,
        }

        last_result = None
        done = False
        while not done:
            action = self.act(obs, last_result)
            next_obs, reward, done, info = env.step(action)
            last_result = info.get("tool_result", {})

            trajectory["observations"].append(next_obs)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            trajectory["infos"].append(info)
            trajectory["total_reward"] += reward
            obs = next_obs

        trajectory["summary"] = env.get_episode_summary()
        return trajectory
