"""
SFT Data Generator
==================
Generates expert trajectory datasets for supervised fine-tuning (warmup phase).
Uses the RuleBasedAgent to produce correct demonstrations across all difficulty levels.

Run this BEFORE RL training to create data/sft_dataset.json
"""

import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.env import AdaptAssistEnv
from agent.agent import RuleBasedAgent, PromptBuilder, SYSTEM_PROMPT


def generate_sft_conversation(env: AdaptAssistEnv, agent: RuleBasedAgent,
                               prompt_builder: PromptBuilder) -> dict:
    """
    Run one expert episode and convert to conversation format for SFT.

    Returns a HuggingFace-compatible messages dict:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ],
        "reward": float,
        "difficulty": int,
    }
    """
    agent.reset()
    obs = env.reset()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Initial user message
    initial_msg = prompt_builder.build_initial_prompt(obs)
    messages.append({"role": "user", "content": initial_msg})

    last_result = None
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(obs, last_result)

        # Format the assistant response as clean JSON
        assistant_response = json.dumps({
            "thought": action.get("thought", ""),
            "tool": action["tool"],
            "params": action.get("params", {}),
        }, indent=2)
        messages.append({"role": "assistant", "content": f"```json\n{assistant_response}\n```"})

        # Step environment
        next_obs, reward, done, info = env.step(action)
        last_result = info.get("tool_result", {})
        total_reward = reward
        total_reward = max(-1.0, min(1.0, total_reward))

        # Add result as user message (unless done)
        if not done:
            info["tasks_obs"] = next_obs["tasks"]
            # Backward/forward compatible with env variants.
            info["max_steps"] = getattr(env, "episode_max_length", getattr(env, "max_steps", 0))
            result_msg = prompt_builder.build_tool_result_message(
                action["tool"], action.get("params", {}),
                info["tool_result"], info,
            )
            messages.append({"role": "user", "content": result_msg})

        obs = next_obs

    return {
        "messages": messages,
        "reward": round(total_reward, 4),
        "difficulty": env.difficulty,
        "summary": env.get_episode_summary(),
    }


def generate_dataset(
    n_episodes_per_difficulty: int = 50,
    output_path: str = "data/sft_dataset.json",
    difficulties: list = [1, 2],
    seeds: list = None,
) -> list[dict]:
    """
    Generate the full SFT dataset.

    Args:
        n_episodes_per_difficulty: how many episodes per difficulty level
        output_path: where to save the JSON file
        difficulties: which difficulty levels to include (1=no drift, 2=one drift)
        seeds: optional list of seeds for reproducibility
    """
    prompt_builder = PromptBuilder()
    agent = RuleBasedAgent()
    dataset = []

    total = n_episodes_per_difficulty * len(difficulties)
    print(f"Generating {total} SFT episodes...")

    episode_num = 0
    for difficulty in difficulties:
        print(f"\nDifficulty {difficulty} ({n_episodes_per_difficulty} episodes):")
        for i in range(n_episodes_per_difficulty):
            seed = seeds[episode_num] if seeds else random.randint(0, 99999)
            env = AdaptAssistEnv(
                difficulty=difficulty,
                max_steps=25,
                n_tasks=random.randint(2, 3),
                seed=seed,
            )

            try:
                sample = generate_sft_conversation(env, agent, prompt_builder)
                # Only keep high-quality demonstrations
                if sample["reward"] >= 0.1:
                    dataset.append(sample)
                    summary = sample.get("summary", {})
                    steps_used = summary.get("steps_used", summary.get("steps", "?"))
                    tasks_completed = summary.get("tasks_completed")
                    if tasks_completed is None:
                        # Newer env summary exposes completion rate instead of raw counts.
                        rate = summary.get("task_completion_rate")
                        total_tasks = summary.get("total_tasks", env.n_tasks)
                        if isinstance(rate, (int, float)):
                            tasks_completed = int(round(rate * total_tasks))
                        else:
                            tasks_completed = "?"
                    total_tasks = summary.get("total_tasks", env.n_tasks)
                    print(f"  [{i+1}/{n_episodes_per_difficulty}] reward={sample['reward']:.3f} "
                          f"steps={steps_used} "
                          f"tasks={tasks_completed}/{total_tasks}")
                else:
                    print(f"  [{i+1}/{n_episodes_per_difficulty}] SKIPPED (low reward: {sample['reward']:.3f})")
            except Exception as e:
                print(f"  [{i+1}/{n_episodes_per_difficulty}] ERROR: {e}")

            episode_num += 1

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    avg_reward = sum(s["reward"] for s in dataset) / max(len(dataset), 1)
    print(f"\nDataset saved: {output_path}")
    print(f"Total samples: {len(dataset)}")
    print(f"Average reward: {avg_reward:.4f}")

    return dataset


def convert_to_hf_format(dataset: list[dict], output_path: str = "data/sft_hf.json"):
    """
    Convert to HuggingFace datasets format for direct use with TRL SFTTrainer.
    Each sample has a flat 'text' field with the full conversation.
    """
    # We format as text without a tokenizer (tokenizer applied during training)
    hf_samples = []
    for sample in dataset:
        # Create a text representation of the conversation
        text_parts = []
        for msg in sample["messages"]:
            role = msg["role"].upper()
            text_parts.append(f"<|{role}|>\n{msg['content']}\n<|END_{role}|>")
        hf_samples.append({
            "text": "\n".join(text_parts),
            "messages": sample["messages"],
            "reward": sample["reward"],
            "difficulty": sample["difficulty"],
        })

    with open(output_path, "w") as f:
        json.dump(hf_samples, f, indent=2)

    print(f"HF-format dataset saved: {output_path} ({len(hf_samples)} samples)")
    return hf_samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SFT dataset for AdaptAssist")
    parser.add_argument("--n", type=int, default=50, help="Episodes per difficulty level")
    parser.add_argument("--output", type=str, default="data/sft_dataset.json")
    parser.add_argument("--difficulties", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    dataset = generate_dataset(
        n_episodes_per_difficulty=args.n,
        output_path=args.output,
        difficulties=args.difficulties,
    )
    convert_to_hf_format(dataset, output_path=args.output.replace(".json", "_hf.json"))
