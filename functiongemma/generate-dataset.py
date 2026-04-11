#!/usr/bin/env python3
"""
Generate the FunctionGemma fine-tuning dataset for Ari.

Combines:
1. Ari built-in skill paraphrases (extracted from ari-engine via the
   `export-utterances` Cargo binary — single source of truth)
2. Google mobile-actions dataset (9,654 examples)
3. Negative examples (general knowledge, should NOT trigger any function)

Output: JSONL on stdout in the same format as google/mobile-actions,
ready for SFTTrainer fine-tuning with FunctionGemma's chat template.

Usage:
    # Auto-discover ari-engine (assumes sibling clone or env var):
    python3 generate-dataset.py > dataset.jsonl

    # Or specify the engine path:
    ARI_ENGINE_DIR=/path/to/ari-engine python3 generate-dataset.py > dataset.jsonl
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

random.seed(42)

# ── Locate ari-engine and run the export binary ────────────────────────

def find_engine_dir() -> Path:
    """Find ari-engine relative to this script or via env var."""
    if env := os.environ.get("ARI_ENGINE_DIR"):
        p = Path(env)
        if (p / "Cargo.toml").is_file():
            return p
        sys.exit(f"ERROR: ARI_ENGINE_DIR={env} doesn't contain Cargo.toml")

    # Look for sibling clones in common locations
    here = Path(__file__).resolve().parent
    candidates = [
        # Sibling under same parent (e.g. ~/projects/ari-tools and ~/projects/ari-engine)
        here.parent.parent / "ari-engine",
        # Sibling under the local Ari workspace
        here.parent.parent.parent / "ari-engine",
        # Same directory level
        here.parent / "ari-engine",
    ]
    for c in candidates:
        if (c / "Cargo.toml").is_file():
            return c

    sys.exit(
        "ERROR: could not find ari-engine. Set ARI_ENGINE_DIR=/path/to/ari-engine "
        "or clone it as a sibling of ari-tools."
    )


def export_skills(engine_dir: Path) -> list:
    """Run `cargo run -p ari-skills --bin export-utterances` and parse JSON."""
    print(f"Exporting skills from {engine_dir}...", file=sys.stderr)
    result = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "ari-skills", "--bin", "export-utterances"],
        cwd=engine_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


# ── Negative examples (should NOT match any skill) ─────────────────────

NEGATIVE_EXAMPLES = [
    "what is the capital of France",
    "who wrote Romeo and Juliet",
    "why is the sky blue",
    "tell me a joke",
    "how far is the moon from earth",
    "what's the meaning of life",
    "who is the president of the United States",
    "how many continents are there",
    "what's the speed of light",
    "explain quantum physics",
    "what is photosynthesis",
    "who painted the Mona Lisa",
    "what language do they speak in Brazil",
    "how many bones are in the human body",
    "what's the tallest building in the world",
    "when was the internet invented",
    "who discovered gravity",
    "what causes earthquakes",
    "how does wifi work",
    "what is the biggest ocean",
    "tell me about dinosaurs",
    "what's the boiling point of water",
    "who is Albert Einstein",
    "what year did world war 2 end",
    "how do birds fly",
    "what is DNA",
    "who invented the light bulb",
    "what is climate change",
    "how many planets are in the solar system",
    "what is the longest river in the world",
    "explain how a car engine works",
    "what is the most spoken language",
    "who was the first person on the moon",
    "what does AI stand for",
    "how do vaccines work",
    "what is the smallest country in the world",
    "who composed the four seasons",
    "what is blockchain",
    "how do airplanes stay in the air",
    "what is the great wall of china",
    "tell me something interesting",
    "what's a good book to read",
    "recommend a movie",
    "what should I have for dinner",
    "I'm bored",
    "do you like music",
    "what's your favourite colour",
    "are you a robot",
    "what can you do",
    "thank you",
    "thanks ari",
    "never mind",
    "forget it",
    "that's all",
    "goodbye",
    "see you later",
    "talk to you later",
    "I'm done",
    "nothing",
    "okay",
]

SYSTEM_PROMPT = "You are a model that can do function calling with the following functions"


# ── Build training samples ─────────────────────────────────────────────

def build_tools(skills: list) -> list:
    """Convert exported skills into FunctionGemma tool declarations."""
    tools = []
    for s in skills:
        tools.append({
            "type": "function",
            "function": {
                "name": s["id"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        })
    return tools


def build_sample(user_text, tool_calls, tools):
    """Build one training sample in the mobile-actions format."""
    messages = [
        {"role": "developer", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    if tool_calls:
        messages.append({
            "role": "assistant",
            "tool_calls": tool_calls,
        })
    else:
        messages.append({
            "role": "assistant",
            "content": (
                "I cannot assist with that request. My available tools are for "
                "specific tasks like checking the time, doing math, opening apps, "
                "searching the web, and flipping coins."
            ),
        })
    return {
        "metadata": "train",
        "tools": tools,
        "messages": messages,
    }


def generate_skill_samples(skills: list, tools: list) -> list:
    samples = []
    total_examples = 0
    for skill in skills:
        for ex in skill["examples"]:
            tool_calls = [{
                "type": "function",
                "function": {
                    "name": skill["id"],
                    "arguments": ex["args"],
                },
            }]
            samples.append(build_sample(ex["text"], tool_calls, tools))
            total_examples += 1
    print(f"  {total_examples} skill paraphrases", file=sys.stderr)
    return samples


def generate_negative_samples(tools: list) -> list:
    samples = []
    for text in NEGATIVE_EXAMPLES:
        samples.append(build_sample(text, None, tools))
    print(f"  {len(samples)} negative examples", file=sys.stderr)
    return samples


def load_mobile_actions() -> list:
    """Load the Google mobile-actions dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "WARNING: 'datasets' library not available, skipping mobile-actions",
            file=sys.stderr,
        )
        return []

    print("Loading google/mobile-actions...", file=sys.stderr)
    ds = load_dataset("google/mobile-actions", split="train")
    samples = []
    for row in ds:
        samples.append({
            "metadata": row["metadata"],
            "tools": row["tools"],
            "messages": row["messages"],
        })
    print(f"  {len(samples)} mobile action examples", file=sys.stderr)
    return samples


def main():
    engine_dir = find_engine_dir()
    skills = export_skills(engine_dir)
    print(f"  {len(skills)} skills found", file=sys.stderr)

    tools = build_tools(skills)

    print("Generating samples:", file=sys.stderr)
    skill_samples = generate_skill_samples(skills, tools)
    negative_samples = generate_negative_samples(tools)
    mobile_samples = load_mobile_actions()

    all_samples = skill_samples + negative_samples + mobile_samples
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.9)
    train = all_samples[:split_idx]
    eval_ = all_samples[split_idx:]

    for s in train:
        s["metadata"] = "train"
    for s in eval_:
        s["metadata"] = "eval"

    combined = train + eval_
    print(
        f"\nTotal: {len(combined)} samples ({len(train)} train, {len(eval_)} eval)",
        file=sys.stderr,
    )

    for sample in combined:
        print(json.dumps(sample, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
