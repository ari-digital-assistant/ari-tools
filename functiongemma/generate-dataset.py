#!/usr/bin/env python3
"""
Generate the FunctionGemma fine-tuning dataset for Ari.

Combines:
1. Ari skill paraphrases (synthetic, ~200 per skill)
2. Google mobile-actions dataset (9,654 examples)
3. Negative examples (general knowledge, should NOT trigger any function)

Output: JSONL in the same format as google/mobile-actions, ready for
MLX fine-tuning with the FunctionGemma chat template.
"""

import json
import random
import sys

random.seed(42)

# ── Ari skill definitions ──────────────────────────────────────────────

ARI_SKILLS = [
    {
        "name": "current_time",
        "description": "Tells the current time. Use when the user asks what time it is, what hour it is, whether it is morning or afternoon, or anything about the current time of day.",
        "parameters": {"type": "OBJECT", "properties": {}},
        "paraphrases": [
            "what time is it",
            "what's the time",
            "tell me the time",
            "what time do you have",
            "do you know what time it is",
            "what hour is it",
            "can you tell me the time",
            "what's the current time",
            "is it morning or afternoon",
            "how late is it",
            "what time is it right now",
            "got the time",
            "what's the time now",
            "could you tell me the time please",
            "I need to know what time it is",
            "time please",
            "what time have you got",
            "is it late",
            "am or pm right now",
            "check the time for me",
            "I wonder what time it is",
            "any idea what time it is",
            "do you have the time",
            "quick, what time is it",
            "the time?",
            "is it still early",
            "how early is it",
            "tell me the current time",
            "what's the clock say",
            "current time please",
        ],
    },
    {
        "name": "current_date",
        "description": "Tells today's date. Use when the user asks what day it is, what date it is, which day of the week it is, or anything about today's date.",
        "parameters": {"type": "OBJECT", "properties": {}},
        "paraphrases": [
            "what's the date today",
            "what day is it",
            "what's today's date",
            "which day of the week is it",
            "what date is it",
            "tell me today's date",
            "what day are we on",
            "is it Monday today",
            "what's the date",
            "do you know today's date",
            "can you tell me the date",
            "what day of the week is it today",
            "I need to know the date",
            "the date please",
            "is today a weekday",
            "what's today",
            "which day is today",
            "tell me what day it is",
            "date please",
            "current date",
            "what is today's date",
            "is it the weekend",
            "what day is today",
            "do you know what day it is",
            "I forgot what day it is",
            "is it still Tuesday",
            "what's the day today",
            "today's date please",
            "check the date for me",
            "could you tell me the date",
        ],
    },
    {
        "name": "calculator",
        "description": "Evaluates math expressions. Use when the user asks to calculate, compute, or figure out any mathematical expression, percentage, division, multiplication, addition, subtraction, or arithmetic.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "expression": {
                    "type": "STRING",
                    "description": "The math expression to evaluate.",
                },
            },
            "required": ["expression"],
        },
        "paraphrases": [
            ("calculate 5 + 3", {"expression": "5 + 3"}),
            ("what's 99 divided by 3", {"expression": "99 / 3"}),
            ("how much is fifteen percent of two hundred", {"expression": "15% of 200"}),
            ("compute 12 times 8", {"expression": "12 * 8"}),
            ("what's 100 minus 37", {"expression": "100 - 37"}),
            ("figure out 2 to the power of 10", {"expression": "2^10"}),
            ("what is 144 divided by 12", {"expression": "144 / 12"}),
            ("25 plus 75", {"expression": "25 + 75"}),
            ("multiply 9 by 6", {"expression": "9 * 6"}),
            ("what's the square root of 81", {"expression": "sqrt(81)"}),
            ("how much is 20 percent of 50", {"expression": "20% of 50"}),
            ("subtract 15 from 100", {"expression": "100 - 15"}),
            ("what does 7 times 7 equal", {"expression": "7 * 7"}),
            ("divide 200 by 8", {"expression": "200 / 8"}),
            ("add 33 and 67", {"expression": "33 + 67"}),
            ("what's 10 percent of 500", {"expression": "10% of 500"}),
            ("calculate the sum of 14 and 28", {"expression": "14 + 28"}),
            ("how much is 3.14 times 2", {"expression": "3.14 * 2"}),
            ("what is 1000 divided by 7", {"expression": "1000 / 7"}),
            ("compute 50 plus 50", {"expression": "50 + 50"}),
            ("figure out 8 squared", {"expression": "8^2"}),
            ("what's half of 246", {"expression": "246 / 2"}),
            ("9 plus 10", {"expression": "9 + 10"}),
            ("how much is a quarter of 80", {"expression": "80 / 4"}),
            ("what's 5 factorial", {"expression": "5!"}),
            ("calculate 999 minus 1", {"expression": "999 - 1"}),
            ("what is 45 times 3", {"expression": "45 * 3"}),
            ("18 divided by 3", {"expression": "18 / 3"}),
            ("what's 75 plus 25", {"expression": "75 + 25"}),
            ("do the math on 6 times 9", {"expression": "6 * 9"}),
        ],
    },
    {
        "name": "greeting",
        "description": "Responds to greetings. Use when the user says hello, hi, hey, good morning, good evening, howdy, what's up, or asks how Ari is doing.",
        "parameters": {"type": "OBJECT", "properties": {}},
        "paraphrases": [
            "hello",
            "hi",
            "hey",
            "hey there",
            "howdy",
            "good morning",
            "good afternoon",
            "good evening",
            "yo",
            "sup",
            "what's up",
            "hiya",
            "heya",
            "hello ari",
            "hi ari",
            "hey ari",
            "good morning ari",
            "greetings",
            "how are you",
            "how are you doing",
            "how's it going",
            "what's going on",
            "how do you do",
            "nice to meet you",
            "hey there ari",
            "morning",
            "evening",
            "how are things",
            "how you doing",
            "what's happening",
        ],
    },
    {
        "name": "open_app",
        "description": "Opens or launches apps by name. Use when the user asks to open, launch, start, run, or fire up an application or app.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {
                    "type": "STRING",
                    "description": "Name of the app to open.",
                },
            },
            "required": ["app_name"],
        },
        "paraphrases": [
            ("open spotify", {"app_name": "Spotify"}),
            ("launch the camera", {"app_name": "Camera"}),
            ("start the browser", {"app_name": "Browser"}),
            ("open youtube", {"app_name": "YouTube"}),
            ("can you open settings", {"app_name": "Settings"}),
            ("launch maps", {"app_name": "Maps"}),
            ("fire up the music player", {"app_name": "Music Player"}),
            ("run chrome", {"app_name": "Chrome"}),
            ("open my email", {"app_name": "Email"}),
            ("start whatsapp", {"app_name": "WhatsApp"}),
            ("open the calculator app", {"app_name": "Calculator"}),
            ("launch instagram", {"app_name": "Instagram"}),
            ("can you start the camera app", {"app_name": "Camera"}),
            ("open netflix", {"app_name": "Netflix"}),
            ("fire up spotify", {"app_name": "Spotify"}),
            ("launch my music player", {"app_name": "Music Player"}),
            ("run the gallery", {"app_name": "Gallery"}),
            ("open telegram", {"app_name": "Telegram"}),
            ("start firefox", {"app_name": "Firefox"}),
            ("open the clock app", {"app_name": "Clock"}),
            ("launch the phone app", {"app_name": "Phone"}),
            ("open messages", {"app_name": "Messages"}),
            ("can you open twitter", {"app_name": "Twitter"}),
            ("start the notes app", {"app_name": "Notes"}),
            ("open slack", {"app_name": "Slack"}),
            ("launch the calendar", {"app_name": "Calendar"}),
            ("fire up the weather app", {"app_name": "Weather"}),
            ("open reddit", {"app_name": "Reddit"}),
            ("run discord", {"app_name": "Discord"}),
            ("open the files app", {"app_name": "Files"}),
        ],
    },
    {
        "name": "search",
        "description": "Searches the web. Use when the user asks to search, look up, find information about something, or google something.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
        },
        "paraphrases": [
            ("search for python tutorials", {"query": "python tutorials"}),
            ("look up the weather in London", {"query": "weather in London"}),
            ("google how to make pasta", {"query": "how to make pasta"}),
            ("find information about black holes", {"query": "black holes"}),
            ("search for nearby restaurants", {"query": "nearby restaurants"}),
            ("look up who won the world cup", {"query": "who won the world cup"}),
            ("find me a recipe for brownies", {"query": "recipe for brownies"}),
            ("search how tall is mount everest", {"query": "how tall is mount everest"}),
            ("google the latest news", {"query": "latest news"}),
            ("look up train times to London", {"query": "train times to London"}),
            ("find out about the Mars rover", {"query": "Mars rover"}),
            ("search for cheap flights to Tokyo", {"query": "cheap flights to Tokyo"}),
            ("I need to look something up about batteries", {"query": "batteries"}),
            ("can you google that for me", {"query": "that"}),
            ("search the web for Ari digital assistant", {"query": "Ari digital assistant"}),
            ("find directions to the airport", {"query": "directions to the airport"}),
            ("look up symptoms of a cold", {"query": "symptoms of a cold"}),
            ("google how to change a tyre", {"query": "how to change a tyre"}),
            ("search for best programming languages 2026", {"query": "best programming languages 2026"}),
            ("find reviews for the pixel phone", {"query": "reviews for the pixel phone"}),
            ("look up the population of Malta", {"query": "population of Malta"}),
            ("search for hiking trails near me", {"query": "hiking trails near me"}),
            ("google what time the shops close", {"query": "what time the shops close"}),
            ("find out when the next bus is", {"query": "when the next bus is"}),
            ("search for the meaning of serendipity", {"query": "meaning of serendipity"}),
            ("look up how to tie a tie", {"query": "how to tie a tie"}),
            ("find me a good pizza place", {"query": "good pizza place"}),
            ("google who invented the telephone", {"query": "who invented the telephone"}),
            ("search for free online courses", {"query": "free online courses"}),
            ("look up currency exchange rates", {"query": "currency exchange rates"}),
        ],
    },
    {
        "name": "coin_flip",
        "description": "Flips a virtual coin and returns heads or tails. Use when the user asks to flip a coin, toss a coin, or make a random heads or tails choice.",
        "parameters": {"type": "OBJECT", "properties": {}},
        "paraphrases": [
            "flip a coin",
            "toss a coin",
            "heads or tails",
            "coin flip",
            "can you flip a coin for me",
            "toss a coin please",
            "let's flip for it",
            "I need a coin flip",
            "random coin toss",
            "flip it",
            "heads or tails please",
            "give me a coin flip",
            "let's leave it to chance",
            "coin toss",
            "can you toss a coin",
            "flip a coin for me",
            "do a coin flip",
            "let's do heads or tails",
            "I'll flip a coin",
            "random flip",
            "make a random choice for me",
            "pick heads or tails",
            "toss it",
            "should I or shouldn't I, flip a coin",
            "let chance decide",
            "quick coin flip",
            "help me decide with a coin flip",
            "flip",
            "give me heads or tails",
            "let's toss for it",
        ],
    },
]

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

# ── Build the dataset ──────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a model that can do function calling with the following functions"


def build_tool_def(skill):
    """Build a tool definition dict in FunctionGemma format."""
    return {
        "type": "function",
        "function": {
            "name": skill["name"],
            "description": skill["description"],
            "parameters": skill["parameters"],
        },
    }


def build_ari_tools():
    """All Ari skills as tool definitions."""
    return [build_tool_def(s) for s in ARI_SKILLS]


def build_function_call(name, arguments=None):
    """Build a tool_calls entry."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments or {},
        },
    }


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
            "content": "I cannot assist with that request. My available tools are for specific tasks like checking the time, doing math, opening apps, searching the web, and flipping coins.",
        })
    return {
        "metadata": "train",
        "tools": tools,
        "messages": messages,
    }


def generate_ari_samples():
    """Generate training samples from Ari skill paraphrases."""
    tools = build_ari_tools()
    samples = []

    for skill in ARI_SKILLS:
        for phrase in skill["paraphrases"]:
            if isinstance(phrase, tuple):
                user_text, args = phrase
                tc = [build_function_call(skill["name"], args)]
            else:
                user_text = phrase
                tc = [build_function_call(skill["name"])]
            samples.append(build_sample(user_text, tc, tools))

    return samples


def generate_negative_samples():
    """Generate samples where no function should be called."""
    tools = build_ari_tools()
    samples = []
    for text in NEGATIVE_EXAMPLES:
        samples.append(build_sample(text, None, tools))
    return samples


def load_mobile_actions():
    """Load the Google mobile-actions dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("google/mobile-actions", split="train")
        samples = []
        for row in ds:
            samples.append({
                "metadata": row["metadata"],
                "tools": row["tools"],
                "messages": row["messages"],
            })
        return samples
    except ImportError:
        print("WARNING: datasets library not available, skipping mobile-actions", file=sys.stderr)
        return []


def main():
    print("Generating Ari skill samples...", file=sys.stderr)
    ari_samples = generate_ari_samples()
    print(f"  {len(ari_samples)} skill paraphrases", file=sys.stderr)

    print("Generating negative samples...", file=sys.stderr)
    neg_samples = generate_negative_samples()
    print(f"  {len(neg_samples)} negative examples", file=sys.stderr)

    print("Loading mobile-actions dataset...", file=sys.stderr)
    mobile_samples = load_mobile_actions()
    print(f"  {len(mobile_samples)} mobile action examples", file=sys.stderr)

    # Split: 90% train, 10% eval
    all_samples = ari_samples + neg_samples + mobile_samples
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.9)
    train = all_samples[:split_idx]
    eval_ = all_samples[split_idx:]

    for s in train:
        s["metadata"] = "train"
    for s in eval_:
        s["metadata"] = "eval"

    combined = train + eval_

    print(f"\nTotal: {len(combined)} samples ({len(train)} train, {len(eval_)} eval)", file=sys.stderr)

    # Write JSONL
    for sample in combined:
        print(json.dumps(sample, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
