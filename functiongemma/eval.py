#!/usr/bin/env python3
"""
Quick eval of FunctionGemma 270M for Ari skill routing.

Uses the model's native prompt format (not llama-cpp-python's
chat_completion wrapper, which doesn't understand FunctionGemma's
custom control tokens).

Usage:
    python3 eval.py <path-to-gguf>

    # Example: eval the fine-tuned model downloaded by launch-aws.sh
    python3 eval.py ./output/ari-functiongemma-q4_k_m.gguf

    # Or eval the base model for comparison
    python3 eval.py ~/models/functiongemma-270m-it-Q4_K_M.gguf

    # Eval a per-locale router against that locale's cases (default: en)
    python3 eval.py ./output/ari-functiongemma-it-q4_k_m.gguf --locale it

Prints one line per test case plus a summary grouped by difficulty.
"""

import argparse
import re
import sys
import time
import types

# Stub out sqlite3/diskcache — some pyenv builds are missing _sqlite3
_diskcache = types.ModuleType("diskcache")
_diskcache.Cache = type("Cache", (), {"__init__": lambda *a, **kw: None})
sys.modules.setdefault("sqlite3", types.ModuleType("sqlite3"))
sys.modules.setdefault("sqlite3.dbapi2", types.ModuleType("sqlite3.dbapi2"))
sys.modules.setdefault("diskcache", _diskcache)
sys.modules.setdefault("diskcache.core", _diskcache)

from llama_cpp import Llama

E = "<escape>"  # string delimiter token

# Ari's skills as FunctionGemma declarations.
# Descriptions enriched with semantic keywords per the best practices doc.
SKILL_DECLARATIONS = [
    f"<start_function_declaration>declaration:current_time{{description:{E}Tells the current time. Use when the user asks what time it is, what hour it is, whether it is morning or afternoon, or anything related to the current time of day.{E},parameters:{{type:{E}OBJECT{E}}}}}<end_function_declaration>",
    f"<start_function_declaration>declaration:date{{description:{E}Tells today's date. Use when the user asks what day it is, what date it is, which day of the week, or anything about today's date.{E},parameters:{{type:{E}OBJECT{E}}}}}<end_function_declaration>",
    f"<start_function_declaration>declaration:calculator{{description:{E}Evaluates math expressions. Use when the user asks to calculate, compute, or figure out any mathematical expression, percentage, division, multiplication, or arithmetic.{E},parameters:{{properties:{{expression:{{description:{E}The math expression to evaluate{E},type:{E}STRING{E}}}}},required:[{E}expression{E}],type:{E}OBJECT{E}}}}}<end_function_declaration>",
    f"<start_function_declaration>declaration:greeting{{description:{E}Responds to greetings. Use when the user says hello, hi, hey, good morning, good evening, howdy, or asks how Ari is doing.{E},parameters:{{type:{E}OBJECT{E}}}}}<end_function_declaration>",
    f"<start_function_declaration>declaration:open_app{{description:{E}Opens or launches apps by name. Use when the user asks to open, launch, start, or run an application or app.{E},parameters:{{properties:{{app_name:{{description:{E}Name of the app to open{E},type:{E}STRING{E}}}}},required:[{E}app_name{E}],type:{E}OBJECT{E}}}}}<end_function_declaration>",
    f"<start_function_declaration>declaration:search{{description:{E}Searches the web. Use when the user asks to search, look up, find information, or google something.{E},parameters:{{properties:{{query:{{description:{E}The search query{E},type:{E}STRING{E}}}}},required:[{E}query{E}],type:{E}OBJECT{E}}}}}<end_function_declaration>",
    f"<start_function_declaration>declaration:coin_flip{{description:{E}Flips a virtual coin. Use when the user wants to flip a coin, toss a coin, or make a random heads or tails choice.{E},parameters:{{type:{E}OBJECT{E}}}}}<end_function_declaration>",
]

SKILL_NAMES = {"current_time", "date", "calculator", "greeting", "open_app", "search", "coin_flip"}

# Test cases: (input, expected_skill, difficulty)
TEST_CASES = [
    # Easy — keyword matcher would handle these
    ("what time is it", "current_time", "easy"),
    ("what's the date today", "date", "easy"),
    ("calculate 5 + 3", "calculator", "easy"),
    ("hello", "greeting", "easy"),
    ("open spotify", "open_app", "easy"),
    ("search for python tutorials", "search", "easy"),
    ("flip a coin", "coin_flip", "easy"),

    # Hard — paraphrases the keyword matcher would miss
    ("do you know what hour it is", "current_time", "hard"),
    ("is it morning or afternoon right now", "current_time", "hard"),
    ("which day of the week is it", "date", "hard"),
    ("how much is fifteen percent of two hundred", "calculator", "hard"),
    ("what's 99 divided by 3", "calculator", "hard"),
    ("hey there, how are you", "greeting", "hard"),
    ("good morning ari", "greeting", "hard"),
    ("can you start the camera app", "open_app", "hard"),
    ("launch my music player", "open_app", "hard"),
    ("I need to look something up online", "search", "hard"),
    ("find me information about black holes", "search", "hard"),
    ("let's leave it to chance, heads or tails", "coin_flip", "hard"),

    # None — should NOT match any skill (general knowledge)
    ("what is the capital of France", None, "none"),
    ("who wrote Romeo and Juliet", None, "none"),
    ("why is the sky blue", None, "none"),
    ("tell me a joke", None, "none"),
    ("how far is the moon from earth", None, "none"),
]

# Italian counterpart of TEST_CASES, for the per-locale router model.
# Held-out like routing-eval.it.jsonl: none of these appear in the Italian
# training data (the built-in *_EXAMPLES_IT consts, the SKILL.it.md
# examples, or generate-dataset.py's Italian negative pool), so a pass
# means the model generalised. The "none" cases carry no live Italian skill
# trigger, so none of them is an utterance a keyword skill should own.
IT_TEST_CASES = [
    # Easy — keyword matcher would handle these
    ("che ore sono di preciso", "current_time", "easy"),
    ("che data abbiamo oggi", "date", "easy"),
    ("calcola 63 meno 28", "calculator", "easy"),
    ("buondì ari", "greeting", "easy"),
    ("apri signal", "open_app", "easy"),
    ("cerca dei tutorial di python", "search", "easy"),
    ("lancia una monetina", "coin_flip", "easy"),

    # Hard — paraphrases the keyword matcher would miss
    ("sapresti dirmi l'ora", "current_time", "hard"),
    ("è ancora mattina o siamo nel pomeriggio", "current_time", "hard"),
    ("a quanti ne siamo oggi", "date", "hard"),
    ("quanto viene il 30 percento di 90", "calculator", "hard"),
    ("fammi la somma di 128 e 256", "calculator", "hard"),
    ("come te la cavi", "greeting", "hard"),
    ("buon pomeriggio ari", "greeting", "hard"),
    ("aprimi duolingo", "open_app", "hard"),
    ("avvia l'app della banca", "open_app", "hard"),
    ("vorrei cercare una cosa su internet", "search", "hard"),
    ("trovami delle informazioni sui buchi neri", "search", "hard"),
    ("decidiamo a sorte, testa o croce", "coin_flip", "hard"),

    # None — should NOT match any skill (general knowledge)
    ("chi ha scritto la Divina Commedia", None, "none"),
    ("perché le zebre hanno le strisce", None, "none"),
    ("qual è il pianeta più vicino al sole", None, "none"),
    ("che cosa mangiano i panda", None, "none"),
    ("quanto pesa un elefante africano", None, "none"),
]

# eval.py runs English by default; --locale it selects the Italian cases.
TEST_CASES_BY_LOCALE = {
    "en": TEST_CASES,
    "it": IT_TEST_CASES,
}


def build_prompt(user_input: str) -> str:
    """Build the FunctionGemma prompt with tool declarations."""
    declarations = "".join(SKILL_DECLARATIONS)
    return (
        f"<start_of_turn>developer\n"
        f"You are a model that can do function calling with the following functions"
        f"{declarations}<end_of_turn>\n"
        f"<start_of_turn>user\n"
        f"{user_input}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def parse_first_call(raw: str) -> str | None:
    """Extract the function name from the first call block."""
    m = re.search(r"<start_function_call>call:(\w+)\{", raw)
    if m:
        name = m.group(1)
        return name if name in SKILL_NAMES else f"__hallucinated:{name}__"
    return None


def run_test(model_path: str, locale: str = "en"):
    test_cases = TEST_CASES_BY_LOCALE[locale]

    print(f"Loading model from {model_path}...")
    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Locale: {locale} ({len(test_cases)} cases)\n")

    results = {"easy": [], "hard": [], "none": []}
    total_time = 0

    for user_input, expected, difficulty in test_cases:
        prompt = build_prompt(user_input)

        t1 = time.time()
        output = llm(
            prompt,
            max_tokens=60,
            stop=["<end_of_turn>", "<start_function_response>"],
            temperature=0.0,
        )
        elapsed = time.time() - t1
        total_time += elapsed

        raw = output["choices"][0]["text"].strip()

        matched_skill = parse_first_call(raw)
        if matched_skill is None:
            # No function call — model declined. This is correct for "none" cases.
            matched_skill = "__none__"

        correct = (
            (expected is None and matched_skill == "__none__")
            or (expected is not None and matched_skill == expected)
        )
        mark = "PASS" if correct else "FAIL"
        results[difficulty].append(correct)

        print(
            f"  [{mark}] ({elapsed:.2f}s) {difficulty:4s} | "
            f"input: {user_input!r:50s} | "
            f"expected: {str(expected):15s} | "
            f"got: {matched_skill:20s} | "
            f"raw: {raw[:80]}"
        )

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    total_correct = 0
    total_count = 0
    for difficulty in ("easy", "hard", "none"):
        n = len(results[difficulty])
        c = sum(results[difficulty])
        total_correct += c
        total_count += n
        print(f"  {difficulty:5s}: {c}/{n} ({100*c/n:.0f}%)")

    print(f"  {'total':5s}: {total_correct}/{total_count} ({100*total_correct/total_count:.0f}%)")
    print(f"\n  Average inference time: {total_time/total_count:.3f}s per query")
    print(f"  Total time: {total_time:.1f}s for {total_count} queries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path", help="Path to the GGUF model file")
    parser.add_argument(
        "--locale",
        default="en",
        choices=sorted(TEST_CASES_BY_LOCALE),
        help="Which test-case set to run (default: en)",
    )
    args = parser.parse_args()
    run_test(args.model_path, args.locale)
