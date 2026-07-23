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

The tool declarations are built from generate-dataset.py's own catalogue, so
this needs the ari-engine and ari-skills checkouts the generator needs
(ARI_ENGINE_DIR / ARI_SKILLS_DIR, or sibling clones).
"""

import argparse
import importlib.util
import re
import sys
import time
import types
from pathlib import Path

# Stub out sqlite3/diskcache — some pyenv builds are missing _sqlite3
_diskcache = types.ModuleType("diskcache")
_diskcache.Cache = type("Cache", (), {"__init__": lambda *a, **kw: None})
sys.modules.setdefault("sqlite3", types.ModuleType("sqlite3"))
sys.modules.setdefault("sqlite3.dbapi2", types.ModuleType("sqlite3.dbapi2"))
sys.modules.setdefault("diskcache", _diskcache)
sys.modules.setdefault("diskcache.core", _diskcache)

from llama_cpp import Llama

# The generator owns the skill catalogue. Its filename has a hyphen, so
# load it explicitly — same trick test_generate_dataset.py uses.
_spec = importlib.util.spec_from_file_location(
    "gends", Path(__file__).parent / "generate-dataset.py"
)
gends = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gends)

E = "<escape>"  # string delimiter token


def load_tools(locale: str) -> list:
    """The exact tool list the router was trained on and is offered at
    inference, built by the generator's own functions.

    This used to be a hand-maintained copy of the declarations, and it had
    drifted: it declared `date`, `open_app` and `coin_flip` while the corpus
    taught `current_date`, `open` and `coinflip`, so every test case in those
    categories was scored as a hallucination no matter what the model did.
    It also declared `search`, which `router_eligible_skills` removes — the
    router is never offered it.
    """
    engine_dir = gends.find_engine_dir()
    skills_dir = gends.find_skills_dir()
    builtin = gends.export_skills(engine_dir, locale)
    community = gends.load_community_skills(skills_dir, locale) if skills_dir else []
    skills = gends.router_eligible_skills(gends.merge_skills(builtin, community))
    gends.assign_aliases(skills)
    return gends.build_tools(skills)


def render_schema(value) -> str:
    """Render a JSON schema in FunctionGemma's declaration syntax: bare
    identifier keys, string values upper-cased and wrapped in <escape>.

    Mirrors `render_funcgemma_value` in ari-llm, which builds the same
    prompt on device.
    """
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{render_schema(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(render_schema(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return f"{E}NULL{E}"
    return f"{E}{value.upper()}{E}"


def build_declarations(tools: list) -> str:
    return "".join(
        f"<start_function_declaration>declaration:{t['function']['name']}"
        f"{{description:{E}{t['function']['description']}{E},"
        f"parameters:{render_schema(t['function']['parameters'])}}}"
        f"<end_function_declaration>"
        for t in tools
    )


# Test cases: (input, expected_skill, difficulty). Skill names are the
# router's aliases — the final id segment, as `assign_aliases` computes it.
# No `search` cases: it is router-ineligible, so it is never declared and
# routing to it is not a thing the model can be asked to do.
TEST_CASES = [
    # Easy — keyword matcher would handle these
    ("what time is it", "current_time", "easy"),
    ("what's the date today", "current_date", "easy"),
    ("calculate 5 + 3", "calculator", "easy"),
    ("hello", "greeting", "easy"),
    ("open spotify", "open", "easy"),
    ("flip a coin", "coinflip", "easy"),

    # Hard — paraphrases the keyword matcher would miss
    ("do you know what hour it is", "current_time", "hard"),
    ("is it morning or afternoon right now", "current_time", "hard"),
    ("which day of the week is it", "current_date", "hard"),
    ("how much is fifteen percent of two hundred", "calculator", "hard"),
    ("what's 99 divided by 3", "calculator", "hard"),
    ("hey there, how are you", "greeting", "hard"),
    ("good morning ari", "greeting", "hard"),
    ("can you start the camera app", "open", "hard"),
    ("launch my music player", "open", "hard"),
    ("let's leave it to chance, heads or tails", "coinflip", "hard"),

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
    ("che data abbiamo oggi", "current_date", "easy"),
    ("calcola 63 meno 28", "calculator", "easy"),
    ("buondì ari", "greeting", "easy"),
    ("apri signal", "open", "easy"),
    ("lancia una monetina", "coinflip", "easy"),

    # Hard — paraphrases the keyword matcher would miss
    ("sapresti dirmi l'ora", "current_time", "hard"),
    ("è ancora mattina o siamo nel pomeriggio", "current_time", "hard"),
    ("quanti ne abbiamo oggi", "current_date", "hard"),
    ("quanto viene il 30 percento di 90", "calculator", "hard"),
    ("fammi la somma di 128 e 256", "calculator", "hard"),
    ("come va ultimamente", "greeting", "hard"),
    ("buon pomeriggio ari", "greeting", "hard"),
    ("aprimi duolingo", "open", "hard"),
    ("avvia l'app della banca", "open", "hard"),
    ("decidiamo a sorte, testa o croce", "coinflip", "hard"),

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


def build_prompt(user_input: str, declarations: str) -> str:
    """Build the FunctionGemma prompt with tool declarations."""
    return (
        f"<start_of_turn>developer\n"
        f"You are a model that can do function calling with the following functions"
        f"{declarations}<end_of_turn>\n"
        f"<start_of_turn>user\n"
        f"{user_input}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def parse_first_call(raw: str, skill_names: set) -> str | None:
    """Extract the function name from the first call block."""
    m = re.search(r"<start_function_call>call:(\w+)\{", raw)
    if m:
        name = m.group(1)
        return name if name in skill_names else f"__hallucinated:{name}__"
    return None


def run_test(model_path: str, locale: str = "en"):
    test_cases = TEST_CASES_BY_LOCALE[locale]
    tools = load_tools(locale)
    declarations = build_declarations(tools)
    skill_names = {t["function"]["name"] for t in tools}

    unknown = {expected for _, expected, _ in test_cases if expected} - skill_names
    if unknown:
        sys.exit(
            f"ERROR: test case(s) expect {sorted(unknown)}, which the router is "
            f"never offered. Declared: {sorted(skill_names)}"
        )

    print(f"Loading model from {model_path}...")
    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Locale: {locale} ({len(test_cases)} cases, {len(tools)} tools declared)\n")

    results = {"easy": [], "hard": [], "none": []}
    total_time = 0

    for user_input, expected, difficulty in test_cases:
        prompt = build_prompt(user_input, declarations)

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

        matched_skill = parse_first_call(raw, skill_names)
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
