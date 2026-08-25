#!/usr/bin/env python3
"""Re-grade the committed routing-eval banks against the LIVE keyword scorer.

Why this exists — the 2026-08-18 incident. `generate-eval.py` filters every
candidate through the keyword oracle at generation time, so a bank is clean
on the day it is written. Nothing re-checked it afterwards. Weather 0.2.0
then added a `\\bhumid(ity)?\\b` pattern, which turned the already-committed
case "Is the humidity going to be bad in Atlanta this afternoon?" from a
keyword-MISS into a keyword-HIT. route-eval refuses to score a polluted bank
(exit 3), so the nightly train died — after two hours of Modal GPU, because
route-eval loads the model before it checks the bank. Seven nights in a row.

The banks are not wrong. They are older than the skills they test. A skill
that grows a trigger word silently swallows the oblique cases written for
it, and there is no way to notice except by asking the scorer again.

So: same oracle route-eval uses (`engine.keyword_decision`, reached through
generate-dataset.keyword_hits), same rule (ANY hit is pollution, whatever
skill wins it), but no model and no GPU. Cheap enough to run in ari-tools CI
and as a pre-flight in the nightly, before a GPU-second is spent.

Usage:
    check_eval_hygiene.py --locale en BANK [BANK ...]
    check_eval_hygiene.py --all          # every committed bank, both locales

Exit 0 clean, 1 polluted, 2 usage/harness error.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# generate-dataset.py has a hyphen in its name, so it can't be imported by
# name. Loading it explicitly is what keeps this script on the SAME oracle
# as the corpus filter — a reimplementation here would drift, and drift is
# exactly the failure this script exists to catch.
_spec = importlib.util.spec_from_file_location("gends", HERE / "generate-dataset.py")
gd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gd)

# The nightly's banks, by locale. Kept here rather than globbed so a stray
# scratch file in functiongemma/ can't quietly become a checked artefact.
BANKS = {
    "en": ["routing-eval.jsonl", "routing-eval.gen.jsonl"],
    "it": ["routing-eval.it.jsonl", "routing-eval.gen.it.jsonl"],
}


def load_cases(path: Path) -> list:
    """Read a bank. Same tolerances as route-eval: blank lines and `//`
    comments are skipped, everything else must be a well-formed case."""
    cases = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"ERROR: {path}:{n}: {e}")
        if "utterance" not in case or "expect" not in case:
            sys.exit(f"ERROR: {path}:{n}: case needs both `utterance` and `expect`")
        cases.append((n, case["utterance"], case["expect"]))
    return cases


def check(path: Path, locale: str, engine_dir: Path, skills_dir: Path | None) -> list:
    """Return the polluted cases in one bank as (line, utterance, expect)."""
    cases = load_cases(path)
    if not cases:
        return []
    # Pass RAW text — keyword_hits normalises internally, exactly as
    # production does with raw user input.
    hits = gd.keyword_hits(engine_dir, [c[1] for c in cases], locale,
                           skills_dir=skills_dir)
    return [c for c, hit in zip(cases, hits) if hit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("banks", nargs="*", type=Path)
    ap.add_argument("--locale", default="en", help="locale for the banks named on the command line")
    ap.add_argument("--all", action="store_true", help="check every committed bank in both locales")
    args = ap.parse_args()

    if args.all == bool(args.banks):
        ap.error("give bank paths, or --all, but not both")

    engine_dir = gd.find_engine_dir()
    skills_dir = gd.find_skills_dir()
    if skills_dir is None:
        # Without the community manifests the oracle sees only the built-ins,
        # so a community skill's own patterns are never in the running and
        # this check would pass a bank route-eval will reject. A false green
        # here is worse than no check at all.
        print("ERROR: could not find ari-skills. Set ARI_SKILLS_DIR — without the "
              "community manifests this check cannot see the patterns that "
              "actually poison a bank.", file=sys.stderr)
        return 2

    if args.all:
        work = [(locale, HERE / name) for locale, names in BANKS.items() for name in names]
    else:
        work = [(args.locale, p) for p in args.banks]

    polluted = 0
    for locale, path in work:
        if not path.is_file():
            # A missing generated bank is a legitimate state — the nightly
            # falls back to the spine when one isn't there. Say so and move on.
            print(f"– {path.name} ({locale}): not present, skipped")
            continue
        offenders = check(path, locale, engine_dir, skills_dir)
        if not offenders:
            print(f"✓ {path.name} ({locale}): clean")
            continue
        polluted += len(offenders)
        print(f"✗ {path.name} ({locale}): {len(offenders)} polluted case(s)")
        for line, utterance, expect in offenders:
            print(f"    {path.name}:{line}  {utterance!r}  expect={expect}")

    if polluted:
        print(
            f"\n{polluted} case(s) are already won by the keyword scorer, so the "
            f"router never sees them in production.\nA router promotion gate must "
            f"contain only keyword-MISSES — route-eval will refuse these (exit 3).\n"
            f"Re-run generate-eval.yml to rewrite the banks against the current skills.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
