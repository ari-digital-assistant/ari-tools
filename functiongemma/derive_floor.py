#!/usr/bin/env python3
"""Derive the per-model confidence floor from route-eval VERBOSE dumps.

route-eval with ROUTE_EVAL_VERBOSE=1 prints one line per eval case:

    VERBOSE\t<category>\t<conf>\t<expect>\t<raw pick>\t<utterance>

where <raw pick>/<conf> are the router's PRE-threshold emission (pick
"NONE" + conf NaN when it produced no call). That makes every candidate
floor evaluable offline: this script sweeps them and picks the LOWEST
floor at which, over the spine + generated-eval union, BOTH gate bars
still hold — precision-when-firing >= 90% AND abstention >= 90%. The
constraint is deliberately joint: lowering the floor also lets NONE-case
emissions through (measured 2026-07-20: IT at -0.10 lost abstention to
28/30), so a precision-only sweep can ship a floor that fails the
abstention gate.

Safety rails, in order:
  - derivation requires the GENERATED dump. The 40-positive spine alone
    is noise for this purpose (one flipped case moves precision 5-6
    points); no generated dump => fallback, never a spine-only sweep.
  - a viable floor must fire on >= MIN_FIRED positives (no vacuous
    precision) and sit above FLOOR_MIN (a mean log-prob below that is a
    router guessing, whatever the sweep says).
  - the winner is VALIDATED on the spine alone: if the spine graded at
    the derived floor fails either bar, the derivation is not trusted.
  - every fallback path emits the compiled constant (parsed from the
    spine dump's own "threshold:" line) with floor_source=fallback. The
    nightly never fails because of this script's judgement — only
    because of unusable inputs (missing/unparseable spine dump).

Output: a human summary on stdout, and min_confidence= / floor_source=
lines appended to $GITHUB_OUTPUT when set.

Usage:
    derive_floor.py --spine spine.log [--generated gen.log]
                    [--precision-min 0.90] [--abstain-min 0.90]
                    [--min-fired 10] [--floor-min -0.5]
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path

THRESHOLD_RE = re.compile(r"MIN_ROUTER_CONFIDENCE\s*=\s*(-?[0-9.]+)")


def parse_dump(path: Path) -> tuple:
    """Returns (cases, compiled_threshold|None).

    Each case: {"expect": str, "pick": str|None, "conf": float|None} with
    pick/conf None when the router emitted nothing (raw abstention).
    """
    cases, threshold = [], None
    for line in path.read_text().splitlines():
        m = THRESHOLD_RE.search(line)
        if m:
            threshold = float(m.group(1))
        if not line.startswith("VERBOSE\t"):
            continue
        parts = line.split("\t", 5)
        if len(parts) != 6:
            sys.exit(f"ERROR: malformed VERBOSE line in {path}: {line!r}")
        _, _category, conf_s, expect, pick, _utt = parts
        conf = float(conf_s)
        if pick == "NONE" or math.isnan(conf):
            cases.append({"expect": expect, "pick": None, "conf": None})
        else:
            cases.append({"expect": expect, "pick": pick, "conf": conf})
    return cases, threshold


def metrics(cases: list, floor: float) -> dict:
    """Gate metrics for `cases` graded at `floor` — same definitions as
    route-eval: abstention over NONE cases, precision over fired positives,
    recall over all positives. Vacuous precision (nothing fired) is 1.0,
    matching route-eval; the min-fired guard in derive() is what keeps the
    sweep from exploiting that."""
    abst_total = abst_pass = pos_total = pos_fired = pos_correct = 0
    for c in cases:
        firing = c["pick"] is not None and c["conf"] >= floor
        if c["expect"].upper() == "NONE":
            abst_total += 1
            if not firing:
                abst_pass += 1
        else:
            pos_total += 1
            if firing:
                pos_fired += 1
                if c["pick"] == c["expect"]:
                    pos_correct += 1
    return {
        "abstention": abst_pass / abst_total if abst_total else 1.0,
        "abst_pass": abst_pass, "abst_total": abst_total,
        "precision": pos_correct / pos_fired if pos_fired else 1.0,
        "fired": pos_fired, "correct": pos_correct,
        "recall": pos_correct / pos_total if pos_total else 0.0,
        "pos_total": pos_total,
    }


def derive(cases: list, precision_min: float, abstain_min: float,
           min_fired: int, floor_min: float):
    """Lowest candidate floor meeting the joint constraint, or None.

    Candidates are the distinct observed confidences — between two observed
    values every floor grades identically, so nothing else needs sweeping.
    """
    candidates = sorted({c["conf"] for c in cases if c["conf"] is not None})
    viable = None
    for t in candidates:                      # ascending: first hit is lowest
        if t < floor_min:
            continue
        m = metrics(cases, t)
        if (m["precision"] >= precision_min and m["abstention"] >= abstain_min
                and m["fired"] >= min_fired):
            viable = (t, m)
            break
    return viable


def fmt(label: str, m: dict, floor: float) -> str:
    return (f"{label} @ {floor:.4f}: abstention {m['abst_pass']}/"
            f"{m['abst_total']} ({m['abstention']:.0%}), precision "
            f"{m['correct']}/{m['fired']} fired ({m['precision']:.0%}), "
            f"recall {m['correct']}/{m['pos_total']} ({m['recall']:.0%})")


def emit(min_confidence: float, source: str) -> None:
    print(f"floor: min_confidence={min_confidence} source={source}")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"min_confidence={min_confidence}\n")
            f.write(f"floor_source={source}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spine", required=True, type=Path)
    ap.add_argument("--generated", type=Path, default=None)
    ap.add_argument("--precision-min", type=float, default=0.90)
    ap.add_argument("--abstain-min", type=float, default=0.90)
    ap.add_argument("--min-fired", type=int, default=10)
    ap.add_argument("--floor-min", type=float, default=-0.5)
    args = ap.parse_args()

    if not args.spine.is_file():
        sys.exit(f"ERROR: spine dump not found: {args.spine}")
    spine_cases, compiled = parse_dump(args.spine)
    if compiled is None:
        sys.exit(f"ERROR: no 'MIN_ROUTER_CONFIDENCE =' line in {args.spine} — "
                 f"cannot establish the fallback floor")
    if not spine_cases:
        sys.exit(f"ERROR: no VERBOSE lines in {args.spine} — was route-eval "
                 f"run with ROUTE_EVAL_VERBOSE=1?")

    if args.generated is None or not args.generated.is_file():
        print("no generated dump — derivation needs generated-eval scale; "
              "falling back to the compiled constant.")
        emit(compiled, "fallback")
        return
    gen_cases, _ = parse_dump(args.generated)
    if not gen_cases:
        print("generated dump has no VERBOSE lines — falling back.")
        emit(compiled, "fallback")
        return

    union = spine_cases + gen_cases
    result = derive(union, args.precision_min, args.abstain_min,
                    args.min_fired, args.floor_min)
    if result is None:
        print(f"no floor in [{args.floor_min}, 0] meets precision>="
              f"{args.precision_min:.0%} AND abstention>="
              f"{args.abstain_min:.0%} with >={args.min_fired} fired — "
              f"falling back to the compiled constant.")
        emit(compiled, "fallback")
        return
    floor, union_m = result

    spine_m = metrics(spine_cases, floor)
    print(fmt("union", union_m, floor))
    print(fmt("spine", spine_m, floor))
    if (spine_m["precision"] < args.precision_min
            or spine_m["abstention"] < args.abstain_min):
        print(f"derived floor {floor:.4f} FAILS spine validation — the "
              f"generated set skewed the sweep; falling back to the "
              f"compiled constant.")
        emit(compiled, "fallback")
        return

    print(fmt("spine@compiled", metrics(spine_cases, compiled), compiled))
    emit(floor, "derived")


if __name__ == "__main__":
    main()
