"""Tests for derive_floor.py — exact fixtures, exact expected floors.

The fixture geometry matters: the joint-constraint test is built so a
precision-only sweep would pick a LOWER floor than the abstention bar
allows, which is precisely the bug the joint constraint exists to prevent
(IT at -0.10 measured 82% precision AND 93% abstention on 2026-07-20 —
both bars move).
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name.removesuffix(".py"), HERE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


df = _load("derive_floor.py")


def dump(path: Path, cases, threshold=-0.06):
    """Write a route-eval-shaped log. cases = [(expect, pick, conf)] with
    pick None for a raw abstention."""
    lines = [f"threshold:  MIN_ROUTER_CONFIDENCE = {threshold}"]
    for expect, pick, conf in cases:
        if pick is None:
            lines.append(f"VERBOSE\tCAT\tNaN\t{expect}\tNONE\tutt")
        else:
            lines.append(f"VERBOSE\tCAT\t{conf:.4f}\t{expect}\t{pick}\tutt")
    path.write_text("\n".join(lines) + "\n")


def C(cases):
    """Tuple fixtures → parse_dump's case-dict shape, for direct calls."""
    return [{"expect": e, "pick": p, "conf": c if p is not None else None}
            for e, p, c in cases]


def pos_ok(conf):
    return ("current_time", "current_time", conf)


def none_quiet():
    return ("NONE", None, None)


def none_fire(conf):
    return ("NONE", "current_time", conf)


# ── parse_dump ─────────────────────────────────────────────────────────

def test_parse_threshold_and_nan(tmp_path):
    p = tmp_path / "d.log"
    dump(p, [pos_ok(-0.05), none_quiet()], threshold=-0.06)
    cases, threshold = df.parse_dump(p)
    assert threshold == -0.06
    assert cases[0] == {"expect": "current_time", "pick": "current_time",
                        "conf": -0.05}
    assert cases[1] == {"expect": "NONE", "pick": None, "conf": None}


# ── metrics ────────────────────────────────────────────────────────────

def test_metrics_match_route_eval_definitions(tmp_path):
    cases = [
        pos_ok(-0.02),                            # fires, correct
        ("calculator", "open", -0.03),            # fires, wrong
        ("calculator", "calculator", -0.30),      # below floor: recall miss
        ("greeting", None, None),                 # raw abstention on positive
        none_quiet(),
        none_fire(-0.04),                         # NONE fires above floor
    ]
    m = df.metrics(C(cases), -0.10)
    assert m["fired"] == 2
    assert m["correct"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.25          # 1 of 4 positives
    assert m["abst_pass"] == 1
    assert m["abst_total"] == 2
    assert m["abstention"] == 0.5


# ── derive: the joint constraint ───────────────────────────────────────

def test_joint_constraint_raises_floor_above_precision_only_answer():
    # 12 correct-firing positives at -0.01..-0.12; two NONE emissions at
    # -0.11 / -0.115 among 10 NONE. Precision is 100% everywhere, so a
    # precision-only sweep picks -0.12. Abstention: at -0.115 or lower two
    # NONE cases fire (8/10 = 80%); at -0.11 exactly one fires (9/10 = 90%).
    cases = [pos_ok(round(-0.01 * i, 4)) for i in range(1, 13)]
    cases += [none_quiet()] * 8 + [none_fire(-0.11), none_fire(-0.115)]
    floor, m = df.derive(C(cases), 0.90, 0.90, min_fired=10, floor_min=-0.5)
    assert floor == -0.11
    assert m["abstention"] == 0.9
    assert m["precision"] == 1.0
    assert m["fired"] == 11             # the -0.12 positive sits below


def test_lowest_viable_floor_wins_and_floor_min_excludes():
    # 15 correct fires at -0.01..-0.15 plus one at -0.60 (below floor_min).
    # No NONE emissions. Everything from -0.10 down is viable; the answer
    # must be the LOWEST in-range candidate, -0.15 — not the first viable
    # (-0.10), and never -0.60.
    cases = [pos_ok(round(-0.01 * i, 4)) for i in range(1, 16)]
    cases += [pos_ok(-0.60)]
    cases += [none_quiet()] * 5
    floor, m = df.derive(C(cases), 0.90, 0.90, min_fired=10, floor_min=-0.5)
    assert floor == -0.15
    assert m["fired"] == 15


def test_min_fired_guard_blocks_vacuous_floors():
    cases = [pos_ok(-0.01), pos_ok(-0.02), pos_ok(-0.03), pos_ok(-0.04)]
    cases += [none_quiet()] * 20
    assert df.derive(C(cases), 0.90, 0.90, min_fired=10, floor_min=-0.5) is None


# ── main(): fallback semantics + GITHUB_OUTPUT (the workflow's interface) ──

def run_main(tmp_path, spine_cases, gen_cases=None, threshold=-0.06):
    spine = tmp_path / "spine.log"
    dump(spine, spine_cases, threshold)
    argv = [sys.executable, str(HERE / "derive_floor.py"), "--spine", str(spine)]
    if gen_cases is not None:
        gen = tmp_path / "gen.log"
        dump(gen, gen_cases, threshold)
        argv += ["--generated", str(gen)]
    gh_out = tmp_path / "gh_output"
    env = dict(os.environ, GITHUB_OUTPUT=str(gh_out))
    res = subprocess.run(argv, capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    outputs = dict(l.split("=", 1) for l in gh_out.read_text().splitlines())
    return outputs, res.stdout


def test_derived_floor_reaches_github_output(tmp_path):
    spine = [pos_ok(round(-0.01 * i, 4)) for i in range(1, 6)]
    spine += [none_quiet()] * 3
    gen = [pos_ok(round(-0.01 * i, 4)) for i in range(1, 13)]
    gen += [none_quiet()] * 8 + [none_fire(-0.11), none_fire(-0.115)]
    outputs, _ = run_main(tmp_path, spine, gen)
    assert outputs == {"min_confidence": "-0.11", "floor_source": "derived"}


def test_spine_validation_failure_falls_back(tmp_path):
    # Spine has a wrong fire at -0.04 (3/4 = 75% precision at any floor
    # <= -0.04); the generated set is clean enough that the union sweep
    # still passes. Validation must catch it and emit the compiled constant.
    spine = [pos_ok(-0.01), pos_ok(-0.02), pos_ok(-0.03),
             ("current_time", "calculator", -0.04)]
    spine += [none_quiet()] * 3
    gen = [pos_ok(round(-0.001 * i, 4)) for i in range(1, 31)]
    gen += [none_quiet()] * 10
    outputs, stdout = run_main(tmp_path, spine, gen)
    assert outputs == {"min_confidence": "-0.06", "floor_source": "fallback"}
    assert "FAILS spine validation" in stdout


def test_missing_generated_dump_falls_back(tmp_path):
    spine = [pos_ok(-0.01)] + [none_quiet()] * 3
    outputs, stdout = run_main(tmp_path, spine, gen_cases=None)
    assert outputs == {"min_confidence": "-0.06", "floor_source": "fallback"}
    assert "no generated dump" in stdout


def test_no_viable_floor_falls_back(tmp_path):
    spine = [pos_ok(-0.01)] + [none_quiet()] * 3
    # Generated: every fired positive is wrong — no floor can reach 90%.
    gen = [("current_time", "calculator", round(-0.01 * i, 4))
           for i in range(1, 15)]
    gen += [none_quiet()] * 10
    outputs, stdout = run_main(tmp_path, spine, gen)
    assert outputs == {"min_confidence": "-0.06", "floor_source": "fallback"}
    assert "no floor" in stdout
