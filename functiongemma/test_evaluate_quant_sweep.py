"""Tests for the same-runner quant-sweep reporter."""

import argparse
import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "evaluate_quant_sweep", HERE / "evaluate_quant_sweep.py"
)
eqs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eqs)


def test_parse_model():
    assert eqs.parse_model("q4_k_m=/tmp/model.gguf") == (
        "q4_k_m",
        Path("/tmp/model.gguf"),
    )
    with pytest.raises(argparse.ArgumentTypeError, match="LABEL=PATH"):
        eqs.parse_model("missing-path")
    with pytest.raises(argparse.ArgumentTypeError, match="lowercase"):
        eqs.parse_model("Q4=/tmp/model.gguf")


def test_render_summary_contains_exact_counts_and_gate():
    results = [
        {
            "label": "q4_k_m",
            "bytes": 253 * 1024 * 1024,
            "floor": -0.06,
            "floor_source": "fallback",
            "metrics": {
                "abst_pass": 49,
                "abst_total": 50,
                "abstention": 0.98,
                "correct": 25,
                "fired": 43,
                "precision": 25 / 43,
                "pos_total": 320,
                "recall": 25 / 320,
            },
            "passed": False,
        }
    ]
    summary = eqs.render_summary(results)
    assert "| q4_k_m | 253.0 | -0.0600 | fallback" in summary
    assert "| 49/50 (98%) | 25/43 (58%) | 25/320 (8%) | FAIL |" in summary


def test_parse_route_eval_summary_uses_executed_counts(tmp_path):
    log = tmp_path / "route-eval.log"
    log.write_text(
        "threshold:  MIN_ROUTER_CONFIDENCE = -0.003\n"
        "abstention: 50/50 (100%, min 90%)\n"
        "precision:  12/14 fired (86%, min 85%)\n"
        "recall:     12/320 (4%) — coverage KPI, tracked not gated\n"
    )

    assert eqs.parse_route_eval_summary(log) == {
        "abstention": 1.0,
        "abst_pass": 50,
        "abst_total": 50,
        "precision": 12 / 14,
        "fired": 14,
        "correct": 12,
        "recall": 12 / 320,
        "pos_total": 320,
    }
