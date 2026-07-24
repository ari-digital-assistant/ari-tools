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
