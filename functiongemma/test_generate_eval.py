"""Tests for generate-eval.py's filter pipeline and output format.

The Gemini call and the engine oracles are injected/stubbed — these tests
pin the logic that decides what enters the committed eval: dedupe, keyword
filtering, collision filtering (raw AND normalised key), and the exact file
format route-eval and load_eval_keys will parse.
"""

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_").removesuffix(".py"), HERE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ge = _load("generate-eval.py")
from corpus_expander import load_eval_keys, loose_key  # noqa: E402


def _filter(cands, avoid=(), corpus=(), keep=10, hits=None, norms=None):
    """Run filter_candidates with fully stubbed oracles."""
    hit_map = hits or {}
    norm_map = norms or {}
    return ge.filter_candidates(
        cands, "en", Path("/nonexistent"), None,
        set(avoid), set(corpus), keep,
        oracle=lambda texts: [hit_map.get(t, False) for t in texts],
        normalizer=lambda texts: [norm_map.get(t, t) for t in texts],
    )


def test_dedupe_and_whitespace_hygiene():
    kept, stats = _filter([
        "  what   time is\tit in  Tokyo ",
        "what time is it in Tokyo",   # loose-key dupe of the first
        "",
        "   ",
    ])
    assert kept == ["what time is it in Tokyo"]
    assert stats["raw"] == 4
    assert stats["deduped"] == 1
    assert stats["kept"] == 1


def test_keyword_hits_are_dropped():
    kept, stats = _filter(
        ["open the camera", "get the camera going"],
        hits={"open the camera": True},
    )
    assert kept == ["get the camera going"]
    assert stats["keyword_hits"] == 1


def test_eval_collision_dropped_on_raw_key():
    kept, stats = _filter(
        ["Who painted Starry Night?", "who sculpted David"],
        avoid={loose_key("who painted starry night")},
    )
    assert kept == ["who sculpted David"]
    assert stats["eval_collisions"] == 1


def test_corpus_collision_dropped_on_normalised_key():
    # The raw key differs from the corpus key; only the normalised form
    # collides — exactly the near-miss the nightly guard would let into
    # training while route-eval still measured the case.
    kept, stats = _filter(
        ["what's the damage", "how much altogether"],
        corpus={loose_key("whats the damage")},
        norms={"what's the damage": "whats the damage"},
    )
    assert kept == ["how much altogether"]
    assert stats["corpus_collisions"] == 1


def test_keep_cap_is_exact():
    kept, _ = _filter([f"utterance number {i}" for i in range(30)], keep=7)
    assert len(kept) == 7
    assert kept[0] == "utterance number 0"


def test_write_eval_format_is_parseable_and_exact(tmp_path):
    out = tmp_path / "routing-eval.gen.jsonl"
    ge.write_eval(
        out,
        {"dev.heyari.timer": ["count me down from five minutes"],
         "current_time": ["is it beer o clock yet"]},
        ["who sculpted david"],
        "en", "gemini-3.5-flash",
    )
    text = out.read_text()
    lines = [l for l in text.splitlines() if l and not l.startswith("//")]
    # Sorted by skill id, NONE last.
    assert json.loads(lines[0]) == {
        "utterance": "is it beer o clock yet", "expect": "current_time"}
    assert json.loads(lines[1]) == {
        "utterance": "count me down from five minutes",
        "expect": "dev.heyari.timer"}
    assert json.loads(lines[2]) == {
        "utterance": "who sculpted david", "expect": "NONE"}
    # load_eval_keys (the decontamination reader) must see all three.
    assert load_eval_keys([out]) == {
        loose_key("is it beer o clock yet"),
        loose_key("count me down from five minutes"),
        loose_key("who sculpted david"),
    }
    assert "gemini-3.5-flash" in text


def test_positive_prompt_carries_spec_and_avoid_list():
    skill = {"id": "dev.heyari.timer", "description": "Set countdown timers",
             "parameters": {"type": "object"},
             "examples": [{"text": "set a timer for 5 minutes", "args": {}}]}
    p = ge.positive_prompt(skill, ["start a {N1} minute timer"], 60, "en")
    assert "dev.heyari.timer" in p
    assert "Set countdown timers" in p
    assert "start a {N1} minute timer" in p
    assert "60 utterances" in p


def test_none_prompt_lists_every_skill():
    skills = [{"id": "current_time", "description": "Tells the time"},
              {"id": "dev.heyari.weather", "description": "Weather forecasts"}]
    p = ge.none_prompt(skills, ["what is the capital of Denmark"], 150, "it")
    assert "current_time: Tells the time" in p
    assert "dev.heyari.weather: Weather forecasts" in p
    assert "what is the capital of Denmark" in p
    assert "accents" in p  # Italian language rule engaged
