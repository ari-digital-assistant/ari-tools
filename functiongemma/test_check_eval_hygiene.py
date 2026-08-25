"""Unit tests for the eval-bank hygiene checker.

Hermetic on purpose: the oracle is stubbed. Whether a given utterance is a
keyword-hit is `keyword-hit`'s business and it has its own tests in
ari-engine — what's tested here is that this script reads a bank the way
route-eval reads one, and reports what it finds.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "hygiene", Path(__file__).parent / "check_eval_hygiene.py"
)
hygiene = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hygiene)


def _bank(tmp_path, *lines):
    p = tmp_path / "bank.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def _case(utterance, expect="dev.heyari.weather"):
    return json.dumps({"utterance": utterance, "expect": expect})


def test_blank_lines_and_comments_are_skipped(tmp_path):
    # route-eval tolerates both, so the generated banks carry a comment
    # header. Counting them as cases would misalign every line number.
    p = _bank(tmp_path, "// generated set", "", _case("hello"), "   ")
    assert [c[1] for c in hygiene.load_cases(p)] == ["hello"]


def test_line_numbers_survive_the_skipped_lines(tmp_path):
    # The offender report is only actionable if the number points at the
    # line a human would edit.
    p = _bank(tmp_path, "// header", "", _case("first"), _case("second"))
    assert [(c[0], c[1]) for c in hygiene.load_cases(p)] == [(3, "first"), (4, "second")]


def test_malformed_json_is_fatal(tmp_path):
    p = _bank(tmp_path, "{not json")
    with pytest.raises(SystemExit):
        hygiene.load_cases(p)


def test_a_case_missing_its_fields_is_fatal(tmp_path):
    # Silently skipping one would shrink the graded set without saying so.
    p = _bank(tmp_path, json.dumps({"utterance": "hello"}))
    with pytest.raises(SystemExit):
        hygiene.load_cases(p)


def test_check_reports_only_the_hits(tmp_path, monkeypatch):
    p = _bank(tmp_path, _case("a miss"), _case("a hit"), _case("another miss"))
    monkeypatch.setattr(hygiene.gd, "keyword_hits",
                        lambda engine, texts, locale, skills_dir=None:
                        [t == "a hit" for t in texts])
    offenders = hygiene.check(p, "en", Path("/engine"), Path("/skills"))
    assert [(o[0], o[1]) for o in offenders] == [(2, "a hit")]


def test_a_clean_bank_reports_nothing(tmp_path, monkeypatch):
    p = _bank(tmp_path, _case("one"), _case("two"))
    monkeypatch.setattr(hygiene.gd, "keyword_hits",
                        lambda engine, texts, locale, skills_dir=None:
                        [False] * len(texts))
    assert hygiene.check(p, "en", Path("/engine"), Path("/skills")) == []


def test_an_empty_bank_never_shells_out(tmp_path):
    # No oracle stub here: calling it would try to run cargo. An empty bank
    # must short-circuit before that.
    p = tmp_path / "empty.jsonl"
    p.write_text("// nothing but a header\n")
    assert hygiene.check(p, "en", Path("/engine"), Path("/skills")) == []


def test_raw_text_reaches_the_oracle(tmp_path, monkeypatch):
    # keyword_hits normalises internally, exactly as production does with
    # raw user input. Pre-normalising here would grade a different string
    # than route-eval does.
    seen = {}
    p = _bank(tmp_path, _case("Is the HUMIDITY bad in Atlanta?"))
    monkeypatch.setattr(hygiene.gd, "keyword_hits",
                        lambda engine, texts, locale, skills_dir=None:
                        seen.setdefault("texts", texts) and [] or [False] * len(texts))
    hygiene.check(p, "en", Path("/engine"), Path("/skills"))
    assert seen["texts"] == ["Is the HUMIDITY bad in Atlanta?"]


def test_every_committed_bank_is_covered():
    # The table is hand-maintained so a scratch file can't become a checked
    # artefact. The cost is that a new bank has to be added here — this is
    # the reminder.
    listed = {name for names in hygiene.BANKS.values() for name in names}
    on_disk = {p.name for p in hygiene.HERE.glob("routing-eval*.jsonl")}
    assert on_disk - listed == set(), (
        "a routing-eval bank exists that check_eval_hygiene.py never grades"
    )
