"""Tests for the manifest-resolution seam in author-frames.py.

The nightly drafts banks for whatever check_banks.py reports as stale, in
whatever locale the matrix leg is running. That list is derived from the
skills the ROUTER sees, and the engine registers every skill on every device
regardless of the `languages` it declares — so the drafter is routinely
asked for Italian banks for an English-only skill and must cope.
"""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "authorframes", Path(__file__).parent / "author-frames.py"
)
af = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(af)


def _skill(root, name, locale, skill_id):
    d = root / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"SKILL.{locale}.md").write_text(
        "---\n"
        f"description: {name} in {locale}\n"
        "metadata:\n"
        "  ari:\n"
        f"    id: {skill_id}\n"
        "---\nbody\n"
    )


@pytest.fixture
def skills_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(af, "find_skills_dir", lambda: tmp_path)
    return tmp_path


def test_the_locales_own_manifest_wins(skills_dir):
    _skill(skills_dir, "weather", "en", "dev.heyari.weather")
    _skill(skills_dir, "weather", "it", "dev.heyari.weather")
    assert "weather in it" in af.skill_spec("dev.heyari.weather", "it")


def test_an_english_only_skill_falls_back_instead_of_dying(skills_dir, capsys):
    # dev.heyari.message, 2026-08-21 → 2026-08-25. Four nightlies died here.
    _skill(skills_dir, "message", "en", "dev.heyari.message")
    spec = af.skill_spec("dev.heyari.message", "it")
    assert "message in en" in spec
    assert "no SKILL.it.md" in capsys.readouterr().err, (
        "falling back silently would hide a skill that needs translating"
    )


def test_english_lookup_does_not_warn(skills_dir, capsys):
    _skill(skills_dir, "message", "en", "dev.heyari.message")
    af.skill_spec("dev.heyari.message", "en")
    assert capsys.readouterr().err == ""


def test_a_skill_with_no_manifest_at_all_is_still_fatal(skills_dir):
    _skill(skills_dir, "weather", "en", "dev.heyari.weather")
    with pytest.raises(SystemExit):
        af.skill_spec("dev.heyari.nonexistent", "it")


def test_the_id_must_match_exactly(skills_dir):
    # A prefix match would hand back the wrong skill's ground truth and the
    # drafter would write a bank of confidently wrong frames.
    _skill(skills_dir, "message", "en", "dev.heyari.message")
    with pytest.raises(SystemExit):
        af.skill_spec("dev.heyari.mess", "it")
