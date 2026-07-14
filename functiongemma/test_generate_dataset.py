"""Unit tests for the locale seams in generate-dataset.py."""
import importlib.util
from pathlib import Path

# The module filename has a hyphen, so load it explicitly.
_spec = importlib.util.spec_from_file_location(
    "gends", Path(__file__).parent / "generate-dataset.py"
)
gends = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gends)


def _write_manifest(dir_: Path, locale: str, skill_id: str) -> None:
    (dir_ / f"SKILL.{locale}.md").write_text(
        "---\n"
        f"description: desc-{locale}\n"
        "metadata:\n"
        "  ari:\n"
        f"    id: {skill_id}\n"
        "    type: wasm\n"
        "    examples:\n"
        f"      - text: utterance-{locale}\n"
        "---\nbody\n"
    )


def test_load_community_skills_prefers_requested_locale(tmp_path):
    skills_root = tmp_path / "skills" / "demo"
    skills_root.mkdir(parents=True)
    _write_manifest(skills_root, "en", "demo")
    _write_manifest(skills_root, "it", "demo")

    out = gends.load_community_skills(tmp_path, "it")

    assert len(out) == 1
    assert out[0]["id"] == "demo"
    assert out[0]["examples"][0]["text"] == "utterance-it"


def test_load_community_skills_falls_back_to_english(tmp_path):
    skills_root = tmp_path / "skills" / "demo"
    skills_root.mkdir(parents=True)
    _write_manifest(skills_root, "en", "demo")  # no it.md

    out = gends.load_community_skills(tmp_path, "it")

    assert out[0]["examples"][0]["text"] == "utterance-en"


def test_negatives_for_locale_en_matches_english_pool():
    expected = gends.NEGATIVE_EXAMPLES + gends.generate_factual_negatives()
    assert gends.negatives_for_locale("en") == expected
