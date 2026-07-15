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


def test_negatives_for_locale_en_includes_curated_general_knowledge():
    pool = gends.negatives_for_locale("en")
    # Real behaviour: the English negative pool teaches the router to abstain on
    # general-knowledge questions. Pin specific curated members (from
    # NEGATIVE_EXAMPLES) rather than restating the implementation expression.
    assert "what is the capital of France" in pool
    assert "who wrote Romeo and Juliet" in pool
    assert "what language do they speak in Brazil" in pool
    # The templated factual expander must also contribute — e.g. capitals of
    # other countries beyond the curated France entry.
    assert any(
        n.startswith("what is the capital of ") and n != "what is the capital of France"
        for n in pool
    )
    # Every entry is a non-empty string.
    assert all(isinstance(n, str) and n for n in pool)


def test_negatives_for_locale_it_returns_italian_pool():
    pool = gends.negatives_for_locale("it")
    en_pool = gends.negatives_for_locale("en")
    assert pool != en_pool, "Italian pool is not the English pool"
    assert "qual è la capitale della Francia" in pool
    assert "chi ha scritto Romeo e Giulietta" in pool
    # Templated Italian factuals contribute beyond the curated list.
    assert any(
        n.startswith("qual è la capitale") and n != "qual è la capitale della Francia"
        for n in pool
    )
    assert all(isinstance(n, str) and n for n in pool)


def test_italian_negatives_do_not_collide_with_skill_triggers():
    # A negative containing a live Italian trigger is a poisoned sample: it
    # teaches the router to abstain on an utterance a skill should own.
    triggers = [
        "ricorda", "timer", "sveglia", "meteo", "che ora", "che giorno",
        "accendi", "spegni", "calcola",
    ]
    offenders = [
        n for n in gends.negatives_for_locale("it")
        if any(t in n.lower() for t in triggers)
    ]
    assert offenders == [], f"Italian negatives collide with skill triggers: {offenders[:5]}"


def test_italian_negatives_clear_the_real_trigger_rules():
    # The substring list above is a floor. This checks the pool against the
    # engine's ACTUAL matching rules (whole-word for the union dictionaries
    # and SKILL.it.md keywords, substring for calculator.rs), which catch
    # collisions the plain substring list cannot — e.g. date's bag-of-words
    # ["che","giorno"] firing on "in che giorno è nato Napoleone", search's
    # bare `trova` owning "dove si trova X", or weather's `tempo`.
    offenders = gends._it_trigger_offenders(gends.negatives_for_locale("it"))
    assert offenders == [], f"Italian negatives hit live triggers: {offenders[:5]}"


def test_it_trigger_offenders_catches_a_poisoned_sample():
    # Guard the guard: the checker must actually reject the collisions that
    # motivated it, or the test above passes vacuously.
    assert gends._it_trigger_offenders(["in che giorno è nato Napoleone"])  # date
    assert gends._it_trigger_offenders(["dove si trova la Torre Eiffel"])  # search
    assert gends._it_trigger_offenders(["quanto tempo dura un anno su Marte"])  # weather
    assert gends._it_trigger_offenders(["che cos'è la musica medievale"])  # calculator "eval"
    assert gends._it_trigger_offenders(["chi ha inventato la lancia"])  # open
    # ...and must not reject a clean general-knowledge question.
    assert gends._it_trigger_offenders(["qual è la capitale della Francia"]) == []


def test_italian_pool_is_large_enough_to_scale_negatives():
    # generate_negative_samples scales negatives to the positive count and
    # warns below target. An undersized Italian pool = the r70→r75
    # abstention regression, in Italian.
    it_uniques = len(dict.fromkeys(gends.negatives_for_locale("it")))
    en_uniques = len(dict.fromkeys(gends.negatives_for_locale("en")))
    assert it_uniques >= 500, f"Italian pool too small: {it_uniques}"
    assert it_uniques >= en_uniques * 0.9, f"it {it_uniques} << en {en_uniques}"


def test_negatives_for_locale_unknown_falls_back_to_english():
    assert gends.negatives_for_locale("de") == gends.negatives_for_locale("en")


def test_export_skills_passes_locale_after_double_dash(monkeypatch):
    # Pins the cross-repo contract: cargo needs `--` before binary args, and
    # the binary expects `--locale <xx>`. Getting this wrong silently exports
    # English for every locale.
    captured = {}

    class _Result:
        stdout = "[]"

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(gends.subprocess, "run", fake_run)
    gends.export_skills(Path("/tmp/engine"), "it")

    cmd = captured["cmd"]
    assert "--" in cmd, "cargo requires -- before binary args"
    assert cmd[cmd.index("--") + 1 :] == ["--locale", "it"]
    assert "export-utterances" in cmd
