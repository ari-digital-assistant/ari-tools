"""Unit tests for the locale seams in generate-dataset.py."""
import importlib.util
from pathlib import Path

import pytest

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


def test_normalize_texts_matches_the_engine(tmp_path):
    # Not a replica test: this shells out to the engine's own normalize bin,
    # which is the whole point — one source of truth for train/serve parity.
    engine = Path(__file__).resolve().parents[2] / "ari-engine"
    out = gends.normalize_texts(engine, ["what's the time", "quick, what time is it?"], "en")
    assert out == ["what is the time", "quick what time is it"]


def test_normalize_texts_preserves_order_and_count():
    engine = Path(__file__).resolve().parents[2] / "ari-engine"
    # Deliberately NOT number words: English normalisation converts those to
    # digits (see the test below), which would muddy an order/count check.
    texts = ["hello", "WORLD", "stop!"]
    out = gends.normalize_texts(engine, texts, "en")
    assert out == ["hello", "world", "stop"]


def test_english_number_words_become_digits_but_italian_is_untouched():
    # normalize_input runs replace_number_words for "en" ONLY. This is the
    # single most surprising thing normalisation does to the English corpus —
    # pin it so it's a documented decision, not a shock in a diff.
    engine = Path(__file__).resolve().parents[2] / "ari-engine"
    assert gends.normalize_texts(
        engine, ["how much is fifteen percent of two hundred"], "en"
    ) == ["how much is 15 percent of 200"]
    # Italian gets no number-word replacement, so its equivalent is unchanged.
    assert gends.normalize_texts(
        engine, ["quanto fa il quindici percento di duecento"], "it"
    ) == ["quanto fa il quindici percento di duecento"]


def test_normalize_texts_is_idempotent():
    # Normalising already-normalised text must be a no-op, or the corpus
    # would depend on how many times the pipeline ran.
    engine = Path(__file__).resolve().parents[2] / "ari-engine"
    once = gends.normalize_texts(engine, ["What's the TIME?"], "en")
    twice = gends.normalize_texts(engine, once, "en")
    assert once == twice


def test_router_ineligible_skills_are_excluded():
    # search sets router_eligible=false, so router_catalog() never offers it.
    # Training it teaches the model to call a function that isn't on the menu.
    skills = [
        {"id": "search", "description": "d", "router_eligible": False,
         "parameters": {}, "examples": [{"text": "find x", "args": {}}]},
        {"id": "current_time", "description": "d", "router_eligible": True,
         "parameters": {}, "examples": [{"text": "what time", "args": {}}]},
    ]
    kept = gends.router_eligible_skills(skills)
    assert [s["id"] for s in kept] == ["current_time"]


def test_skills_without_the_flag_default_to_eligible():
    # Community skills come from SKILL.md manifests, which have no
    # router_eligible concept — they must not be silently dropped.
    skills = [{"id": "dev.heyari.weather", "description": "d",
               "parameters": {}, "examples": [{"text": "meteo", "args": {}}]}]
    assert gends.router_eligible_skills(skills) == skills


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


def _fake_keyword_hit(monkeypatch, captured, verdicts: str):
    class _Result:
        stdout = verdicts

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(gends.subprocess, "run", fake_run)


def test_keyword_hits_points_the_oracle_at_the_skills_root(monkeypatch):
    # Pins the cross-repo contract with keyword-hit's --skills-dir. The
    # binary wants the `skills/` ROOT (the directory whose children are skill
    # folders), but find_skills_dir returns the repo checkout one level up.
    # Passing the checkout would make the loader find zero skills and the
    # filter would silently keep every community example the scorer wins.
    captured = {}
    _fake_keyword_hit(monkeypatch, captured, "true\nfalse\n")

    got = gends.keyword_hits(
        Path("/tmp/engine"), ["a", "b"], "it", Path("/tmp/ari-skills")
    )

    assert got == [True, False]
    cmd = captured["cmd"]
    assert cmd[cmd.index("--") + 1 :] == [
        "--locale", "it", "--skills-dir", "/tmp/ari-skills/skills",
    ]
    assert "--no-default-features" in cmd, (
        "the training container has no clang; the llm feature must stay off"
    )


def test_keyword_hits_omits_the_flag_when_there_is_no_skills_checkout(monkeypatch):
    # No checkout means builtin-only verdicts — degraded but valid. The flag
    # must be absent entirely rather than passed as an empty string, which
    # the binary would treat as a real path and fail on.
    captured = {}
    _fake_keyword_hit(monkeypatch, captured, "false\n")

    assert gends.keyword_hits(Path("/tmp/engine"), ["a"], "en", None) == [False]
    cmd = captured["cmd"]
    assert "--skills-dir" not in cmd
    assert cmd[cmd.index("--") + 1 :] == ["--locale", "en"]


def _skill(skill_id: str, *texts: str) -> dict:
    return {"id": skill_id, "examples": [{"text": t} for t in texts]}


def _oracle(*verdicts):
    """Oracle stub returning a fixed verdict list, recording what it was asked."""
    seen = []

    def call(texts):
        seen.append(list(texts))
        return list(verdicts)

    call.seen = seen
    return call


def test_drop_keyword_hits_pairs_verdicts_across_multiple_skills():
    # The verdicts are flat but the structure is nested, so the hazard is a
    # skill boundary shifting the pairing. Three skills of differing sizes,
    # with the hits deliberately straddling every boundary.
    skills = [
        _skill("one", "a1", "a2"),
        _skill("two", "b1", "b2", "b3"),
        _skill("three", "c1"),
    ]
    #                a1     a2    b1     b2     b3     c1
    gends.drop_keyword_hits(skills, _oracle(False, True, True, False, False, False))

    assert [e["text"] for e in skills[0]["examples"]] == ["a1"]
    assert [e["text"] for e in skills[1]["examples"]] == ["b2", "b3"]
    assert [e["text"] for e in skills[2]["examples"]] == ["c1"]


def test_drop_keyword_hits_asks_the_oracle_in_all_skills_order():
    # The oracle is a subprocess whose reply is matched positionally, so the
    # order the texts go out in IS the contract. Pin it.
    skills = [_skill("one", "a1", "a2"), _skill("two", "b1")]
    oracle = _oracle(False, False, False)

    gends.drop_keyword_hits(skills, oracle)

    assert oracle.seen == [["a1", "a2", "b1"]]


def test_drop_keyword_hits_keeps_everything_when_no_verdict_is_a_hit():
    skills = [_skill("one", "a1", "a2"), _skill("two", "b1")]

    gends.drop_keyword_hits(skills, _oracle(False, False, False))

    assert [e["text"] for e in skills[0]["examples"]] == ["a1", "a2"]
    assert [e["text"] for e in skills[1]["examples"]] == ["b1"]


def test_drop_keyword_hits_can_empty_a_skill():
    # main() exits on an emptied skill; the filter itself must still produce
    # that state rather than silently retaining an example.
    skills = [_skill("one", "a1", "a2")]

    gends.drop_keyword_hits(skills, _oracle(True, True))

    assert skills[0]["examples"] == []


def test_drop_keyword_hits_raises_when_verdicts_run_short():
    # Fewer verdicts than examples. The tail must NOT be quietly kept.
    skills = [_skill("one", "a1", "a2"), _skill("two", "b1")]

    with pytest.raises(ValueError, match="ran out at skill 'two'"):
        gends.drop_keyword_hits(skills, _oracle(False, False))


def test_drop_keyword_hits_raises_when_verdicts_are_left_over():
    # More verdicts than examples — without the exhaustion check this returned
    # normally, having mis-paired every example after the divergence.
    skills = [_skill("one", "a1"), _skill("two", "b1")]

    with pytest.raises(ValueError, match="verdicts but all_skills holds fewer"):
        gends.drop_keyword_hits(skills, _oracle(False, False, False))


def test_drop_keyword_hits_treats_a_false_verdict_as_kept_not_as_exhaustion():
    # The exhaustion sentinel must be distinguishable from a legitimate False.
    # `next(flags, False)` would have made an all-False run indistinguishable
    # from a short one.
    skills = [_skill("one", "a1", "a2", "a3")]

    gends.drop_keyword_hits(skills, _oracle(False, False, False))

    assert [e["text"] for e in skills[0]["examples"]] == ["a1", "a2", "a3"]
