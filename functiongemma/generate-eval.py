#!/usr/bin/env python3
"""Generate the GENERATED routing eval bank with Gemini, for commit.

The hand-written evals (routing-eval.jsonl / routing-eval.it.jsonl) are the
locked promotion-gate spine: 40 positives + 30 NONE per locale, 5 built-in
skills, comparable run-to-run. This script writes the BREADTH set next to
them — routing-eval.gen.<locale>.jsonl — covering every router-eligible
skill (built-in + community) plus generated NONE cases, used by
derive_floor.py to derive the per-model confidence floor at a sample size
the spine cannot provide. The gate never runs on this file's verdict.

Independence from training is the whole design (see zobb.md): banks are
authored on OpenAI (author-frames.py), this eval on Gemini — different
model families, different priors. Three filters keep the output honest,
each using the engine's own oracle so generator and pipeline cannot
disagree:

  1. keyword-miss: candidates the keyword scorer claims are dropped
     (`keyword-hit` bin — same oracle as generate-dataset.py's filter and
     route-eval's pollution guardrail, which exits 3 on violations).
  2. eval-collision: candidates colliding (loose_key) with ANY existing
     eval case, either locale, spine or generated, are dropped.
  3. corpus-collision: candidates colliding with any training text that
     generate-dataset.py --augment corpus produces, EITHER locale, on the
     raw or normalised loose_key, are dropped — generate-dataset.py's
     held-out guard sys.exit()s on such a collision, so an unlucky case
     here would kill every future nightly.

Usage:
    GEMINI_API_KEY=... python3 generate-eval.py --locale en
    python3 generate-eval.py --locale it --dry-run   # no API, plumbing only

The model is configurable (`--model`, or the `EVAL_MODEL` env var), same
convention as author-frames.py's FRAMES_MODEL: names move faster than
scripts. Italian output is machine-drafted and goes through Keith's
italian-review queue (standing project rule) — the workflow files the
issue.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
# `or` (not a .get default): the workflow exports EVAL_MODEL="" when the
# dispatch input is blank, and an empty string must still mean the default.
DEFAULT_MODEL = os.environ.get("EVAL_MODEL") or "gemini-3.5-flash"

EVAL_FILES = [
    HERE / "routing-eval.jsonl",
    HERE / "routing-eval.it.jsonl",
    HERE / "routing-eval.gen.jsonl",
    HERE / "routing-eval.gen.it.jsonl",
]

# Below this many surviving cases the skill's numbers would be noise and the
# run is refusing to pretend otherwise. Between HARD_FLOOR and the warn line
# the case count is thin but usable — derive_floor.py works on the union.
HARD_FLOOR = 3
WARN_FLOOR = 10


def _load_generate_dataset():
    """Import generate-dataset.py despite the dash in its name.

    Module-level side effects are just constants and random.seed(42) — safe.
    Reusing its functions (skill discovery, oracles) rather than copying them
    is what keeps this generator incapable of disagreeing with the pipeline.
    """
    spec = importlib.util.spec_from_file_location(
        "generate_dataset", HERE / "generate-dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gd = _load_generate_dataset()
from corpus_expander import _load_union, load_eval_keys, loose_key  # noqa: E402


def corpus_keys(engine_dir: Path, skills_dir: Path | None) -> set:
    """loose_key set of every training text the corpus produces, BOTH locales.

    generate-dataset.py's held-out guard compares its (normalised) texts
    against ALL eval keys regardless of locale, so a new EN eval case that
    collides with an IT training text hard-fails the IT nightly. Running the
    real generator is the only oracle that cannot drift from the guard.
    """
    keys = set()
    for locale in ("en", "it"):
        env = dict(os.environ)
        env["ARI_ENGINE_DIR"] = str(engine_dir)
        if skills_dir is not None:
            env["ARI_SKILLS_DIR"] = str(skills_dir)
        print(f"  corpus oracle: generate-dataset --locale {locale} "
              f"--augment corpus ...", file=sys.stderr)
        res = subprocess.run(
            [sys.executable, str(HERE / "generate-dataset.py"),
             "--locale", locale, "--augment", str(CORPUS)],
            capture_output=True, text=True, env=env)
        if res.returncode != 0:
            sys.exit(f"ERROR: generate-dataset.py --locale {locale} failed — "
                     f"cannot decontaminate without the corpus oracle:\n"
                     f"{res.stderr[-2000:]}")
        for line in res.stdout.splitlines():
            sample = json.loads(line)
            user = next(m["content"] for m in sample["messages"]
                        if m["role"] == "user")
            keys.add(loose_key(user))
    return keys


def merged_skills(engine_dir: Path, skills_dir: Path | None, locale: str) -> list:
    """Router-eligible skills, builtins + community, same merge as training."""
    builtin = gd.export_skills(engine_dir, locale)
    community = gd.load_community_skills(skills_dir, locale) if skills_dir else []
    builtin_ids = {s["id"] for s in builtin}
    merged = builtin + [s for s in community if s["id"] not in builtin_ids]
    return gd.router_eligible_skills(merged)


def bank_frame_texts(locale: str) -> dict:
    """{skill_id: [frame text, ...]} from the committed frame banks."""
    frames_doc = _load_union(CORPUS, f"frames*.{locale}.json", "skill")
    return {sid: [f["text"] for f in entry.get("frames", [])]
            for sid, entry in frames_doc.items()}


LANG_RULES = {
    "en": "Utterances must be natural spoken English.",
    "it": ("Utterances must be idiomatic spoken Italian (never translated "
           "English), with correct accents and apostrophes."),
}


def positive_prompt(skill: dict, avoid: list, n: int, locale: str) -> str:
    spec = {
        "id": skill["id"],
        "description": skill["description"],
        "parameters": skill.get("parameters", {}),
        "example_requests": [e["text"] for e in skill.get("examples", [])][:20],
    }
    avoid_block = "\n".join(f"- {a}" for a in avoid[:150])
    return f"""You are writing HELD-OUT evaluation utterances for a voice-assistant skill router.

The router is a FALLBACK tier: it only ever sees utterances that a keyword matcher MISSED. Evaluation cases must therefore be oblique, indirect, natural spoken phrasings that avoid the skill's obvious trigger words — what a real person says when they don't use the canonical command.

The skill under test. Every utterance must be a genuine, unambiguous request for THIS skill, answerable with these parameters:

{json.dumps(spec, ensure_ascii=False, indent=2)}

{LANG_RULES[locale]}

Do NOT reuse or lightly paraphrase any of the following known phrasings — they are the training bank, and your job is to test generalisation BEYOND them (some contain {{SLOT}} placeholders; treat those as whole template families to avoid):

{avoid_block}

Produce {n} utterances spread across genuinely different framings: indirect questions, statements that imply the request, colloquial idioms, situational phrasings, polite forms, terse forms. No two utterances from the same template. Nothing a real person wouldn't say to an assistant. Concrete values only — no placeholders. One line each; no newlines inside an utterance.

Reply with ONLY a JSON object: {{"cases": ["...", "..."]}}"""


def none_prompt(skills: list, avoid: list, n: int, locale: str) -> str:
    skill_lines = "\n".join(f"- {s['id']}: {s['description']}" for s in skills)
    avoid_block = "\n".join(f"- {a}" for a in avoid[:100])
    return f"""You are writing HELD-OUT "must abstain" cases for a voice-assistant skill router. For every one of these utterances the router must emit NOTHING: they are general-knowledge questions, chit-chat, opinions, and requests OUTSIDE every installed skill.

The assistant DOES have the following skills. Your utterances must NOT be a request for any of them, even indirectly or partially:

{skill_lines}

{LANG_RULES[locale]}

Do NOT reuse or lightly paraphrase any of these existing cases:

{avoid_block}

Produce {n} varied utterances: factual questions, explanations, recommendations, small talk, opinions, conversation closers. Nothing a real person wouldn't say to an assistant. One line each; no newlines inside an utterance.

Reply with ONLY a JSON object: {{"cases": ["...", "..."]}}"""


CASES_SCHEMA = {
    "type": "object",
    "properties": {
        "cases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["cases"],
}


def call_gemini(prompt: str, model: str) -> list:
    """One structured-output call. Returns the candidate list.

    Import inside the function so the test suite (which stubs this) and
    --dry-run work without the google-genai package installed.
    """
    from google import genai
    client = genai.Client()  # reads GEMINI_API_KEY
    last_err = None
    # 5 attempts, capped exponential backoff: observed 2026-07-20, the
    # flash tier under "high demand" throws alternating 500s and silent
    # connection drops for 10+ minutes — 3 thin retries lost a whole run.
    for attempt in range(5):
        try:
            interaction = client.interactions.create(
                model=model,
                input=prompt,
                generation_config={"temperature": 1.0},
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": CASES_SCHEMA,
                },
            )
            doc = json.loads(interaction.output_text)
            cases = doc.get("cases")
            if not isinstance(cases, list):
                raise ValueError(f"no 'cases' array in reply: "
                                 f"{interaction.output_text[:200]}")
            return [c for c in cases if isinstance(c, str)]
        except Exception as e:  # noqa: BLE001 — retried, then fatal with detail
            # A 404 ("model not found" / "not available to new users") will
            # never heal on retry — fail fast with the model-picking hint.
            # 429s stay retryable: per-minute throttles recover in seconds.
            if "not_found" in str(e) or "404" in str(e):
                sys.exit(f"ERROR: Gemini model {model!r} is not usable on "
                         f"this API key: {e!r} — pick another with "
                         f"--list-models.")
            last_err = e
            wait = min(5 * (2 ** attempt), 60)
            print(f"  Gemini attempt {attempt + 1} failed ({e!r}); "
                  f"retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    sys.exit(f"ERROR: Gemini ({model}) failed after 5 attempts: {last_err!r} — "
             f"if the model is overloaded, re-run with EVAL_MODEL (or "
             f"--model) set to another GA model, e.g. gemini-2.5-flash; the "
             f"output header records which model wrote the bank.")


def filter_candidates(cands: list, locale: str, engine_dir: Path,
                      skills_dir: Path | None, avoid_keys: set,
                      corpus_keyset: set, keep: int,
                      oracle=None, normalizer=None) -> tuple:
    """Hygiene → dedupe → keyword-miss → collision filters. Returns
    (kept, stats). `oracle`/`normalizer` are injectable for tests; default
    to the engine subprocesses."""
    oracle = oracle or (lambda texts: gd.keyword_hits(
        engine_dir, texts, locale, skills_dir))
    normalizer = normalizer or (lambda texts: gd.normalize_texts(
        engine_dir, texts, locale))

    seen, clean = set(), []
    for c in cands:
        c = " ".join(c.split())
        k = loose_key(c)
        if not c or not k or k in seen:
            continue
        seen.add(k)
        clean.append(c)

    hits = oracle(clean) if clean else []
    misses = [c for c, h in zip(clean, hits) if not h]
    norms = normalizer(misses) if misses else []

    kept, coll_eval, coll_corpus = [], 0, 0
    for c, n in zip(misses, norms):
        keys = {loose_key(c), loose_key(n)}
        if keys & avoid_keys:
            coll_eval += 1
            continue
        if keys & corpus_keyset:
            coll_corpus += 1
            continue
        kept.append(c)
        if len(kept) >= keep:
            break

    stats = {
        "raw": len(cands),
        "deduped": len(clean),
        "keyword_hits": len(clean) - len(misses),
        "eval_collisions": coll_eval,
        "corpus_collisions": coll_corpus,
        "kept": len(kept),
    }
    return kept, stats


def write_eval(path: Path, per_skill: dict, none_cases: list,
               locale: str, model: str) -> None:
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    total_pos = sum(len(v) for v in per_skill.values())
    lines = [
        f"// GENERATED routing eval ({locale}) — written by generate-eval.py.",
        f"// Model: {model}  Run: {run_id}  At: {stamp}",
        f"// {total_pos} positives across {len(per_skill)} skills + "
        f"{len(none_cases)} NONE.",
        "//",
        "// Regenerated wholesale — do not hand-edit (edits are overwritten).",
        "// The hand-written spine in routing-eval[.it].jsonl is the promotion",
        "// gate; this file adds breadth for derive_floor.py and community-",
        "// skill coverage. Every case is a keyword-MISS and decontaminated",
        "// against the training corpus at generation time.",
    ]
    for sid in sorted(per_skill):
        for c in per_skill[sid]:
            lines.append(json.dumps({"utterance": c, "expect": sid},
                                    ensure_ascii=False))
    for c in none_cases:
        lines.append(json.dumps({"utterance": c, "expect": "NONE"},
                                ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--locale", required=True, choices=["en", "it"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--per-skill", type=int, default=20,
                    help="kept cases per skill (surplus is requested 3x)")
    ap.add_argument("--none-count", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true",
                    help="No API calls: run discovery, the corpus oracle and "
                         "the filters over frame-bank texts as stand-in "
                         "candidates. Writes nothing.")
    ap.add_argument("--list-models", action="store_true",
                    help="Print the models this GEMINI_API_KEY can use, then "
                         "exit. For choosing EVAL_MODEL when a tier is "
                         "quota-capped or gated (2026-07-20: 3.5-flash free "
                         "tier is 20 req/day; 2.5-flash 404s for new keys).")
    args = ap.parse_args()

    if args.list_models:
        if not os.environ.get("GEMINI_API_KEY"):
            sys.exit("ERROR: GEMINI_API_KEY not set")
        from google import genai
        client = genai.Client()
        for m in client.models.list():
            name = getattr(m, "name", m)
            extra = (getattr(m, "supported_actions", None)
                     or getattr(m, "supported_generation_methods", None) or "")
            print(f"{name}  {extra}")
        return

    if not args.dry_run and not os.environ.get("GEMINI_API_KEY"):
        sys.exit("ERROR: GEMINI_API_KEY not set (or use --dry-run)")

    engine_dir = gd.find_engine_dir()
    skills_dir = gd.find_skills_dir()
    skills = merged_skills(engine_dir, skills_dir, args.locale)
    print(f"  {len(skills)} router-eligible skills ({args.locale})",
          file=sys.stderr)

    avoid_keys = load_eval_keys(EVAL_FILES)
    corpus_keyset = corpus_keys(engine_dir, skills_dir)
    print(f"  {len(avoid_keys)} eval keys, {len(corpus_keyset)} corpus keys",
          file=sys.stderr)

    banks = bank_frame_texts(args.locale)
    per_skill, thin = {}, []
    # Keys of everything kept SO FAR this run — a case generated for one
    # skill (or for NONE, below) must not reappear under another label in
    # the same file: two contradictory expectations for one utterance.
    session_keys = set()
    for skill in skills:
        sid = skill["id"]
        avoid = banks.get(sid, []) + [e["text"] for e in skill.get("examples", [])]
        want = args.per_skill * 3
        if args.dry_run:
            cands = (banks.get(sid) or [e["text"] for e in skill["examples"]])[:5]
        else:
            cands = call_gemini(
                positive_prompt(skill, avoid, want, args.locale), args.model)
        kept, stats = filter_candidates(
            cands, args.locale, engine_dir, skills_dir,
            avoid_keys | session_keys, corpus_keyset, args.per_skill)
        if not args.dry_run and len(kept) < args.per_skill:
            # One top-up round: ask again, with the survivors AND the
            # rejects in the avoid list so the model explores elsewhere.
            more = call_gemini(
                positive_prompt(skill, avoid + cands, want, args.locale),
                args.model)
            extra, stats2 = filter_candidates(
                more, args.locale, engine_dir, skills_dir,
                avoid_keys | session_keys | {loose_key(k) for k in kept},
                corpus_keyset, args.per_skill - len(kept))
            kept += extra
            for k in stats:
                stats[k] += stats2[k]
        session_keys |= {loose_key(k) for k in kept}
        per_skill[sid] = kept
        flag = "" if len(kept) >= WARN_FLOOR else "  << THIN"
        print(f"    {sid}: kept {len(kept)}/{stats['raw']} "
              f"(kw-hit {stats['keyword_hits']}, eval-coll "
              f"{stats['eval_collisions']}, corpus-coll "
              f"{stats['corpus_collisions']}){flag}", file=sys.stderr)
        if len(kept) < HARD_FLOOR:
            thin.append(sid)

    spine = HERE / ("routing-eval.jsonl" if args.locale == "en"
                    else f"routing-eval.{args.locale}.jsonl")
    spine_none = []
    for line in spine.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        case = json.loads(line)
        if case["expect"].upper() == "NONE":
            spine_none.append(case["utterance"])

    if args.dry_run:
        none_cands = ["placeholder general knowledge question"]
    else:
        none_cands = call_gemini(
            none_prompt(skills, spine_none, args.none_count * 3, args.locale),
            args.model)
    none_kept, none_stats = filter_candidates(
        none_cands, args.locale, engine_dir, skills_dir,
        avoid_keys | session_keys, corpus_keyset, args.none_count)
    print(f"    NONE: kept {len(none_kept)}/{none_stats['raw']}",
          file=sys.stderr)

    if thin and not args.dry_run:
        sys.exit(f"ERROR: fewer than {HARD_FLOOR} surviving cases for "
                 f"{thin} — the skill's phrasings may be inherently "
                 f"keyword-bound. Investigate before committing a set that "
                 f"pretends to measure it.")
    if not args.dry_run and len(none_kept) < 20:
        sys.exit(f"ERROR: only {len(none_kept)} NONE cases survived — "
                 f"abstention derivation needs at least 20.")

    if args.dry_run:
        print("  dry run complete — plumbing OK, nothing written.",
              file=sys.stderr)
        return

    out = HERE / (f"routing-eval.gen.jsonl" if args.locale == "en"
                  else f"routing-eval.gen.{args.locale}.jsonl")
    write_eval(out, per_skill, none_kept, args.locale, args.model)
    total = sum(len(v) for v in per_skill.values())
    print(f"wrote {out.name}: {total} positives / {len(per_skill)} skills "
          f"+ {len(none_kept)} NONE")


if __name__ == "__main__":
    main()
