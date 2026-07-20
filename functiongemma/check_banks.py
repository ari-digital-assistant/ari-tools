"""Detect skills whose corpus banks are missing or stale.

The training corpus is generated from committed frame/slot banks
(`corpus/`), which are authored per skill. When a skill is added to the
registry — or its description, parameters or examples change materially —
its banks must be (re)drafted, or the model trains on a stale idea of what
that skill does. `generate-dataset.py --augment` already hard-errors on a
MISSING bank; this module additionally catches SILENT staleness, and gives
the nightly a machine-readable list to act on.

Fingerprint: sha256 over the parts of a skill that determine what its
frames should say — id, description, parameter schema, and the text of its
manifest examples. Deliberately NOT the whole manifest: a version bump, a
wasm rebuild or a prose edit to the README section must not invalidate
perfectly good banks.

Recorded in `corpus/bank-sources.json`:

    {"<skill_id>": {"en": "<sha256>", "it": "<sha256>"}}

A skill is:
  - MISSING  — no frame bank entry for it in frames*.<locale>.json
  - STALE    — bank exists but the recorded fingerprint differs from now
  - ORPHANED — bank exists for a skill no longer in the registry
  - OK       — fingerprint matches

Orphans are reported, never auto-deleted: a skill can vanish from a
checkout for boring reasons (a failed clone, a sparse-checkout typo), and
silently deleting hand-reviewed content on that basis would be the kind of
irreversible helpfulness nobody asked for.
"""

import hashlib
import json
import sys
from pathlib import Path

SOURCES_FILE = "bank-sources.json"


def fingerprint(skill: dict) -> str:
    """Stable hash of the skill facts that determine its frames."""
    payload = {
        "id": skill.get("id", ""),
        "description": skill.get("description", ""),
        "parameters": skill.get("parameters", {}),
        # Example TEXT only — args are canonical and already covered by the
        # parameter schema; ordering is preserved because reordering
        # examples is not a semantic change worth a redraft.
        "examples": sorted(e.get("text", "") for e in skill.get("examples", [])),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_sources(banks_dir: Path) -> dict:
    path = banks_dir / SOURCES_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_sources(banks_dir: Path, sources: dict) -> None:
    path = banks_dir / SOURCES_FILE
    path.write_text(json.dumps(sources, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n")


def banked_skill_ids(banks_dir: Path, locale: str) -> set:
    """Skill ids with a frame bank, ignoring unreviewed draft-* files."""
    ids = set()
    for p in sorted(banks_dir.glob(f"frames*.{locale}.json")):
        if p.name.startswith("draft-"):
            continue
        ids |= set(json.loads(p.read_text()).keys())
    return ids


def audit(banks_dir: Path, locale: str, skills: list) -> dict:
    """Classify every router-eligible skill for `locale`.

    `skills` is the merged builtin+community list AFTER router-eligibility
    filtering — the same list the generator trains on, so a skill the
    router never sees is never reported as needing banks.
    """
    sources = load_sources(banks_dir)
    banked = banked_skill_ids(banks_dir, locale)

    missing, stale, ok = [], [], []
    for skill in skills:
        sid = skill["id"]
        now = fingerprint(skill)
        if sid not in banked:
            missing.append(sid)
        elif sources.get(sid, {}).get(locale) != now:
            stale.append(sid)
        else:
            ok.append(sid)

    live_ids = {s["id"] for s in skills}
    orphaned = sorted(banked - live_ids)

    return {"missing": sorted(missing), "stale": sorted(stale),
            "ok": sorted(ok), "orphaned": orphaned}


def record(banks_dir: Path, locale: str, skills: list, only: list = None) -> None:
    """Write current fingerprints. `only` limits the update to those ids —
    used after a successful redraft so an unrelated stale skill is not
    silently marked fresh without its banks being touched."""
    sources = load_sources(banks_dir)
    for skill in skills:
        sid = skill["id"]
        if only is not None and sid not in only:
            continue
        sources.setdefault(sid, {})[locale] = fingerprint(skill)
    save_sources(banks_dir, sources)


def _load_skills(locale: str) -> list:
    """Reuse the generator's own loaders — one source of truth for what a
    skill IS and which skills are router-eligible."""
    sys.path.insert(0, str(Path(__file__).parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gen", Path(__file__).parent / "generate-dataset.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    engine_dir = gen.find_engine_dir()
    builtins_ = gen.export_skills(engine_dir, locale)
    skills_dir = gen.find_skills_dir()
    community = gen.load_community_skills(skills_dir, locale) if skills_dir else []
    builtin_ids = {s["id"] for s in builtins_}
    merged = builtins_ + [s for s in community if s["id"] not in builtin_ids]
    return gen.router_eligible_skills(merged)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--locale", required=True)
    ap.add_argument("--banks-dir", default=str(Path(__file__).parent / "corpus"))
    ap.add_argument("--record", action="store_true",
                    help="Write current fingerprints for ALL skills and exit "
                         "(use after a bulk redraft, or to adopt existing banks).")
    ap.add_argument("--github-output", action="store_true",
                    help="Emit needs_drafting=<csv> / has_drift=<bool> for CI.")
    args = ap.parse_args()

    banks_dir = Path(args.banks_dir)
    skills = _load_skills(args.locale)

    if args.record:
        record(banks_dir, args.locale, skills)
        print(f"recorded fingerprints for {len(skills)} skill(s) ({args.locale})",
              file=sys.stderr)
        return

    result = audit(banks_dir, args.locale, skills)
    for kind in ("missing", "stale", "orphaned"):
        if result[kind]:
            print(f"  {kind}: {', '.join(result[kind])}", file=sys.stderr)
    print(f"  ok: {len(result['ok'])} skill(s) up to date", file=sys.stderr)

    needs = result["missing"] + result["stale"]
    if args.github_output:
        import os
        out = os.environ.get("GITHUB_OUTPUT")
        if out:
            with open(out, "a") as fh:
                fh.write(f"needs_drafting={','.join(needs)}\n")
                fh.write(f"has_drift={'true' if needs else 'false'}\n")
    print(",".join(needs))


if __name__ == "__main__":
    main()
