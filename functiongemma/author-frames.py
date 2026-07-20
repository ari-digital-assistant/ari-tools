#!/usr/bin/env python3
"""Draft frame/slot banks for one skill with an LLM, for human review.

This is the reproducible path for NEW or updated skills: the corpus banks
under corpus/ are hand-reviewed committed content, and this script produces
the first draft the same way every time — same prompt contract, same rules
(it embeds corpus/README.md verbatim), same validation — instead of an ad-hoc
chat with whichever model is nearby.

    ANTHROPIC_API_KEY=... python3 author-frames.py \
        --skill dev.heyari.timer --locale en

Output: corpus/draft-frames.<skill>.<locale>.json (+ draft-slots.* if the
model needed new slot banks). Draft files are IGNORED by the expander until a
human reviews, renames (drop the draft- prefix, merge slots), and commits.
Italian drafts additionally need native-speaker review — standing project
rule.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-opus-4-8"

BUILTIN_IDS = {"current_time", "current_date", "calculator", "greeting", "open"}


def find_engine_dir() -> Path:
    if env := os.environ.get("ARI_ENGINE_DIR"):
        return Path(env)
    p = HERE.parent.parent / "ari-engine"
    if p.exists():
        return p
    sys.exit("ERROR: set ARI_ENGINE_DIR (needed for built-in skill specs)")


def find_skills_dir() -> Path:
    if env := os.environ.get("ARI_SKILLS_DIR"):
        return Path(env)
    p = HERE.parent.parent / "ari-skills"
    if p.exists():
        return p
    sys.exit("ERROR: set ARI_SKILLS_DIR (needed for community skill specs)")


def skill_spec(skill_id: str, locale: str) -> str:
    """The skill's ground truth: description, parameters, existing examples."""
    if skill_id in BUILTIN_IDS:
        out = subprocess.run(
            ["cargo", "run", "--quiet", "-p", "ari-skills", "--bin",
             "export-utterances", "--", "--locale", locale],
            cwd=find_engine_dir(), capture_output=True, text=True, check=True,
        ).stdout
        for entry in json.loads(out):
            if entry["id"] == skill_id:
                return json.dumps(entry, ensure_ascii=False, indent=2)
        sys.exit(f"ERROR: {skill_id} not in export-utterances output")
    root = find_skills_dir() / "skills"
    for manifest in sorted(root.glob(f"*/SKILL.{locale}.md")):
        text = manifest.read_text()
        if re.search(rf"^\s*id:\s*{re.escape(skill_id)}\s*$", text, re.M):
            return text
    sys.exit(f"ERROR: no SKILL.{locale}.md declares id {skill_id} under {root}")


def existing_slot_names(locale: str) -> list:
    names = []
    for p in sorted(CORPUS.glob(f"slots*.{locale}.json")):
        if not p.name.startswith("draft-"):
            names.extend(json.loads(p.read_text()).keys())
    return names


def build_prompt(skill_id: str, locale: str, spec: str) -> str:
    readme = (CORPUS / "README.md").read_text()
    slots = existing_slot_names(locale)
    lang_rule = (
        "Frames must be idiomatic spoken Italian (never translated English), "
        "with correct accents and apostrophes." if locale == "it" else
        "Frames must be natural spoken English.")
    return f"""You are authoring FunctionGemma training frame banks for ONE Ari skill.

The rules document (conform to it exactly):

{readme}

The skill's ground truth (description, parameter schema, existing examples —
your frames' args must match these arg shapes exactly):

{spec}

{lang_rule}

Existing slot bank names you may reference without redefining: {slots}

Produce 55 frames if the skill takes arguments, 150 if it is slotless.
Cover the full register range: commands, questions, statements implying a
request, indirect idioms. Nothing a real user wouldn't say to an assistant.

Reply with ONLY a JSON object, no prose, of the shape:
{{"frames": {{"{skill_id}": {{"target": 500, "cap": 12, "frames": [...]}}}},
 "slots": {{...ONLY newly introduced slot banks, prefixed with the skill's
            short name to avoid collisions...}}}}"""


def call_model(prompt: str, model: str) -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({
            "model": model,
            "max_tokens": 16000,
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        body = json.load(resp)
    text = "".join(b["text"] for b in body["content"] if b["type"] == "text")
    # Tolerate a fenced reply, but nothing looser.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


def validate(doc: dict, skill_id: str, locale: str) -> None:
    frames = doc["frames"][skill_id]["frames"]
    known = set(existing_slot_names(locale)) | set(doc.get("slots", {}))
    ph = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")
    for f in frames:
        refs = set(ph.findall(f["text"]))
        for v in (f.get("args") or {}).values():
            if isinstance(v, str):
                refs |= set(ph.findall(v))
        unknown = refs - known
        if unknown:
            sys.exit(f"ERROR: frame {f['text']!r} references undefined "
                     f"slot(s) {sorted(unknown)}")
    print(f"  validated: {len(frames)} frames, "
          f"{len(doc.get('slots', {}))} new slot bank(s)", file=sys.stderr)


def apply_draft(doc: dict, skill_id: str, locale: str) -> list:
    """Activate a draft in place: drop any existing bank for this skill, then
    write the new one into the auto-authored union files.

    Auto-drafted content lives in `frames.auto.<locale>.json` /
    `slots.auto.<locale>.json`, kept separate from hand-authored banks so a
    human can always see which frames a machine wrote. The expander unions
    all `frames*.<locale>.json`, so activation needs no merge step — but it
    DOES need the stale entry removed first, because a skill defined in two
    files is a hard error by design.

    Returns the list of files changed.
    """
    changed = []
    # Remove the skill from wherever it currently lives (hand-authored or a
    # previous auto draft) so the union stays single-owner per skill.
    for p in sorted(CORPUS.glob(f"frames*.{locale}.json")):
        if p.name.startswith("draft-"):
            continue
        doc_existing = json.loads(p.read_text())
        if skill_id in doc_existing:
            del doc_existing[skill_id]
            p.write_text(json.dumps(doc_existing, ensure_ascii=False, indent=2) + "\n")
            changed.append(p.name)

    frames_path = CORPUS / f"frames.auto.{locale}.json"
    merged = json.loads(frames_path.read_text()) if frames_path.exists() else {}
    merged.update(doc["frames"])
    frames_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    changed.append(frames_path.name)

    if doc.get("slots"):
        slots_path = CORPUS / f"slots.auto.{locale}.json"
        slots = json.loads(slots_path.read_text()) if slots_path.exists() else {}
        # Never clobber a slot bank another author owns — the expander
        # rejects duplicate slot keys across files, so only add genuinely new
        # ones and let a collision surface as an authoring error.
        existing_names = set(existing_slot_names(locale))
        for name, bank in doc["slots"].items():
            if name in existing_names and name not in slots:
                print(f"  skipping slot {name!r}: already owned by another bank file",
                      file=sys.stderr)
                continue
            slots[name] = bank
        slots_path.write_text(json.dumps(slots, ensure_ascii=False, indent=2) + "\n")
        changed.append(slots_path.name)

    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skill", required=True, help="skill id, e.g. dev.heyari.timer")
    ap.add_argument("--locale", required=True, choices=["en", "it"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--apply", action="store_true",
                    help="Activate the draft immediately (write into "
                         "frames.auto.<locale>.json) instead of leaving a "
                         "draft-* file for review. Used by the nightly.")
    args = ap.parse_args()

    spec = skill_spec(args.skill, args.locale)
    doc = call_model(build_prompt(args.skill, args.locale, spec), args.model)
    validate(doc, args.skill, args.locale)

    if args.apply:
        for name in apply_draft(doc, args.skill, args.locale):
            print(f"updated {name}")
        return

    frames_out = CORPUS / f"draft-frames.{args.skill}.{args.locale}.json"
    frames_out.write_text(json.dumps(doc["frames"], ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {frames_out}")
    if doc.get("slots"):
        slots_out = CORPUS / f"draft-slots.{args.skill}.{args.locale}.json"
        slots_out.write_text(json.dumps(doc["slots"], ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {slots_out}")
    print("Review the draft, drop the draft- prefix (merging slots into an "
          "existing slots file if you prefer), and commit. Italian drafts "
          "need native review before merging.")


if __name__ == "__main__":
    main()
