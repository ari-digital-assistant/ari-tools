#!/usr/bin/env python3
"""Draft frame/slot banks for one skill with an LLM, for human review.

This is the reproducible path for NEW or updated skills: the corpus banks
under corpus/ are hand-reviewed committed content, and this script produces
the first draft the same way every time — same prompt contract, same rules
(it embeds corpus/README.md verbatim), same validation — instead of an ad-hoc
chat with whichever model is nearby.

    OPENAI_API_KEY=... python3 author-frames.py \
        --skill dev.heyari.timer --locale en

Output: corpus/draft-frames.<skill>.<locale>.json (+ draft-slots.* if the
model needed new slot banks). Draft files are IGNORED by the expander until a
human reviews, renames (drop the draft- prefix, merge slots), and commits.
`--apply` instead activates the draft immediately (what the nightly does).
Italian drafts additionally need native-speaker review — standing project
rule.

Uses the OpenAI API. The model is configurable (`--model`, or the
`FRAMES_MODEL` env var) because model names move faster than this script:
if the default is retired, override it rather than editing code.
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("FRAMES_MODEL", "gpt-5.6-sol")

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


def sibling_catalogue(locale: str, exclude_id: str) -> list:
    """(id, description) for every OTHER router-eligible skill.

    The drafting model sees one skill's spec; without this list it cannot
    know which surface shapes its siblings own, and it WILL wander into
    them — counter's `somma {ITCNT} al contatore` sat next to calculator's
    `somma {N1} e {N2}` until 2026-07-21, and the resulting model routed
    arithmetic to counter and failed the promotion gate two nights running.
    """
    spec = importlib.util.spec_from_file_location(
        "generate_dataset", HERE / "generate-dataset.py")
    gd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gd)
    builtin = gd.export_skills(find_engine_dir(), locale)
    community = gd.load_community_skills(find_skills_dir(), locale)
    builtin_ids = {s["id"] for s in builtin}
    merged = builtin + [s for s in community if s["id"] not in builtin_ids]
    return [(s["id"], s["description"])
            for s in gd.router_eligible_skills(merged) if s["id"] != exclude_id]


def build_prompt(skill_id: str, locale: str, spec: str, siblings: list) -> str:
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

Other skills installed on the same device (id: description):

{chr(10).join(f"- {i}: {d}" for i, d in siblings)}

STAY IN YOUR LANE. Every frame must be unambiguously a request for
{skill_id}, with zero surrounding context. If a phrasing could plausibly be
read as a request for ANY sibling above, anchor it with a noun naming this
skill's own domain, or discard it. In particular, bare arithmetic shapes —
a generic verb (add, sum, più, somma, aggiungi, ...) plus numbers and
nothing else — belong to the calculator skill only; a frame of yours that
mentions a number or numeric slot must also contain a noun naming your
domain. When in doubt, discard the frame.

Produce 55 frames if the skill takes arguments, 150 if it is slotless.
Cover the full register range: commands, questions, statements implying a
request, indirect idioms. Nothing a real user wouldn't say to an assistant.

Reply with ONLY a JSON object, no prose, of the shape:
{{"frames": {{"{skill_id}": {{"target": 500, "cap": 12, "frames": [...]}}}},
 "slots": {{...ONLY newly introduced slot banks, prefixed with the skill's
            short name to avoid collisions...}}}}"""


def _post(payload: dict, key: str) -> dict:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def call_model(prompt: str, model: str) -> dict:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("ERROR: OPENAI_API_KEY not set")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # The prompt demands a bare JSON object; asking the API to enforce
        # it removes the single most common failure mode (prose preamble).
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 32000,
    }

    # Parameter names drift between model generations, and this script has
    # to keep working when they do. Retry once without whichever parameter
    # the API rejected, rather than failing a nightly over a schema nit.
    for attempt in range(4):
        try:
            body = _post(payload, key)
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            if e.code == 400 and "max_completion_tokens" in detail and \
                    "max_completion_tokens" in payload:
                payload["max_tokens"] = payload.pop("max_completion_tokens")
                continue
            if e.code == 400 and "max_tokens" in detail and "max_tokens" in payload:
                payload.pop("max_tokens")
                continue
            if e.code == 400 and "response_format" in detail:
                payload.pop("response_format", None)
                continue
            sys.exit(f"ERROR: OpenAI API {e.code} for model {model!r}: {detail[:600]}")
    else:
        sys.exit(f"ERROR: OpenAI API kept rejecting the request for model {model!r}")

    choice = body["choices"][0]
    text = (choice["message"].get("content") or "").strip()
    if not text:
        sys.exit(f"ERROR: model {model!r} returned no content "
                 f"(finish_reason={choice.get('finish_reason')!r}) — if this is "
                 f"'length', the bank is too large for one reply; lower the "
                 f"per-skill frame quota or raise the token limit.")
    if choice.get("finish_reason") == "length":
        sys.exit("ERROR: reply was truncated mid-JSON (finish_reason=length). "
                 "Re-run with a smaller quota rather than committing a partial bank.")
    # Tolerate a fenced reply, but nothing looser.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        sys.exit(f"ERROR: model {model!r} did not return valid JSON: {e}\n"
                 f"first 400 chars: {text[:400]}")


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
    siblings = sibling_catalogue(args.locale, args.skill)
    doc = call_model(build_prompt(args.skill, args.locale, spec, siblings), args.model)
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
