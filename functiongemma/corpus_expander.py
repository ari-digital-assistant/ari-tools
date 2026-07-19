"""Deterministic frame×slot corpus expansion for FunctionGemma training.

This is the volume half of Google's mobile-actions recipe (9,650 rows for
SEVEN functions — ~1,380 per function) applied to Ari's own skills. Frames
and slot banks are committed, human-reviewed JSON under `corpus/`; this
module multiplies them mechanically. The LLM never does the maths and the
expansion never invents content — every string in the output is a reviewed
frame with reviewed slot values substituted in.

Layout (per locale `xx`):
  corpus/frames.xx.json     {"<skill_id>": {"target": 500, "cap": 12,
                              "frames": [{"text": "...{APP}...",
                                          "args": {"app_name": "{APP}"}}]}}
  corpus/slots.xx.json      {"APP": [{"surface": "la fotocamera",
                                      "canonical": "Camera"}, ...],
                             "N1": {"type": "int", "min": 2, "max": 999}}
  corpus/negatives.xx.json  {"target": 8000, "cap": 12,
                             "frames": [{"text": "chi ha {PVERB} {ENTITY}"}]}

Rules:
  - `{SLOT}` in `text` substitutes the slot's SURFACE form; the same `{SLOT}`
    inside an args value substitutes its CANONICAL form (args stay canonical:
    "la fotocamera" -> "Camera"). A frame needing two values from the same
    conceptual bank uses two bank keys (N1, N2) — no aliasing syntax.
  - Numeric banks ({"type": "int"}) are generated, not enumerated;
    surface == canonical == str(n).
  - Expansion is seeded (SEED) and therefore reproducible in CI.
  - Per-frame cap: one frame repeated hundreds of times with different
    numbers teaches the frame, not the skill.
  - Dedupe on a loose key (lowercase, punctuation stripped) within the
    expansion AND against the skill's existing manifest examples.
  - Decontamination: any expansion whose loose key appears in a routing-eval
    file is dropped and reported. The eval must stay held out — this is the
    same rule the hand-authored corpus lives under.
"""

import json
import random
import re
import sys
from pathlib import Path

SEED = 20260719
PLACEHOLDER = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")
DEFAULT_CAP = 12


def loose_key(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — near-dupe key."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def load_eval_keys(eval_paths: list) -> set:
    keys = set()
    for p in eval_paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            keys.add(loose_key(json.loads(line)["utterance"]))
    return keys


def _load_union(banks_dir: Path, pattern: str, kind: str) -> dict:
    """Union all JSON files matching `pattern` (sorted — deterministic).

    Multiple authors contribute split files (frames.builtin.en.json,
    frames.community.en.json, ...); the merge is mechanical so the pipeline
    has no by-hand step. A key defined twice is an authoring error, not a
    merge policy decision — fail loudly.
    Files named draft-* are never picked up (author-frames.py output awaiting
    review).
    """
    merged = {}
    files = sorted(p for p in banks_dir.glob(pattern)
                   if not p.name.startswith("draft-"))
    if not files:
        sys.exit(f"ERROR: --augment: no files match {pattern} under {banks_dir}")
    for p in files:
        doc = json.loads(p.read_text())
        for k, v in doc.items():
            if k in merged:
                sys.exit(f"ERROR: {kind} {k!r} defined in more than one file "
                         f"(latest: {p.name}) — every key must have exactly "
                         f"one owner")
            merged[k] = v
    return merged


def _slot_values(bank, rng: random.Random, want: int) -> list:
    """Return up to `want` (surface, canonical) pairs from a bank."""
    if isinstance(bank, dict) and bank.get("type") == "int":
        lo, hi = bank["min"], bank["max"]
        span = hi - lo + 1
        ns = rng.sample(range(lo, hi + 1), min(want, span))
        return [(str(n), str(n)) for n in ns]
    entries = list(bank)
    rng.shuffle(entries)
    return [(e["surface"], e["canonical"]) for e in entries[:want]]


def _expand_frame(frame: dict, slots: dict, rng: random.Random, cap: int) -> list:
    """All (text, args) expansions of one frame, up to cap, order-stable."""
    text = frame["text"]
    args = frame.get("args") or {}
    names = PLACEHOLDER.findall(text)
    for v in args.values():
        if isinstance(v, str):
            names.extend(PLACEHOLDER.findall(v))
    names = list(dict.fromkeys(names))  # unique, insertion order
    if not names:
        return [(text, dict(args))]

    for n in names:
        if n not in slots:
            sys.exit(f"ERROR: frame {text!r} references slot {{{n}}} which is "
                     f"not in the slot bank")

    # Draw cap independent combinations: one value per slot per row.
    per_slot = {n: _slot_values(slots[n], rng, cap) for n in names}
    rows = []
    depth = min(cap, max(len(v) for v in per_slot.values()))
    for i in range(depth):
        sub_s = {n: per_slot[n][i % len(per_slot[n])][0] for n in names}
        sub_c = {n: per_slot[n][i % len(per_slot[n])][1] for n in names}
        t = PLACEHOLDER.sub(lambda m: sub_s[m.group(1)], text)
        a = {
            k: (PLACEHOLDER.sub(lambda m: sub_c[m.group(1)], v)
                if isinstance(v, str) else v)
            for k, v in args.items()
        }
        rows.append((t, a))
    return rows


def expand_skills(banks_dir: Path, locale: str, all_skills: list,
                  eval_keys: set, allow_missing: bool = False) -> dict:
    """Expand frames for every router-eligible skill. Returns
    {skill_id: [{"text":..., "args":...}]} of NEW examples (deduped,
    decontaminated, not already in the manifest examples)."""
    frames_doc = _load_union(banks_dir, f"frames*.{locale}.json", "skill")
    slots = _load_union(banks_dir, f"slots*.{locale}.json", "slot")

    missing = [s["id"] for s in all_skills if s["id"] not in frames_doc]
    if missing and not allow_missing:
        sys.exit(
            f"ERROR: no frame bank for router-eligible skill(s) {missing} "
            f"under {banks_dir}. Every trainable skill needs a bank — run "
            f"author-frames.py for it, review, and commit. "
            f"(--allow-missing-banks to override.)")
    if missing:
        print(f"  WARNING: skills without frame banks (thin training): "
              f"{missing}", file=sys.stderr)

    rng = random.Random(SEED)
    out = {}
    dropped_eval_total = 0
    for skill in all_skills:
        sid = skill["id"]
        entry = frames_doc.get(sid)
        if not entry:
            continue
        target = entry.get("target", 500)
        cap = entry.get("cap", DEFAULT_CAP)
        existing = {loose_key(e["text"]) for e in skill["examples"]}
        seen, rows, dropped_eval = set(existing), [], 0

        pool = []
        for frame in entry["frames"]:
            pool.extend(_expand_frame(frame, slots, rng, cap))
        rng.shuffle(pool)

        for text, args in pool:
            if len(rows) >= target:
                break
            k = loose_key(text)
            if k in seen:
                continue
            if k in eval_keys:
                dropped_eval += 1
                continue
            seen.add(k)
            rows.append({"text": text, "args": args})

        dropped_eval_total += dropped_eval
        short = "" if len(rows) >= target else \
            f"  << SHORTFALL (cross-product too small — add frames)"
        print(f"    {sid}: +{len(rows)} expanded "
              f"(target {target}, frames {len(entry['frames'])}, "
              f"eval-dropped {dropped_eval}){short}", file=sys.stderr)
        out[sid] = rows

    if dropped_eval_total:
        print(f"  decontamination: {dropped_eval_total} expansion(s) collided "
              f"with eval cases and were dropped", file=sys.stderr)
    return out


def expand_negatives(banks_dir: Path, locale: str, eval_keys: set) -> list:
    """Expand the negative frames. Returns a list of texts."""
    path = banks_dir / f"negatives.{locale}.json"
    if not path.exists():
        sys.exit(f"ERROR: --augment given but {path.name} missing under {banks_dir}")
    doc = json.loads(path.read_text())
    slots = _load_union(banks_dir, f"slots*.{locale}.json", "slot")
    target = doc.get("target", 8000)
    cap = doc.get("cap", DEFAULT_CAP)

    rng = random.Random(SEED + 1)
    pool = []
    for frame in doc["frames"]:
        pool.extend(_expand_frame(frame, slots, rng, cap))
    rng.shuffle(pool)

    seen, rows, dropped_eval = set(), [], 0
    for text, _ in pool:
        if len(rows) >= target:
            break
        k = loose_key(text)
        if k in seen:
            continue
        if k in eval_keys:
            dropped_eval += 1
            continue
        seen.add(k)
        rows.append(text)

    short = "" if len(rows) >= target else "  << SHORTFALL (add negative frames)"
    print(f"    negatives: +{len(rows)} expanded (target {target}, "
          f"frames {len(doc['frames'])}, eval-dropped {dropped_eval}){short}",
          file=sys.stderr)
    return rows
