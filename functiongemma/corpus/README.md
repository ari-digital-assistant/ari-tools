# FunctionGemma corpus banks

The volume half of Google's mobile-actions recipe (9,650 rows for seven
functions — ~1,380 per function) applied to Ari's skills. Humans (or an LLM
with human review) author **frames** and **slot banks**; `corpus_expander.py`
multiplies them deterministically at dataset-generation time
(`generate-dataset.py --augment functiongemma/corpus`). The LLM never does the
maths; the expander never invents content.

**Every router-eligible skill MUST have a frame bank entry** — generation
fails naming the skill otherwise. Adding a skill to the registry means adding
its frames here (draft with `author-frames.py`, review, commit).

## Files (per locale `xx` — currently `en`, `it`)

Split files are fine and encouraged: the expander unions everything matching
`frames*.xx.json` and `slots*.xx.json` (sorted, deterministic), and fails
loudly if the same skill or slot is defined twice. Files named `draft-*` are
ignored (that's `author-frames.py` output awaiting human review — review,
rename, commit).

### `frames.xx.json`
```json
{
  "dev.heyari.timer": {
    "target": 500,
    "cap": 12,
    "frames": [
      {"text": "set a {N1} minute timer for the pasta",
       "args": {"duration_minutes": "{N1}", "name": "pasta"}},
      {"text": "how long is left on the {THING} timer",
       "args": {"action": "query", "name": "{THING}"}}
    ]
  }
}
```
- `target` (default 500): expansions to keep for this skill after dedupe.
- `cap` (default 12): max expansions per frame — one frame repeated 400
  times with different numbers teaches the frame, not the skill.
- `{SLOT}` in `text` → the slot's **surface** form; `{SLOT}` inside an args
  value → its **canonical** form. Args stay canonical English
  (`"la fotocamera"` → `"Camera"`, expressions in evaluator syntax).
- A frame needing two values from the same conceptual bank uses two bank
  keys (`{N1}`, `{N2}`) — there is no aliasing syntax.
- Slotless skills (greeting, current_time…): a frame is exactly one example —
  compensate with more frames (150–250) and accept the reported shortfall.

### `slots.xx.json`
```json
{
  "APP":  [{"surface": "spotify", "canonical": "Spotify"},
           {"surface": "la fotocamera", "canonical": "Camera"}],
  "N1":   {"type": "int", "min": 2, "max": 999}
}
```
Surface forms are locale-idiomatic; canonicals identical across locales.
Numeric banks are generated, never enumerated.

### `negatives.xx.json`
Same frame machinery, no `args` — general-knowledge questions the router
must ABSTAIN on. Must include **attractor traps** at volume: the shapes the
failure map shows pulling false routes (who-/chi-questions near `greeting`,
historical dates near `current_date`, sentences containing time/tempo that
are still general knowledge, quantity questions near `calculator`).

## Rules that are not style preferences

1. **Natural language only.** A frame no real user would say is noise, not
   signal (measured lesson, 2026-07-19). Write what people say to an
   assistant, in that locale's idiom — Italian frames are idiomatic Italian,
   never translated English.
2. **Do NOT filter by keyword-hit status.** Canonical AND oblique phrasings
   both belong — training on keyword-won shapes was proven harmless and
   removing them was proven harmful (falsified 2026-07-19). Obliques carry
   the generalisation signal; write plenty of both.
3. **The evals are held out.** The expander decontaminates automatically,
   but do not author frames copied from `routing-eval*.jsonl`.
4. **Canonical args never localise.** App names, expression syntax, enum
   values (`when: now|today|tomorrow|this week`) stay exactly as the skill's
   manifest declares them.
5. Reproducibility: expansion is seeded — same banks, same corpus, anywhere
   including CI.

## Automation — what the nightly does for you

You should not normally author banks by hand. When a skill is added, or its
description / parameters / examples change, the nightly training run:

1. **Detects the drift** — `check_banks.py` fingerprints each router-eligible
   skill (id + description + parameters + example texts, deliberately NOT the
   whole manifest, so a version bump or prose edit doesn't invalidate good
   banks) and compares against `bank-sources.json`.
2. **Drafts the banks** — `author-frames.py --apply` per changed skill, under
   a fixed prompt contract that embeds this file verbatim.
3. **Proves the corpus still builds** before committing or spending a
   GPU-second.
4. **Commits to main** as `ari-frames-bot`, into `frames.auto.<locale>.json` /
   `slots.auto.<locale>.json` — kept separate from hand-authored banks so it
   is always obvious which frames a machine wrote.
5. **Trains and gates.** The promotion gate is the safety net: bad banks make
   a model that fails Gate v3 and publishes nothing.

Italian drafts additionally open a review issue (label `italian-review`).
That is a queue, not a blocker — edit `frames.auto.it.json` in place and
commit; there is no need to re-draft.

**Requires the `OPENAI_API_KEY` secret.** Without it the nightly fails
loudly naming the stale skills rather than training on a stale corpus.

Manual escape hatches, unchanged: `author-frames.py --skill X --locale en`
writes a `draft-*` file for review (the expander ignores `draft-*`), and the
`author-frames` workflow_dispatch does the same via a PR.
