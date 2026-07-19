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
