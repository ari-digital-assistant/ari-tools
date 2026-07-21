"""Cross-skill ambiguity guards for the frame banks.

Written after the 2026-07-21 incident: counter's banks carried bare
verb+number shapes (`più uno`, `somma {ITCNT} al contatore`) that share
calculator's core surface, and the trained model routed arithmetic to
counter with high confidence — failing the promotion gate two nights
running. corpus/README.md's "stay in your lane" section states the rule for
authors (human and machine); this file makes the incident's specific shape
a hard error rather than a hope.
"""

import json
import re
from pathlib import Path

CORPUS = Path(__file__).parent / "corpus"

# A counter frame may mention numbers ONLY alongside a noun (or the counting
# verb) that names its own domain. Substring match on purpose: "conta"
# covers contatore/conteggio/conta.
# "tick" and "increment" count as anchors: they are counting-domain
# vocabulary no sibling uses — arithmetic never ticks.
COUNTER_ANCHORS = {
    "en": ("counter", "count", "tally", "tick", "increment"),
    "it": ("conta", "conteggio", "incrementa"),
}

# The exact shapes the incident model learned arithmetic from. They must
# never return, in any file, under any skill except calculator.
BANNED_OUTSIDE_CALCULATOR = {
    "en": ("add {CN1} to the tally", "add another {CN1} to the count"),
    "it": ("più uno", "aggiungi altri {ITCNT}", "fai {ITCNT} in più",
           "somma {ITCNT} al contatore"),
}

NUMERIC = re.compile(r"\{[A-Z][A-Z0-9_]*\}|\d|\buno\b|\bone\b")


def _frames(locale):
    for p in sorted(CORPUS.glob(f"frames*.{locale}.json")):
        if p.name.startswith("draft-"):
            continue
        for sid, entry in json.loads(p.read_text()).items():
            for f in entry.get("frames", []):
                yield p.name, sid, f["text"]


def test_counter_number_frames_are_anchored():
    for locale, anchors in COUNTER_ANCHORS.items():
        offenders = [
            text
            for _, sid, text in _frames(locale)
            if sid == "dev.heyari.counter"
            and NUMERIC.search(text)
            and not any(a in text.lower() for a in anchors)
        ]
        assert offenders == [], (
            f"[{locale}] counter frames mention a number without naming the "
            f"counter domain — that surface belongs to calculator and trains "
            f"misrouting: {offenders}"
        )


def test_incident_shapes_never_return():
    for locale, banned in BANNED_OUTSIDE_CALCULATOR.items():
        offenders = [
            (sid, text)
            for _, sid, text in _frames(locale)
            if sid != "calculator" and text in banned
        ]
        assert offenders == [], (
            f"[{locale}] 2026-07-21 incident shapes are back in a "
            f"non-calculator bank: {offenders}"
        )
