#!/usr/bin/env python3
"""
Generate synthetic speech clips with Piper's LibriTTS-R generator.

Used for BOTH sides of the corpus, because the job is the same either way:
positives are the wake phrase, near-miss negatives are the phrases the rejection
log shows actually confuse the model. Only the text differs.

This is step one of two. It produces CLEAN speech only — many voices, many
speeds, all of it loud and close-mic'd. Do not train on the output of this
script. `../train/prepare_clips.py` does the resampling and the volume/room
augmentation, and that second step is not optional:

    Raw generator output is uniformly loud — measured across 40 clips from 40
    different speakers, RMS varies by 4.6 dB. Real speech across a room varies
    by 20-40 dB, so training on this unaltered would teach the model about a
    world that does not exist.

A six-sample run of this script shows the problem in its own output: six
different speakers, peaks of 32767 / 30899 / 27790, RMS clustered inside a 20%
band. That uniformity is the thing to destroy downstream.

Speaker diversity is the whole reason for the LibriTTS-R generator over a plain
Piper voice: it carries 904 speakers and can interpolate BETWEEN them via SLERP,
so the corpus is not limited to the speakers that exist. `--max-speakers`
defaults below 904 deliberately — upstream warns that the thinnest-represented
speakers produce artefacts.

Usage:
  # Positives:
  ./generate_speech.py --model ../models/en_US-libritts_r-medium.pt \\
      --count 2000 --output-dir ../corpus/positives-raw

  # Near-miss negatives, from observed confusions rather than guesses:
  ./generate_speech.py --phrases-file ../negatives/near_misses.txt \\
      --model ... --count 3000 --output-dir ../corpus/near-misses-raw

Requires the `generate` venv, and a clone of the upstream repo at
`upstream/` — `piper_train` is missing from every published artifact, so the
clone is the only way to get it. See ../README.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Prosody variants of the same phrase. TTS reads punctuation, so these produce
# genuinely different intonation contours rather than near-duplicate clips —
# a statement, a question, a shout. Real people do all three.
PROSODY_SUFFIXES = (".", "?", "!", "")

# Upstream warns that the thinnest-represented of the 904 LibriTTS-R speakers
# produce artefacts, and a corpus of artefacts teaches artefacts.
DEFAULT_MAX_SPEAKERS = 800

# Wider than upstream's (0.75, 1, 1.25). People say a wake phrase briskly when
# they are used to it and slowly when it has just ignored them twice.
DEFAULT_LENGTH_SCALES = (0.6, 0.75, 0.9, 1.0, 1.15, 1.3)

# 0.5 is an even blend of two speakers; the outliers lean towards one of them.
# Cycling these makes the interpolated voices spread out instead of all landing
# halfway between the same pairs.
DEFAULT_SLERP_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _import_generator():
    """Import upstream's generator, with `piper_train` supplied from the clone.

    `piper-sample-generator` imports `piper_train.vits.commons`, but its
    packaging includes only `piper_sample_generator*`, so no published artifact
    ships it. Handled here rather than in a PYTHONPATH the caller has to
    remember, because forgetting it fails at import time with a stack trace
    that names the wrong problem.
    """
    upstream = HERE / "upstream"
    if not (upstream / "piper_train").is_dir():
        sys.exit(
            f"Missing {upstream}/piper_train.\n"
            "Clone the pinned upstream tag (see ../README.md):\n"
            "  git clone --depth 1 --branch v3.2.0 "
            "https://github.com/rhasspy/piper-sample-generator.git upstream"
        )
    sys.path.insert(0, str(upstream))
    from piper_sample_generator.__main__ import generate_samples

    return generate_samples


def _revision(path: Path) -> str:
    """The upstream commit the clone is parked on, for the manifest."""
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--phrase", default="hey ari", help="Single phrase, lowercase")
    ap.add_argument(
        "--phrases-file",
        type=Path,
        help="One phrase per line; blank lines and # comments ignored. "
        "Overrides --phrase. Prosody suffixes are NOT added to these — a "
        "near-miss list is already a list of exact things somebody said.",
    )
    ap.add_argument("--model", required=True, type=Path, help="LibriTTS-R .pt")
    ap.add_argument("--count", type=int, required=True, help="Samples to generate")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16, help="Raise on a GPU")
    ap.add_argument("--max-speakers", type=int, default=DEFAULT_MAX_SPEAKERS)
    ap.add_argument(
        "--length-scales", type=float, nargs="+", default=list(DEFAULT_LENGTH_SCALES)
    )
    ap.add_argument(
        "--slerp-weights", type=float, nargs="+", default=list(DEFAULT_SLERP_WEIGHTS)
    )
    args = ap.parse_args()

    if not args.model.is_file():
        sys.exit(f"No generator model at {args.model}")
    # The generator also wants <model>.pt.json beside the checkpoint, and that
    # file ships in the upstream repo rather than the GitHub release — so a
    # correctly downloaded .pt on its own still fails, several minutes in.
    if not args.model.with_suffix(".pt.json").is_file():
        sys.exit(
            f"No config at {args.model.with_suffix('.pt.json')}\n"
            "It ships in the upstream repo, not the release:\n"
            f"  cp {HERE}/upstream/models/{args.model.stem}.pt.json {args.model.parent}/"
        )

    generate_samples = _import_generator()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.phrases_file:
        lines = args.phrases_file.read_text(encoding="utf-8").splitlines()
        texts = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
        if not texts:
            sys.exit(f"No phrases in {args.phrases_file}")
    else:
        texts = [args.phrase + suffix for suffix in PROSODY_SUFFIXES]
    started = time.monotonic()
    generate_samples(
        text=texts,
        output_dir=args.output_dir,
        model=args.model,
        max_samples=args.count,
        batch_size=args.batch_size,
        max_speakers=args.max_speakers,
        length_scales=tuple(args.length_scales),
        slerp_weights=tuple(args.slerp_weights),
    )
    elapsed = time.monotonic() - started

    produced = sorted(args.output_dir.glob("*.wav"))
    # The settings ARE the corpus. Without this, a model trained six weeks from
    # now cannot be compared against one trained today, because nobody can say
    # what differed between them.
    manifest = {
        "phrase": None if args.phrases_file else args.phrase,
        "phrases_file": str(args.phrases_file) if args.phrases_file else None,
        "texts": texts,
        "count_requested": args.count,
        "count_produced": len(produced),
        "max_speakers": args.max_speakers,
        "length_scales": args.length_scales,
        "slerp_weights": args.slerp_weights,
        "model": args.model.name,
        "generator_revision": _revision(HERE / "upstream"),
        "sample_rate_hz": 22050,
        "augmented": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(elapsed, 1),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"{len(produced)} clips in {args.output_dir} ({elapsed:.1f}s)")
    print("NOT training data yet — run ../train/prepare_clips.py next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
