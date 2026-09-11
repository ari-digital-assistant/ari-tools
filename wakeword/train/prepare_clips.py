#!/usr/bin/env python3
"""
Turn audio into training clips: resample to 16 kHz, then break the loudness
shortcut.

Used for every corpus — synthetic positives, synthetic near-miss negatives, and
real captured negatives. Also has an eval mode (`--no-augment`) that resamples
and nothing else, because held-out material has to stay exactly as recorded.

This is the step whose absence produced the bug. `hey_ari.tflite` was trained on
raw generator output, which is uniformly loud — measured over 40 clips from 40
different speakers, RMS varied by 13% and every clip peaked near full scale. In
that training set amplitude genuinely IS discriminative, so the model learned
loudness as a proxy for the wake phrase and never had to learn the phrase. It
then answered to any loud noise and ignored anyone speaking normally from across
a room. See the retrain design, section 2.3.

Real speech reaching a phone across a room varies by 20-40 dB. That is a 10-100x
amplitude range against the generator's 1.7x. Closing that gap is the point of
this script, and `Gain` is therefore the one transform applied to EVERY clip
rather than probabilistically.

Nothing here normalises. A normalisation step anywhere in this chain would
quietly undo the entire exercise and leave the corpus looking fine.

Usage:
  ./prepare_positives.py --input ../corpus/positives-raw \\
      --output ../corpus/positives --variants 5

Each input clip becomes [--variants] different clips: same words, different
room, distance, and level.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from audiomentations import (
    AddGaussianSNR,
    AirAbsorption,
    ApplyImpulseResponse,
    Compose,
    Gain,
)

HERE = Path(__file__).resolve().parent
TARGET_RATE = 16000

# Piper ships eight impulse responses, but most are studio effects rather than
# rooms — "Fat Bass", "Reverse Gate", "Blatty Plate" and "Symphonic" model a
# mixing desk, not a kitchen. Training on those teaches the model what a reverb
# plugin sounds like. Only the room-like ones are used.
#
# A real room-IR corpus (the MIT Acoustical Reverberation Survey, OpenAIR) would
# be a straight upgrade and is the obvious next improvement here.
ROOM_IMPULSES = (
    "Accoustic2_Impulse.wav",
    "Concrete Room.wav",
    "Derlon Sanctuary.wav",
    "ir_bathroom1.wav",
)

# -30 dB is a person across a room; +3 dB is someone leaning into the handset.
# Wide on purpose: the shortcut only breaks if loudness stops correlating with
# "this is the wake phrase".
MIN_GAIN_DB = -30.0
MAX_GAIN_DB = 3.0


def build_chain(impulse_dir: Path) -> Compose:
    """The augmentation chain, in physical order.

    Source -> room -> air -> distance attenuation -> microphone noise floor.
    The order is not cosmetic: attenuating BEFORE adding reverb would model a
    quiet person in a small room rather than a normal person far away, and
    adding the noise floor before the gain would scale the noise with the
    signal, which is exactly what real microphones do not do.
    """
    impulses = [impulse_dir / name for name in ROOM_IMPULSES]
    missing = [p for p in impulses if not p.is_file()]
    if missing:
        sys.exit(
            "Missing impulse responses:\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\nThey ship inside the generate venv's upstream clone."
        )
    return Compose(
        [
            ApplyImpulseResponse(ir_path=impulses, p=0.5),
            AirAbsorption(min_distance=1.0, max_distance=10.0, p=0.3),
            Gain(min_gain_db=MIN_GAIN_DB, max_gain_db=MAX_GAIN_DB, p=1.0),
            AddGaussianSNR(min_snr_db=8.0, max_snr_db=45.0, p=0.6),
        ]
    )


def load_16k(path: Path) -> np.ndarray:
    """Read a WAV as float32 mono at 16 kHz, resampling if needed."""
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != TARGET_RATE:
        audio = soxr.resample(audio, rate, TARGET_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--input", required=True, type=Path, help="Raw generator output")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--variants", type=int, default=5, help="Clips per input clip")
    ap.add_argument(
        "--impulse-dir",
        type=Path,
        default=HERE.parent / "generate/upstream/piper_sample_generator/impulses",
    )
    ap.add_argument("--seed", type=int, default=0, help="For a reproducible corpus")
    ap.add_argument(
        "--no-augment",
        action="store_true",
        help="Resample only, one clip in one clip out. THE EVAL PATH: held-out "
        "material has to stay exactly as recorded. Augmenting it would measure "
        "the model against audio no microphone ever produced, and would flatter "
        "whichever model happened to suit the augmentation.",
    )
    args = ap.parse_args()

    sources = sorted(args.input.glob("*.wav"))
    if not sources:
        sys.exit(f"No WAVs in {args.input}")

    if args.no_augment and args.variants != 1:
        sys.exit("--no-augment makes --variants meaningless; drop it or pass --variants 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    augment = None if args.no_augment else build_chain(args.impulse_dir)
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    written = 0
    clipped = 0
    for source in sources:
        audio = load_16k(source)
        for variant in range(args.variants):
            out = audio if augment is None else augment(samples=audio, sample_rate=TARGET_RATE)
            peak = float(np.max(np.abs(out))) if out.size else 0.0
            # Clip rather than rescale. Rescaling to fit is normalisation by
            # another name, and would hand back the loudness cue this whole
            # script exists to remove.
            if peak > 1.0:
                out = np.clip(out, -1.0, 1.0)
                clipped += 1
            sf.write(
                args.output / f"{source.stem}-{variant}.wav",
                out,
                TARGET_RATE,
                subtype="PCM_16",
            )
            written += 1
    elapsed = time.monotonic() - started

    manifest = {
        "source_dir": str(args.input),
        "source_clips": len(sources),
        "variants_per_clip": args.variants,
        "clips_written": written,
        "clipped_clips": clipped,
        "sample_rate_hz": TARGET_RATE,
        "gain_db_range": None if args.no_augment else [MIN_GAIN_DB, MAX_GAIN_DB],
        "room_impulses": [] if args.no_augment else list(ROOM_IMPULSES),
        "normalised": False,
        "augmented_chain": not args.no_augment,
        "seed": args.seed,
        "augmented": not args.no_augment,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(elapsed, 1),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"{written} clips from {len(sources)} sources in {elapsed:.1f}s")
    if clipped:
        print(f"{clipped} clipped at the ceiling (expected: max gain is +3 dB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
