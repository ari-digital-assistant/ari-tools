#!/usr/bin/env python3
"""
Turn the corpus into the mmap spectrogram banks microWakeWord trains on.

Feeds upstream's own Clips -> Augmentation -> SpectrogramGeneration path rather
than a parallel one, because that is what the trainer consumes and because its
Augmentation mixes real backgrounds into each clip at generation time — which a
directory of pre-augmented WAVs cannot do.

That is why this reads `positives-raw`, NOT `positives`. The prepared clips from
prepare_clips.py are already augmented; running them through again would
compound two augmentation chains and model a room inside a room. prepare_clips
remains the tool for eval material and for standalone resampling, where exactly
one pass is what you want.

Everything runs at step_ms=10, matching the app's feature_step_size. Upstream
defaults to 20. Getting that wrong halves the frame rate and silently changes
what a sliding window of 10 means.

Usage:
  ./make_features.py --corpus ../corpus --out ../features
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream"))

# 3.2 s of context around a ~1 s phrase, as upstream uses. Long enough for the
# jitter and the background to be meaningful, short enough to stay cheap.
AUGMENTATION_DURATION_S = 3.2

# Gain at 1.0, against the incumbent trainer's 0.8: every clip gets a level,
# none escape at full volume. Backgrounds down to -5 dB SNR so speech is
# sometimes genuinely buried — the incumbent's floor was +5, which never
# really tested it.
AUGMENTATION_PROBABILITIES = {
    "SevenBandParametricEQ": 0.1,
    "TanhDistortion": 0.1,
    "PitchShift": 0.1,
    "BandStopFilter": 0.1,
    "AddColorNoise": 0.1,
    "AddBackgroundNoise": 0.75,
    "Gain": 1.0,
    "RIR": 0.5,
}

SPLITS = {
    # slide_frames simulates streaming inference by reusing one spectrogram
    # shifted a frame at a time. The test set runs the streaming model for
    # real, so it needs no artificial repetition.
    "training": {"split": "train", "repetition": 2, "slide_frames": 10},
    "validation": {"split": "validation", "repetition": 1, "slide_frames": 10},
    "testing": {"split": "test", "repetition": 1, "slide_frames": 1},
}


def build(name: str, input_dir: Path, out_root: Path, augmenter, max_duration_s: float | None) -> None:
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap

    clips = Clips(
        input_directory=str(input_dir),
        file_pattern="*.wav",
        max_clip_duration_s=max_duration_s,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )
    for split_dir, cfg in SPLITS.items():
        destination = out_root / name / split_dir
        destination.mkdir(parents=True, exist_ok=True)
        generation = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=cfg["slide_frames"],
            step_ms=10,
        )
        started = time.monotonic()
        RaggedMmap.from_generator(
            out_dir=str(destination / "wakeword_mmap"),
            sample_generator=generation.spectrogram_generator(
                split=cfg["split"], repeat=cfg["repetition"]
            ),
            batch_size=100,
            verbose=False,
        )
        written = len(RaggedMmap(str(destination / "wakeword_mmap")))
        if written == 0:
            sys.exit(
                f"{name}/{split_dir} came out EMPTY.\n"
                f"Clips filters on `duration < max_clip_duration_s` (strictly "
                f"less), so a cap equal to the clip length drops everything. "
                f"Current cap: {max_duration_s}. Check {input_dir}."
            )
        print(f"  {name}/{split_dir} {written:6d} spectrograms "
              f"({time.monotonic() - started:.0f}s)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--impulses", type=Path, default=HERE.parent / "models/mit_rirs")
    args = ap.parse_args()

    from microwakeword.audio.augmentation import Augmentation

    backgrounds = [
        p for p in (args.corpus / "negatives").iterdir() if p.is_dir()
    ] if (args.corpus / "negatives").is_dir() else []
    if not backgrounds:
        sys.exit(f"No background directories under {args.corpus / 'negatives'}")
    if not args.impulses.is_dir():
        sys.exit(f"No impulse responses at {args.impulses}")

    print(f"impulses   {len(list(args.impulses.glob('*.wav')))}")
    print(f"backgrounds {', '.join(p.name for p in backgrounds)}")

    augmenter = Augmentation(
        augmentation_duration_s=AUGMENTATION_DURATION_S,
        augmentation_probabilities=AUGMENTATION_PROBABILITIES,
        impulse_paths=[str(args.impulses)],
        background_paths=[str(p) for p in backgrounds],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )

    sets = [
        ("positive", args.corpus / "positives-raw", None),
        ("near_miss", args.corpus / "near-misses-raw", None),
        # Caps sit ABOVE the real clip length on purpose. Clips filters with
        # `duration < max_clip_duration_s` — strictly less — so a cap of 10.0
        # against AudioSet's exactly-10 s clips excludes every one of them and
        # builds an empty feature bank without a word of complaint. Found the
        # hard way; see the emptiness check in build().
        ("speech_mt", args.corpus / "negatives/common-voice-mt", 12.0),
        ("speech_fr", args.corpus / "negatives/common-voice-fr", 12.0),
        ("ambient", args.corpus / "negatives/audioset", 12.0),
    ]
    for name, directory, max_duration in sets:
        if not directory.is_dir():
            print(f"  (skipping {name}: no {directory})")
            continue
        print(f"{name} <- {directory.name}", flush=True)
        build(name, directory, args.out, augmenter, max_duration)

    print(f"\nfeatures in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
