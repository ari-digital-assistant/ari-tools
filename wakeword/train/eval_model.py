#!/usr/bin/env python3
"""
Score a wake-word model the way the phone actually runs it.

This is the ship gate from the retrain design (section 5). A model only ships
if it beats the incumbent on BOTH numbers, measured on audio neither model was
trained on:

  1. false accepts — how often it wakes when nobody called it
  2. recall        — how often it wakes when somebody did

One without the other is not a win. A model that never false-fires because it
never fires is not an improvement, and it is the easy mistake to make when you
are chasing a single number.

The detection rule here is a deliberate port of the app's, not a convenient
approximation. `MicroWakeWordEngine.cpp` fires when

    sum(recent probabilities) > probability_cutoff * sliding_window_size

i.e. the MEAN probability across the whole window clears the cutoff — and then
holds off for a refractory period so one event is not counted many times. Score
a model any other way and the number is about the scoring script rather than
about the phone.

Features must be generated at the model's own `feature_step_size` (10 ms for
hey_ari; upstream's default is 20). Getting that wrong halves the frame rate and
quietly changes what a "window of 10" means.

Usage:
  ./eval_model.py --model ../../../ari-android/app/src/main/assets/hey_ari.tflite \\
      --positives ~/captures/hey-ari --negatives ~/captures/rejected

  # Sweep the operating point instead of using the app's presets:
  ./eval_model.py --model ... --positives ... --negatives ... --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Upstream's packaging drops microwakeword.audio and .layers (neither has an
# __init__.py, so find_packages skips them), which makes the installed package
# unable to run inference at all. The clone supplies them. See ../README.md.
sys.path.insert(0, str(HERE / "upstream"))

# The three settings the app actually ships, from WakeWordSensitivity.kt. These
# are what a user is running, so these are what the numbers should describe.
APP_OPERATING_POINTS = {
    "HIGH": (0.95, 5),
    "MEDIUM": (0.985, 10),
    "LOW": (0.99, 14),
}

# WakeWordService.detectionDebounceMs. One shout must not score as six wakes.
REFRACTORY_MS = 4000


@dataclass(frozen=True)
class Score:
    clips: int
    detected_clips: int
    detections: int
    audio_seconds: float

    @property
    def clip_rate(self) -> float:
        return self.detected_clips / self.clips if self.clips else 0.0

    @property
    def per_hour(self) -> float:
        hours = self.audio_seconds / 3600.0
        return self.detections / hours if hours else 0.0


def read_wav_16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(
                f"{path.name}: need 16 kHz mono 16-bit, got "
                f"{w.getframerate()} Hz / {w.getnchannels()}ch / {w.getsampwidth()*8}-bit"
            )
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def count_detections(probs: np.ndarray, cutoff: float, window: int, step_ms: int) -> int:
    """Detections in one clip under the app's sliding-window mean rule."""
    if len(probs) < window:
        return 0
    means = np.convolve(probs, np.ones(window) / window, mode="valid")
    refractory_frames = REFRACTORY_MS // step_ms
    detections = 0
    index = 0
    while index < len(means):
        if means[index] > cutoff:
            detections += 1
            index += refractory_frames
        else:
            index += 1
    return detections


def score_dir(model, clips: list[tuple[Path, np.ndarray]], cutoff, window, step_ms) -> Score:
    detected = 0
    total = 0
    seconds = 0.0
    for _, probs_and_len in clips:
        probs, samples = probs_and_len
        hits = count_detections(probs, cutoff, window, step_ms)
        detected += 1 if hits else 0
        total += hits
        seconds += samples / 16000.0
    return Score(len(clips), detected, total, seconds)


def load_probs(model, directory: Path, step_ms: int):
    out = []
    for path in sorted(directory.glob("*.wav")):
        audio = read_wav_16k(path)
        probs = np.asarray(model.predict_clip(audio, step_ms=step_ms), dtype=np.float64)
        out.append((path, (probs, len(audio))))
    if not out:
        sys.exit(f"No WAVs in {directory}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True, type=Path, help="A .tflite wake model")
    ap.add_argument("--positives", required=True, type=Path, help="Clips that SHOULD wake it")
    ap.add_argument("--negatives", required=True, type=Path, help="Clips that should NOT")
    ap.add_argument("--step-ms", type=int, default=10, help="Model's feature_step_size")
    ap.add_argument("--sweep", action="store_true", help="Sweep cutoffs instead of presets")
    ap.add_argument("--window", type=int, default=10, help="Sliding window for --sweep")
    ap.add_argument("--json", type=Path, help="Write results here as well")
    args = ap.parse_args()

    from microwakeword.inference import Model

    model = Model(str(args.model))
    positives = load_probs(model, args.positives, args.step_ms)
    negatives = load_probs(model, args.negatives, args.step_ms)

    if args.sweep:
        points = {f"{c:.4f}": (c, args.window) for c in
                  (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.985, 0.99, 0.995, 0.999)}
    else:
        points = APP_OPERATING_POINTS

    print(f"model     {args.model.name}")
    print(f"positives {len(positives)} clips  ({sum(p[1][1] for p in positives)/16000:.1f}s)")
    print(f"negatives {len(negatives)} clips  ({sum(n[1][1] for n in negatives)/16000:.1f}s)")
    print()
    print(f"{'point':>8} {'cutoff':>7} {'win':>4} | {'recall':>7} {'missed':>7} | "
          f"{'neg fired':>10} {'FA/hour':>8}")
    print("-" * 66)

    results = {}
    for name, (cutoff, window) in points.items():
        pos = score_dir(model, positives, cutoff, window, args.step_ms)
        neg = score_dir(model, negatives, cutoff, window, args.step_ms)
        print(
            f"{name:>8} {cutoff:7.4f} {window:4d} | {pos.clip_rate:6.1%} "
            f"{pos.clips - pos.detected_clips:7d} | "
            f"{neg.detected_clips:4d}/{neg.clips:<5d} {neg.per_hour:8.1f}"
        )
        results[name] = {
            "cutoff": cutoff,
            "sliding_window_size": window,
            "recall": round(pos.clip_rate, 4),
            "positives_missed": pos.clips - pos.detected_clips,
            "negatives_fired": neg.detected_clips,
            "negatives_total": neg.clips,
            "false_accepts_per_hour": round(neg.per_hour, 2),
        }

    print()
    print("FA/hour here is measured on KNOWN-HARD negatives — clips selected")
    print("precisely because they already fooled a model. It is a worst case,")
    print("not the rate a user experiences. An ambient corpus is needed for that.")

    if args.json:
        args.json.write_text(json.dumps(
            {"model": args.model.name, "step_ms": args.step_ms,
             "positives_dir": str(args.positives), "negatives_dir": str(args.negatives),
             "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
