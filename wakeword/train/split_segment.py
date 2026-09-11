#!/usr/bin/env python3
"""
Split a recorder segment into individual takes.

The Android recorder captures one continuous stretch per position — the speaker
says the phrase several times without touching the phone, because the
across-the-room takes are the ones that matter and nobody can reach a button
from there. Splitting is deliberately left until here, where thresholds can be
chosen after hearing the room rather than guessed on a phone, and where a bad
split can be redone without asking anybody to record anything again.

Energy-based, not clever. A wake phrase is short and the gaps around it are
long, so the hard part is not detection, it is not merging two takes into one
or cutting one take in half. The defaults err towards keeping takes whole.

The segment's `.txt` sidecar is copied alongside each take, so the room,
distance and background labels survive the split. Losing those would leave a
pile of clips nobody can group by anything.

Usage:
  ./split_segment.py files/wake-samples/sample-...-gail-bedroom.wav --out ../corpus/gail
  ./split_segment.py seg.wav --out dir --threshold-db -32   # noisier room
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME_MS = 20
# A wake phrase is ~0.5-1.0 s. Anything briefer is a cough or a door.
MIN_TAKE_MS = 350
# Two bursts closer than this are one utterance ("hey" ... "ari"), not two.
MERGE_GAP_MS = 350
# Keep the onset: /h/ is turbulent air and barely registers as energy, so a
# tight cut removes exactly the part of "Hey Ari" the detector needs most.
PAD_MS = 250


def frame_energies_db(audio: np.ndarray, rate: int) -> tuple[np.ndarray, int]:
    frame = int(rate * FRAME_MS / 1000)
    usable = len(audio) - (len(audio) % frame)
    frames = audio[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-9)), frame


def find_takes(audio: np.ndarray, rate: int, threshold_db: float | None) -> list[tuple[int, int]]:
    energies, frame = frame_energies_db(audio, rate)
    if energies.size == 0:
        return []
    if threshold_db is None:
        # Relative to the room, not to an absolute level: the same speaker at
        # the same volume lands 30 dB apart near-field and across a room.
        floor = float(np.percentile(energies, 20))
        peak = float(np.percentile(energies, 99))
        threshold_db = floor + 0.35 * (peak - floor)

    loud = energies > threshold_db
    runs: list[list[int]] = []
    for index, is_loud in enumerate(loud):
        if is_loud:
            if runs and index - runs[-1][1] <= MERGE_GAP_MS // FRAME_MS:
                runs[-1][1] = index
            else:
                runs.append([index, index])

    pad = int(PAD_MS / 1000 * rate)
    takes = []
    for start, end in runs:
        if (end - start + 1) * FRAME_MS < MIN_TAKE_MS:
            continue
        takes.append((max(0, start * frame - pad), min(len(audio), (end + 1) * frame + pad)))
    return takes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("segment", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--threshold-db",
        type=float,
        help="Absolute dBFS gate. Default adapts to the room; set this when a "
        "noisy background defeats that.",
    )
    args = ap.parse_args()

    audio, rate = sf.read(args.segment, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    takes = find_takes(audio, rate, args.threshold_db)
    if not takes:
        sys.exit("No takes found. Try a lower --threshold-db.")

    args.out.mkdir(parents=True, exist_ok=True)
    sidecar = args.segment.with_suffix(".txt")
    for index, (start, end) in enumerate(takes):
        name = f"{args.segment.stem}-take{index:02d}"
        sf.write(args.out / f"{name}.wav", audio[start:end], rate, subtype="PCM_16")
        if sidecar.is_file():
            shutil.copyfile(sidecar, args.out / f"{name}.txt")
        print(f"  {name}.wav  {start/rate:6.2f}s - {end/rate:6.2f}s  ({(end-start)/rate:.2f}s)")

    print(f"\n{len(takes)} takes from {len(audio)/rate:.1f}s")
    print("Check the count against what was actually said before trusting any recall number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
