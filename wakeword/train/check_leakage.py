#!/usr/bin/env python3
"""
Refuse to train if the exam material is in the lessons.

The ship gate (retrain design section 5) is only worth having if the held-out
set is genuinely held out. The failure mode is not dishonesty, it is drift:
after several failed runs it becomes very tempting to let a few eval clips into
training because the numbers improve. This makes that a hard error rather than
a discipline.

Three routes, because no one check catches all of them:

  1. IDENTICAL — the same PCM in both sets. The obvious case, and the only one
     a hash finds.

  2. SHARED SOURCE — `0-3.wav` in training and `0-7.wav` in eval are different
     augmentations of the SAME original clip. Different bytes, different hash,
     same utterance underneath. prepare_clips.py names variants this way, so
     this is the route a corpus built by this pipeline will actually leak by.

  3. SAME UTTERANCE, DIFFERENT CAPTURE PATH — the one that would have got us.
     WakeCaptureStore's firehose and UtteranceCaptureStore both record an
     accepted turn through different windows: 2 s against 4.2 s of the same
     moment. Different lengths, different directories, different features,
     neither aware of the other. The 2026-09-10 export contained 22 such pairs.
     They are only detectable by the epoch-ms timestamp in the filename stem.

Exit status is 1 if anything is found, so this can gate a pipeline rather than
print advice nobody reads.

Usage:
  ./check_leakage.py --train ../corpus/positives --eval ../corpus/eval-positives
  ./check_leakage.py --train a b c --eval d      # several directories per side
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import wave
from collections import defaultdict
from pathlib import Path

# `wake-00001788516322743-accepted.wav` / `utterance-00001788516330864-answered.wav`
# — the fixed-width epoch-ms that AudioClipStore.clipStem writes.
CAPTURE_STEM = re.compile(r"^[a-z]+-(\d{17})-")

# `0-3.wav` from prepare_clips.py: source stem, then the variant index.
VARIANT_SUFFIX = re.compile(r"^(.*)-(\d+)$")

# Two capture paths recording one turn are seconds apart, never minutes. Wide
# enough for a 4 s turn plus the pre-roll and the write, tight enough that two
# genuinely separate utterances do not collide.
SAME_UTTERANCE_MS = 15_000


def content_hash(path: Path) -> str:
    """Hash the PCM, not the file: a differing WAV header is not a difference."""
    with wave.open(str(path)) as handle:
        return hashlib.sha256(handle.readframes(handle.getnframes())).hexdigest()


def source_stem(path: Path) -> str:
    """The clip a variant came from. `0-3` -> `0`; anything else unchanged."""
    match = VARIANT_SUFFIX.match(path.stem)
    return match.group(1) if match else path.stem


def capture_timestamp(path: Path) -> int | None:
    """Epoch ms from a capture-store filename, or None if it is not one."""
    match = CAPTURE_STEM.match(path.name)
    return int(match.group(1)) if match else None


def wavs(directories: list[Path]) -> list[Path]:
    found: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            sys.exit(f"Not a directory: {directory}")
        found.extend(sorted(directory.rglob("*.wav")))
    return found


def find_leaks(train: list[Path], evaluation: list[Path]) -> list[tuple[str, Path, Path]]:
    leaks: list[tuple[str, Path, Path]] = []

    by_hash: dict[str, Path] = {}
    for path in train:
        by_hash.setdefault(content_hash(path), path)
    for path in evaluation:
        twin = by_hash.get(content_hash(path))
        if twin is not None:
            leaks.append(("identical", twin, path))

    by_source: dict[str, Path] = {}
    for path in train:
        by_source.setdefault(source_stem(path), path)
    for path in evaluation:
        twin = by_source.get(source_stem(path))
        if twin is not None:
            leaks.append(("shared-source", twin, path))

    train_stamps = sorted(
        (stamp, path) for path in train if (stamp := capture_timestamp(path)) is not None
    )
    for path in evaluation:
        stamp = capture_timestamp(path)
        if stamp is None:
            continue
        for other_stamp, other in train_stamps:
            if abs(other_stamp - stamp) <= SAME_UTTERANCE_MS:
                leaks.append(("same-utterance", other, path))
                break

    return leaks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--train", required=True, nargs="+", type=Path)
    ap.add_argument("--eval", required=True, nargs="+", type=Path, dest="evaluation")
    args = ap.parse_args()

    train = wavs(args.train)
    evaluation = wavs(args.evaluation)
    if not train or not evaluation:
        sys.exit("Both sides need at least one clip")

    leaks = find_leaks(train, evaluation)
    print(f"train {len(train)} clips, eval {len(evaluation)} clips")

    if not leaks:
        print("clean — no eval material found in training")
        return 0

    by_kind: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for kind, left, right in leaks:
        by_kind[kind].append((left, right))

    for kind, pairs in by_kind.items():
        print(f"\n{kind}: {len(pairs)}")
        for left, right in pairs[:10]:
            print(f"  train {left.name}\n  eval  {right.name}")
        if len(pairs) > 10:
            print(f"  ... and {len(pairs) - 10} more")

    print(f"\nLEAKAGE: {len(leaks)} pair(s). The ship gate is meaningless until this is fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
