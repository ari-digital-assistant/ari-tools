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

Recall is reported per speaker whenever the positives carry marks, because one
number across a household hides the only failure that matters. The ladder this
app shipped from April to September served the developer 13 takes in 16 and the
quietest speaker in his house 18 in 46. Averaged, that is 51%, and 51% reads
like a model problem rather than a threshold that answers one person.

Usage:
  ./eval_model.py --model ../../../ari-android/app/src/main/assets/hey_ari.tflite \\
      --positives ~/captures/hey-ari --negatives ~/captures/rejected

  # Sweep the operating point instead of using the app's presets:
  ./eval_model.py --model ... --positives ... --negatives ... --sweep
"""

from __future__ import annotations

import argparse
import json
import re
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

# Where the engine keeps the ladder the app actually ships.
REGISTRY_RS = (
    HERE.parent.parent.parent / "ari-engine/crates/ari-voice/src/wake_model.rs"
)


def app_operating_points(model_id: str = "hey_ari") -> dict[str, tuple[float, int]]:
    """The three settings a user is actually running, read from the engine source.

    Not copied into a constant here, because a copy is exactly what went wrong
    twice. This harness is the ship gate: it decides whether a model is good
    enough to put in front of the household, and it answers with whatever
    ladder it believes the app uses. Both times the app's ladder moved, the
    copy stayed, and the gate went on grading against a setting nobody was
    running — once reporting a model uniformly better than the one it replaced
    while the phone said otherwise.

    Parsing Rust with a regex is ugly and it is still the right trade: the
    alternative is a number that agrees with the app only as long as somebody
    remembers, and twice now nobody did. If the parse fails this raises rather
    than falling back to a guess, because a wrong ladder here is invisible in
    the output — every row still prints, just about the wrong app.
    """
    src = REGISTRY_RS.read_text()
    block = re.search(
        rf'\bid:\s*"{re.escape(model_id)}"\.into\(\)(.*?)^\s*\}}',
        src,
        re.DOTALL | re.MULTILINE,
    )
    if not block and re.search(rf'\bunmeasured\(\s*"{re.escape(model_id)}"', src):
        # The models with no household recordings share one block via the
        # `unmeasured` helper, so they have no `id: "..."` line of their own.
        block = re.search(
            r"fn unmeasured\((.*?)^\s*\}", src, re.DOTALL | re.MULTILINE
        )
    if not block:
        raise SystemExit(f"No WakeWordModel block for {model_id!r} in {REGISTRY_RS}")
    points = {}
    for level in ("high", "medium", "low"):
        m = re.search(
            rf"\b{level}:\s*OperatingPoint::new\(\s*([0-9.]+)\s*,\s*(\d+)\s*\)",
            block.group(1),
        )
        if not m:
            raise SystemExit(f"No {level} OperatingPoint for {model_id!r} in {REGISTRY_RS}")
        points[level.upper()] = (float(m.group(1)), int(m.group(2)))
    return points

# WakeWordService.detectionDebounceMs. One shout must not score as six wakes.
REFRACTORY_MS = 4000
# How long before the speaker's tap a detection may land and still be that take.
# The phrase itself plus however long a hand takes to reach the button.
MARK_WINDOW_MS = 2500


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


@dataclass(frozen=True)
class MarkedSegment:
    """One continuous recording and the taps the speaker made inside it."""

    path: Path
    speaker: str
    marks: list[int]
    probs: np.ndarray


def read_wav_16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(
                f"{path.name}: need 16 kHz mono 16-bit, got "
                f"{w.getframerate()} Hz / {w.getnchannels()}ch / {w.getsampwidth()*8}-bit"
            )
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def count_detections(probs: np.ndarray, cutoff: float, window: int, frame_ms: int) -> int:
    """Detections in one clip under the app's sliding-window mean rule.

    [frame_ms] is the audio a single probability covers, which is NOT the
    feature step size: MicroWakeWordEngine accumulates `stride` feature frames
    before each Invoke, so a 10 ms feature step on a stride-2 model yields one
    probability per 20 ms. Reading it as 10 doubles the refractory period and
    silently swallows real repeat wakes.
    """
    if len(probs) < window:
        return 0
    means = np.convolve(probs, np.ones(window) / window, mode="valid")
    refractory_frames = REFRACTORY_MS // frame_ms
    detections = 0
    index = 0
    while index < len(means):
        if means[index] > cutoff:
            detections += 1
            index += refractory_frames
        else:
            index += 1
    return detections


def score_dir(clips: list[tuple[Path, np.ndarray]], cutoff, window, frame_ms) -> Score:
    detected = 0
    total = 0
    seconds = 0.0
    for _, probs_and_len in clips:
        probs, samples = probs_and_len
        hits = count_detections(probs, cutoff, window, frame_ms)
        detected += 1 if hits else 0
        total += hits
        seconds += samples / 16000.0
    return Score(len(clips), detected, total, seconds)


def ambient_floor(negatives, window: int) -> float:
    """The highest window mean the negatives ever reach.

    The number every cutoff in `wake_model.rs` is quoted against, and the
    one `WakeWordSensitivityTest` asserts on. FA/hour answers "did it fire on
    this recording"; on twenty minutes of audio the answer is usually no, at
    every cutoff, which says nothing about whether the next twenty minutes will
    — the last two ladders both read zero here and one of them false-fired in a
    house within two days. The margin is the number that degrades gracefully:
    it says by how much the recording missed.
    """
    return max(
        float(np.convolve(probs, np.ones(window) / window, mode="valid").max())
        for _, (probs, _) in negatives
    )


def load_probs(model_path: Path, directory: Path, step_ms: int):
    """Score every clip in [directory] from a cold start.

    A new Model per clip, because nothing short of that resets one. The tflite
    model is a streaming model whose recurrent state survives
    `reset_all_variables()`, `allocate_tensors()` and zeroing every input
    tensor — scoring the same clip twice on one instance already gives two
    answers. Left shared, a clip inherits whatever the previous clip left
    behind: the ten-minute kitchen negative opens at 0.02 on a fresh model and
    at 0.9961 straight after a wake phrase, so a directory's results depended on
    the alphabetical order of its filenames.

    Construction costs about a millisecond, which is nothing beside inference.
    """
    from microwakeword.inference import Model

    out = []
    for path in sorted(directory.glob("*.wav")):
        audio = read_wav_16k(path)
        probs = np.asarray(
            Model(str(model_path)).predict_clip(audio, step_ms=step_ms), dtype=np.float64
        )
        out.append((path, (probs, len(audio))))
    if not out:
        sys.exit(f"No WAVs in {directory}")
    return out


def score_segment(
    probs: np.ndarray, marks: list[int], cutoff: float, window: int, frame_ms: int
) -> tuple[int, int]:
    """Takes caught, and detections that answered to no take.

    The faithful measurement. On the phone the detector runs across one
    continuous stream with its state carried from whatever the room was doing a
    moment ago; it never sees a clip boundary and never starts cold. Splitting a
    segment into takes and scoring each on its own tests neither of those, and
    which artefact you get depends on whether the model instance is shared.

    A mark is tapped just after the phrase, so a take counts as caught when the
    window mean clears the cutoff somewhere in the [MARK_WINDOW_MS] before the
    tap. Deliberately without the refractory period: a speaker recording takes
    says the phrase every two or three seconds, so a 4 s debounce would score
    most of them as misses when the question being asked is whether the model
    heard each one. The refractory is real on the phone, but nobody talks to an
    assistant in bursts of fifteen.

    Detections outside every mark window are counted with the refractory, since
    there the debounce is exactly what a user would experience. They are false
    wakes inside a positive recording, which clip scoring cannot see at all.
    """
    means = np.convolve(probs, np.ones(window) / window, mode="valid")

    # Each take owns the stretch between the previous tap and its own. Without
    # that clamp a 2.5 s lookback reaches back over the take before it — the
    # speakers here tap every two seconds — and one clear phrase gets counted
    # twice while the mumbled one beside it scores as a hit.
    windows = []
    previous = 0
    for mark in marks:
        start = max(previous, mark - MARK_WINDOW_MS)
        windows.append((max(0, start // frame_ms), min(len(means), mark // frame_ms + 1)))
        previous = mark

    caught = 0
    for start, end in windows:
        if end > start and means[start:end].max() > cutoff:
            caught += 1

    in_take = np.zeros(len(means), dtype=bool)
    for start, end in windows:
        in_take[start:end] = True
    refractory = REFRACTORY_MS // frame_ms
    unclaimed = 0
    index = 0
    while index < len(means):
        if means[index] > cutoff:
            if not in_take[index]:
                unclaimed += 1
            index += refractory
        else:
            index += 1
    return caught, unclaimed


def read_marks(wav: Path) -> tuple[str, list[int]] | None:
    """Speaker and tap times from a WakeSampleRecorder sidecar, or None.

    None when there is no sidecar or it carries no marks, which is how an
    unmarked corpus falls back to clip scoring instead of reporting a recall of
    zero against takes nobody recorded.

    The speaker comes from the sidecar's `set:` rather than the filename: the
    recorder writes the name the speaker typed, and a filename is only ever a
    convention somebody can rename.

    That field is free text, and the same person does not type their name the
    same way twice — this corpus holds both "Gail" and "gail". Left as typed,
    her 46 takes report as a 31 and a 15, and the breakdown that exists to stop
    one speaker's recall hiding inside a household average hides it inside two
    spellings instead. Hence the fold to a single key.
    """
    sidecar = wav.with_suffix(".txt")
    if not sidecar.exists():
        return None
    speaker, marks = "unknown", []
    for line in sidecar.read_text().splitlines():
        key, _, value = line.partition(":")
        if key == "set" and value.strip():
            speaker = " ".join(value.split()).casefold()
        elif key == "marks":
            marks = [int(m) for m in value.split(",") if m.strip()]
    return (speaker, marks) if marks else None


def marked_segments(positives) -> list[MarkedSegment]:
    """Pair each scored positive with its marks, dropping those that have none.

    Built from probabilities already computed for [positives] rather than
    re-running inference, which is most of a run's cost.
    """
    segments = []
    for path, (probs, _) in positives:
        parsed = read_marks(path)
        if parsed is None:
            continue
        speaker, marks = parsed
        segments.append(MarkedSegment(path, speaker, marks, probs))
    return segments


def score_speakers(
    segments: list[MarkedSegment], cutoff: float, window: int, frame_ms: int
) -> dict[str, tuple[int, int, int]]:
    """Per speaker: takes caught, takes attempted, and false wakes inside them."""
    per: dict[str, tuple[int, int, int]] = {}
    for segment in segments:
        caught, unclaimed = score_segment(
            segment.probs, segment.marks, cutoff, window, frame_ms
        )
        had = per.get(segment.speaker, (0, 0, 0))
        per[segment.speaker] = (
            had[0] + caught,
            had[1] + len(segment.marks),
            had[2] + unclaimed,
        )
    return per


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
    # int() because the stride comes off the tflite input shape as a numpy
    # int32, which propagates into the JSON and fails the write at the very end
    # of a run, after all the compute is spent.
    frame_ms = args.step_ms * int(model.stride)
    positives = load_probs(args.model, args.positives, args.step_ms)
    negatives = load_probs(args.model, args.negatives, args.step_ms)
    segments = marked_segments(positives)
    speakers = sorted({segment.speaker for segment in segments})
    takes = sum(len(segment.marks) for segment in segments)

    if args.sweep:
        # Spread across the range a ladder has actually occupied rather than
        # bunched at the ceiling, which is where they all sat when the only
        # cutoff anyone had tried was 0.985.
        points = {f"{c:.4f}": (c, args.window) for c in
                  (0.45, 0.55, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95, 0.985)}
    else:
        points = app_operating_points()

    print(f"model     {args.model.name}  (stride {model.stride}, {frame_ms} ms per probability)")
    if segments:
        unmarked = len(positives) - len(segments)
        note = f", {unmarked} unmarked clip(s) ignored" if unmarked else ""
        print(f"positives {takes} marked takes across {len(segments)} segments{note}")
    else:
        print(f"positives {len(positives)} clips  ({sum(p[1][1] for p in positives)/16000:.1f}s)")
    print(f"negatives {len(negatives)} clips  ({sum(n[1][1] for n in negatives)/16000:.1f}s)")
    print()

    # Two layouts, because clip recall is not merely coarser on a marked
    # corpus, it is wrong: a segment holding sixteen takes counts as one clip
    # and scores 100% the moment any single one of them lands.
    if segments:
        header = (f"{'point':>8} {'cutoff':>7} {'win':>4} {'margin':>7} | "
                  + " ".join(f"{s:>11}" for s in speakers)
                  + f" | {'all':>10} | {'in-take':>7} {'FA/hour':>8}")
    else:
        header = (f"{'point':>8} {'cutoff':>7} {'win':>4} {'margin':>7} | "
                  f"{'recall':>7} {'missed':>7} | {'neg fired':>10} {'FA/hour':>8}")
    print(header)
    print("-" * len(header))

    results = {}
    for name, (cutoff, window) in points.items():
        neg = score_dir(negatives, cutoff, window, frame_ms)
        floor = ambient_floor(negatives, window)
        results[name] = {
            "cutoff": cutoff,
            "sliding_window_size": window,
            "ambient_floor": round(floor, 4),
            "margin": round(cutoff - floor, 4),
            "negatives_fired": int(neg.detected_clips),
            "negatives_total": int(neg.clips),
            "false_accepts_per_hour": round(neg.per_hour, 2),
        }
        if segments:
            per = score_speakers(segments, cutoff, window, frame_ms)
            caught = sum(per[s][0] for s in speakers)
            in_take = sum(per[s][2] for s in speakers)
            cells = " ".join(
                f"{per[s][0]:>3}/{per[s][1]:<3}{per[s][0]/per[s][1]:>4.0%}" for s in speakers
            )
            print(
                f"{name:>8} {cutoff:7.4f} {window:4d} {cutoff - floor:+7.2f} | {cells} | "
                f"{caught:>3}/{takes:<3}{caught/takes:>4.0%} | {in_take:7d} {neg.per_hour:8.1f}"
            )
            results[name] |= {
                "recall": round(caught / takes, 4),
                "takes_caught": caught,
                "takes_total": takes,
                "by_speaker": {
                    s: {"caught": per[s][0], "takes": per[s][1], "false_wakes": per[s][2]}
                    for s in speakers
                },
                "in_take_false_wakes": in_take,
            }
        else:
            pos = score_dir(positives, cutoff, window, frame_ms)
            print(
                f"{name:>8} {cutoff:7.4f} {window:4d} {cutoff - floor:+7.2f} | "
                f"{pos.clip_rate:6.1%} {pos.clips - pos.detected_clips:7d} | "
                f"{neg.detected_clips:4d}/{neg.clips:<5d} {neg.per_hour:8.1f}"
            )
            results[name] |= {
                "recall": round(pos.clip_rate, 4),
                # int() because these come off numpy sums as int32/int64, which
                # the json encoder refuses — and it refuses at the very end,
                # after the whole run has been spent.
                "positives_missed": int(pos.clips - pos.detected_clips),
            }

    print()
    if segments:
        print("margin is the cutoff's gap above the loudest window mean the negatives")
        print("reach. It is the unit the ladder in wake_model.rs is quoted in, and")
        print("unlike FA/hour it still carries information when nothing fired at all.")
        print()
    # The note used to assert the negatives were hand-picked near-misses, which
    # stopped being true the moment an ambient corpus existed to pass instead.
    negative_seconds = sum(n[1][1] for n in negatives) / 16000
    if len(negatives) == 1 and negative_seconds > 120:
        print(f"FA/hour is measured on {negative_seconds/60:.0f} minutes of continuous audio")
        print(f"from {args.negatives.name}. It is a rate a user would experience in")
        print("that room, with that noise, and nowhere else.")
    else:
        print(f"FA/hour is extrapolated from {negative_seconds:.0f}s of clips. If those were")
        print("selected because they already fooled a model, it is a worst case rather")
        print("than a rate anybody experiences — use a continuous ambient recording for that.")

    if args.json:
        args.json.write_text(json.dumps(
            {"model": args.model.name, "step_ms": args.step_ms, "frame_ms": frame_ms,
             "positives_dir": str(args.positives), "negatives_dir": str(args.negatives),
             "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
