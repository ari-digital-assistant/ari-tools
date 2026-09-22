"""Tests for the detection rule. This is a port of MicroWakeWordEngine.cpp;
if it drifts, every number the ship gate produces is about this script rather
than about the phone."""

from __future__ import annotations

import numpy as np
import pytest

from pathlib import Path

from eval_model import (
    REFRACTORY_MS,
    MarkedSegment,
    Score,
    ambient_floor,
    count_detections,
    marked_segments,
    read_marks,
    score_segment,
    score_speakers,
)

# One probability covers stride x feature-step ms of audio; the rule itself
# is period-agnostic, so the tests pick a round number.
FRAME_MS = 10


def test_sustained_high_probability_fires_once():
    probs = np.full(50, 0.99)

    assert count_detections(probs, cutoff=0.985, window=10, frame_ms=FRAME_MS) == 1


def test_nothing_fires_below_the_cutoff():
    probs = np.full(50, 0.98)

    assert count_detections(probs, cutoff=0.985, window=10, frame_ms=FRAME_MS) == 0


def test_the_rule_is_the_window_MEAN_not_every_frame():
    """Five frames at 1.0 and five at 0.98 average 0.99, which clears 0.985.

    The engine sums across the window and compares to cutoff * window, so a
    single dipping frame does not veto a detection. A per-frame rule would
    score this as a miss and quietly make every model look worse.
    """
    probs = np.concatenate([np.full(5, 1.0), np.full(5, 0.98)])

    assert count_detections(probs, cutoff=0.985, window=10, frame_ms=FRAME_MS) == 1


def test_a_window_short_of_frames_cannot_fire():
    probs = np.full(9, 1.0)

    assert count_detections(probs, cutoff=0.985, window=10, frame_ms=FRAME_MS) == 0


def test_one_long_shout_counts_once_not_once_per_frame():
    """Without the refractory period a two-second wake scores as ~190 wakes,
    and false-accepts-per-hour becomes meaningless."""
    probs = np.full(200, 0.999)

    assert count_detections(probs, cutoff=0.985, window=10, frame_ms=FRAME_MS) == 1


def test_two_events_separated_by_more_than_the_refractory_count_twice():
    quiet = np.zeros(REFRACTORY_MS // FRAME_MS + 20)
    probs = np.concatenate([np.full(20, 0.999), quiet, np.full(20, 0.999)])

    assert count_detections(probs, cutoff=0.985, window=10, frame_ms=FRAME_MS) == 2


def test_score_reports_rate_per_hour_and_per_clip():
    score = Score(clips=10, detected_clips=4, detections=6, audio_seconds=1800.0)

    assert score.clip_rate == 0.4
    assert score.per_hour == 12.0


def test_score_of_nothing_does_not_divide_by_zero():
    empty = Score(clips=0, detected_clips=0, detections=0, audio_seconds=0.0)

    assert empty.clip_rate == 0.0
    assert empty.per_hour == 0.0


FRAME = 20  # this model's probability period


def _probs(length_ms: int, bursts: list[tuple[int, int]]) -> np.ndarray:
    """A probability stream with 1.0 across each (start_ms, end_ms) burst."""
    p = np.zeros(length_ms // FRAME)
    for start, end in bursts:
        p[start // FRAME : end // FRAME] = 1.0
    return p


def test_a_take_counts_when_the_model_fired_just_before_the_tap():
    # Phrase at 1.0-1.6 s, speaker taps at 2.0 s.
    probs = _probs(4000, [(1000, 1600)])

    assert score_segment(probs, [2000], cutoff=0.9, window=5, frame_ms=FRAME) == (1, 0)


def test_a_silent_take_is_a_miss_not_a_false_wake():
    probs = _probs(4000, [])

    assert score_segment(probs, [2000], cutoff=0.9, window=5, frame_ms=FRAME) == (0, 0)


def test_one_phrase_cannot_be_credited_to_two_takes():
    """Marks two seconds apart with a 2.5 s lookback would otherwise overlap.

    The second take was silent; without clamping each window to start at the
    previous tap it would borrow the first take's phrase and score 2/2.
    """
    probs = _probs(6000, [(1000, 1600)])

    assert score_segment(probs, [2000, 4000], cutoff=0.9, window=5, frame_ms=FRAME) == (1, 0)


def test_a_detection_away_from_every_take_is_a_false_wake():
    # Take at 2 s answered; something else fires at 8 s with nobody speaking.
    probs = _probs(12000, [(1000, 1600), (8000, 8600)])

    assert score_segment(probs, [2000], cutoff=0.9, window=5, frame_ms=FRAME) == (1, 1)


def test_a_long_false_wake_is_counted_once():
    """The refractory applies to the unclaimed detections, where a user would
    genuinely feel the debounce, and not to the per-take recall."""
    probs = _probs(20000, [(8000, 15000)])

    caught, unclaimed = score_segment(probs, [2000], cutoff=0.9, window=5, frame_ms=FRAME)
    assert (caught, unclaimed) == (0, 2)  # 7 s of firing, 4 s refractory


def test_the_refractory_does_not_suppress_rapid_takes():
    """Three takes two seconds apart all count, though a 4 s debounce would
    let only the first through on a real phone. Recording takes in quick
    succession is a protocol artefact, not how anybody speaks to an assistant."""
    probs = _probs(8000, [(1000, 1600), (3000, 3600), (5000, 5600)])

    assert score_segment(probs, [2000, 4000, 6000], cutoff=0.9, window=5, frame_ms=FRAME) == (3, 0)


def sidecar(tmp_path, name, body):
    (tmp_path / f"{name}.wav").write_bytes(b"")
    (tmp_path / f"{name}.txt").write_text(body)
    return tmp_path / f"{name}.wav"


def test_marks_come_from_the_sidecar_not_the_filename():
    """A filename is a convention; `set:` is what the speaker typed."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        wav = sidecar(Path(d), "sample-0001-keith-kitchen",
                      "set: Gail\nmarks: 4059,6321,8455\n")

        assert read_marks(wav) == ("gail", [4059, 6321, 8455])


def test_one_speaker_who_typed_their_name_two_ways_is_one_speaker():
    """The real corpus holds both "Gail" and "gail". Split, she reports as a
    31/46 and a 15/46 and neither number looks like the problem it is."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        shouty = sidecar(Path(d), "one", "set:  GAIL \nmarks: 2000\n")
        quiet = sidecar(Path(d), "two", "set: gail\nmarks: 2000\n")
        probs = np.zeros(10)

        segments = marked_segments([(shouty, (probs, 0)), (quiet, (probs, 0))])

        assert {s.speaker for s in segments} == {"gail"}


def test_a_clip_with_no_sidecar_is_unmarked_rather_than_a_miss():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "lonely.wav"
        wav.write_bytes(b"")

        assert read_marks(wav) is None


def test_a_sidecar_with_an_empty_marks_line_is_unmarked():
    """The recorder writes the key whether or not anybody tapped.

    Treating that as zero takes would report a recall of 0% against a corpus
    nobody marked, which reads exactly like a broken model.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        wav = sidecar(Path(d), "untapped", "set: Gail\nmarks: \n")

        assert read_marks(wav) is None


def test_unmarked_clips_are_dropped_from_the_marked_corpus():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        marked = sidecar(Path(d), "marked", "set: joe\nmarks: 2000\n")
        unmarked = Path(d) / "plain.wav"
        unmarked.write_bytes(b"")
        probs = np.zeros(10)

        segments = marked_segments([(marked, (probs, 0)), (unmarked, (probs, 0))])

        assert [s.speaker for s in segments] == ["joe"]


def test_each_speaker_is_counted_apart_across_segments():
    """The whole point. One number across a household hid a 18/46 for months."""
    quiet = np.concatenate([np.zeros(100), np.full(10, 0.99), np.zeros(100)])
    silent = np.zeros(210)
    segments = [
        MarkedSegment(Path("a.wav"), "gail", [1100], quiet),
        MarkedSegment(Path("b.wav"), "gail", [1100], silent),
        MarkedSegment(Path("c.wav"), "keith", [1100], quiet),
    ]

    per = score_speakers(segments, cutoff=0.9, window=5, frame_ms=FRAME_MS)

    assert per == {"gail": (1, 2, 0), "keith": (1, 1, 0)}


def test_the_ambient_floor_is_the_loudest_window_mean_any_negative_reaches():
    """Not the loudest frame: the engine fires on the window mean, so a floor
    quoted per frame would price headroom the detector never has to clear."""
    spike = np.concatenate([np.zeros(20), np.full(5, 0.8), np.zeros(20)])
    flat = np.full(45, 0.2)

    floor = ambient_floor([(Path("a.wav"), (spike, 0)), (Path("b.wav"), (flat, 0))], window=10)

    assert floor == pytest.approx(0.4)
