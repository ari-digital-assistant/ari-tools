"""Tests for the detection rule. This is a port of MicroWakeWordEngine.cpp;
if it drifts, every number the ship gate produces is about this script rather
than about the phone."""

from __future__ import annotations

import numpy as np

from eval_model import REFRACTORY_MS, Score, count_detections

STEP_MS = 10


def test_sustained_high_probability_fires_once():
    probs = np.full(50, 0.99)

    assert count_detections(probs, cutoff=0.985, window=10, step_ms=STEP_MS) == 1


def test_nothing_fires_below_the_cutoff():
    probs = np.full(50, 0.98)

    assert count_detections(probs, cutoff=0.985, window=10, step_ms=STEP_MS) == 0


def test_the_rule_is_the_window_MEAN_not_every_frame():
    """Five frames at 1.0 and five at 0.98 average 0.99, which clears 0.985.

    The engine sums across the window and compares to cutoff * window, so a
    single dipping frame does not veto a detection. A per-frame rule would
    score this as a miss and quietly make every model look worse.
    """
    probs = np.concatenate([np.full(5, 1.0), np.full(5, 0.98)])

    assert count_detections(probs, cutoff=0.985, window=10, step_ms=STEP_MS) == 1


def test_a_window_short_of_frames_cannot_fire():
    probs = np.full(9, 1.0)

    assert count_detections(probs, cutoff=0.985, window=10, step_ms=STEP_MS) == 0


def test_one_long_shout_counts_once_not_once_per_frame():
    """Without the refractory period a two-second wake scores as ~190 wakes,
    and false-accepts-per-hour becomes meaningless."""
    probs = np.full(200, 0.999)

    assert count_detections(probs, cutoff=0.985, window=10, step_ms=STEP_MS) == 1


def test_two_events_separated_by_more_than_the_refractory_count_twice():
    quiet = np.zeros(REFRACTORY_MS // STEP_MS + 20)
    probs = np.concatenate([np.full(20, 0.999), quiet, np.full(20, 0.999)])

    assert count_detections(probs, cutoff=0.985, window=10, step_ms=STEP_MS) == 2


def test_score_reports_rate_per_hour_and_per_clip():
    score = Score(clips=10, detected_clips=4, detections=6, audio_seconds=1800.0)

    assert score.clip_rate == 0.4
    assert score.per_hour == 12.0


def test_score_of_nothing_does_not_divide_by_zero():
    empty = Score(clips=0, detected_clips=0, detections=0, audio_seconds=0.0)

    assert empty.clip_rate == 0.0
    assert empty.per_hour == 0.0
