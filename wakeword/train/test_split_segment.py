"""Tests for segment splitting.

The failure modes that matter are merging two takes into one and cutting one
take in half — both silently corrupt a recall number, because the count of
takes IS the denominator.
"""

from __future__ import annotations

import numpy as np

from split_segment import find_takes, read_marks, takes_from_marks

RATE = 16000


def _segment(*, bursts: list[tuple[float, float]], total: float = 10.0) -> np.ndarray:
    audio = np.random.default_rng(0).normal(0, 0.0005, int(total * RATE)).astype(np.float32)
    t = np.arange(len(audio)) / RATE
    for start, duration in bursts:
        span = (t >= start) & (t < start + duration)
        audio[span] += 0.3 * np.sin(2 * np.pi * 220 * t[span]).astype(np.float32)
    return audio


def test_silence_yields_nothing():
    audio = np.random.default_rng(0).normal(0, 0.0005, 5 * RATE).astype(np.float32)

    assert find_takes(audio, RATE, threshold_db=-40.0) == []


def test_one_utterance_is_one_take():
    audio = _segment(bursts=[(2.0, 0.8)])

    assert len(find_takes(audio, RATE, threshold_db=-40.0)) == 1


def test_two_well_separated_utterances_stay_separate():
    audio = _segment(bursts=[(1.0, 0.8), (5.0, 0.8)])

    assert len(find_takes(audio, RATE, threshold_db=-40.0)) == 2


def test_a_gap_inside_one_phrase_does_not_split_it():
    """"Hey" then a beat then "Ari" is one take, not two."""
    audio = _segment(bursts=[(2.0, 0.4), (2.55, 0.4)])

    assert len(find_takes(audio, RATE, threshold_db=-40.0)) == 1


def test_a_click_is_not_a_take():
    """Shorter than MIN_TAKE_MS: a door, a cough, a tap on the table."""
    audio = _segment(bursts=[(2.0, 0.1)])

    assert find_takes(audio, RATE, threshold_db=-40.0) == []


def test_takes_are_padded_so_a_soft_onset_survives():
    """/h/ is turbulent air and barely registers; a tight cut removes the part
    of "Hey Ari" the detector most needs."""
    audio = _segment(bursts=[(2.0, 0.8)])

    (start, end), = find_takes(audio, RATE, threshold_db=-40.0)

    assert start < int(2.0 * RATE)
    assert end > int(2.8 * RATE)


def test_marks_are_read_back_in_tap_order(tmp_path):
    sidecar = tmp_path / "seg.txt"
    sidecar.write_text("room: kitchen\nmarks: 1200,5400,9000\napp: 0.1.0\n")

    assert read_marks(sidecar) == [1200, 5400, 9000]


def test_an_unmarked_sidecar_reads_as_no_marks(tmp_path):
    sidecar = tmp_path / "seg.txt"
    sidecar.write_text("room: kitchen\napp: 0.1.0\n")

    assert read_marks(sidecar) == []


def test_a_mark_snaps_onto_the_burst_that_prompted_it():
    audio = _segment(bursts=[(1.0, 0.8), (5.0, 0.8)])
    runs = find_takes(audio, RATE, threshold_db=-40.0)

    # Tapped 0.4 s after each phrase ended.
    takes, fallbacks = takes_from_marks([2200, 6200], runs, RATE, len(audio))

    assert fallbacks == 0
    assert takes == runs


def test_a_mark_with_no_detectable_burst_still_yields_a_take():
    """The phrase was said too quietly for the energy gate to see it.

    Dropping it would remove exactly the takes a wake model fails on, which
    silently inflates every recall number computed from the split.
    """
    audio = _segment(bursts=[(1.0, 0.8)])
    runs = find_takes(audio, RATE, threshold_db=-40.0)

    takes, fallbacks = takes_from_marks([2200, 8000], runs, RATE, len(audio))

    assert fallbacks == 1
    assert len(takes) == 2
    start, end = takes[1]
    assert start == int(8.0 * RATE) - int(2.5 * RATE)
    assert end == int(8.0 * RATE) + int(0.25 * RATE)


def test_one_take_per_mark_even_when_energy_finds_more():
    """Somebody else talking, or the telly, is not a take.

    The mark count is the denominator; energy bursts are not.
    """
    audio = _segment(bursts=[(1.0, 0.8), (3.0, 0.8), (5.0, 0.8), (7.0, 0.8)])
    runs = find_takes(audio, RATE, threshold_db=-40.0)
    assert len(runs) == 4

    takes, _ = takes_from_marks([2200, 6200], runs, RATE, len(audio))

    assert len(takes) == 2
