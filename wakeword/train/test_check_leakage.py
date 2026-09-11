"""Tests for the leakage gate.

Each test covers one of the three routes. The third is the one that matters
most: it is the only check that would have caught the 22 duplicate pairs in the
2026-09-10 export, which a hash or a filename comparison sails straight past.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from check_leakage import capture_timestamp, find_leaks, source_stem


def _write(path: Path, seconds: float = 0.2, freq: float = 220.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 16000
    t = np.linspace(0.0, seconds, int(seconds * rate), endpoint=False)
    pcm = (np.sin(2 * math.pi * freq * t) * 8000).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return path


def test_source_stem_strips_the_augmentation_variant():
    assert source_stem(Path("0-3.wav")) == "0"
    assert source_stem(Path("sample-17-11.wav")) == "sample-17"


def test_source_stem_leaves_names_that_are_not_variants():
    assert source_stem(Path("cv-mt-common_voice_mt_21107689.wav")) == (
        "cv-mt-common_voice_mt_21107689"
    )


def test_capture_timestamp_reads_the_clip_stem():
    assert capture_timestamp(Path("wake-00001788516322743-accepted.wav")) == 1788516322743
    assert capture_timestamp(Path("utterance-00001788516330864-answered.wav")) == 1788516330864


def test_capture_timestamp_is_none_for_anything_else():
    assert capture_timestamp(Path("0-3.wav")) is None
    assert capture_timestamp(Path("cv-mt-common_voice_mt_21107689.wav")) is None


def test_identical_pcm_is_caught(tmp_path):
    train = [_write(tmp_path / "train/a.wav")]
    evaluation = [_write(tmp_path / "eval/b.wav")]

    kinds = [kind for kind, _, _ in find_leaks(train, evaluation)]

    assert "identical" in kinds


def test_two_augmentations_of_one_source_are_caught(tmp_path):
    """Different bytes, different hash, same utterance underneath."""
    train = [_write(tmp_path / "train/0-3.wav", freq=220.0)]
    evaluation = [_write(tmp_path / "eval/0-7.wav", freq=440.0)]

    kinds = [kind for kind, _, _ in find_leaks(train, evaluation)]

    assert "shared-source" in kinds
    assert "identical" not in kinds


def test_one_utterance_through_two_capture_paths_is_caught(tmp_path):
    """The 2026-09-10 case: 8.2 s apart, different lengths, different stores."""
    train = [_write(tmp_path / "train/wake-00001788516322743-accepted.wav", seconds=0.2)]
    evaluation = [
        _write(tmp_path / "eval/utterance-00001788516330864-answered.wav", seconds=0.42)
    ]

    kinds = [kind for kind, _, _ in find_leaks(train, evaluation)]

    assert kinds == ["same-utterance"]


def test_captures_minutes_apart_are_left_alone(tmp_path):
    """Otherwise every clip from a recording session pairs with every other."""
    train = [_write(tmp_path / "train/wake-00001788516322743-accepted.wav", seconds=0.2)]
    evaluation = [
        _write(tmp_path / "eval/utterance-00001788516400000-answered.wav", seconds=0.42)
    ]

    assert find_leaks(train, evaluation) == []


def test_genuinely_separate_clips_are_clean(tmp_path):
    train = [_write(tmp_path / "train/cv-mt-one.wav", freq=220.0)]
    evaluation = [_write(tmp_path / "eval/cv-fr-two.wav", freq=330.0, seconds=0.3)]

    assert find_leaks(train, evaluation) == []
