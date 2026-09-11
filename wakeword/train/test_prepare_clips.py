"""Tests for the augmentation step that breaks the loudness shortcut."""

from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile as sf

from prepare_clips import (
    MAX_GAIN_DB,
    MIN_GAIN_DB,
    TARGET_RATE,
    build_chain,
    load_16k,
)


def _tone(seconds: float, rate: int, channels: int = 1) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * rate), endpoint=False)
    wave = (0.25 * np.sin(2 * math.pi * 220.0 * t)).astype(np.float32)
    return np.stack([wave, wave], axis=1) if channels == 2 else wave


def _dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio))))
    return 20 * math.log10(max(rms, 1e-12))


def test_resamples_22050_to_16000(tmp_path):
    path = tmp_path / "in.wav"
    sf.write(path, _tone(1.0, 22050), 22050)

    audio = load_16k(path)

    assert audio.dtype == np.float32
    assert len(audio) == TARGET_RATE


def test_leaves_16k_audio_at_its_own_length(tmp_path):
    path = tmp_path / "in.wav"
    sf.write(path, _tone(0.5, TARGET_RATE), TARGET_RATE)

    assert len(load_16k(path)) == TARGET_RATE // 2


def test_downmixes_stereo_to_mono(tmp_path):
    path = tmp_path / "stereo.wav"
    sf.write(path, _tone(0.25, TARGET_RATE, channels=2), TARGET_RATE)

    audio = load_16k(path)

    assert audio.ndim == 1
    assert len(audio) == TARGET_RATE // 4


def test_missing_impulses_exit_with_the_path(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        build_chain(tmp_path / "nowhere")

    assert "Accoustic2_Impulse.wav" in str(excinfo.value)


def test_augmentation_spans_the_dynamic_range_of_a_real_room(impulse_dir):
    """The single assertion this whole script exists to satisfy.

    The threshold is the specification, not a loose bound: speech reaching a
    phone across a room varies by 20-40 dB, while raw generator output varies
    by 4.6 dB. A corpus that does not span that range is a corpus that teaches
    loudness as a proxy for the wake phrase, which is the bug being fixed.
    """
    np.random.seed(0)
    augment = build_chain(impulse_dir)
    source = _tone(0.6, TARGET_RATE)

    levels = [_dbfs(augment(samples=source, sample_rate=TARGET_RATE)) for _ in range(60)]

    assert max(levels) - min(levels) > 20.0


def test_the_corpus_leans_quiet(impulse_dir):
    """Gain has p=1.0 and its range is lopsided towards attenuation, so most
    clips come out quieter than they went in.

    Guards the specific regression: if Gain were made probabilistic, or a
    Normalize crept into the chain, the corpus would look augmented while every
    clip sat back at full level.

    Note there is deliberately NO upper bound asserted here. Reverb adds energy,
    so the chain's ceiling is not gain-limited — an earlier version of this test
    asserted `max <= reference + MAX_GAIN_DB` and was simply wrong about the
    physics. It passed on one seed and failed on the next.
    """
    np.random.seed(0)
    augment = build_chain(impulse_dir)
    source = _tone(0.6, TARGET_RATE)
    reference = _dbfs(source)

    levels = sorted(_dbfs(augment(samples=source, sample_rate=TARGET_RATE)) for _ in range(60))

    assert min(levels) < reference + MIN_GAIN_DB / 2
    assert levels[len(levels) // 2] < reference


def test_no_augment_preserves_the_signal_exactly(tmp_path):
    """The eval path must hand back what it was given, resampled and no more.

    Held-out material is the only honest measurement in the project. If this
    ever starts augmenting, the ship gate silently begins scoring models
    against audio no microphone produced.
    """
    source = _tone(0.4, TARGET_RATE)
    path = tmp_path / "in.wav"
    sf.write(path, source, TARGET_RATE)

    loaded = load_16k(path)

    assert len(loaded) == len(source)
    assert _dbfs(loaded) == pytest.approx(_dbfs(source), abs=0.01)
