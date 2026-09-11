"""Tests for the audio conversion in the negatives fetcher.

Common Voice ships 32 kHz mono MP3; AudioSet ships 48 kHz stereo FLAC. Both
have to land as 16 kHz mono or the trainer silently consumes the wrong thing.
"""

from __future__ import annotations

import io
import math

import numpy as np
import soundfile as sf

from fetch_negatives import TARGET_RATE, to_wav_16k


def _encode(seconds: float, rate: int, channels: int, fmt: str = "WAV") -> bytes:
    t = np.linspace(0.0, seconds, int(seconds * rate), endpoint=False)
    wave = (0.3 * np.sin(2 * math.pi * 220.0 * t)).astype(np.float32)
    if channels == 2:
        wave = np.stack([wave, wave], axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, wave, rate, format=fmt, subtype="PCM_16")
    return buffer.getvalue()


def test_downmixes_and_resamples_the_audioset_shape():
    """48 kHz stereo in, 16 kHz mono out."""
    out = to_wav_16k(_encode(1.0, 48000, channels=2))

    assert out is not None
    assert out.ndim == 1
    assert len(out) == TARGET_RATE
    assert out.dtype == np.float32


def test_resamples_the_common_voice_shape():
    out = to_wav_16k(_encode(2.0, 32000, channels=1))

    assert out is not None
    assert len(out) == TARGET_RATE * 2


def test_passes_through_audio_already_at_16k():
    out = to_wav_16k(_encode(0.5, TARGET_RATE, channels=1))

    assert len(out) == TARGET_RATE // 2


def test_unreadable_bytes_return_none_rather_than_raising():
    """One corrupt member must not abandon the rest of a 3,000-clip shard."""
    assert to_wav_16k(b"this is not audio") is None


def test_empty_audio_returns_none():
    assert to_wav_16k(_encode(0.0, TARGET_RATE, channels=1)) is None
