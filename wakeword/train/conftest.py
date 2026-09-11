from pathlib import Path

import pytest

IMPULSES = (
    Path(__file__).resolve().parent.parent
    / "generate/upstream/piper_sample_generator/impulses"
)


@pytest.fixture
def impulse_dir() -> Path:
    if not IMPULSES.is_dir():
        pytest.skip(f"upstream clone absent: {IMPULSES}")
    return IMPULSES
