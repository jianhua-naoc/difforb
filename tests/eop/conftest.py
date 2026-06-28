"""Pytest fixtures for Earth-orientation tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def eopc04_sample_path() -> Path:
    """Return the committed compact IERS EOP C04 sample path."""

    return Path(__file__).resolve().parents[1] / "data" / "eop" / "eopc04_sample.txt"
