from pathlib import Path

import pytest

import difforb.spk as spk
from difforb.spk.spk import Ephemeris


DE441_SPK_PATH = Path(__file__).resolve().parents[2] / "data" / "spk" / "de441_2017_2025_excerpt.bsp"
SB441_SPK_PATH = Path(__file__).resolve().parents[2] / "data" / "spk" / "sb441_2017_2025_excerpt.bsp"


@pytest.fixture
def default_ephemeris():
    if not DE441_SPK_PATH.exists() or not SB441_SPK_PATH.exists():
        pytest.skip("local DE441/SB441 SPK excerpts are not installed")
    ephemeris = Ephemeris([str(DE441_SPK_PATH), str(SB441_SPK_PATH)])
    spk.set_default_ephemeris(ephemeris)
    yield ephemeris
    spk.clear_default_ephemeris()
