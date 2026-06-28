"""Earth Orientation Parameter (EOP) access for the DiffOrb core package."""

from pathlib import Path

from difforb.core.config import get_data_path
from difforb.core.eop.container import EarthOrientationData
from . import loaders as _eop_loaders

load_iers_eopc04 = _eop_loaders.load_iers_eopc04

_DEFAULT_EOP_FILE: EarthOrientationData | None = None


def load_default_eop_file() -> EarthOrientationData:
    """Return the cached default EOP table without performing network access."""
    global _DEFAULT_EOP_FILE
    if _DEFAULT_EOP_FILE is None:
        filepath = Path(get_data_path("iers/eopc04.dPsi_dEps.1962-now.txt", dataset="eop"))
        _DEFAULT_EOP_FILE = _eop_loaders.parse_iers_eopc04(str(filepath))
    return _DEFAULT_EOP_FILE


def update_eop() -> EarthOrientationData:
    """Download the latest default EOP table and replace the cached table."""
    global _DEFAULT_EOP_FILE
    _DEFAULT_EOP_FILE = load_iers_eopc04(force_update=True)
    return _DEFAULT_EOP_FILE


__all__ = [
    "EarthOrientationData",
    "load_default_eop_file",
    "load_iers_eopc04",
    "update_eop",
]
