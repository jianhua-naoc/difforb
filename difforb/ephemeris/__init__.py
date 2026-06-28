"""Public ephemeris table and product generation helpers."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from difforb.ephemeris.core import (
        ApsidesTable,
        CloseApproachTable,
        OpticalTable,
        RadarTable,
        VectorTable,
    )
    from difforb.ephemeris.generator import EphemerisGenerator

_EXPORTS = {
    "ApsidesTable": ("difforb.ephemeris.core", "ApsidesTable"),
    "CloseApproachTable": ("difforb.ephemeris.core", "CloseApproachTable"),
    "EphemerisGenerator": ("difforb.ephemeris.generator", "EphemerisGenerator"),
    "OpticalTable": ("difforb.ephemeris.core", "OpticalTable"),
    "RadarTable": ("difforb.ephemeris.core", "RadarTable"),
    "VectorTable": ("difforb.ephemeris.core", "VectorTable"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
