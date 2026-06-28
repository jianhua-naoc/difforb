"""User-facing time-scale and time-delta objects for DiffOrb.

This package exposes object-level time views for ``TT``, ``TDB``, ``TAI``, ``UT1``, and ``UTC``. Low-level calendar and time-scale kernels remain in their implementation modules.
"""

from importlib import import_module

_EXPORTS = {
    "TAIView": ("difforb.core.time.timescale", "TAIView"),
    "TDBView": ("difforb.core.time.timescale", "TDBView"),
    "TTView": ("difforb.core.time.timescale", "TTView"),
    "Time": ("difforb.core.time.timescale", "Time"),
    "TimeDelta": ("difforb.core.time.timedelta", "TimeDelta"),
    "TimeView": ("difforb.core.time.timescale", "TimeView"),
    "UTCView": ("difforb.core.time.timescale", "UTCView"),
    "UT1View": ("difforb.core.time.timescale", "UT1View"),
    "UTView": ("difforb.core.time.timescale", "UTView"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
