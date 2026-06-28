"""User-facing core time, state, element, frame, and site-coordinate objects."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from difforb.core.element import KepElement
    from difforb.core.geo import ITRF, ITRS, WGS84
    from difforb.core.state import (
        Axes,
        BCRS,
        Frame,
        GCRS,
        HELIO_ECLIP_J2000,
        HELIO_ICRS,
        HELIO_J2000,
        Origin,
        RelativeState,
        State,
    )
    from difforb.core.time.timedelta import TimeDelta
    from difforb.core.time.timescale import (
        TAIView,
        TDBView,
        TTView,
        Time,
        TimeView,
        UT1View,
        UTCView,
        UTView,
    )

_EXPORTS = {
    "Axes": ("difforb.core.state", "Axes"),
    "BCRS": ("difforb.core.state", "BCRS"),
    "Frame": ("difforb.core.state", "Frame"),
    "GCRS": ("difforb.core.state", "GCRS"),
    "HELIO_ECLIP_J2000": ("difforb.core.state", "HELIO_ECLIP_J2000"),
    "HELIO_ICRS": ("difforb.core.state", "HELIO_ICRS"),
    "HELIO_J2000": ("difforb.core.state", "HELIO_J2000"),
    "ITRF": ("difforb.core.geo", "ITRF"),
    "ITRS": ("difforb.core.geo", "ITRS"),
    "KepElement": ("difforb.core.element", "KepElement"),
    "Origin": ("difforb.core.state", "Origin"),
    "RelativeState": ("difforb.core.state", "RelativeState"),
    "State": ("difforb.core.state", "State"),
    "TAIView": ("difforb.core.time", "TAIView"),
    "TDBView": ("difforb.core.time", "TDBView"),
    "TTView": ("difforb.core.time", "TTView"),
    "Time": ("difforb.core.time", "Time"),
    "TimeDelta": ("difforb.core.time", "TimeDelta"),
    "TimeView": ("difforb.core.time", "TimeView"),
    "UTCView": ("difforb.core.time", "UTCView"),
    "UT1View": ("difforb.core.time", "UT1View"),
    "UTView": ("difforb.core.time", "UTView"),
    "WGS84": ("difforb.core.geo", "WGS84"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
