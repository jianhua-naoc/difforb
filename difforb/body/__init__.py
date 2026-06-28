"""User-facing body, orbit, and observing-site objects."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from difforb.body.ephbody import EphemerisBody
    from difforb.body.site import Site
    from difforb.body.smallbody import Orbit, SmallBody

_EXPORTS = {
    "EphemerisBody": ("difforb.body.ephbody", "EphemerisBody"),
    "Orbit": ("difforb.body.smallbody", "Orbit"),
    "Site": ("difforb.body.site", "Site"),
    "SmallBody": ("difforb.body.smallbody", "SmallBody"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
