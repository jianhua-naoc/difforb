"""DiffOrb package namespace.

Most user-facing APIs live in domain subpackages such as ``difforb.core``,
``difforb.body``, ``difforb.astrometry``, and ``difforb.od``.
"""

from importlib import import_module

_SUBPACKAGES = {
    "astrometry": "difforb.astrometry",
    "body": "difforb.body",
    "core": "difforb.core",
    "dynamics": "difforb.dynamics",
    "ephemeris": "difforb.ephemeris",
    "integrator": "difforb.integrator",
    "od": "difforb.od",
    "spk": "difforb.spk",
}

__all__ = sorted(_SUBPACKAGES)


def __getattr__(name):
    if name in _SUBPACKAGES:
        value = import_module(_SUBPACKAGES[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
