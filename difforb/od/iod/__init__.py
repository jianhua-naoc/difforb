"""User-facing initial-orbit-determination solver and result types."""

from importlib import import_module

_EXPORTS = {
    "IODResult": ("difforb.od.iod.result", "IODResult"),
    "IODSolver": ("difforb.od.iod.solver", "IODSolver"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
