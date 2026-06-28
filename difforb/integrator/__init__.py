"""User-facing numerical integration objects."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from difforb.integrator.integrator import BiDirectionalInterpolator, NumericalIntegrator

_EXPORTS = {
    "BiDirectionalInterpolator": ("difforb.integrator.integrator", "BiDirectionalInterpolator"),
    "NumericalIntegrator": ("difforb.integrator.integrator", "NumericalIntegrator"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
