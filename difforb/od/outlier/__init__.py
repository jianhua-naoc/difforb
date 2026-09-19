"""User-facing outlier rejection policies and result types."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Chi2OutlierRejecter": ("difforb.od.outlier.chi2", "Chi2OutlierRejecter"),
    "CompiledOutlierPolicy": ("difforb.od.outlier.policy", "CompiledOutlierPolicy"),
    "InteractiveOutlierPolicy": ("difforb.od.outlier.policy", "InteractiveOutlierPolicy"),
    "OutlierRejecter": ("difforb.od.outlier.outlier", "OutlierRejecter"),
    "RejResult": ("difforb.od.outlier.outlier", "RejResult"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
