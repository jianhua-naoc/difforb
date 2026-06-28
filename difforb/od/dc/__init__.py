"""User-facing differential-correction solver and result types."""

from importlib import import_module

_EXPORTS = {
    "DCBucketPolicy": ("difforb.od.dc.bucket", "DCBucketPolicy"),
    "DCEstimate": ("difforb.od.dc.result", "DCEstimate"),
    "DCResult": ("difforb.od.dc.result", "DCResult"),
    "DCSolver": ("difforb.od.dc.solver", "DCSolver"),
    "LSQDiagnostics": ("difforb.od.dc.result", "LSQDiagnostics"),
    "OpticalResult": ("difforb.od.dc.result", "OpticalResult"),
    "PhotocenterCorrection": ("difforb.astrometry.reduction.photocenter", "PhotocenterCorrection"),
    "RadarResult": ("difforb.od.dc.result", "RadarResult"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
