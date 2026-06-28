"""User-facing astrometric reduction result and configuration objects."""

from importlib import import_module

_EXPORTS = {
    "CometNuclearMagModel": ("difforb.astrometry.reduction.photometry", "CometNuclearMagModel"),
    "CometTotalMagModel": ("difforb.astrometry.reduction.photometry", "CometTotalMagModel"),
    "HGModel": ("difforb.astrometry.reduction.photometry", "HGModel"),
    "LightPath": ("difforb.astrometry.reduction.lt", "LightPath"),
    "LightTimeContext": ("difforb.astrometry.reduction.lt", "LightTimeContext"),
    "MagModel": ("difforb.astrometry.reduction.photometry", "MagModel"),
    "PhotocenterCorrection": ("difforb.astrometry.reduction.photocenter", "PhotocenterCorrection"),
    "RadarObservation": ("difforb.astrometry.reduction.radar", "RadarObservation"),
    "WeatherParams": ("difforb.astrometry.reduction.refraction", "WeatherParams"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
