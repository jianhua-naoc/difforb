"""User-facing astrometry data, loading, debiasing, and weighting objects."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from difforb.astrometry.data import (
        ObsMode,
        ObsType,
        ObservationData,
        ObservationLayout,
        ObserverType,
        OpticalObservationData,
        RadarObservationData,
    )
    from difforb.astrometry.debias import (
        AstrometricDebiasMap,
        DebiasPolicy,
        DebiasResult,
        EgglDebiasPolicy,
        NoDebiasPolicy,
    )
    from difforb.astrometry.io import load_local_observations, load_online_observations
    from difforb.astrometry.reduction.photocenter import PhotocenterCorrection
    from difforb.astrometry.weight import (
        ADESWeightPolicy,
        InteractiveWeightPolicy,
        UnitWeightPolicy,
        VFCC17WeightPolicy,
        WeightPolicy,
        WeightResult,
        WeightRule,
    )

_EXPORTS = {
    "ADESWeightPolicy": ("difforb.astrometry.weight", "ADESWeightPolicy"),
    "AstrometricDebiasMap": ("difforb.astrometry.debias", "AstrometricDebiasMap"),
    "DebiasPolicy": ("difforb.astrometry.debias", "DebiasPolicy"),
    "DebiasResult": ("difforb.astrometry.debias", "DebiasResult"),
    "EgglDebiasPolicy": ("difforb.astrometry.debias", "EgglDebiasPolicy"),
    "InteractiveWeightPolicy": ("difforb.astrometry.weight", "InteractiveWeightPolicy"),
    "NoDebiasPolicy": ("difforb.astrometry.debias", "NoDebiasPolicy"),
    "ObsMode": ("difforb.astrometry.data", "ObsMode"),
    "ObsType": ("difforb.astrometry.data", "ObsType"),
    "ObservationData": ("difforb.astrometry.data", "ObservationData"),
    "ObservationLayout": ("difforb.astrometry.data", "ObservationLayout"),
    "ObserverType": ("difforb.astrometry.data", "ObserverType"),
    "OpticalObservationData": ("difforb.astrometry.data", "OpticalObservationData"),
    "PhotocenterCorrection": ("difforb.astrometry.reduction.photocenter", "PhotocenterCorrection"),
    "RadarObservationData": ("difforb.astrometry.data", "RadarObservationData"),
    "UnitWeightPolicy": ("difforb.astrometry.weight", "UnitWeightPolicy"),
    "VFCC17WeightPolicy": ("difforb.astrometry.weight", "VFCC17WeightPolicy"),
    "WeightPolicy": ("difforb.astrometry.weight", "WeightPolicy"),
    "WeightResult": ("difforb.astrometry.weight", "WeightResult"),
    "WeightRule": ("difforb.astrometry.weight", "WeightRule"),
    "load_local_observations": ("difforb.astrometry.io", "load_local_observations"),
    "load_online_observations": ("difforb.astrometry.io", "load_online_observations"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
