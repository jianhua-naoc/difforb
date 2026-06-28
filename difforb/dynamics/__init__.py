"""User-facing dynamic-system and force-model objects."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from difforb.dynamics.dynamic_system import DynamicSystem
    from difforb.dynamics.force_model import (
        CometOutgassingEffect,
        EmpiricalRadiationPressure,
        EmpiricalYarkovskyEffect,
        EarthJ2Perturbation,
        Force,
        ForceModel,
        NewtonianGravity,
        PPNGravity,
        ParametrizedForce,
        RTNDistanceLawNonGravEffect,
        SolarJ2Perturbation,
    )

_EXPORTS = {
    "CometOutgassingEffect": ("difforb.dynamics.force_model", "CometOutgassingEffect"),
    "DynamicSystem": ("difforb.dynamics.dynamic_system", "DynamicSystem"),
    "EmpiricalRadiationPressure": ("difforb.dynamics.force_model", "EmpiricalRadiationPressure"),
    "EmpiricalYarkovskyEffect": ("difforb.dynamics.force_model", "EmpiricalYarkovskyEffect"),
    "EarthJ2Perturbation": ("difforb.dynamics.force_model", "EarthJ2Perturbation"),
    "Force": ("difforb.dynamics.force_model", "Force"),
    "ForceModel": ("difforb.dynamics.force_model", "ForceModel"),
    "NewtonianGravity": ("difforb.dynamics.force_model", "NewtonianGravity"),
    "PPNGravity": ("difforb.dynamics.force_model", "PPNGravity"),
    "ParametrizedForce": ("difforb.dynamics.force_model", "ParametrizedForce"),
    "RTNDistanceLawNonGravEffect": ("difforb.dynamics.force_model", "RTNDistanceLawNonGravEffect"),
    "SolarJ2Perturbation": ("difforb.dynamics.force_model", "SolarJ2Perturbation"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
