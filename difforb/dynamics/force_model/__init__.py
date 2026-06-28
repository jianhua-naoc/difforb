"""Force-model interfaces and built-in force terms.

This package preserves the historical :mod:`difforb.dynamics.force_model` import path while splitting the implementation by responsibility. Unless noted otherwise, positions are in ``au``, velocities are in ``au / day``, accelerations are in ``au / day^2``, and epochs are in ``TDB``.
"""

from importlib import import_module
from typing import TYPE_CHECKING

import jax

jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from difforb.dynamics.force_model.base import Force, ParametrizedForce
    from difforb.dynamics.force_model.gravity import (
        NewtonianGravity,
        PPNGravity,
        compute_newtonian_acceleration,
        compute_planetary_potentials,
        compute_ppn_acceleration,
    )
    from difforb.dynamics.force_model.j2 import EarthJ2Perturbation, SolarJ2Perturbation
    from difforb.dynamics.force_model.model import ForceModel
    from difforb.dynamics.force_model.nongrav_rtn import (
        CometOutgassingEffect,
        EmpiricalRadiationPressure,
        EmpiricalYarkovskyEffect,
        RTNDistanceLawNonGravEffect,
        compute_rtn_distance_law_non_grav_acceleration,
    )

_EXPORTS = {
    "CometOutgassingEffect": ("difforb.dynamics.force_model.nongrav_rtn", "CometOutgassingEffect"),
    "EmpiricalRadiationPressure": ("difforb.dynamics.force_model.nongrav_rtn", "EmpiricalRadiationPressure"),
    "EmpiricalYarkovskyEffect": ("difforb.dynamics.force_model.nongrav_rtn", "EmpiricalYarkovskyEffect"),
    "EarthJ2Perturbation": ("difforb.dynamics.force_model.j2", "EarthJ2Perturbation"),
    "Force": ("difforb.dynamics.force_model.base", "Force"),
    "ForceModel": ("difforb.dynamics.force_model.model", "ForceModel"),
    "NewtonianGravity": ("difforb.dynamics.force_model.gravity", "NewtonianGravity"),
    "PPNGravity": ("difforb.dynamics.force_model.gravity", "PPNGravity"),
    "ParametrizedForce": ("difforb.dynamics.force_model.base", "ParametrizedForce"),
    "RTNDistanceLawNonGravEffect": ("difforb.dynamics.force_model.nongrav_rtn", "RTNDistanceLawNonGravEffect"),
    "SolarJ2Perturbation": ("difforb.dynamics.force_model.j2", "SolarJ2Perturbation"),
    "compute_newtonian_acceleration": ("difforb.dynamics.force_model.gravity", "compute_newtonian_acceleration"),
    "compute_planetary_potentials": ("difforb.dynamics.force_model.gravity", "compute_planetary_potentials"),
    "compute_ppn_acceleration": ("difforb.dynamics.force_model.gravity", "compute_ppn_acceleration"),
    "compute_rtn_distance_law_non_grav_acceleration": ("difforb.dynamics.force_model.nongrav_rtn", "compute_rtn_distance_law_non_grav_acceleration"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
