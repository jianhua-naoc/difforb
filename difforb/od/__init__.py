"""User-facing orbit-determination solvers, result objects, and policies."""

from importlib import import_module
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from difforb.od.analysis import ODAnalysis
    from difforb.od.dc.result import (
        DCEstimate,
        DCResult,
        LSQDiagnostics,
        OpticalResult,
        RadarResult,
    )
    from difforb.od.dc.solver import DCSolver
    from difforb.astrometry.reduction.photocenter import PhotocenterCorrection
    from difforb.od.iod.result import IODResult
    from difforb.od.iod.solver import IODSolver
    from difforb.od.od import (
        DCStageSelection,
        DCStrategy,
        DCEpochStrategy,
        IODStrategy,
        ODSolver,
        recenter_orbit_to_dc_epoch,
        select_dc_stage_observations,
    )
    from difforb.od.outlier.chi2 import Chi2OutlierRejecter
    from difforb.od.outlier.outlier import OutlierRejecter, RejResult
    from difforb.od.outlier.policy import (
        CompiledOutlierPolicy,
        InteractiveOutlierPolicy,
    )
    from difforb.od.progress import SolverReporter
    from difforb.od.result import DCStageRecord, ODResult

_EXPORTS = {
    "Chi2OutlierRejecter": ("difforb.od.outlier", "Chi2OutlierRejecter"),
    "CompiledOutlierPolicy": ("difforb.od.outlier", "CompiledOutlierPolicy"),
    "DCEstimate": ("difforb.od.dc", "DCEstimate"),
    "DCResult": ("difforb.od.dc", "DCResult"),
    "DCSolver": ("difforb.od.dc", "DCSolver"),
    "DCStageRecord": ("difforb.od.result", "DCStageRecord"),
    "DCStageSelection": ("difforb.od.od", "DCStageSelection"),
    "DCStrategy": ("difforb.od.od", "DCStrategy"),
    "DCEpochStrategy": ("difforb.od.od", "DCEpochStrategy"),
    "IODResult": ("difforb.od.iod", "IODResult"),
    "IODSolver": ("difforb.od.iod", "IODSolver"),
    "IODStrategy": ("difforb.od.od", "IODStrategy"),
    "InteractiveOutlierPolicy": ("difforb.od.outlier", "InteractiveOutlierPolicy"),
    "LSQDiagnostics": ("difforb.od.dc", "LSQDiagnostics"),
    "ODResult": ("difforb.od.result", "ODResult"),
    "ODAnalysis": ("difforb.od.analysis", "ODAnalysis"),
    "ODSolver": ("difforb.od.od", "ODSolver"),
    "OpticalResult": ("difforb.od.dc", "OpticalResult"),
    "OutlierRejecter": ("difforb.od.outlier", "OutlierRejecter"),
    "PhotocenterCorrection": ("difforb.astrometry.reduction.photocenter", "PhotocenterCorrection"),
    "RadarResult": ("difforb.od.dc", "RadarResult"),
    "RejResult": ("difforb.od.outlier", "RejResult"),
    "SolverReporter": ("difforb.od.progress", "SolverReporter"),
    "print_solver_progress": ("difforb.od.progress", "print_solver_progress"),
    "recenter_orbit_to_dc_epoch": ("difforb.od.od", "recenter_orbit_to_dc_epoch"),
    "select_dc_stage_observations": ("difforb.od.od", "select_dc_stage_observations"),
    "solver_progress_reporter": ("difforb.od.progress", "solver_progress_reporter"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
