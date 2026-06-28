from typing import Literal, NamedTuple

from jax import Array
from jaxtyping import Int

from difforb.astrometry.data import ObservationData
from difforb.core.time.timescale import Time
from difforb.od.dc.result import DCResult
from difforb.od.iod.result import IODResult
from difforb.report.text import build_repr, format_float_array


class DCStageRecord(NamedTuple):
    """Stage-level differential-correction orchestration summary."""

    stage_index: int
    configured_arc_days: float
    status: Literal["completed", "skipped_too_few_observations", "skipped_unchanged_observation_mask"]
    selected_observation_count: int
    actual_observation_arc_days: float | None = None


class ODResult(NamedTuple):
    """High-level orbit-determination workflow result.

    Parameters
    ----------
    iod_result : IODResult or None
        Initial orbit-determination result, if the workflow ran an IOD stage.
    dc_result : DCResult or None
        Final differential-correction result. Use ``dc_result.estimate`` to access the solved orbit and posterior covariance; ``None`` means no differential-correction covariance is available.
    dc_stage_records : tuple[DCStageRecord, ...], optional
        Per-stage orchestration records for the differential-correction workflow.
    """

    iod_result: IODResult | None
    dc_result: DCResult | None
    dc_stage_records: tuple[DCStageRecord, ...] = ()
