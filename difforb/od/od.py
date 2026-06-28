"""High-level orbit-determination orchestration utilities."""

from typing import Literal, NamedTuple

import numpy as np

from difforb.astrometry.debias import DebiasPolicy
from difforb.astrometry.data import ObservationData
from difforb.astrometry.reduction.photocenter import PhotocenterCorrection
from difforb.astrometry.weight import WeightPolicy
from difforb.body.smallbody import Orbit, SmallBody
from difforb.core.time.timescale import Time
from difforb.dynamics.force_model import ForceModel
from difforb.integrator.integrator import NumericalIntegrator
from difforb.od.dc.solver import DCSolver
from difforb.od.events import (SolverEventHandler, SolverEventLogger, SolverLogDetail,
                               print_solver_event)
from difforb.od.iod.solver import IODSolver
from difforb.od.outlier.policy import InteractiveOutlierPolicy
from difforb.od.result import DCStageRecord, ODResult
from difforb.report.text import build_repr

DC_EPOCH_CENTER_ITERATIONS = 2
DCEpochStrategy = Literal["keep_initial", "arc_midpoint", "weighted_mean"]


class IODStrategy(NamedTuple):
    arc_days: float = 1.0
    max_candidates: int = 10
    max_iterations: int = 30
    init_rho: tuple[float, float] = (1.0, 1.0)


class DCStrategy(NamedTuple):
    incremental_arc_days: list[float] = [60.0, 180.0, 1e9]
    min_observations: int = 3
    epoch_strategy: DCEpochStrategy = "keep_initial"


class DCStageSelection(NamedTuple):
    """Selected observations and epoch for one differential-correction stage."""

    epoch: Time
    observations: ObservationData
    active_mask: tuple[np.ndarray, np.ndarray]


class ODSolver:
    """
    Coordinate the high-level orbit-determination workflow.

    The solver delegates numerical work to an initial-orbit-determination
    solver and a differential-correction solver, while this class manages the
    observation-arc staging logic and prints plain-text progress messages. The default workflow
    starts from a short IOD arc, then expands the usable observation span in
    staged differential-correction passes while keeping the IOD epoch as the
    orbit epoch. Alternative strategies may recenter each pass to the selected
    observation-arc midpoint or to the selected observations' weighted mean
    epoch before solving that stage.

    Parameters
    ----------
    iod_solver : IODSolver
        Solver used to generate one or more initial orbit candidates from the
        input observation arc.
    dc_solver : DCSolver
        Solver used to refine the current orbit estimate on progressively larger
        observation subsets.

    See Also
    --------
    IODSolver
        Initial-orbit-determination backend.
    DCSolver
        Differential-correction backend.

    Examples
    --------
    >>> # solver = ODSolver(iod_solver, dc_solver)
    >>> # result = solver.solve(obs, force_model, integrator)
    """

    def __init__(self,
                 iod_solver: IODSolver,
                 dc_solver: DCSolver):
        """Initialize the workflow wrapper around the concrete OD solvers."""
        self.iod_solver = iod_solver
        self.dc_solver = dc_solver

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("iod_solver", self.iod_solver.__class__.__name__),
                ("dc_solver", self.dc_solver.__class__.__name__),
            ],
        )

    def solve(self, obs: ObservationData, force_model: ForceModel,
              integrator: NumericalIntegrator, weight_policy: WeightPolicy, debias_policy: DebiasPolicy,
              outlier_policy: InteractiveOutlierPolicy,
              iod_strategy: IODStrategy = IODStrategy(), dc_strategy: DCStrategy = DCStrategy(),
              photocenter_correction: PhotocenterCorrection | None = None,
              event_handler: SolverEventHandler | None = None,
              log_detail: SolverLogDetail = "iter",
              event_logger: SolverEventLogger | None = None):
        """
        Execute the full orbit-determination pipeline.

        Parameters
        ----------
        obs : ObservationData
            Full observation set available to the orbit-determination run.
        force_model : ForceModel
            Dynamical model passed to the differential-correction stage.
        integrator : NumericalIntegrator
            Numerical propagator used by the differential-correction solver. The
            required integrator type depends on the configured DC solver.
        weight_policy : WeightPolicy
            Policy for observation weights in each differential-correction stage.
        debias_policy : DebiasPolicy
            Policy for optical astrometric debias corrections in each differential-correction stage.
        outlier_policy : InteractiveOutlierPolicy
            Policy for inlier masks in each differential-correction stage.
        iod_strategy : IODStrategy, default=IODStrategy()
            IOD settings. They include the arc width, candidate limit, and initial topocentric range guesses.
        dc_strategy : DCStrategy, default=DCStrategy()
            DC staging settings. The solver uses ``incremental_arc_days`` exactly as the requested centered arc widths
            in days. By default, ``epoch_strategy`` keeps the orbit epoch at the IOD epoch.
        photocenter_correction : PhotocenterCorrection or None, optional
            Optional optical center-of-light correction passed to each
            differential-correction stage.
        event_handler : SolverEventHandler or None, optional
            Optional callback for differential-correction progress events. If
            omitted, events are printed with the surrounding workflow log.
        log_detail : {"quiet", "summary", "iter", "trial"}, default="iter"
            Minimum differential-correction log detail emitted to ``event_handler``.
        event_logger : SolverEventLogger or None, optional
            Structured event logger used for staged differential-correction
            events. If supplied, it takes precedence over ``event_handler`` and
            ``log_detail`` for nested solver logs.
        Returns
        -------
        ODResult
            High-level workflow result containing the IOD output, all staged
            differential-correction summaries, and the final DC result.

        Notes
        -----
        The stage epoch is snapped to a ``TDB`` Julian date whose fractional
        part is ``0.5``. With the default ``epoch_strategy="keep_initial"``,
        the final orbit remains expressed at the snapped IOD epoch. With
        ``epoch_strategy="weighted_mean"``, optical row weights are the sum of
        the right-ascension and declination inverse variances returned by
        ``weight_policy``. Stages with too few valid observations are skipped,
        and stages that would reuse the exact same active observation mask
        without an epoch recentering are not recomputed.
        """
        _validate_dc_epoch_strategy(dc_strategy.epoch_strategy)

        stage_arc_days = [float(span) for span in dc_strategy.incremental_arc_days]
        strategy_str = (
            f"IOD({iod_strategy.arc_days}d) -> "
            f"DC({len(stage_arc_days)} stages, epoch={dc_strategy.epoch_strategy})"
        )
        print("=== Orbit Determination Pipeline ===")
        print(f"Total observations: {len(obs)}")
        print(f"Strategy: {strategy_str}")
        print()

        print("=== Step 1: Initial Orbit Determination (IOD) ===")
        print("Running IOD solver...")
        self.iod_solver.max_iter = int(iod_strategy.max_iterations)
        iod_result = self.iod_solver.solve(
            obs,
            iod_strategy.arc_days,
            iod_strategy.max_candidates,
            iod_strategy.init_rho,
        )
        cur_orbit = iod_result.initial_orbit
        print(f"IOD finished: iterations={iod_result.iter_num}, err={float(iod_result.err):.2E}")
        print(repr(iod_result))
        print()

        print("=== Step 2: Differential Correction (DC) ===")
        solver_logger = event_logger if event_logger is not None else SolverEventLogger(
            event_handler if event_handler is not None else print_solver_event,
            log_detail,
        )
        cur_force_model = force_model
        cur_photocenter_correction = (
            photocenter_correction if photocenter_correction is not None else PhotocenterCorrection()
        )
        cur_cor_result = None
        dc_stage_records: list[DCStageRecord] = []
        prev_active_mask = None
        initial_epoch = _snap_to_midnight_epoch(cur_orbit.tdb.jd)
        for i, span in enumerate(stage_arc_days):
            stage_selection = select_dc_stage_observations(
                obs,
                cur_orbit.tdb.time,
                initial_epoch,
                span,
                dc_strategy,
                weight_policy,
            )
            stage_epoch = stage_selection.epoch
            cur_obs = stage_selection.observations
            active_mask = stage_selection.active_mask
            valid_obs_num = len(cur_obs)
            stage_label = f"Stage {i + 1}"
            recenter_days = _dc_epoch_shift_days(cur_orbit, stage_epoch)
            should_recenter = recenter_days > 1.0e-9
            if valid_obs_num < dc_strategy.min_observations:
                dc_stage_records.append(
                    DCStageRecord(
                        stage_index=i + 1,
                        configured_arc_days=float(span),
                        status="skipped_too_few_observations",
                        selected_observation_count=valid_obs_num,
                        actual_observation_arc_days=_observation_arc_days(cur_obs),
                    )
                )
                print(f"{stage_label} failed: arc={span:.1f} d, obs={valid_obs_num}, reason=Not enough observations")
                print()
                prev_active_mask = active_mask
                continue
            if (
                prev_active_mask is not None
                and not should_recenter
                and all(
                    np.array_equal(cur_mask, prev_mask)
                    for cur_mask, prev_mask in zip(active_mask, prev_active_mask)
                )
            ):
                dc_stage_records.append(
                    DCStageRecord(
                        stage_index=i + 1,
                        configured_arc_days=float(span),
                        status="skipped_unchanged_observation_mask",
                        selected_observation_count=valid_obs_num,
                        actual_observation_arc_days=_observation_arc_days(cur_obs),
                    )
                )
                print(
                    f"{stage_label} skipped: arc={span:.1f} d, obs={valid_obs_num}, "
                    "reason=Unchanged observation mask"
                )
                print()
                continue
            if should_recenter:
                cur_orbit, _ = recenter_orbit_to_dc_epoch(cur_orbit, stage_epoch, cur_force_model, integrator)
                target_tdb = stage_epoch.tdb()
                print(
                    f"{stage_label} recentered: epoch_jd={float(np.asarray(target_tdb.jd).item()):.9f}, "
                    f"shift={recenter_days:.1f} d"
                )
            print(
                f"{stage_label} start: arc={span:.1f} d, obs={valid_obs_num}, "
                f"epoch_jd={float(np.asarray(cur_orbit.tdb.jd).item()):.9f}"
            )
            cur_cor_result = self.dc_solver.solve(
                cur_obs,
                cur_orbit,
                cur_force_model,
                integrator,
                weight_policy,
                debias_policy,
                outlier_policy,
                photocenter_correction=cur_photocenter_correction,
                event_logger=solver_logger.bind(stage=i + 1),
            )
            cur_orbit = cur_cor_result.estimate.orbit
            n_force_params = cur_force_model.get_all_estimated_params().shape[-1]
            cur_force_model = cur_force_model.update_estimated_params(
                cur_cor_result.estimate.model_params[:n_force_params]
            )
            cur_photocenter_correction = cur_photocenter_correction.update_estimated_params(
                cur_cor_result.estimate.model_params[n_force_params:]
            )
            prev_active_mask = active_mask
            dc_stage_records.append(
                DCStageRecord(
                    stage_index=i + 1,
                    configured_arc_days=float(span),
                    status="completed",
                    selected_observation_count=valid_obs_num,
                    actual_observation_arc_days=_observation_arc_days(cur_obs),
                )
            )
            print(
                f"{stage_label} finished: normalized residual RMS={float(cur_cor_result.normalized_residual_rms):.5f}, "
                f"iters={cur_cor_result.lsq_diagnostics.lsq_iterations}/{cur_cor_result.lsq_diagnostics.outlier_iterations}, "
                f"optical obs={cur_cor_result.optical.n_inliers}/{cur_cor_result.optical.n_obs}, "
                f"radar obs={cur_cor_result.radar.n_inliers}/{cur_cor_result.radar.n_obs}"
            )
            print()

        print("=== Orbit Determination Result ===")
        if cur_cor_result is None:
            print("No final solution.")
        else:
            print(repr(cur_cor_result))
        return ODResult(
            iod_result=iod_result,
            dc_result=cur_cor_result,
            dc_stage_records=tuple(dc_stage_records),
        )


def select_dc_stage_observations(
        obs: ObservationData,
        current_epoch: Time,
        initial_epoch: Time,
        span: float,
        dc_strategy: DCStrategy,
        weight_policy: WeightPolicy,
) -> DCStageSelection:
    """Select the epoch-centered observation subset for one DC stage.

    Parameters
    ----------
    obs : ObservationData
        Full observation set available to the staged differential-correction run.
    current_epoch : Time
        Current orbit epoch. For recentered strategies this is the starting
        point for the fixed-point epoch update.
    initial_epoch : Time
        Initial orbit epoch. The ``keep_initial`` strategy snaps this epoch to a
        ``TDB`` Julian date whose fractional part is ``0.5``.
    span : float
        Symmetric stage-window width in days.
    dc_strategy : DCStrategy
        Differential-correction staging settings, including epoch strategy and
        minimum observation count.
    weight_policy : WeightPolicy
        Weight policy used only when ``epoch_strategy`` is ``weighted_mean``.

    Returns
    -------
    DCStageSelection
        Snapped stage epoch, selected observations, and optical/radar active
        masks relative to ``obs``.

    Raises
    ------
    ValueError
        Raised when the epoch strategy is unsupported or weighted-mean epoch
        selection has no finite positive observation weights.

    Notes
    -----
    The weighted-mean epoch uses the per-row sum of RA and Dec inverse
    variances for optical observations and the scalar inverse variance for
    radar observations. No declination cosine factor is applied.
    """
    _validate_dc_epoch_strategy(dc_strategy.epoch_strategy)
    stage_epoch = (
        _snap_to_midnight_epoch(initial_epoch.tdb().jd)
        if dc_strategy.epoch_strategy == "keep_initial"
        else current_epoch
    )
    if dc_strategy.epoch_strategy != "keep_initial":
        for _ in range(DC_EPOCH_CENTER_ITERATIONS):
            candidate_obs, _ = _select_centered_observations(obs, stage_epoch, span)
            if len(candidate_obs) < dc_strategy.min_observations:
                break
            next_stage_epoch = _observation_stage_epoch(candidate_obs, dc_strategy.epoch_strategy, weight_policy)
            epoch_shift_days = abs(
                float(np.asarray(next_stage_epoch.tdb().jd).item()) -
                float(np.asarray(stage_epoch.tdb().jd).item())
            )
            stage_epoch = next_stage_epoch
            if epoch_shift_days <= 1.0e-9:
                break
    cur_obs, active_mask = _select_centered_observations(obs, stage_epoch, span)
    return DCStageSelection(epoch=stage_epoch, observations=cur_obs, active_mask=active_mask)


def recenter_orbit_to_dc_epoch(
        cur_orbit: Orbit,
        stage_epoch: Time,
        force_model: ForceModel,
        integrator: NumericalIntegrator,
) -> tuple[Orbit, float]:
    """Propagate an orbit to a differential-correction stage epoch when needed.

    Parameters
    ----------
    cur_orbit : Orbit
        Current orbit estimate.
    stage_epoch : Time
        Target stage epoch. The epoch is interpreted in ``TDB``.
    force_model : ForceModel
        Dynamical model used for the epoch propagation.
    integrator : NumericalIntegrator
        Numerical integrator used for the epoch propagation.

    Returns
    -------
    orbit : Orbit
        ``cur_orbit`` if the epoch shift is negligible, otherwise the propagated
        orbit state at ``stage_epoch``.
    shift_days : float
        Absolute shift between the current orbit epoch and the target stage
        epoch, in days.
    """
    target_tdb = stage_epoch.tdb()
    recenter_days = _dc_epoch_shift_days(cur_orbit, stage_epoch)
    if recenter_days <= 1.0e-9:
        return cur_orbit, recenter_days
    return (
        SmallBody.create(cur_orbit).propagate(
            cur_orbit.tdb,
            target_tdb,
            force_model,
            integrator,
            grid=False,
        ).state(target_tdb),
        recenter_days,
    )


def _dc_epoch_shift_days(cur_orbit: Orbit, stage_epoch: Time) -> float:
    target_tdb = stage_epoch.tdb()
    return abs(
        float(np.asarray(target_tdb.jd).item()) -
        float(np.asarray(cur_orbit.tdb.jd).item())
    )


def _validate_dc_epoch_strategy(epoch_strategy: str) -> None:
    if epoch_strategy not in ("keep_initial", "arc_midpoint", "weighted_mean"):
        raise ValueError(
            "dc_strategy.epoch_strategy must be one of 'keep_initial', 'arc_midpoint', or 'weighted_mean'."
        )


def _observation_stage_epoch(
        obs: ObservationData,
        epoch_strategy: DCEpochStrategy,
        weight_policy: WeightPolicy,
) -> Time:
    if epoch_strategy == "arc_midpoint":
        t_start_tdb = obs.t_start.tdb()
        t_end_tdb = obs.t_end.tdb()
        midpoint_jd = 0.5 * (
            float(np.asarray(t_start_tdb.jd).item()) +
            float(np.asarray(t_end_tdb.jd).item())
        )
        return _snap_to_midnight_epoch(midpoint_jd)

    weight_result = weight_policy.weights(obs)
    optical_epoch_jd = np.asarray(obs.optical.t.tdb().jd, dtype=float)
    radar_epoch_jd = np.asarray(obs.radar.t.tdb().jd, dtype=float)
    optical_weight_matrices = np.asarray(weight_result.optical_weight_matrices, dtype=float)
    optical_epoch_weights = np.trace(optical_weight_matrices, axis1=1, axis2=2)
    radar_epoch_weights = np.asarray(weight_result.radar_weights, dtype=float)
    epoch_jd = np.concatenate([optical_epoch_jd, radar_epoch_jd])
    epoch_weights = np.concatenate([optical_epoch_weights, radar_epoch_weights])
    valid_weight_mask = np.isfinite(epoch_jd) & np.isfinite(epoch_weights) & (epoch_weights > 0.0)
    if not np.any(valid_weight_mask):
        raise ValueError(
            "Weighted DC epoch selection requires at least one finite positive observation weight."
        )
    weighted_epoch_jd = np.average(
        epoch_jd[valid_weight_mask],
        weights=epoch_weights[valid_weight_mask],
    )
    return _snap_to_midnight_epoch(weighted_epoch_jd)


def _observation_arc_days(obs: ObservationData) -> float | None:
    """Return the observation span of one selected stage, in days."""
    if len(obs) == 0:
        return None
    try:
        return max(0.0, float(np.asarray(obs.t_end.tt.jd).item() - np.asarray(obs.t_start.tt.jd).item()))
    except Exception:
        return None


def _snap_to_midnight_epoch(tdb_jd) -> Time:
    """Return the nearest ``TDB`` midnight epoch with Julian-date fraction ``0.5``."""
    midnight_jd = np.floor(float(np.asarray(tdb_jd).item())) + 0.5
    midnight_jd1 = np.floor(midnight_jd)
    midnight_jd2 = midnight_jd - midnight_jd1
    return Time.from_tdb_jd(midnight_jd1, midnight_jd2)


def _select_centered_observations(obs: ObservationData, epoch: Time, span: float):
    """Return observations selected by a symmetric stage window around ``epoch``."""
    t_start = epoch - span / 2.0
    t_end = epoch + span / 2.0
    optical_active_mask = np.asarray((obs.optical.t > t_start) & (obs.optical.t < t_end))
    radar_active_mask = np.asarray((obs.radar.t > t_start) & (obs.radar.t < t_end))
    active_mask = (optical_active_mask, radar_active_mask)
    return (
        ObservationData(
            name=obs.name,
            optical=obs.optical[optical_active_mask],
            radar=obs.radar[radar_active_mask],
        ),
        active_mask,
    )
