"""High-level initial orbit determination from sampled optical triplets.

This module exposes the public IOD solver used by ``difforb.od``. The solver
collects the optical subtables from one single-target observation bundle,
builds a globally time-sorted angle-only arc, samples candidate triplets inside
a configurable time window, solves each candidate with the Double-r iteration,
scores candidates against the full sampling window, and returns the best
heliocentric solution transformed to ``BCRS``.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array

from difforb.astrometry.data import ObservationData
from difforb.body.ephbody import EphemerisBody
from difforb.core.constants import C
from difforb.core.state.frame import BCRS, HELIO_ICRS
from difforb.core.state.state import State
from difforb.core.time.timescale import Time
from difforb.dynamics.two_body import kepler_propagate
from difforb.od.iod.double_r import DoubleRIODResult, double_r_iod
from difforb.od.iod.result import IODResult
from difforb.od.iod.sampling import (
    IODSamplingWindow,
    OpticalIODInputs,
    build_optical_iod_inputs,
    build_triplet_batch,
    resolve_sampling_window,
    sample_triplet_indices,
)
from difforb.report.text import build_repr, format_float_array

jax.config.update("jax_enable_x64", True)

DEFAULT_IOD_SEED = 116
IOD_RANGE_FLOOR_AU = 1.0e-4


def _select_best_candidate(candidate_result: DoubleRIODResult) -> tuple[Array, Array]:
    """Select the lowest-residual valid candidate from a batched Double-r run."""
    valid_pos = jnp.all(jnp.isfinite(candidate_result.pos_t2), axis=-1)
    valid_vel = jnp.all(jnp.isfinite(candidate_result.vel_t2), axis=-1)
    valid_residual = jnp.isfinite(candidate_result.residual_norm)
    valid_mask = valid_pos & valid_vel & valid_residual

    safe_residual_norm = jnp.where(valid_mask, candidate_result.residual_norm, jnp.inf)
    best_candidate_idx = jnp.argmin(safe_residual_norm)
    best_residual_norm = safe_residual_norm[best_candidate_idx]
    return best_candidate_idx, best_residual_norm


def _score_candidates_on_window(
        candidate_result: DoubleRIODResult,
        optical_inputs: OpticalIODInputs,
        sampling_window: IODSamplingWindow,
        mu: Array,
) -> Array:
    """Score candidate states against all optical observations in the IOD window.

    Parameters
    ----------
    candidate_result : DoubleRIODResult
        Batched Double-r candidate states in ``HELIO_ICRS`` at their middle
        triplet epochs.
    optical_inputs : OpticalIODInputs
        Time-sorted observer positions and line-of-sight vectors in
        ``HELIO_ICRS``.
    sampling_window : IODSamplingWindow
        Inclusive observation-index bounds used to build the candidate pool.
    mu : Array
        Central-body gravitational parameter in ``au^3 / day^2``.

    Returns
    -------
    Array
        Root-mean-square angular residual for each candidate, in radians.

    Notes
    -----
    The score uses two-body propagation and a short fixed-point down-leg
    light-time correction. It is intended as a cheap consistency check across
    the sampled window, not as a replacement for full differential correction.
    """
    window_tdb_jd = optical_inputs.tdb_jd[sampling_window.start_idx:sampling_window.end_idx + 1]
    window_site_pos = optical_inputs.site_pos[sampling_window.start_idx:sampling_window.end_idx + 1]
    window_los_unit = optical_inputs.los_unit[sampling_window.start_idx:sampling_window.end_idx + 1]

    candidate_valid = (
        jnp.all(jnp.isfinite(candidate_result.pos_t2), axis=-1)
        & jnp.all(jnp.isfinite(candidate_result.vel_t2), axis=-1)
        & jnp.isfinite(candidate_result.residual_norm)
    )
    safe_pos = jnp.where(candidate_valid[:, None], candidate_result.pos_t2, jnp.asarray([1.0, 0.0, 0.0]))
    safe_vel = jnp.where(candidate_valid[:, None], candidate_result.vel_t2, jnp.asarray([0.0, jnp.sqrt(mu), 0.0]))

    candidate_epoch_tdb_jd = candidate_result.epoch_tdb_jd1 + candidate_result.epoch_tdb_jd2
    dt_to_window = window_tdb_jd[None, :] - candidate_epoch_tdb_jd[:, None]

    candidate_count = candidate_result.pos_t2.shape[0]
    observation_count = window_tdb_jd.shape[0]
    receive_pos, _ = kepler_propagate(
        jnp.broadcast_to(safe_pos[:, None, :], (candidate_count, observation_count, 3)).reshape(-1, 3),
        jnp.broadcast_to(safe_vel[:, None, :], (candidate_count, observation_count, 3)).reshape(-1, 3),
        dt_to_window.reshape(-1),
        mu=mu,
    )
    receive_pos = receive_pos.reshape(candidate_count, observation_count, 3)

    topocentric_pos = receive_pos - window_site_pos[None, :, :]
    topocentric_dist = jnp.linalg.norm(topocentric_pos, axis=-1, keepdims=True)
    finite_dist = jnp.isfinite(topocentric_dist) & (topocentric_dist > 0.0)
    light_time = jnp.where(finite_dist[..., 0], topocentric_dist[..., 0] / C, 0.0)

    for _ in range(2):
        transmit_dt = (dt_to_window - light_time).reshape(-1)
        transmit_pos, _ = kepler_propagate(
            jnp.broadcast_to(safe_pos[:, None, :], (candidate_count, observation_count, 3)).reshape(-1, 3),
            jnp.broadcast_to(safe_vel[:, None, :], (candidate_count, observation_count, 3)).reshape(-1, 3),
            transmit_dt,
            mu=mu,
        )
        transmit_pos = transmit_pos.reshape(candidate_count, observation_count, 3)
        topocentric_pos = transmit_pos - window_site_pos[None, :, :]
        topocentric_dist = jnp.linalg.norm(topocentric_pos, axis=-1, keepdims=True)
        finite_dist = jnp.isfinite(topocentric_dist) & (topocentric_dist > 0.0)
        light_time = jnp.where(finite_dist[..., 0], topocentric_dist[..., 0] / C, light_time)

    pred_los_unit = topocentric_pos / jnp.where(finite_dist, topocentric_dist, 1.0)

    dot = jnp.sum(pred_los_unit * window_los_unit[None, :, :], axis=-1)
    cross_norm = jnp.linalg.norm(jnp.cross(pred_los_unit, window_los_unit[None, :, :], axis=-1), axis=-1)
    angular_residual = jnp.arctan2(cross_norm, jnp.clip(dot, -1.0, 1.0))
    rms_score = jnp.sqrt(jnp.mean(jnp.square(angular_residual), axis=-1))

    score_valid = (
        candidate_valid
        & jnp.all(jnp.isfinite(angular_residual), axis=-1)
        & jnp.all(finite_dist[..., 0], axis=-1)
    )
    return jnp.where(score_valid, rms_score, jnp.inf)


def _select_best_scored_candidate(candidate_result: DoubleRIODResult, candidate_score: Array) -> tuple[Array, Array]:
    """Select the finite candidate with the lowest full-window angular score."""
    valid_pos = jnp.all(jnp.isfinite(candidate_result.pos_t2), axis=-1)
    valid_vel = jnp.all(jnp.isfinite(candidate_result.vel_t2), axis=-1)
    valid_score = jnp.isfinite(candidate_score)
    valid_mask = valid_pos & valid_vel & valid_score

    safe_score = jnp.where(valid_mask, candidate_score, jnp.inf)
    best_candidate_idx = jnp.argmin(safe_score)
    best_score = safe_score[best_candidate_idx]
    return best_candidate_idx, best_score


class IODSolver:
    """Initial orbit determination solver based on sampled angle-only triplets.

    The current implementation samples candidate observation triplets from one
    prepared optical arc, solves each candidate with the Double-r method, and
    returns the solution with the smallest full-window angular score.
    """

    def __init__(self, max_iter: int = 30, tol: float = 1e-10):
        """Initialize an initial-orbit-determination solver.

        Parameters
        ----------
        max_iter : int, default=30
            Maximum number of Newton iterations allowed in the underlying IOD
            algorithm.
        tol : float, default=1e-10
            Convergence threshold for the final angular residual norm, in
            radians.

        Raises
        ------
        ValueError
            Raised when a numeric parameter falls outside its allowed range.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.key = jrandom.PRNGKey(DEFAULT_IOD_SEED)

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("max_iter", str(self.max_iter)),
                ("tol", format_float_array(self.tol)),
            ],
        )

    def solve(self, data: ObservationData, max_arc_days: float = 1.0, candidates_num: int = 10,
              init_rho: tuple[float, float] = (1.0, 1.0)) -> IODResult:
        """Solve for an initial orbit by sampling observation triplets.

        Parameters
        ----------
        data : ObservationData
            Single-target observation bundle. Ground-based optical and
            space-based optical observations are used by this stage. Radar
            observations are ignored.
        max_arc_days : float, default=1.0
            Width of the sampling window, in days. The window is centered on
            the middle sorted observation and is used to restrict the pool of
            optical observations available for IOD triplets. If the nominal
            window contains fewer than three observations, the full filtered
            optical arc is used instead.
        candidates_num : int, default=10
            Number of random triplets to sample and solve.
        init_rho : tuple[float, float], default=(1.0, 1.0)
            Initial line-of-sight range guesses for the first and third
            observations of each sampled triplet, in ``au``.

        Returns
        -------
        IODResult
            Initial orbit estimate, iteration count, full-window angular score,
            and the selected triplet indices in the original mixed input order.

        Raises
        ------
        ValueError
            Raised when the filtered observation set or the selected window does
            not contain enough observations, or when a public argument violates
            its allowed range.
        RuntimeError
            Raised when no valid IOD candidate can be produced.
        """

        # -------------------------------------------------------------------------
        # Step 1: Build the prepared optical candidate arrays
        # -------------------------------------------------------------------------
        sun = EphemerisBody("sun")
        optical_inputs = build_optical_iod_inputs(data, sun)

        sampling_window = resolve_sampling_window(
            optical_inputs.tdb_jd,
            max_arc_days,
        )
        self.key, solve_key = jrandom.split(self.key)
        candidate_triplet_indices = sample_triplet_indices(
            sampling_window,
            candidates_num,
            solve_key,
        )
        candidate_triplets = build_triplet_batch(optical_inputs, candidate_triplet_indices)

        # -------------------------------------------------------------------------
        # Step 2: Solve all sampled triplets with Double-r
        # -------------------------------------------------------------------------
        init_rho_batch = jnp.broadcast_to(jnp.asarray(init_rho, dtype=jnp.float64), (candidates_num, 2))
        candidate_result = double_r_iod(
            candidate_triplets.site_pos,
            candidate_triplets.los_unit,
            candidate_triplets.tdb_jd1,
            candidate_triplets.tdb_jd2,
            mu=sun.gm,
            init_rho=init_rho_batch,
            min_rho=IOD_RANGE_FLOOR_AU,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        # -------------------------------------------------------------------------
        # Step 3: Select the best valid candidate and resolve the output epoch
        # -------------------------------------------------------------------------
        candidate_score = _score_candidates_on_window(
            candidate_result,
            optical_inputs,
            sampling_window,
            sun.gm,
        )
        best_candidate_idx, best_score = _select_best_scored_candidate(candidate_result, candidate_score)
        if not bool(jnp.isfinite(best_score)):
            raise RuntimeError("IOD failed to produce any valid candidate orbit.")

        best_epoch_tdb_jd1 = candidate_result.epoch_tdb_jd1[best_candidate_idx]
        best_epoch_tdb_jd2 = candidate_result.epoch_tdb_jd2[best_candidate_idx]
        best_pos_helio = candidate_result.pos_t2[best_candidate_idx]
        best_vel_helio = candidate_result.vel_t2[best_candidate_idx]

        solution_epoch_tdb = jnp.round((best_epoch_tdb_jd1 + best_epoch_tdb_jd2) * 2.0) / 2.0
        solution_epoch_tdb_jd1 = jnp.floor(solution_epoch_tdb)
        solution_epoch_tdb_jd2 = solution_epoch_tdb - solution_epoch_tdb_jd1

        # -------------------------------------------------------------------------
        # Step 4: Propagate the best candidate and package the public result
        # -------------------------------------------------------------------------
        dt_to_solution_epoch = (solution_epoch_tdb_jd1 - best_epoch_tdb_jd1) + (solution_epoch_tdb_jd2 - best_epoch_tdb_jd2)
        solution_pos_helio, solution_vel_helio = kepler_propagate(
            best_pos_helio[None, :],
            best_vel_helio[None, :],
            jnp.atleast_1d(dt_to_solution_epoch),
            mu=sun.gm,
        )
        solution_time = Time.from_tdb_jd(solution_epoch_tdb_jd1, solution_epoch_tdb_jd2)
        solution_state_helio = State(solution_time.tdb(), solution_pos_helio[0], solution_vel_helio[0], HELIO_ICRS)
        solution_state_bcrs = solution_state_helio.to(BCRS, sun=sun)

        return IODResult(
            initial_orbit=solution_state_bcrs,
            iter_num=candidate_result.iter_num,
            err=float(best_score),
            used_indices=candidate_triplets.input_indices[best_candidate_idx],
        )
