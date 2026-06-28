import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from difforb.astrometry.data import (
    ObsMode,
    ObsType,
    ObservationData,
    OpticalObservationData,
    RadarObservationData,
)
from difforb.astrometry.debias import NoDebiasPolicy
from difforb.astrometry.reduction.lt import LightTimeContext
from difforb.astrometry.reduction.optical import compute_astrometric_vector, correct_light_bending
from difforb.astrometry.weight import ADESWeightPolicy
from difforb.body.ephbody import EphemerisBody
from difforb.body.site import Site
from difforb.body.smallbody import SmallBody
from difforb.core.element import KepElement
from difforb.core.state.frame import BCRS
from difforb.core.state.state import State
from difforb.core.time.timescale import Time
from difforb.dynamics.dynamic_system import DynamicSystem
from difforb.integrator.integrator import NumericalIntegrator
from difforb.od.dc.bucket import DCBucketPolicy
from difforb.od.dc.result import DCResult
from difforb.od.dc.solver import DCSolver
from difforb.od.outlier.policy import InteractiveOutlierPolicy
from difforb.utils import car2sph
from tests.assertions import assert_allclose

EPOCH_TDB_JD = 2460690.5
OBSERVATION_OFFSETS = jnp.asarray([-18.0, -12.0, -7.0, -3.0, 0.0, 4.0, 9.0, 15.0, 21.0], dtype=jnp.float64)
OBSERVATORY_CODES = np.asarray(["568", "G96", "F51", "703", "I41", "691", "568", "G96", "F51"], dtype=str)


def empty_radar_data():
    return RadarObservationData(
        t=Time.from_tdb_jd(jnp.asarray([], dtype=float), jnp.asarray([], dtype=float)),
        obs_type_ids=np.asarray([], dtype=int),
        obs_mode_ids=np.asarray([], dtype=int),
        values=np.empty((0,), dtype=float),
        uncertainties=np.empty((0,), dtype=float),
        rx_codes=np.asarray([], dtype=object),
        tx_codes=np.asarray([], dtype=object),
        tx_freq=np.empty((0,), dtype=float),
        input_indices=np.asarray([], dtype=int),
    )


def optical_data(t, values, rx_codes, *, time_uncertainty_s=np.nan):
    count = len(values)
    time_uncertainties = np.full(count, time_uncertainty_s, dtype=float)
    return OpticalObservationData(
        t=t,
        trk_ids=np.asarray([f"T{i}" for i in range(count)], dtype=object),
        obs_type_ids=np.full(count, ObsType.OPTICAL.id, dtype=int),
        obs_mode_ids=np.full(count, ObsMode.CCD.id, dtype=int),
        values=np.asarray(values, dtype=float),
        uncertainties=np.full((count, 2), np.deg2rad(0.2 / 3600.0), dtype=float),
        correlations=np.zeros(count, dtype=float),
        time_uncertainties=time_uncertainties,
        rx_codes=np.asarray(rx_codes, dtype=str),
        program_codes=np.asarray([""] * count, dtype=object),
        catalog_codes=np.asarray([""] * count, dtype=object),
        note_codes=np.asarray([""] * count, dtype=object),
        magnitudes=np.full(count, np.nan, dtype=float),
        band_codes=np.asarray([""] * count, dtype=object),
        sub_frames=np.asarray(["ICRF"] * count, dtype=object),
        input_indices=np.arange(count, dtype=int),
    )


def build_target_state(default_ephemeris):
    sun = EphemerisBody("sun", eph=default_ephemeris)
    tdb0 = Time.from_tdb_jd(EPOCH_TDB_JD, 0.0).tdb()
    element = KepElement.from_classical(
        tdb=tdb0,
        a=2.35,
        e=0.28,
        inc=14.0,
        node=83.0,
        peri=126.0,
        m=42.0,
    )
    return SmallBody.create(element, sun=sun).orbit0


def optical_values_from_state(state, t_obs, rx_codes, default_ephemeris, force_model, integrator):
    sun = EphemerisBody("sun", eph=default_ephemeris)
    earth = EphemerisBody("earth", eph=default_ephemeris)
    target = SmallBody.create(state).propagate(
        Time.from_tdb_jd(EPOCH_TDB_JD - 70.0, 0.0).tdb(),
        Time.from_tdb_jd(EPOCH_TDB_JD + 70.0, 0.0).tdb(),
        force_model,
        integrator,
    )
    path = compute_astrometric_vector(
        t_obs,
        Site.from_code(rx_codes),
        target,
        LightTimeContext(sun=sun, earth=earth, shapiro_bodies=(sun,)),
    )
    bent_pos = correct_light_bending(sun, path)
    ra, dec = car2sph(bent_pos)
    return jnp.stack([ra, dec], axis=1)


def ground_optical_case(default_ephemeris, *, time_uncertainty_s=np.nan):
    sun = EphemerisBody("sun", eph=default_ephemeris)
    earth = EphemerisBody("earth", eph=default_ephemeris)
    force_model = DynamicSystem.from_extended_system(default_ephemeris).build_force_model()
    integrator = NumericalIntegrator(method="IAS15", tol=1e-12, max_steps=4096)
    t_obs = Time.from_tdb_jd(EPOCH_TDB_JD + OBSERVATION_OFFSETS, jnp.zeros_like(OBSERVATION_OFFSETS))
    expected_state = build_target_state(default_ephemeris)
    values = optical_values_from_state(expected_state, t_obs, OBSERVATORY_CODES, default_ephemeris, force_model, integrator)
    data = ObservationData(
        name="synthetic-ground-optical",
        optical=optical_data(t_obs, values, OBSERVATORY_CODES, time_uncertainty_s=time_uncertainty_s),
        radar=empty_radar_data(),
    )
    initial_state = State(
        expected_state.tdb,
        expected_state.pos + jnp.asarray([2.0e-7, -1.5e-7, 1.0e-7]),
        expected_state.vel + jnp.asarray([-1.0e-8, 8.0e-9, 5.0e-9]),
        BCRS,
    )
    return sun, earth, force_model, integrator, data, initial_state, expected_state


def solve_ground_optical_case(default_ephemeris, *, bucket_policy=None, time_uncertainty_s=np.nan):
    sun, earth, force_model, integrator, data, initial_state, expected_state = ground_optical_case(
        default_ephemeris,
        time_uncertainty_s=time_uncertainty_s,
    )
    result = DCSolver(
        lsq_tol=1.0e-16,
        lsq_max_iters=20,
        sun=sun,
        earth=earth,
        bucket_policy=bucket_policy,
    ).solve(
        data,
        initial_state,
        force_model,
        integrator,
        ADESWeightPolicy(),
        NoDebiasPolicy(),
        InteractiveOutlierPolicy(auto_rejecter=None, enable_auto_rejecter=False, max_iters=1),
        log_detail="quiet",
    )
    return result, initial_state, expected_state


def test_dc_solver_recovers_ground_optical_arc(default_ephemeris):
    result, initial_state, expected_state = solve_ground_optical_case(default_ephemeris)
    pos_diff = result.estimate.orbit.pos - expected_state.pos
    vel_diff = result.estimate.orbit.vel - expected_state.vel
    initial_pos_diff = initial_state.pos - expected_state.pos
    initial_vel_diff = initial_state.vel - expected_state.vel
    print(
        "[od.dc.solver.ground_optical] "
        f"normalized_rms={result.normalized_residual_rms:.12e} "
        f"pos_norm_diff={float(jnp.linalg.norm(pos_diff)):.12e} au "
        f"vel_norm_diff={float(jnp.linalg.norm(vel_diff)):.12e} au/day"
    )

    assert result.estimate.orbit.frame == BCRS
    assert result.optical.residuals.shape == (len(OBSERVATION_OFFSETS), 2)
    assert result.radar.residuals.shape == (0,)
    assert result.optical.n_inliers == len(OBSERVATION_OFFSETS)
    assert result.radar.n_obs == 0
    assert jnp.linalg.norm(pos_diff) < 0.4 * jnp.linalg.norm(initial_pos_diff)
    assert jnp.linalg.norm(vel_diff) < 0.1 * jnp.linalg.norm(initial_vel_diff)
    assert_allclose(result.estimate.orbit.pos, expected_state.pos, atol=5.0e-9, rtol=0.0)
    assert_allclose(result.estimate.orbit.vel, expected_state.vel, atol=5.0e-11, rtol=0.0)
    assert result.normalized_residual_rms < 1.0e-5


def test_dc_solver_contract(default_ephemeris):
    result, _, _ = solve_ground_optical_case(default_ephemeris)
    n_optical = len(OBSERVATION_OFFSETS)
    n_flat = 2 * n_optical
    n_params = 6

    print(
        "[od.dc.solver.contract] "
        f"flat_jac_shape={result.lsq_diagnostics.flat_jacobian.shape} "
        f"cov_rank={int(result.lsq_diagnostics.cov_rank)} "
        f"cov_condition={float(result.lsq_diagnostics.cov_condition):.12e} "
        f"termination={result.lsq_diagnostics.termination_reason}"
    )

    assert isinstance(result, DCResult)
    assert isinstance(result.estimate.orbit, State)
    assert result.estimate.orbit.frame == BCRS
    assert result.estimate.orbit.shape == ()
    assert result.estimate.orbit.pos.shape == (3,)
    assert result.estimate.orbit.vel.shape == (3,)
    assert result.estimate.model_params.shape == (0,)
    assert result.estimate.model_param_names == []
    assert result.estimate.cov_mat_post.shape == (n_params, n_params)
    assert result.estimate.uncertainties.shape == (n_params,)
    assert bool(jnp.all(jnp.isfinite(result.estimate.orbit.pos)))
    assert bool(jnp.all(jnp.isfinite(result.estimate.orbit.vel)))
    assert bool(jnp.all(jnp.isfinite(result.estimate.cov_mat_post)))
    assert bool(jnp.all(jnp.isfinite(result.estimate.uncertainties)))

    assert result.optical.residuals.shape == (n_optical, 2)
    assert result.optical.normalized_residuals.shape == (n_optical, 2)
    assert result.optical.inlier_masks.shape == (n_optical,)
    assert result.optical.metrics.shape == (n_optical,)
    assert result.optical.n_obs == n_optical
    assert result.optical.n_inliers == n_optical
    assert result.optical.n_outliers == 0
    assert np.isfinite(result.optical.weighted_rms)
    assert np.isfinite(result.optical.unweighted_rms)

    assert result.radar.residuals.shape == (0,)
    assert result.radar.normalized_residuals.shape == (0,)
    assert result.radar.inlier_masks.shape == (0,)
    assert result.radar.metrics.shape == (0,)
    assert result.radar.n_obs == 0
    assert result.radar.n_inliers == 0
    assert result.radar.n_outliers == 0

    diagnostics = result.lsq_diagnostics
    assert diagnostics.flat_jacobian.shape == (n_flat, n_params)
    assert diagnostics.flat_weights.shape == (n_flat,)
    assert diagnostics.cov_mat_prior.shape == (n_params, n_params)
    assert jnp.asarray(diagnostics.cov_rank).shape == ()
    assert int(diagnostics.cov_rank) == n_params
    assert bool(diagnostics.cov_valid)
    assert bool(jnp.isfinite(diagnostics.cov_condition))
    assert bool(jnp.all(jnp.isfinite(diagnostics.flat_jacobian)))
    assert bool(jnp.all(jnp.isfinite(diagnostics.flat_weights)))
    assert bool(jnp.all(jnp.isfinite(diagnostics.cov_mat_prior)))
    assert diagnostics.converged
    assert diagnostics.termination_reason in {"gradient_converged", "step_converged"}
    assert 0 < diagnostics.lsq_iterations <= 20
    assert diagnostics.outlier_iterations >= 0
    assert np.isfinite(result.normalized_residual_rms)


def test_dc_solver_applies_optical_time_uncertainty_weights(default_ephemeris):
    result, _, _ = solve_ground_optical_case(default_ephemeris, time_uncertainty_s=60.0)
    covariances = np.linalg.inv(np.asarray(result.lsq_diagnostics.optical_weight_matrices))
    base_variance = np.deg2rad(0.2 / 3600.0) ** 2

    assert np.all(np.isfinite(covariances))
    assert np.any(np.diagonal(covariances, axis1=1, axis2=2) > base_variance)


def test_dc_solver_with_bucket_policy(default_ephemeris):
    result, _, expected_state = solve_ground_optical_case(
        default_ephemeris,
        bucket_policy=DCBucketPolicy(optical_buckets=(12,), radar_buckets=(1,)),
    )

    pos_diff = result.estimate.orbit.pos - expected_state.pos
    vel_diff = result.estimate.orbit.vel - expected_state.vel
    print(
        "[od.dc.solver.bucket] "
        f"normalized_rms={result.normalized_residual_rms:.12e} "
        f"pos_norm_diff={float(jnp.linalg.norm(pos_diff)):.12e} au "
        f"vel_norm_diff={float(jnp.linalg.norm(vel_diff)):.12e} au/day "
        f"flat_jac_shape={result.lsq_diagnostics.flat_jacobian.shape}"
    )

    assert result.estimate.orbit.frame == BCRS
    assert result.optical.residuals.shape == (len(OBSERVATION_OFFSETS), 2)
    assert result.optical.normalized_residuals.shape == (len(OBSERVATION_OFFSETS), 2)
    assert result.optical.inlier_masks.shape == (len(OBSERVATION_OFFSETS),)
    assert result.optical.metrics.shape == (len(OBSERVATION_OFFSETS),)
    assert result.radar.residuals.shape == (0,)
    assert result.lsq_diagnostics.flat_jacobian.shape == (2 * len(OBSERVATION_OFFSETS), 6)
    assert result.lsq_diagnostics.flat_weights.shape == (2 * len(OBSERVATION_OFFSETS),)
    assert result.optical.n_inliers == len(OBSERVATION_OFFSETS)
    assert_allclose(result.estimate.orbit.pos, expected_state.pos, atol=5.0e-9, rtol=0.0)
    assert_allclose(result.estimate.orbit.vel, expected_state.vel, atol=5.0e-11, rtol=0.0)
    assert result.normalized_residual_rms < 1.0e-5
