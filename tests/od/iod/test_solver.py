import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from difforb.astrometry.data import (
    ObsMode,
    ObsType,
    ObservationData,
    OpticalObservationData,
    RadarObservationData,
)
from difforb.body.ephbody import EphemerisBody
from difforb.body.site import Site
from difforb.core.constants import C, GM_SUN
from difforb.core.state.frame import BCRS, HELIO_ICRS
from difforb.core.state.state import State
from difforb.core.time.timescale import Time
from difforb.dynamics.two_body import kepler_propagate
from difforb.od.iod.double_r import DoubleRIODResult
from difforb.od.iod.sampling import IODSamplingWindow, OpticalIODInputs
from difforb.od.iod.solver import IODSolver, _score_candidates_on_window, _select_best_scored_candidate
from difforb.utils import car2sph
from tests.od.iod.reference import propagate_reference_arc, reference_t2_state, reference_tdb
from tests.assertions import assert_allclose, assert_array_equal


def empty_optical_data():
    return OpticalObservationData(
        t=Time.from_tdb_jd(jnp.asarray([], dtype=float), jnp.asarray([], dtype=float)),
        trk_ids=np.asarray([], dtype=object),
        obs_type_ids=np.asarray([], dtype=int),
        obs_mode_ids=np.asarray([], dtype=int),
        values=np.empty((0, 2), dtype=float),
        uncertainties=np.empty((0, 2), dtype=float),
        correlations=np.empty((0,), dtype=float),
        time_uncertainties=np.empty((0,), dtype=float),
        rx_codes=np.asarray([], dtype=str),
        program_codes=np.asarray([], dtype=object),
        catalog_codes=np.asarray([], dtype=object),
        note_codes=np.asarray([], dtype=object),
        magnitudes=np.empty((0,), dtype=float),
        band_codes=np.asarray([], dtype=object),
        sub_frames=np.asarray([], dtype=object),
        input_indices=np.asarray([], dtype=int),
    )


def empty_radar_data():
    return RadarObservationData(
        t=Time.from_tdb_jd(jnp.asarray([], dtype=float), jnp.asarray([], dtype=float)),
        obs_type_ids=np.asarray([], dtype=int),
        obs_mode_ids=np.asarray([], dtype=int),
        values=np.empty((0,), dtype=float),
        uncertainties=np.empty((0,), dtype=float),
        rx_codes=np.asarray([], dtype=str),
        tx_codes=np.asarray([], dtype=str),
        tx_freq=np.empty((0,), dtype=float),
        input_indices=np.asarray([], dtype=int),
    )


def optical_data(t, values, rx_codes, input_indices):
    count = len(values)
    return OpticalObservationData(
        t=t,
        trk_ids=np.asarray([f"T{i}" for i in range(count)], dtype=object),
        obs_type_ids=np.full(count, ObsType.OPTICAL.id, dtype=int),
        obs_mode_ids=np.full(count, ObsMode.CCD.id, dtype=int),
        values=np.asarray(values, dtype=float),
        uncertainties=np.full((count, 2), np.deg2rad(1.0 / 3600.0), dtype=float),
        correlations=np.zeros(count, dtype=float),
        time_uncertainties=np.full(count, np.nan, dtype=float),
        rx_codes=np.asarray(rx_codes, dtype=str),
        program_codes=np.asarray([""] * count, dtype=object),
        catalog_codes=np.asarray([""] * count, dtype=object),
        note_codes=np.asarray([""] * count, dtype=object),
        magnitudes=np.full(count, np.nan, dtype=float),
        band_codes=np.asarray([""] * count, dtype=object),
        sub_frames=np.asarray([""] * count, dtype=object),
        input_indices=np.asarray(input_indices, dtype=int),
    )


def observation_data(optical):
    return ObservationData(
        name="synthetic",
        optical=optical,
        radar=empty_radar_data(),
    )


def synthetic_arc(default_ephemeris):
    sun = EphemerisBody("sun", eph=default_ephemeris)
    earth = EphemerisBody("earth", eph=default_ephemeris)
    t_obs = reference_tdb()
    target_pos, _ = propagate_reference_arc()
    rx_codes = np.asarray(["000", "568", "G96"], dtype=str)
    site_pos = Site.from_code(rx_codes).state(
        t_obs,
        frame=HELIO_ICRS,
        sun=sun,
        earth=earth,
    ).pos
    topocentric_pos = target_pos - site_pos
    topocentric_distance = jnp.linalg.norm(topocentric_pos, axis=1)
    ra, dec = car2sph(topocentric_pos)

    optical = optical_data(
        t_obs,
        values=jnp.stack([ra, dec], axis=1),
        rx_codes=rx_codes,
        input_indices=[0, 1, 2],
    )
    expected_orbit = reference_t2_state().to(BCRS, sun=sun)

    return observation_data(optical), expected_orbit, (float(topocentric_distance[0]), float(topocentric_distance[2]))


def test_iod_solver_recovers_arc(default_ephemeris):
    data, expected_orbit, init_rho = synthetic_arc(default_ephemeris)

    result = IODSolver(max_iter=30, tol=1.0e-10).solve(
        data,
        max_arc_days=10.0,
        candidates_num=1,
        init_rho=init_rho,
    )

    print(
        "[od.iod.solver.arc] "
        f"err={result.err:.12e} rad "
        f"pos_norm_diff={float(jnp.linalg.norm(result.initial_orbit.pos - expected_orbit.pos)):.12e} au "
        f"vel_norm_diff={float(jnp.linalg.norm(result.initial_orbit.vel - expected_orbit.vel)):.12e} au/day"
    )

    assert result.initial_orbit.frame == BCRS
    assert_allclose(result.initial_orbit.tdb.jd, expected_orbit.tdb.jd, atol=0.0, rtol=0.0)
    assert_allclose(result.initial_orbit.pos, expected_orbit.pos, atol=1.0e-9, rtol=0.0)
    assert_allclose(result.initial_orbit.vel, expected_orbit.vel, atol=1.0e-9, rtol=0.0)
    assert_array_equal(result.used_indices, jnp.asarray([0, 1, 2]))


def test_iod_solver_requires_three(default_ephemeris):
    t_obs = Time.from_tdb_jd(jnp.asarray([2460000.0, 2460001.0]), jnp.zeros(2, dtype=float))
    optical = optical_data(
        t_obs,
        values=np.deg2rad(np.asarray([[30.0, 5.0], [35.0, 6.0]])),
        rx_codes=["000", "568"],
        input_indices=[0, 1],
    )

    with pytest.raises(ValueError, match="at least 3 observations"):
        IODSolver(max_iter=5, tol=1.0e-8).solve(
            observation_data(optical),
            max_arc_days=10.0,
            candidates_num=1,
            init_rho=(1.0, 1.0),
        )


def test_iod_solver_contract(default_ephemeris):
    data, _, init_rho = synthetic_arc(default_ephemeris)

    result = IODSolver(max_iter=30, tol=1.0e-10).solve(
        data,
        max_arc_days=10.0,
        candidates_num=1,
        init_rho=init_rho,
    )

    assert isinstance(result.initial_orbit, State)
    assert result.initial_orbit.frame == BCRS
    assert result.shape == ()
    assert result.used_indices.shape == (3,)
    assert int(result.iter_num) <= 30
    assert np.isfinite(result.err)
    assert bool(jnp.all(jnp.isfinite(result.initial_orbit.pos)))
    assert bool(jnp.all(jnp.isfinite(result.initial_orbit.vel)))


def test_iod_solver_scores_candidates_on_full_window():
    epoch_tdb_jd = 2460000.0
    window_tdb_jd = epoch_tdb_jd + jnp.asarray([0.0, 1.0, 2.0])
    reference_pos = jnp.asarray([1.30, 0.20, 0.10], dtype=jnp.float64)
    reference_vel = jnp.asarray([-0.004, 0.012, 0.001], dtype=jnp.float64)
    dt = jnp.asarray([0.0, 1.0, 2.0])
    target_pos, _ = kepler_propagate(
        jnp.broadcast_to(reference_pos, (3, 3)),
        jnp.broadcast_to(reference_vel, (3, 3)),
        dt,
        mu=GM_SUN,
    )
    light_time = jnp.linalg.norm(target_pos, axis=-1) / C
    for _ in range(2):
        target_pos, _ = kepler_propagate(
            jnp.broadcast_to(reference_pos, (3, 3)),
            jnp.broadcast_to(reference_vel, (3, 3)),
            dt - light_time,
            mu=GM_SUN,
        )
        light_time = jnp.linalg.norm(target_pos, axis=-1) / C
    los_unit = target_pos / jnp.linalg.norm(target_pos, axis=-1, keepdims=True)
    tdb_jd1 = jnp.floor(window_tdb_jd)
    tdb_jd2 = window_tdb_jd - tdb_jd1
    optical_inputs = OpticalIODInputs(
        tdb_jd1=tdb_jd1,
        tdb_jd2=tdb_jd2,
        tdb_jd=window_tdb_jd,
        site_pos=jnp.zeros((3, 3), dtype=jnp.float64),
        los_unit=los_unit,
        input_indices=jnp.asarray([10, 20, 30], dtype=jnp.int32),
    )
    candidate_result = DoubleRIODResult(
        pos_t2=jnp.stack([
            jnp.asarray([0.80, 0.50, -0.20], dtype=jnp.float64),
            reference_pos,
        ]),
        vel_t2=jnp.stack([
            jnp.asarray([0.002, -0.006, 0.003], dtype=jnp.float64),
            reference_vel,
        ]),
        epoch_tdb_jd1=jnp.asarray([epoch_tdb_jd, epoch_tdb_jd], dtype=jnp.float64),
        epoch_tdb_jd2=jnp.zeros(2, dtype=jnp.float64),
        residual_norm=jnp.asarray([1.0e-12, 1.0e-8], dtype=jnp.float64),
        iter_num=3,
    )

    score = _score_candidates_on_window(
        candidate_result,
        optical_inputs,
        IODSamplingWindow(start_idx=0, end_idx=2, center_idx=1),
        GM_SUN,
    )
    best_idx, best_score = _select_best_scored_candidate(candidate_result, score)

    print(
        "[od.iod.solver.window_score] "
        f"scores={jnp.asarray(score)} "
        f"best_idx={int(best_idx)} "
        f"best_score={float(best_score):.12e}"
    )

    assert int(best_idx) == 1
    assert score[1] < 1.0e-12
    assert score[0] > score[1]
