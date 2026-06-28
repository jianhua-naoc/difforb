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
from difforb.astrometry.debias import DebiasResult
from difforb.astrometry.reduction.lt import LightTimeContext
from difforb.astrometry.reduction.optical import compute_astrometric_vector, correct_light_bending
from difforb.astrometry.reduction.radar import compute_radar_obs
from difforb.body.ephbody import EphemerisBody
from difforb.body.site import Site
from difforb.body.smallbody import SmallBody
from difforb.core.element import KepElement
from difforb.core.time.timescale import Time
from difforb.dynamics.dynamic_system import DynamicSystem
from difforb.integrator.integrator import NumericalIntegrator
from difforb.od.dc.prediction import AstrometryMeasurementModel
from difforb.utils import car2sph
from tests.assertions import assert_allclose, assert_array_equal


EPOCH_TDB_JD = 2460690.5
OPTICAL_OFFSETS = jnp.asarray([-3.0, 2.0], dtype=jnp.float64)
RADAR_OFFSETS = jnp.asarray([-1.0, 4.0], dtype=jnp.float64)
OPTICAL_CODES = np.asarray(["568", "G96"], dtype=str)
RADAR_CODES = np.asarray(["-14", "-14"], dtype=str)
RADAR_TX_FREQ = jnp.asarray([8.56e9, 8.56e9], dtype=jnp.float64)


def build_target_state(default_ephemeris):
    sun = EphemerisBody("sun", eph=default_ephemeris)
    element = KepElement.from_classical(
        tdb=Time.from_tdb_jd(EPOCH_TDB_JD, 0.0).tdb(),
        a=2.35,
        e=0.28,
        inc=14.0,
        node=83.0,
        peri=126.0,
        m=42.0,
    )
    return SmallBody.create(element, sun=sun).orbit0


def propagated_target(state, force_model, integrator):
    return SmallBody.create(state).propagate(
        Time.from_tdb_jd(EPOCH_TDB_JD - 60.0, 0.0).tdb(),
        Time.from_tdb_jd(EPOCH_TDB_JD + 60.0, 0.0).tdb(),
        force_model,
        integrator,
    )


def optical_values_from_state(state, t_obs, rx_codes, sun, earth, force_model, integrator):
    target = propagated_target(state, force_model, integrator)
    path = compute_astrometric_vector(
        t_obs,
        Site.from_code(rx_codes),
        target,
        LightTimeContext(sun=sun, earth=earth, shapiro_bodies=(sun,)),
    )
    bent_pos = correct_light_bending(sun, path)
    ra, dec = car2sph(bent_pos)
    return jnp.stack([ra, dec], axis=1)


def radar_values_from_state(state, t_obs, rx_codes, tx_codes, tx_freq, sun, earth, force_model, integrator):
    target = propagated_target(state, force_model, integrator)
    radar_obs = compute_radar_obs(
        t_obs,
        target,
        Site.from_code(rx_codes).require_ground(),
        Site.from_code(tx_codes).require_ground(),
        tx_freq,
        LightTimeContext(
            sun=sun,
            earth=earth,
            atmos_cor_enable=True,
            corona_cor_enable=True,
            shapiro_bodies=(sun,),
        ),
    )
    return jnp.asarray([radar_obs.delay[0], radar_obs.doppler_shift[1]], dtype=jnp.float64)


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


def optical_data(t_obs, values, *, note_codes=None):
    count = len(values)
    return OpticalObservationData(
        t=t_obs,
        trk_ids=np.asarray([f"T{i}" for i in range(count)], dtype=object),
        obs_type_ids=np.full(count, ObsType.OPTICAL.id, dtype=int),
        obs_mode_ids=np.full(count, ObsMode.CCD.id, dtype=int),
        values=np.asarray(values, dtype=float),
        uncertainties=np.full((count, 2), np.deg2rad(0.2 / 3600.0), dtype=float),
        correlations=np.zeros(count, dtype=float),
        time_uncertainties=np.full(count, np.nan, dtype=float),
        rx_codes=OPTICAL_CODES[:count],
        program_codes=np.asarray([""] * count, dtype=object),
        catalog_codes=np.asarray([""] * count, dtype=object),
        note_codes=np.asarray([""] * count if note_codes is None else note_codes, dtype=object),
        magnitudes=np.full(count, np.nan, dtype=float),
        band_codes=np.asarray([""] * count, dtype=object),
        sub_frames=np.asarray(["ICRF"] * count, dtype=object),
        input_indices=np.arange(count, dtype=int),
    )


def radar_data(t_obs, values):
    return RadarObservationData(
        t=t_obs,
        obs_type_ids=np.full(2, ObsType.RADAR.id, dtype=int),
        obs_mode_ids=np.asarray([ObsMode.DELAY_CENTER.id, ObsMode.DOPPLER_CENTER.id], dtype=int),
        values=np.asarray(values, dtype=float),
        uncertainties=np.asarray([10.0, 0.5], dtype=float),
        rx_codes=RADAR_CODES,
        tx_codes=RADAR_CODES,
        tx_freq=np.asarray(RADAR_TX_FREQ, dtype=float),
        input_indices=np.asarray([2, 3], dtype=int),
    )


def mixed_prediction_case(default_ephemeris, *, optical_bias=None, note_codes=None):
    sun = EphemerisBody("sun", eph=default_ephemeris)
    earth = EphemerisBody("earth", eph=default_ephemeris)
    force_model = DynamicSystem.from_extended_system(default_ephemeris).build_force_model()
    integrator = NumericalIntegrator(method="IAS15", tol=1.0e-12, max_steps=4096)
    state = build_target_state(default_ephemeris)
    optical_t = Time.from_tdb_jd(EPOCH_TDB_JD + OPTICAL_OFFSETS, jnp.zeros_like(OPTICAL_OFFSETS))
    radar_t = Time.from_tdb_jd(EPOCH_TDB_JD + RADAR_OFFSETS, jnp.zeros_like(RADAR_OFFSETS))
    optical_values = optical_values_from_state(state, optical_t, OPTICAL_CODES, sun, earth, force_model, integrator)
    if optical_bias is not None:
        optical_values = optical_values + jnp.asarray(optical_bias, dtype=optical_values.dtype)
    radar_values = radar_values_from_state(state, radar_t, RADAR_CODES, RADAR_CODES, RADAR_TX_FREQ, sun, earth, force_model, integrator)
    data = ObservationData(
        name="synthetic-prediction",
        optical=optical_data(optical_t, optical_values, note_codes=note_codes),
        radar=radar_data(radar_t, radar_values),
    )
    return sun, earth, force_model, integrator, data, state


def build_model(data, state, sun, earth, *, optical_bias=None):
    if optical_bias is None:
        optical_bias = np.zeros((data.num_optical, 2), dtype=float)
    return AstrometryMeasurementModel.build(
        data,
        state.tdb,
        sun,
        earth,
        DebiasResult(np.asarray(optical_bias, dtype=float)),
    )


def test_measurement_model_zero_residuals(default_ephemeris):
    sun, earth, force_model, integrator, data, state = mixed_prediction_case(default_ephemeris)
    model = build_model(data, state, sun, earth)
    params = state.array.squeeze()

    residuals = model.compute_residuals(params, force_model, integrator)

    print(
        "[od.dc.prediction.residuals] "
        f"shape={residuals.shape} "
        f"max_abs={float(jnp.max(jnp.abs(residuals))):.12e} "
        f"radar={jnp.asarray(residuals[-2:])}"
    )

    assert residuals.shape == (2 * data.num_optical + data.num_radar,)
    assert bool(jnp.all(jnp.isfinite(residuals)))
    assert_allclose(residuals[:4], jnp.zeros(4), atol=2.0e-12, rtol=0.0)
    assert_allclose(residuals[4:], jnp.zeros(2), atol=2.0e-5, rtol=0.0)


def test_measurement_model_jacobian_contract(default_ephemeris):
    sun, earth, force_model, integrator, data, state = mixed_prediction_case(default_ephemeris)
    model = build_model(data, state, sun, earth)
    params = state.array.squeeze()

    jacobian, residuals = model.compute_jacobian_with_residuals(params, force_model, integrator)
    direct_residuals = model.compute_residuals(params, force_model, integrator)

    print(
        "[od.dc.prediction.jacobian] "
        f"jac_shape={jacobian.shape} "
        f"jac_norm={float(jnp.linalg.norm(jacobian)):.12e} "
        f"residual_max_abs={float(jnp.max(jnp.abs(residuals))):.12e}"
    )

    assert jacobian.shape == (2 * data.num_optical + data.num_radar, 6)
    assert residuals.shape == (2 * data.num_optical + data.num_radar,)
    assert_allclose(residuals, direct_residuals, atol=1.0e-15, rtol=0.0)
    assert bool(jnp.all(jnp.isfinite(jacobian)))
    assert bool(jnp.linalg.norm(jacobian) > 0.0)


def test_measurement_model_optical_rates_contract(default_ephemeris):
    sun, earth, force_model, integrator, data, state = mixed_prediction_case(default_ephemeris)
    model = build_model(data, state, sun, earth)
    params = state.array.squeeze()

    rates = model.compute_optical_rates(params, force_model, integrator)

    print(
        "[od.dc.prediction.rates] "
        f"shape={rates.shape} "
        f"max_abs={float(jnp.max(jnp.abs(rates))):.12e} rad/day"
    )

    assert rates.shape == (data.num_optical, 2)
    assert bool(jnp.all(jnp.isfinite(rates)))
    assert bool(jnp.any(jnp.abs(rates) > 0.0))


def test_measurement_model_applies_debias_and_photocenter_mask(default_ephemeris):
    optical_bias = jnp.asarray([[1.0e-8, -2.0e-8], [-3.0e-8, 4.0e-8]], dtype=jnp.float64)
    sun, earth, force_model, integrator, data, state = mixed_prediction_case(
        default_ephemeris,
        optical_bias=optical_bias,
        note_codes=["", "e"],
    )
    model = build_model(data, state, sun, earth, optical_bias=optical_bias)
    params = state.array.squeeze()

    residuals = model.compute_residuals(params, force_model, integrator)

    print(
        "[od.dc.prediction.debias] "
        f"photocenter_mask={np.asarray(model.optical_photocenter_mask).tolist()} "
        f"optical_max_abs={float(jnp.max(jnp.abs(residuals[:4]))):.12e}"
    )

    assert_array_equal(model.optical_photocenter_mask, jnp.asarray([True, False]))
    assert_allclose(residuals[:4], jnp.zeros(4), atol=2.0e-12, rtol=0.0)
