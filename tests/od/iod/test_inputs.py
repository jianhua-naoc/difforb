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
from difforb.body.ephbody import EphemerisBody
from difforb.body.site import Site, format_site_key
from difforb.core.state.frame import HELIO_ICRS
from difforb.core.time.timescale import Time
from difforb.od.iod.sampling import build_optical_iod_inputs
from difforb.utils import sph2car
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
        rx_codes=np.asarray([], dtype=object),
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
        rx_codes=np.asarray([], dtype=object),
        tx_codes=np.asarray([], dtype=object),
        tx_freq=np.empty((0,), dtype=float),
        input_indices=np.asarray([], dtype=int),
    )


def optical_data(tdb_jd, values, rx_codes, input_indices):
    count = len(tdb_jd)
    return OpticalObservationData(
        t=Time.from_tdb_jd(jnp.asarray(tdb_jd, dtype=float), jnp.zeros(count, dtype=float)),
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


def observation_data(optical=None):
    return ObservationData(
        name="synthetic",
        optical=optical if optical is not None else empty_optical_data(),
        radar=empty_radar_data(),
    )


def test_iod_inputs_ground_sorted(default_ephemeris):
    sun = EphemerisBody("sun", eph=default_ephemeris)
    tdb_jd = np.asarray([2460003.0, 2460001.0, 2460002.0])
    values = np.deg2rad(np.asarray([[30.0, 5.0], [10.0, -8.0], [70.0, 12.0]]))
    rx_codes = np.asarray(["568", "000", "G96"], dtype=str)
    input_indices = np.asarray([30, 10, 20])
    ground_obs = optical_data(
        tdb_jd,
        values,
        rx_codes,
        input_indices,
    )

    inputs = build_optical_iod_inputs(observation_data(ground_obs), sun)
    sort_idx = np.argsort(tdb_jd)
    expected_site_pos = Site.from_code(rx_codes).state(
        ground_obs.t,
        frame=HELIO_ICRS,
        sun=sun,
        earth=EphemerisBody("earth", eph=default_ephemeris),
    ).pos[sort_idx]

    print(
        "[od.iod.inputs.ground] "
        f"input_indices={jnp.asarray(inputs.input_indices)} "
        f"site_shape={inputs.site_pos.shape}"
    )

    assert inputs.tdb_jd.shape == (3,)
    assert inputs.site_pos.shape == (3, 3)
    assert inputs.los_unit.shape == (3, 3)
    assert_array_equal(inputs.input_indices, input_indices[sort_idx])
    assert_allclose(inputs.tdb_jd, jnp.asarray(tdb_jd[sort_idx]), atol=0.0, rtol=0.0)
    assert_allclose(inputs.los_unit, sph2car(values[sort_idx, 0], values[sort_idx, 1]), atol=1.0e-15, rtol=0.0)
    assert_allclose(inputs.site_pos, expected_site_pos, atol=0.0, rtol=0.0)


def test_iod_inputs_mixed_sorted(default_ephemeris):
    sun = EphemerisBody("sun", eph=default_ephemeris)
    ground_values = np.deg2rad(np.asarray([[30.0, 5.0], [10.0, -8.0]]))
    space_values = np.deg2rad(np.asarray([[70.0, 12.0], [115.0, -3.0]]))
    space_keys = [
        format_site_key("C51", Site.TYPE_SATELLITE, [1.0e-4, 2.0e-4, 3.0e-4]),
        format_site_key("C51", Site.TYPE_SATELLITE, [2.0e-4, 3.0e-4, 4.0e-4]),
    ]
    optical_obs = optical_data(
        [2460003.0, 2460001.0, 2460002.0, 2460004.0],
        np.vstack([ground_values, space_values]),
        ["568", "000", *space_keys],
        [30, 10, 20, 40],
    )

    inputs = build_optical_iod_inputs(observation_data(optical_obs), sun)
    expected_tdb_jd = np.asarray([2460001.0, 2460002.0, 2460003.0, 2460004.0])
    expected_input_indices = np.asarray([10, 20, 30, 40])
    expected_values = np.vstack([ground_values[1], space_values[0], ground_values[0], space_values[1]])
    expected_site_pos = Site.from_code(optical_obs.rx_codes).state(
        optical_obs.t,
        frame=HELIO_ICRS,
        sun=sun,
        earth=EphemerisBody("earth", eph=default_ephemeris),
    ).pos[np.argsort(np.asarray(optical_obs.t.tdb().jd))]

    print(
        "[od.iod.inputs.mixed] "
        f"input_indices={jnp.asarray(inputs.input_indices)} "
        f"site_shape={inputs.site_pos.shape}"
    )

    assert inputs.tdb_jd.shape == (4,)
    assert inputs.site_pos.shape == (4, 3)
    assert inputs.los_unit.shape == (4, 3)
    assert_array_equal(inputs.input_indices, expected_input_indices)
    assert_allclose(inputs.tdb_jd, expected_tdb_jd, atol=0.0, rtol=0.0)
    assert_allclose(inputs.los_unit, sph2car(expected_values[:, 0], expected_values[:, 1]), atol=1.0e-15, rtol=0.0)
    assert_allclose(inputs.site_pos, expected_site_pos, atol=0.0, rtol=0.0)
    assert bool(jnp.all(jnp.isfinite(inputs.site_pos)))


def test_iod_inputs_empty(default_ephemeris):
    inputs = build_optical_iod_inputs(observation_data(), EphemerisBody("sun", eph=default_ephemeris))

    assert inputs.tdb_jd1.shape == (0,)
    assert inputs.tdb_jd2.shape == (0,)
    assert inputs.tdb_jd.shape == (0,)
    assert inputs.site_pos.shape == (0, 3)
    assert inputs.los_unit.shape == (0, 3)
    assert inputs.input_indices.shape == (0,)
