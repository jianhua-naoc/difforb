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
from difforb.core.time.timescale import Time
from difforb.od.dc.bucket import DCBucketPolicy, pack_dc_observations
from tests.assertions import assert_array_equal


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


def optical_data(count: int, *, input_offset: int = 0):
    return OpticalObservationData(
        t=Time.from_tdb_jd(jnp.asarray(2460000.0 + np.arange(count), dtype=float), jnp.zeros(count, dtype=float)),
        trk_ids=np.asarray([f"T{i}" for i in range(count)], dtype=object),
        obs_type_ids=np.full(count, ObsType.OPTICAL.id, dtype=int),
        obs_mode_ids=np.full(count, ObsMode.CCD.id, dtype=int),
        values=np.deg2rad(np.column_stack([np.arange(count, dtype=float), np.arange(count, dtype=float) + 10.0])),
        uncertainties=np.full((count, 2), np.deg2rad(1.0 / 3600.0), dtype=float),
        correlations=np.zeros(count, dtype=float),
        time_uncertainties=np.full(count, np.nan, dtype=float),
        rx_codes=np.asarray(["568"] * count, dtype=str),
        program_codes=np.asarray([""] * count, dtype=object),
        catalog_codes=np.asarray([""] * count, dtype=object),
        note_codes=np.asarray([""] * count, dtype=object),
        magnitudes=np.full(count, np.nan, dtype=float),
        band_codes=np.asarray([""] * count, dtype=object),
        sub_frames=np.asarray([""] * count, dtype=object),
        input_indices=np.arange(input_offset, input_offset + count, dtype=int),
    )


def radar_data(count: int, *, input_offset: int = 100):
    return RadarObservationData(
        t=Time.from_tdb_jd(jnp.asarray(2460100.0 + np.arange(count), dtype=float), jnp.zeros(count, dtype=float)),
        obs_type_ids=np.full(count, ObsType.RADAR.id, dtype=int),
        obs_mode_ids=np.full(count, ObsMode.DELAY_CENTER.id, dtype=int),
        values=np.arange(count, dtype=float),
        uncertainties=np.full(count, 1.0e-6, dtype=float),
        rx_codes=np.asarray(["Arecibo"] * count, dtype=str),
        tx_codes=np.asarray(["Arecibo"] * count, dtype=str),
        tx_freq=np.full(count, 2.38e9, dtype=float),
        input_indices=np.arange(input_offset, input_offset + count, dtype=int),
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


def test_pack_dc_observations_pads_non_empty_modalities():
    data = ObservationData(
        name="synthetic",
        optical=optical_data(3),
        radar=radar_data(2),
    )

    packed = pack_dc_observations(
        data,
        DCBucketPolicy(optical_buckets=(4,), radar_buckets=(4,)),
    )

    assert packed.data.num_optical == 4
    assert packed.data.num_radar == 4
    assert packed.n_optical == 3
    assert packed.n_radar == 2
    assert packed.is_padded
    assert packed.data.optical.input_indices[-1] < 0
    assert packed.data.radar.input_indices[-1] < 0
    assert_array_equal(
        packed.flat_valid_mask,
        jnp.asarray([True, True, True, True, True, True, False, False, True, True, False, False]),
    )
    assert_array_equal(
        packed.observation_valid_mask,
        jnp.asarray([True, True, True, False, True, True, False, False]),
    )
    assert_array_equal(packed.flat_valid_indices, np.asarray([0, 1, 2, 3, 4, 5, 8, 9]))


def test_pack_dc_observations_keeps_exact_bucket_shapes():
    data = ObservationData(
        name="synthetic",
        optical=optical_data(4),
        radar=empty_radar_data(),
    )

    packed = pack_dc_observations(
        data,
        DCBucketPolicy(optical_buckets=(4,), radar_buckets=(4,)),
    )

    assert not packed.is_padded
    assert packed.data.optical is data.optical
    assert_array_equal(packed.flat_valid_mask, jnp.ones(8, dtype=bool))
