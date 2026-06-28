import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jrandom
import pytest

from difforb.od.iod.sampling import (
    DEFAULT_T2_SAMPLING_RADIUS,
    IODSamplingWindow,
    OpticalIODInputs,
    build_triplet_batch,
    resolve_sampling_window,
    sample_triplet_indices,
)
from tests.assertions import assert_array_equal


def test_sampling_window_centered():
    tdb_jd = jnp.asarray([2460000.0, 2460001.0, 2460002.0, 2460003.0, 2460004.0, 2460005.0, 2460006.0])

    window = resolve_sampling_window(tdb_jd, max_arc_days=4.0)

    assert window.start_idx == 1
    assert window.end_idx == 5
    assert window.center_idx == 3


def test_sampling_window_expands():
    tdb_jd = jnp.asarray([2460000.0, 2460010.0, 2460020.0, 2460030.0, 2460040.0])

    window = resolve_sampling_window(tdb_jd, max_arc_days=1.0)

    assert window.start_idx == 0
    assert window.end_idx == 4
    assert window.center_idx == 2


def test_sampling_window_requires_three():
    with pytest.raises(ValueError, match="at least 3 observations"):
        resolve_sampling_window(jnp.asarray([2460000.0, 2460001.0]), max_arc_days=1.0)


def test_sample_triplets_bounds():
    window = IODSamplingWindow(start_idx=2, end_idx=9, center_idx=5)

    triplets = sample_triplet_indices(window, num_candidates=64, solve_key=jrandom.PRNGKey(123))

    t2_min_idx = max(window.start_idx + 1, window.center_idx - DEFAULT_T2_SAMPLING_RADIUS)
    t2_max_idx = min(window.end_idx - 1, window.center_idx + DEFAULT_T2_SAMPLING_RADIUS)

    assert triplets.shape == (64, 3)
    assert bool(jnp.all(triplets[:, 0] >= window.start_idx))
    assert bool(jnp.all(triplets[:, 0] < triplets[:, 1]))
    assert bool(jnp.all(triplets[:, 1] < triplets[:, 2]))
    assert bool(jnp.all(triplets[:, 2] <= window.end_idx))
    assert bool(jnp.all(triplets[:, 1] >= t2_min_idx))
    assert bool(jnp.all(triplets[:, 1] <= t2_max_idx))


def test_triplet_batch():
    optical_inputs = OpticalIODInputs(
        tdb_jd1=jnp.asarray([2460000.0, 2460001.0, 2460002.0, 2460003.0, 2460004.0]),
        tdb_jd2=jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
        tdb_jd=jnp.asarray([2460000.1, 2460001.2, 2460002.3, 2460003.4, 2460004.5]),
        site_pos=jnp.arange(15, dtype=jnp.float64).reshape(5, 3),
        los_unit=(jnp.arange(15, dtype=jnp.float64).reshape(5, 3) + 100.0),
        input_indices=jnp.asarray([10, 12, 14, 16, 18], dtype=jnp.int32),
    )
    triplet_indices = jnp.asarray([[0, 2, 4], [1, 2, 3]], dtype=jnp.int32)

    batch = build_triplet_batch(optical_inputs, triplet_indices)

    assert_array_equal(batch.indices, triplet_indices)
    assert_array_equal(batch.site_pos, jnp.take(optical_inputs.site_pos, triplet_indices, axis=0))
    assert_array_equal(batch.los_unit, jnp.take(optical_inputs.los_unit, triplet_indices, axis=0))
    assert_array_equal(batch.tdb_jd1, jnp.take(optical_inputs.tdb_jd1, triplet_indices, axis=0))
    assert_array_equal(batch.tdb_jd2, jnp.take(optical_inputs.tdb_jd2, triplet_indices, axis=0))
    assert_array_equal(batch.input_indices, jnp.take(optical_inputs.input_indices, triplet_indices, axis=0))
