import jax
import jax.numpy as jnp
import numpy as np
import pytest
from equinox import EquinoxRuntimeError

from difforb.core.constants import DAY_S, J2000
from difforb.spk.spk import (
    Ephemeris,
    MergedSegment,
    TTMinusTDBKernel,
    compute_chebyshev_polynomial,
    compute_position,
    compute_position_single,
    compute_pv,
    compute_pv_single,
    compute_pva,
    compute_pva_single,
    pad_arrays,
)
from tests.assertions import assert_allclose, assert_array_equal

jax.config.update("jax_enable_x64", True)

J2000_JD = float(J2000)
DAY_SECONDS = float(DAY_S)

# Coefficients are ordered from highest Chebyshev degree to constant term.
# They represent:
# x = 10 + 2*T1(s) + 3*T2(s), y = -4 + 5*T1(s), z = 7.
CHEB_COEFFICIENTS = jnp.asarray(
    [
        [3.0, 0.0, 0.0],
        [2.0, 5.0, 0.0],
        [10.0, -4.0, 7.0],
    ],
    dtype=float,
)
SEGMENT_COEFFICIENTS = CHEB_COEFFICIENTS[jnp.newaxis, ..., jnp.newaxis]


def test_pad_arrays():
    first = np.ones((2, 3, 1), dtype=float)
    second = np.full((2, 3, 3), 2.0, dtype=float)

    actual = pad_arrays([first, second])

    assert actual.shape == (2, 2, 3, 3)
    assert_array_equal(actual[0, ..., 0], first[..., 0])
    assert_array_equal(actual[0, ..., 1:], np.zeros((2, 3, 2), dtype=float))
    assert_array_equal(actual[1], second)


def test_compute_chebyshev_polynomial():
    scale = 0.25
    actual = compute_chebyshev_polynomial(scale, CHEB_COEFFICIENTS)
    expected = jnp.asarray(
        [
            10.0 + 2.0 * scale + 3.0 * (2.0 * scale ** 2 - 1.0),
            -4.0 + 5.0 * scale,
            7.0,
        ],
        dtype=float,
    )

    print(
        "[spk.chebyshev] "
        f"scale={scale:+.3f} "
        f"max_abs_diff={float(jnp.max(jnp.abs(actual - expected))):+.12e}"
    )

    assert_allclose(actual, expected, atol=0.0, rtol=0.0)


def test_compute_position_single():
    chunk_offset = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [100.0, 200.0, 300.0],
        ],
        dtype=float,
    )
    coefficients = jnp.stack(
        [CHEB_COEFFICIENTS, CHEB_COEFFICIENTS + chunk_offset],
        axis=-1,
    )[jnp.newaxis, ...]
    scale = -0.5

    actual = compute_position_single(
        jnp.asarray([0.0], dtype=float),
        jnp.asarray([2]),
        jnp.asarray([DAY_SECONDS], dtype=float),
        coefficients,
        J2000_JD,
        1.25,
    )
    expected = jnp.asarray(
        [
            110.0 + 2.0 * scale + 3.0 * (2.0 * scale ** 2 - 1.0),
            196.0 + 5.0 * scale,
            307.0,
        ],
        dtype=float,
    )

    print(
        "[spk.compute_position_single] "
        f"max_abs_diff={float(jnp.max(jnp.abs(actual - expected))):+.12e} km"
    )

    assert_allclose(actual, expected, atol=0.0, rtol=0.0)


def test_compute_pv_single():
    scale = -0.5
    actual_pos, actual_vel = compute_pv_single(
        jnp.asarray([0.0], dtype=float),
        jnp.asarray([1]),
        jnp.asarray([DAY_SECONDS], dtype=float),
        SEGMENT_COEFFICIENTS,
        J2000_JD,
        0.25,
    )
    expected_pos = jnp.asarray(
        [
            10.0 + 2.0 * scale + 3.0 * (2.0 * scale ** 2 - 1.0),
            -4.0 + 5.0 * scale,
            7.0,
        ],
        dtype=float,
    )
    expected_vel = jnp.asarray(
        [
            (2.0 + 4.0 * 3.0 * scale) * 2.0,
            5.0 * 2.0,
            0.0,
        ],
        dtype=float,
    )

    print(
        "[spk.compute_pv_single] "
        f"pos_max_abs_diff={float(jnp.max(jnp.abs(actual_pos - expected_pos))):+.12e} km "
        f"vel_max_abs_diff={float(jnp.max(jnp.abs(actual_vel - expected_vel))):+.12e} km/day"
    )

    assert_allclose(actual_pos, expected_pos, atol=0.0, rtol=0.0)
    assert_allclose(actual_vel, expected_vel, atol=1.0e-14, rtol=0.0)


def test_compute_pva_single():
    scale = 0.5
    actual_pos, actual_vel, actual_acc = compute_pva_single(
        jnp.asarray([0.0], dtype=float),
        jnp.asarray([1]),
        jnp.asarray([DAY_SECONDS], dtype=float),
        SEGMENT_COEFFICIENTS,
        J2000_JD,
        0.75,
    )
    expected_pos = jnp.asarray(
        [
            10.0 + 2.0 * scale + 3.0 * (2.0 * scale ** 2 - 1.0),
            -4.0 + 5.0 * scale,
            7.0,
        ],
        dtype=float,
    )
    expected_vel = jnp.asarray(
        [
            (2.0 + 4.0 * 3.0 * scale) * 2.0,
            5.0 * 2.0,
            0.0,
        ],
        dtype=float,
    )
    expected_acc = jnp.asarray([16.0 * 3.0, 0.0, 0.0], dtype=float)

    print(
        "[spk.compute_pva_single] "
        f"pos_max_abs_diff={float(jnp.max(jnp.abs(actual_pos - expected_pos))):+.12e} km "
        f"vel_max_abs_diff={float(jnp.max(jnp.abs(actual_vel - expected_vel))):+.12e} km/day "
        f"acc_max_abs_diff={float(jnp.max(jnp.abs(actual_acc - expected_acc))):+.12e} km/day^2"
    )

    assert_allclose(actual_pos, expected_pos, atol=0.0, rtol=0.0)
    assert_allclose(actual_vel, expected_vel, atol=1.0e-14, rtol=0.0)
    assert_allclose(actual_acc, expected_acc, atol=1.0e-13, rtol=0.0)


def test_compute_position_pv_pva_batch_shapes():
    tdb_jd1 = jnp.asarray([J2000_JD, J2000_JD], dtype=float)
    tdb_jd2 = jnp.asarray([0.25, 0.75], dtype=float)

    pos = compute_position(
        jnp.asarray([0.0], dtype=float),
        jnp.asarray([1]),
        jnp.asarray([DAY_SECONDS], dtype=float),
        SEGMENT_COEFFICIENTS,
        tdb_jd1,
        tdb_jd2,
    )
    pv_pos, pv_vel = compute_pv(
        jnp.asarray([0.0], dtype=float),
        jnp.asarray([1]),
        jnp.asarray([DAY_SECONDS], dtype=float),
        SEGMENT_COEFFICIENTS,
        tdb_jd1,
        tdb_jd2,
    )
    pva_pos, pva_vel, pva_acc = compute_pva(
        jnp.asarray([0.0], dtype=float),
        jnp.asarray([1]),
        jnp.asarray([DAY_SECONDS], dtype=float),
        SEGMENT_COEFFICIENTS,
        tdb_jd1,
        tdb_jd2,
    )

    print(
        "[spk.compute_batch_shapes] "
        f"pos_shape={pos.shape} "
        f"vel_shape={pv_vel.shape} "
        f"acc_shape={pva_acc.shape}"
    )

    assert pos.shape == (2, 3)
    assert pv_pos.shape == (2, 3)
    assert pv_vel.shape == (2, 3)
    assert pva_pos.shape == (2, 3)
    assert pva_vel.shape == (2, 3)
    assert pva_acc.shape == (2, 3)
    assert_allclose(pv_pos, pos, atol=0.0, rtol=0.0)
    assert_allclose(pva_pos, pos, atol=0.0, rtol=0.0)
    assert_allclose(pva_vel, pv_vel, atol=0.0, rtol=0.0)


def test_merged_segment_pos_state_pva():
    segment = MergedSegment(
        types=jnp.asarray([2]),
        center_ids=jnp.asarray([0]),
        target_ids=jnp.asarray([10]),
        seg_starts_sec=jnp.asarray([0.0], dtype=float),
        seg_ends_sec=jnp.asarray([DAY_SECONDS], dtype=float),
        component_nums=jnp.asarray([3]),
        chunk_nums=jnp.asarray([1]),
        chunk_lengths_sec=jnp.asarray([DAY_SECONDS], dtype=float),
        coefficients=SEGMENT_COEFFICIENTS,
    )
    pos = segment.pos(J2000_JD, 0.25)
    state_pos, state_vel = segment.state(J2000_JD, 0.25)
    pva_pos, pva_vel, pva_acc = segment.pva(J2000_JD, 0.25)

    print(
        "[spk.merged_segment] "
        f"pos_shape={pos.shape} "
        f"covered={bool(segment.is_covered(J2000_JD + 0.5))}"
    )

    assert pos.shape == (3,)
    assert_allclose(state_pos, pos, atol=0.0, rtol=0.0)
    assert_allclose(pva_pos, pos, atol=0.0, rtol=0.0)
    assert_allclose(pva_vel, state_vel, atol=0.0, rtol=0.0)
    assert pva_acc.shape == (3,)
    assert bool(segment.is_covered(J2000_JD + 0.5))
    assert not bool(segment.is_covered(J2000_JD - 0.1))


def test_merged_segment_rejects_epochs_outside_loaded_coverage():
    segment = MergedSegment(
        types=jnp.asarray([2]),
        center_ids=jnp.asarray([0]),
        target_ids=jnp.asarray([10]),
        seg_starts_sec=jnp.asarray([0.0], dtype=float),
        seg_ends_sec=jnp.asarray([DAY_SECONDS], dtype=float),
        component_nums=jnp.asarray([3]),
        chunk_nums=jnp.asarray([1]),
        chunk_lengths_sec=jnp.asarray([DAY_SECONDS], dtype=float),
        coefficients=SEGMENT_COEFFICIENTS,
    )
    message = "outside the loaded SPK coverage"

    with pytest.raises(EquinoxRuntimeError, match=message):
        segment.pos(J2000_JD, -0.25)
    with pytest.raises(EquinoxRuntimeError, match=message):
        segment.state(J2000_JD, 1.25)
    with pytest.raises(EquinoxRuntimeError, match=message):
        segment.pva(jnp.asarray([J2000_JD, J2000_JD]), jnp.asarray([0.25, 2.0]))


def test_merged_segment_is_covered_uses_any_loaded_segment():
    segment = MergedSegment(
        types=jnp.asarray([2, 2]),
        center_ids=jnp.asarray([0, 0]),
        target_ids=jnp.asarray([10, 10]),
        seg_starts_sec=jnp.asarray([0.0, 10.0 * DAY_SECONDS], dtype=float),
        seg_ends_sec=jnp.asarray([DAY_SECONDS, 11.0 * DAY_SECONDS], dtype=float),
        component_nums=jnp.asarray([3, 3]),
        chunk_nums=jnp.asarray([1, 1]),
        chunk_lengths_sec=jnp.asarray([DAY_SECONDS, DAY_SECONDS], dtype=float),
        coefficients=jnp.concatenate([SEGMENT_COEFFICIENTS, SEGMENT_COEFFICIENTS], axis=0),
    )

    actual = segment.is_covered(jnp.asarray([J2000_JD + 0.5, J2000_JD + 10.5, J2000_JD + 5.0]))

    assert_array_equal(actual, jnp.asarray([True, True, False]))


def test_tt_minus_tdb_kernel():
    segment = MergedSegment(
        types=jnp.asarray([2]),
        center_ids=jnp.asarray([1000000000]),
        target_ids=jnp.asarray([1000000001]),
        seg_starts_sec=jnp.asarray([0.0], dtype=float),
        seg_ends_sec=jnp.asarray([DAY_SECONDS], dtype=float),
        component_nums=jnp.asarray([3]),
        chunk_nums=jnp.asarray([1]),
        chunk_lengths_sec=jnp.asarray([DAY_SECONDS], dtype=float),
        coefficients=SEGMENT_COEFFICIENTS,
    )
    kernel = TTMinusTDBKernel(segment)
    scale = -0.5
    expected_offset = 10.0 + 2.0 * scale + 3.0 * (2.0 * scale ** 2 - 1.0)
    expected_rate = ((2.0 + 4.0 * 3.0 * scale) * 2.0) / DAY_SECONDS

    actual_offset = kernel.tt_minus_tdb(J2000_JD, 0.25)
    actual_rate = kernel.dtt_minus_tdb_dtdb(J2000_JD, 0.25)

    print(
        "[spk.tt_minus_tdb_kernel] "
        f"offset_diff={float(actual_offset - expected_offset):+.12e} s "
        f"rate_diff={float(actual_rate - expected_rate):+.12e}"
    )

    assert_allclose(actual_offset, expected_offset, atol=0.0, rtol=0.0)
    assert_allclose(actual_rate, expected_rate, atol=1.0e-18, rtol=0.0)


def test_ephemeris_load_body_path_signs(monkeypatch):
    eph = Ephemeris.__new__(Ephemeris)
    eph.graph = {
        "SOLAR SYSTEM BARYCENTER": ["EARTH"],
        "EARTH": ["SOLAR SYSTEM BARYCENTER", "MOON"],
        "MOON": ["EARTH"],
    }

    def fake_load_path(center_name, target_name):
        if (center_name, target_name) == ("SOLAR SYSTEM BARYCENTER", "EARTH"):
            return "ssb_to_earth"
        if (center_name, target_name) == ("MOON", "EARTH"):
            return "moon_to_earth"
        raise ValueError("missing directed path")

    monkeypatch.setattr(eph, "_load_path", fake_load_path)

    segments, signs = eph.load_body("moon")

    assert segments == ("ssb_to_earth", "moon_to_earth")
    assert signs == (1.0, -1.0)
