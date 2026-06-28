import erfa
import pytest
import jax
import jax.numpy as jnp

from difforb.core.constants import J2000, JULIAN_CENTURY
from difforb.core.earth_rotation.iau import (
    iau_bias_precession_nutation_matrix_single,
    iau_cio_locator_single,
    iau_cip_xyz_single,
    iau_mean_obliquity_single,
    iau_nutation_matrix_single,
    iau_precession_bias_matrix_single,
)
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)

TT_JD_CASES = [
    (2378496.5, 0.0, "1800-01-01 TT"),
    (2415020.5, 0.0, "1900-01-01 TT"),
    (2451545.0, 0.0, "J2000.0"),
    (2453005.25, 0.0, "2004-01-01 18:00:00 TT"),
    (2488070.5, 0.0, "2100-01-01 TT"),
    (2524594.5, 0.0, "2200-01-01 TT"),
]


@pytest.mark.parametrize("tt_jd1, tt_jd2, label", TT_JD_CASES)
def test_iau_mean_obliquity_single_against_erfa(tt_jd1, tt_jd2, label):
    t = ((tt_jd1 - J2000) + tt_jd2) / JULIAN_CENTURY

    actual = iau_mean_obliquity_single(t)
    expected = erfa.obl06(tt_jd1, tt_jd2)
    diff_rad = actual - expected

    print(
        "[iau_mean_obliquity_single] "
        f"label={label:<22} "
        f"diff={float(diff_rad):+.12e} rad"
    )

    assert_allclose(
        actual,
        expected,
        atol=1e-15,
        rtol=0.0,
        msg=f"IAU mean obliquity mismatch for {label}",
    )


@pytest.mark.parametrize("tt_jd1, tt_jd2, label", TT_JD_CASES)
def test_iau_cip_xyz_single_against_erfa(tt_jd1, tt_jd2, label):
    t = ((tt_jd1 - J2000) + tt_jd2) / JULIAN_CENTURY

    actual_x, actual_y, actual_z = iau_cip_xyz_single(t, 0.0, 0.0)
    expected_x, expected_y, _ = erfa.xys06a(tt_jd1, tt_jd2)
    expected_z = jnp.sqrt(1.0 - expected_x * expected_x - expected_y * expected_y)

    diff_x_rad = actual_x - expected_x
    diff_y_rad = actual_y - expected_y
    diff_z = actual_z - expected_z

    print(
        "[iau_cip_xyz_single] "
        f"label={label:<22} "
        f"dx={float(diff_x_rad):+.12e} rad "
        f"dy={float(diff_y_rad):+.12e} rad "
        f"dz={float(diff_z):+.12e}"
    )

    assert_allclose(actual_x, expected_x, atol=1e-12, rtol=0.0, msg=f"IAU CIP x mismatch for {label}")
    assert_allclose(actual_y, expected_y, atol=1e-12, rtol=0.0, msg=f"IAU CIP y mismatch for {label}")
    assert_allclose(actual_z, expected_z, atol=1e-12, rtol=0.0, msg=f"IAU CIP z mismatch for {label}")


@pytest.mark.parametrize("tt_jd1, tt_jd2, label", TT_JD_CASES)
def test_iau_cio_locator_single_against_erfa(tt_jd1, tt_jd2, label):
    t = ((tt_jd1 - J2000) + tt_jd2) / JULIAN_CENTURY

    expected_x, expected_y, expected_s = erfa.xys06a(tt_jd1, tt_jd2)
    actual_s = iau_cio_locator_single(t, expected_x, expected_y)
    diff_rad = actual_s - expected_s

    print(
        "[iau_cio_locator_single] "
        f"label={label:<22} "
        f"diff={float(diff_rad):+.12e} rad"
    )

    assert_allclose(
        actual_s,
        expected_s,
        atol=1e-12,
        rtol=0.0,
        msg=f"IAU CIO locator mismatch for {label}",
    )


@pytest.mark.parametrize("tt_jd1, tt_jd2, label", TT_JD_CASES)
def test_iau_precession_bias_matrix_single_against_erfa(tt_jd1, tt_jd2, label):
    t = ((tt_jd1 - J2000) + tt_jd2) / JULIAN_CENTURY

    actual = iau_precession_bias_matrix_single(t)
    expected = jnp.asarray(erfa.pmat06(tt_jd1, tt_jd2))
    max_abs_diff = jnp.max(jnp.abs(actual - expected))

    print(
        "[iau_precession_bias_matrix_single] "
        f"label={label:<22} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        expected,
        atol=1e-15,
        rtol=0.0,
        msg=f"IAU precession-bias matrix mismatch for {label}",
    )


@pytest.mark.parametrize("tt_jd1, tt_jd2, label", TT_JD_CASES)
def test_iau_nutation_matrix_single_against_erfa(tt_jd1, tt_jd2, label):
    t = ((tt_jd1 - J2000) + tt_jd2) / JULIAN_CENTURY

    actual = iau_nutation_matrix_single(t)
    expected = jnp.asarray(erfa.num06a(tt_jd1, tt_jd2))
    max_abs_diff = jnp.max(jnp.abs(actual - expected))

    print(
        "[iau_nutation_matrix_single] "
        f"label={label:<22} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        expected,
        atol=1e-15,
        rtol=0.0,
        msg=f"IAU nutation matrix mismatch for {label}",
    )


@pytest.mark.parametrize("tt_jd1, tt_jd2, label", TT_JD_CASES)
def test_iau_bias_precession_nutation_matrix_single_against_erfa(tt_jd1, tt_jd2, label):
    t = ((tt_jd1 - J2000) + tt_jd2) / JULIAN_CENTURY

    actual = iau_bias_precession_nutation_matrix_single(t)
    expected = jnp.asarray(erfa.pnm06a(tt_jd1, tt_jd2))
    max_abs_diff = jnp.max(jnp.abs(actual - expected))

    print(
        "[iau_bias_precession_nutation_matrix_single] "
        f"label={label:<22} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        expected,
        atol=1e-15,
        rtol=0.0,
        msg=f"IAU bias-precession-nutation matrix mismatch for {label}",
    )
