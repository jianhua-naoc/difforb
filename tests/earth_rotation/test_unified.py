import erfa
import jax
import jax.numpy as jnp
import pytest

from difforb.core.constants import J2000, JULIAN_CENTURY
from difforb.core.earth_rotation.data import SHORT_TERM_END, SHORT_TERM_START
from difforb.core.earth_rotation.iau import (
    iau_bias_precession_nutation_matrix_single,
    iau_cio_locator_single,
    iau_cip_xyz_single,
    iau_mean_obliquity_single,
    iau_nutation_matrix_single,
    iau_precession_bias_matrix_single,
)
from difforb.core.earth_rotation.unified import (
    bias_precession_nutation_matrix_single,
    cirs_to_gcrs_matrix_single,
    cip_xyz_single,
    cio_locator_single,
    earth_rotation_angle_single,
    gcrs_to_cirs_matrix_single,
    inversed_polar_motion_matrix_single,
    mean_obliquity_single,
    nutation_matrix_single,
    polar_motion_matrix_single,
    precession_bias_matrix_single,
    tio_locator_single,
)
from difforb.core.earth_rotation.vondrak import (
    vondrak_cio_locator_single,
    vondrak_cip_xyz_single,
    vondrak_mean_obliquity_single,
    vondrak_precession_bias_matrix_single,
)
from difforb.utils import R3_single, arcsec_to_rad
from tests.assertions import assert_allclose
from tests.earth_rotation.test_iau import TT_JD_CASES

jax.config.update("jax_enable_x64", True)

UNIFIED_ROUTING_CASES = [
    (0.0, "short-term interior", "iau"),
    (SHORT_TERM_START, "short-term start boundary", "iau"),
    (SHORT_TERM_END, "short-term end boundary", "iau"),
    (SHORT_TERM_START - 1.0, "before short-term range", "vondrak"),
    (SHORT_TERM_END + 1.0, "after short-term range", "vondrak"),
]

UT1_JD_CASES = [
    (2451545.0, 0.0, "J2000.0 UT1"),
    (2453005.0, 0.25, "2004-01-01 18:00:00 UT1"),
    (2453005.5, -0.25, "renormalized split UT1"),
    (2488070.0, 0.5, "2100-01-01 00:00:00 UT1"),
]

POLAR_MOTION_CASES = [
    (0.0, 0.0, "zero pole"),
    (0.2, -0.3, "moderate arcsec"),
    (-0.05, 0.125, "mixed arcsec"),
]


# -------------------------------------------------------------------------
# Helper constructors
# -------------------------------------------------------------------------


def _assemble_gcrs_to_cirs_matrix(x, y, z, s):
    a = 1.0 / (1.0 + z)
    x2 = x * x
    y2 = y * y
    axy = a * x * y
    r_sigma = jnp.array([
        [1.0 - a * x2, -axy, -x],
        [-axy, 1.0 - a * y2, -y],
        [x, y, 1.0 - a * (x2 + y2)],
    ])
    return R3_single(-s) @ r_sigma


def _assemble_cirs_to_gcrs_matrix(x, y, z, s):
    a = 1.0 / (1.0 + z)
    x2 = x * x
    y2 = y * y
    axy = a * x * y
    r_sigma = jnp.array([
        [1.0 - a * x2, -axy, x],
        [-axy, 1.0 - a * y2, y],
        [-x, -y, 1.0 - a * (x2 + y2)],
    ])
    return r_sigma @ R3_single(s)


# -------------------------------------------------------------------------
# Unified routing tests
# -------------------------------------------------------------------------


@pytest.mark.parametrize("t, label, branch", UNIFIED_ROUTING_CASES)
def test_mean_obliquity_single_routes_to_expected_model(t, label, branch):
    actual = mean_obliquity_single(t)
    if branch == "iau":
        expected = iau_mean_obliquity_single(t)
    else:
        expected = vondrak_mean_obliquity_single(t)

    diff_rad = actual - expected

    print(
        "[unified.mean_obliquity_single] "
        f"label={label:<26} "
        f"branch={branch:<7} "
        f"diff={float(diff_rad):+.12e} rad"
    )

    assert_allclose(
        actual,
        expected,
        atol=0.0,
        rtol=0.0,
        msg=f"Unified mean obliquity routed to the wrong model for {label}",
    )


@pytest.mark.parametrize("t, label, branch", UNIFIED_ROUTING_CASES)
def test_cip_xyz_single_routes_to_expected_model(t, label, branch):
    cor_delta_obliquity = 0.0
    cor_delta_longitude = 0.0

    actual_x, actual_y, actual_z = cip_xyz_single(t, cor_delta_obliquity, cor_delta_longitude)
    if branch == "iau":
        expected_x, expected_y, expected_z = iau_cip_xyz_single(t, cor_delta_obliquity, cor_delta_longitude)
    else:
        expected_x, expected_y, expected_z = vondrak_cip_xyz_single(t)

    actual = jnp.array([actual_x, actual_y, actual_z])
    expected = jnp.array([expected_x, expected_y, expected_z])
    max_abs_diff = jnp.max(jnp.abs(actual - expected))

    print(
        "[unified.cip_xyz_single] "
        f"label={label:<26} "
        f"branch={branch:<7} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        expected,
        atol=0.0,
        rtol=0.0,
        msg=f"Unified CIP xyz routed to the wrong model for {label}",
    )


@pytest.mark.parametrize("t, label, branch", UNIFIED_ROUTING_CASES)
def test_cio_locator_single_routes_to_expected_model(t, label, branch):
    cor_delta_obliquity = 0.0
    cor_delta_longitude = 0.0

    if branch == "iau":
        x, y, _ = iau_cip_xyz_single(t, cor_delta_obliquity, cor_delta_longitude)
        expected = iau_cio_locator_single(t, x, y)
    else:
        x, y, _ = vondrak_cip_xyz_single(t)
        expected = vondrak_cio_locator_single(t)

    actual = cio_locator_single(t, x, y)
    diff_rad = actual - expected

    print(
        "[unified.cio_locator_single] "
        f"label={label:<26} "
        f"branch={branch:<7} "
        f"diff={float(diff_rad):+.12e} rad"
    )

    assert_allclose(
        actual,
        expected,
        atol=0.0,
        rtol=0.0,
        msg=f"Unified CIO locator routed to the wrong model for {label}",
    )


@pytest.mark.parametrize("t, label, branch", UNIFIED_ROUTING_CASES)
def test_precession_bias_matrix_single_routes_to_expected_model(t, label, branch):
    actual = precession_bias_matrix_single(t)
    if branch == "iau":
        expected = iau_precession_bias_matrix_single(t)
    else:
        expected = vondrak_precession_bias_matrix_single(t)

    max_abs_diff = jnp.max(jnp.abs(actual - expected))

    print(
        "[unified.precession_bias_matrix_single] "
        f"label={label:<26} "
        f"branch={branch:<7} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        expected,
        atol=0.0,
        rtol=0.0,
        msg=f"Unified precession-bias matrix routed to the wrong model for {label}",
    )


@pytest.mark.parametrize("t, label, branch", UNIFIED_ROUTING_CASES)
def test_nutation_matrix_single_routes_to_expected_model(t, label, branch):
    actual = nutation_matrix_single(t)
    if branch == "iau":
        expected = iau_nutation_matrix_single(t)
    else:
        expected = jnp.eye(3)

    max_abs_diff = jnp.max(jnp.abs(actual - expected))

    print(
        "[unified.nutation_matrix_single] "
        f"label={label:<26} "
        f"branch={branch:<7} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        expected,
        atol=0.0,
        rtol=0.0,
        msg=f"Unified nutation matrix routed to the wrong model for {label}",
    )


@pytest.mark.parametrize("t, label, branch", UNIFIED_ROUTING_CASES)
def test_bias_precession_nutation_matrix_single_routes_to_expected_model(t, label, branch):
    actual = bias_precession_nutation_matrix_single(t)
    if branch == "iau":
        expected = iau_bias_precession_nutation_matrix_single(t)
    else:
        expected = vondrak_precession_bias_matrix_single(t)

    max_abs_diff = jnp.max(jnp.abs(actual - expected))

    print(
        "[unified.bias_precession_nutation_matrix_single] "
        f"label={label:<26} "
        f"branch={branch:<7} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        expected,
        atol=0.0,
        rtol=0.0,
        msg=f"Unified bias-precession-nutation matrix routed to the wrong model for {label}",
    )


@pytest.mark.parametrize("t, label, branch", UNIFIED_ROUTING_CASES)
def test_gcrs_to_cirs_matrix_single_routes_to_expected_model(t, label, branch):
    cor_delta_obliquity = 0.0
    cor_delta_longitude = 0.0

    actual = gcrs_to_cirs_matrix_single(t, cor_delta_obliquity, cor_delta_longitude)
    if branch == "iau":
        x, y, z = iau_cip_xyz_single(t, cor_delta_obliquity, cor_delta_longitude)
        s = iau_cio_locator_single(t, x, y)
    else:
        x, y, z = vondrak_cip_xyz_single(t)
        s = vondrak_cio_locator_single(t)
    expected = _assemble_gcrs_to_cirs_matrix(x, y, z, s)

    max_abs_diff = jnp.max(jnp.abs(actual - expected))

    print(
        "[unified.gcrs_to_cirs_matrix_single] "
        f"label={label:<26} "
        f"branch={branch:<7} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        expected,
        atol=0.0,
        rtol=0.0,
        msg=f"Unified GCRS-to-CIRS matrix routed to the wrong model for {label}",
    )


@pytest.mark.parametrize("t, label, branch", UNIFIED_ROUTING_CASES)
def test_cirs_to_gcrs_matrix_single_routes_to_expected_model(t, label, branch):
    cor_delta_obliquity = 0.0
    cor_delta_longitude = 0.0

    actual = cirs_to_gcrs_matrix_single(t, cor_delta_obliquity, cor_delta_longitude)
    if branch == "iau":
        x, y, z = iau_cip_xyz_single(t, cor_delta_obliquity, cor_delta_longitude)
        s = iau_cio_locator_single(t, x, y)
    else:
        x, y, z = vondrak_cip_xyz_single(t)
        s = vondrak_cio_locator_single(t)
    expected = _assemble_cirs_to_gcrs_matrix(x, y, z, s)

    max_abs_diff = jnp.max(jnp.abs(actual - expected))

    print(
        "[unified.cirs_to_gcrs_matrix_single] "
        f"label={label:<26} "
        f"branch={branch:<7} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        expected,
        atol=0.0,
        rtol=0.0,
        msg=f"Unified CIRS-to-GCRS matrix routed to the wrong model for {label}",
    )


# -------------------------------------------------------------------------
# ERFA external truth tests
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tt_jd1, tt_jd2, label", TT_JD_CASES)
def test_gcrs_to_cirs_matrix_single_short_term_against_erfa(tt_jd1, tt_jd2, label):
    t = ((tt_jd1 - J2000) + tt_jd2) / JULIAN_CENTURY

    actual = gcrs_to_cirs_matrix_single(t, 0.0, 0.0)
    x, y, s = erfa.xys06a(tt_jd1, tt_jd2)
    expected = jnp.asarray(erfa.c2ixys(x, y, s))
    max_abs_diff = jnp.max(jnp.abs(actual - expected))

    print(
        "[unified.gcrs_to_cirs_matrix_single.erfa] "
        f"label={label:<22} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        expected,
        atol=1e-15,
        rtol=0.0,
        msg=f"Unified short-term GCRS-to-CIRS matrix mismatch for {label}",
    )


@pytest.mark.parametrize("tt_jd1, tt_jd2, label", TT_JD_CASES)
def test_cirs_to_gcrs_matrix_single_short_term_against_erfa(tt_jd1, tt_jd2, label):
    t = ((tt_jd1 - J2000) + tt_jd2) / JULIAN_CENTURY

    actual = cirs_to_gcrs_matrix_single(t, 0.0, 0.0)
    x, y, s = erfa.xys06a(tt_jd1, tt_jd2)
    expected = jnp.asarray(erfa.c2ixys(x, y, s)).T
    max_abs_diff = jnp.max(jnp.abs(actual - expected))

    print(
        "[unified.cirs_to_gcrs_matrix_single.erfa] "
        f"label={label:<22} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        expected,
        atol=1e-15,
        rtol=0.0,
        msg=f"Unified short-term CIRS-to-GCRS matrix mismatch for {label}",
    )


@pytest.mark.parametrize("tt_jd1, tt_jd2, label", TT_JD_CASES)
def test_tio_locator_single_against_erfa(tt_jd1, tt_jd2, label):
    t = ((tt_jd1 - J2000) + tt_jd2) / JULIAN_CENTURY

    actual = tio_locator_single(t)
    expected = erfa.sp00(tt_jd1, tt_jd2)
    diff_rad = actual - expected

    print(
        "[unified.tio_locator_single.erfa] "
        f"label={label:<22} "
        f"diff={float(diff_rad):+.12e} rad"
    )

    assert_allclose(
        actual,
        expected,
        atol=1e-15,
        rtol=0.0,
        msg=f"TIO locator mismatch for {label}",
    )


@pytest.mark.parametrize("ut1_jd1, ut1_jd2, label", UT1_JD_CASES)
def test_earth_rotation_angle_single_against_erfa(ut1_jd1, ut1_jd2, label):
    actual = earth_rotation_angle_single(ut1_jd1, ut1_jd2)
    expected = erfa.era00(ut1_jd1, ut1_jd2)
    diff_rad = actual - expected

    print(
        "[unified.earth_rotation_angle_single.erfa] "
        f"label={label:<22} "
        f"diff={float(diff_rad):+.12e} rad"
    )

    assert_allclose(
        actual,
        expected,
        atol=5e-15,
        rtol=0.0,
        msg=f"Earth rotation angle mismatch for {label}",
    )


def test_earth_rotation_angle_single_is_split_invariant():
    jd = 2453005.25
    actual_a = earth_rotation_angle_single(2453005.0, 0.25)
    actual_b = earth_rotation_angle_single(2453004.5, 0.75)
    actual_c = earth_rotation_angle_single(jd, 0.0)

    max_abs_diff = jnp.max(jnp.abs(jnp.array([actual_a - actual_b, actual_a - actual_c])))

    print(
        "[unified.earth_rotation_angle_single.split] "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(actual_a, actual_b, atol=5e-15, rtol=0.0)
    assert_allclose(actual_a, actual_c, atol=5e-15, rtol=0.0)


@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TT_JD_CASES)
@pytest.mark.parametrize("xp_arcsec, yp_arcsec, pole_label", POLAR_MOTION_CASES)
def test_polar_motion_matrix_single_against_erfa(tt_jd1, tt_jd2, time_label, xp_arcsec, yp_arcsec, pole_label):
    t = ((tt_jd1 - J2000) + tt_jd2) / JULIAN_CENTURY
    xp = arcsec_to_rad(xp_arcsec)
    yp = arcsec_to_rad(yp_arcsec)

    actual = polar_motion_matrix_single(t, xp, yp)
    sp = erfa.sp00(tt_jd1, tt_jd2)
    expected = jnp.asarray(erfa.pom00(xp, yp, sp))
    max_abs_diff = jnp.max(jnp.abs(actual - expected))

    print(
        "[unified.polar_motion_matrix_single.erfa] "
        f"time_label={time_label:<22} "
        f"pole_label={pole_label:<15} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        expected,
        atol=1e-15,
        rtol=0.0,
        msg=f"Polar motion matrix mismatch for {time_label} / {pole_label}",
    )


@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TT_JD_CASES)
@pytest.mark.parametrize("xp_arcsec, yp_arcsec, pole_label", POLAR_MOTION_CASES)
def test_inversed_polar_motion_matrix_single_against_erfa(tt_jd1, tt_jd2, time_label, xp_arcsec, yp_arcsec, pole_label):
    t = ((tt_jd1 - J2000) + tt_jd2) / JULIAN_CENTURY
    xp = arcsec_to_rad(xp_arcsec)
    yp = arcsec_to_rad(yp_arcsec)

    actual = inversed_polar_motion_matrix_single(t, xp, yp)
    sp = erfa.sp00(tt_jd1, tt_jd2)
    expected = jnp.asarray(erfa.pom00(xp, yp, sp)).T
    max_abs_diff = jnp.max(jnp.abs(actual - expected))

    print(
        "[unified.inversed_polar_motion_matrix_single.erfa] "
        f"time_label={time_label:<22} "
        f"pole_label={pole_label:<15} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        expected,
        atol=1e-15,
        rtol=0.0,
        msg=f"Inverse polar motion matrix mismatch for {time_label} / {pole_label}",
    )


# -------------------------------------------------------------------------
# Consistency tests
# -------------------------------------------------------------------------


@pytest.mark.parametrize("t, label, branch", UNIFIED_ROUTING_CASES)
def test_gcrs_to_cirs_and_cirs_to_gcrs_are_inverses(t, label, branch):
    forward = gcrs_to_cirs_matrix_single(t, 0.0, 0.0)
    backward = cirs_to_gcrs_matrix_single(t, 0.0, 0.0)
    identity = jnp.eye(3)
    left_product = backward @ forward
    right_product = forward @ backward
    max_abs_diff = jnp.maximum(
        jnp.max(jnp.abs(left_product - identity)),
        jnp.max(jnp.abs(right_product - identity)),
    )

    print(
        "[unified.cirs_gcrs_inverse] "
        f"label={label:<26} "
        f"branch={branch:<7} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        left_product,
        identity,
        atol=1e-15,
        rtol=0.0,
        msg=f"CIRS-to-GCRS and GCRS-to-CIRS are not inverses for {label}",
    )
    assert_allclose(
        right_product,
        identity,
        atol=1e-15,
        rtol=0.0,
        msg=f"GCRS-to-CIRS and CIRS-to-GCRS are not inverses for {label}",
    )


@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TT_JD_CASES)
@pytest.mark.parametrize("xp_arcsec, yp_arcsec, pole_label", POLAR_MOTION_CASES)
def test_polar_motion_and_inverse_are_inverses(tt_jd1, tt_jd2, time_label, xp_arcsec, yp_arcsec, pole_label):
    t = ((tt_jd1 - J2000) + tt_jd2) / JULIAN_CENTURY
    xp = arcsec_to_rad(xp_arcsec)
    yp = arcsec_to_rad(yp_arcsec)

    forward = polar_motion_matrix_single(t, xp, yp)
    backward = inversed_polar_motion_matrix_single(t, xp, yp)
    identity = jnp.eye(3)
    left_product = backward @ forward
    right_product = forward @ backward
    max_abs_diff = jnp.maximum(
        jnp.max(jnp.abs(left_product - identity)),
        jnp.max(jnp.abs(right_product - identity)),
    )

    print(
        "[unified.polar_motion_inverse] "
        f"time_label={time_label:<22} "
        f"pole_label={pole_label:<15} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        left_product,
        identity,
        atol=1e-15,
        rtol=0.0,
        msg=f"Inverse polar motion product mismatch for {time_label} / {pole_label}",
    )
    assert_allclose(
        right_product,
        identity,
        atol=1e-15,
        rtol=0.0,
        msg=f"Polar motion inverse product mismatch for {time_label} / {pole_label}",
    )
