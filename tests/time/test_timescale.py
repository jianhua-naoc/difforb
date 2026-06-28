import jax
import jax.numpy as jnp
import pytest
import equinox as eqx
from astropy.time import Time as AstroTime
import difforb.core.time.timescale as time2_timescale

from difforb.core.eop import load_default_eop_file
from difforb.core.earth_rotation import (
    earth_rotation_angle,
    precession_bias_matrix,
    nutation_matrix,
    polar_motion_matrix,
    inversed_polar_motion_matrix,
    gcrs_to_cirs_matrix,
    cirs_to_gcrs_matrix,
)
from difforb.core.geo import ITRS
from difforb.core.constants import J2000, JULIAN_CENTURY
from difforb.core.time.tai import tai_to_tt, tt_to_tai
from difforb.core.time.timedelta import TimeDelta
from difforb.core.time.timescale import Time
from difforb.core.time.tdb import tt_to_tdb, tt_to_tdb_single
from difforb.core.time.utc import utc_to_tai, split_utc_quasi_jd, julian_date_for_utc
from difforb.core.time.utils import renormalize_split_jd, julian_date, GREGORIAN_START_JD
from difforb.core.time.ut1 import tt_to_ut1_single
from difforb.core.time.utils import ut1_fraction
from difforb.utils import arcsec_to_rad
from tests.assertions import assert_allclose
from tests.time.test_tai import TAI_TT_CASES
from tests.time.test_tdb import TT_CASES as TDB_TT_CASES, LOCATION_CASES as TDB_LOCATION_CASES
from tests.time.test_utc import UTC_TEST_LABELS
from tests.time.test_ut1 import HISTORICAL_TT_CASES, UT1_BOUNDARY_TT_CASES, MODERN_EOP_TT_CASES

jax.config.update("jax_enable_x64", True)

eop = load_default_eop_file()

TT_DATE_CASES = [
    (2000.0, 1.0, 1.0, 12.0, 34.0, 56.0, GREGORIAN_START_JD),
    (2000.0, 1.0, 1.0, 0.0, 0.0, 0.123456789, GREGORIAN_START_JD),
    (1582.0, 10.0, 4.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD),
    (1582.0, 10.0, 15.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD),
    (1582.0, 10.0, 15.0, 0.0, 0.0, 0.0, jnp.inf),
    (1582.0, 10.0, 4.0, 0.0, 0.0, 0.0, -jnp.inf),
]


# =========================================================================
# Earth Orientation Tests
# =========================================================================


@pytest.mark.parametrize(("tt_jd1", "tt_jd2", "label"), MODERN_EOP_TT_CASES)
def test_time_xpole_matches_eop_xpole(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    expected = arcsec_to_rad(eop.xpole(tt_jd1, tt_jd2))

    assert_allclose(actual.xpole, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(("tt_jd1", "tt_jd2", "label"), HISTORICAL_TT_CASES)
def test_time_xpole_is_zero_before_eop_coverage(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert_allclose(actual.xpole, 0.0, atol=0.0, rtol=0.0)


def test_time_xpole_uses_final_predicted_value_beyond_eop_coverage():
    tt_jd1, tt_jd2 = renormalize_split_jd(eop.tt_jds[-1], 100.0)
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    expected = arcsec_to_rad(eop.xpoles[-1])

    assert_allclose(actual.xpole, expected, atol=0.0, rtol=0.0)


def test_time_xpole_shapes():
    tt_jd1 = jnp.asarray([case[0] for case in MODERN_EOP_TT_CASES], dtype=float)
    tt_jd2 = jnp.asarray([case[1] for case in MODERN_EOP_TT_CASES], dtype=float)
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert actual.shape == tt_jd1.shape
    assert actual.xpole.shape == actual.shape


@pytest.mark.parametrize(("tt_jd1", "tt_jd2", "label"), MODERN_EOP_TT_CASES)
def test_time_ypole_matches_eop_ypole(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    expected = arcsec_to_rad(eop.ypole(tt_jd1, tt_jd2))

    assert_allclose(actual.ypole, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(("tt_jd1", "tt_jd2", "label"), HISTORICAL_TT_CASES)
def test_time_ypole_is_zero_before_eop_coverage(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert_allclose(actual.ypole, 0.0, atol=0.0, rtol=0.0)


def test_time_ypole_uses_final_predicted_value_beyond_eop_coverage():
    tt_jd1, tt_jd2 = renormalize_split_jd(eop.tt_jds[-1], 100.0)
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    expected = arcsec_to_rad(eop.ypoles[-1])

    assert_allclose(actual.ypole, expected, atol=0.0, rtol=0.0)


def test_time_ypole_shapes():
    tt_jd1 = jnp.asarray([case[0] for case in MODERN_EOP_TT_CASES], dtype=float)
    tt_jd2 = jnp.asarray([case[1] for case in MODERN_EOP_TT_CASES], dtype=float)
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert actual.shape == tt_jd1.shape
    assert actual.ypole.shape == actual.shape


@pytest.mark.parametrize(("tt_jd1", "tt_jd2", "label"), MODERN_EOP_TT_CASES)
def test_time_cor_delta_longitude_matches_eop_value(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    expected = arcsec_to_rad(eop.cor_delta_longitude(tt_jd1, tt_jd2))

    assert_allclose(actual.cor_delta_longitude, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(("tt_jd1", "tt_jd2", "label"), HISTORICAL_TT_CASES)
def test_time_cor_delta_longitude_is_zero_before_eop_coverage(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert_allclose(actual.cor_delta_longitude, 0.0, atol=0.0, rtol=0.0)


def test_time_cor_delta_longitude_uses_final_predicted_value_beyond_eop_coverage():
    tt_jd1, tt_jd2 = renormalize_split_jd(eop.tt_jds[-1], 100.0)
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    expected = arcsec_to_rad(eop.dpsis[-1])

    assert_allclose(actual.cor_delta_longitude, expected, atol=0.0, rtol=0.0)


def test_time_cor_delta_longitude_shapes():
    tt_jd1 = jnp.asarray([case[0] for case in MODERN_EOP_TT_CASES], dtype=float)
    tt_jd2 = jnp.asarray([case[1] for case in MODERN_EOP_TT_CASES], dtype=float)
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert actual.shape == tt_jd1.shape
    assert actual.cor_delta_longitude.shape == actual.shape


@pytest.mark.parametrize(("tt_jd1", "tt_jd2", "label"), MODERN_EOP_TT_CASES)
def test_time_cor_delta_obliquity_matches_eop_value(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    expected = arcsec_to_rad(eop.cor_delta_obliquity(tt_jd1, tt_jd2))

    assert_allclose(actual.cor_delta_obliquity, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(("tt_jd1", "tt_jd2", "label"), HISTORICAL_TT_CASES)
def test_time_cor_delta_obliquity_is_zero_before_eop_coverage(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert_allclose(actual.cor_delta_obliquity, 0.0, atol=0.0, rtol=0.0)


def test_time_cor_delta_obliquity_uses_final_predicted_value_beyond_eop_coverage():
    tt_jd1, tt_jd2 = renormalize_split_jd(eop.tt_jds[-1], 100.0)
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    expected = arcsec_to_rad(eop.depss[-1])

    assert_allclose(actual.cor_delta_obliquity, expected, atol=0.0, rtol=0.0)


def test_time_cor_delta_obliquity_shapes():
    tt_jd1 = jnp.asarray([case[0] for case in MODERN_EOP_TT_CASES], dtype=float)
    tt_jd2 = jnp.asarray([case[1] for case in MODERN_EOP_TT_CASES], dtype=float)
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert actual.shape == tt_jd1.shape
    assert actual.cor_delta_obliquity.shape == actual.shape


# =========================================================================
# Earth Rotation Matrix And Angle Tests
# =========================================================================


@pytest.mark.parametrize(
    ("tt_jd1", "tt_jd2", "label"),
    [
        (2451545.0, 0.0, "J2000"),
        *MODERN_EOP_TT_CASES,
        *HISTORICAL_TT_CASES[:2],
    ],
)
def test_time_precession_bias_matrix_matches_low_level_function(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    tt_jc_j2000 = ((actual.tt.jd1 - J2000) + actual.tt.jd2) / JULIAN_CENTURY
    expected = precession_bias_matrix(tt_jc_j2000)

    assert_allclose(actual.precession_bias_matrix, expected, atol=0.0, rtol=0.0)


def test_time_precession_bias_matrix_shapes():
    tt_jd1 = jnp.asarray([case[0] for case in MODERN_EOP_TT_CASES], dtype=float)
    tt_jd2 = jnp.asarray([case[1] for case in MODERN_EOP_TT_CASES], dtype=float)
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert actual.shape == tt_jd1.shape
    assert actual.precession_bias_matrix.shape == actual.shape + (3, 3)


@pytest.mark.parametrize(
    ("tt_jd1", "tt_jd2", "label"),
    [
        (2451545.0, 0.0, "J2000"),
        *MODERN_EOP_TT_CASES,
        *HISTORICAL_TT_CASES[:2],
    ],
)
def test_time_precession_bias_matrix_is_orthogonal(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    matrix = actual.precession_bias_matrix
    identity = jnp.eye(3, dtype=matrix.dtype)

    assert_allclose(matrix @ matrix.T, identity, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    ("tt_jd1", "tt_jd2", "label"),
    [
        (2451545.0, 0.0, "J2000"),
        *MODERN_EOP_TT_CASES,
        *HISTORICAL_TT_CASES[:2],
    ],
)
def test_time_nutation_matrix_matches_low_level_function(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    tt_jc_j2000 = ((actual.tt.jd1 - J2000) + actual.tt.jd2) / JULIAN_CENTURY
    expected = nutation_matrix(tt_jc_j2000)

    assert_allclose(actual.nutation_matrix, expected, atol=0.0, rtol=0.0)


def test_time_nutation_matrix_shapes():
    tt_jd1 = jnp.asarray([case[0] for case in MODERN_EOP_TT_CASES], dtype=float)
    tt_jd2 = jnp.asarray([case[1] for case in MODERN_EOP_TT_CASES], dtype=float)
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert actual.shape == tt_jd1.shape
    assert actual.nutation_matrix.shape == actual.shape + (3, 3)


@pytest.mark.parametrize(
    ("tt_jd1", "tt_jd2", "label"),
    [
        (2451545.0, 0.0, "J2000"),
        *MODERN_EOP_TT_CASES,
        *HISTORICAL_TT_CASES[:2],
    ],
)
def test_time_nutation_matrix_is_orthogonal(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    matrix = actual.nutation_matrix
    identity = jnp.eye(3, dtype=matrix.dtype)

    assert_allclose(matrix @ matrix.T, identity, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(("tt_jd1", "tt_jd2", "label"), MODERN_EOP_TT_CASES)
def test_time_polar_motion_matrix_matches_low_level_function(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    tt_jc_j2000 = ((actual.tt.jd1 - J2000) + actual.tt.jd2) / JULIAN_CENTURY
    expected = polar_motion_matrix(tt_jc_j2000, actual.xpole, actual.ypole)

    assert_allclose(actual.polar_motion_matrix, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(("tt_jd1", "tt_jd2", "label"), MODERN_EOP_TT_CASES)
def test_time_inversed_polar_motion_matrix_matches_low_level_function(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    tt_jc_j2000 = ((actual.tt.jd1 - J2000) + actual.tt.jd2) / JULIAN_CENTURY
    expected = inversed_polar_motion_matrix(tt_jc_j2000, actual.xpole, actual.ypole)

    assert_allclose(actual.inversed_polar_motion_matrix, expected, atol=0.0, rtol=0.0)


def test_time_polar_motion_matrix_shapes():
    tt_jd1 = jnp.asarray([case[0] for case in MODERN_EOP_TT_CASES], dtype=float)
    tt_jd2 = jnp.asarray([case[1] for case in MODERN_EOP_TT_CASES], dtype=float)
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert actual.shape == tt_jd1.shape
    assert actual.polar_motion_matrix.shape == actual.shape + (3, 3)
    assert actual.inversed_polar_motion_matrix.shape == actual.shape + (3, 3)


@pytest.mark.parametrize(("tt_jd1", "tt_jd2", "label"), MODERN_EOP_TT_CASES)
def test_time_polar_motion_matrices_are_inverses(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    identity = jnp.eye(3, dtype=actual.polar_motion_matrix.dtype)

    assert_allclose(
        actual.polar_motion_matrix @ actual.inversed_polar_motion_matrix,
        identity,
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("tt_jd1", "tt_jd2", "label"),
    [*HISTORICAL_TT_CASES[:2], *UT1_BOUNDARY_TT_CASES, *MODERN_EOP_TT_CASES],
)
def test_time_era_matches_low_level_function(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    expected_ut1_jd1, expected_ut1_jd2 = time2_timescale.tt_to_ut1(tt_jd1, tt_jd2, eop)
    expected = earth_rotation_angle(expected_ut1_jd1, expected_ut1_jd2)

    assert_allclose(actual.ERA, expected, atol=0.0, rtol=0.0)


def test_time_era_shapes():
    tt_jd1 = jnp.asarray([case[0] for case in MODERN_EOP_TT_CASES], dtype=float)
    tt_jd2 = jnp.asarray([case[1] for case in MODERN_EOP_TT_CASES], dtype=float)
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert actual.shape == tt_jd1.shape
    assert actual.ERA.shape == actual.shape


@pytest.mark.parametrize(
    ("tt_jd1", "tt_jd2", "label"),
    [
        (2451545.0, 0.0, "J2000"),
        *MODERN_EOP_TT_CASES,
        *HISTORICAL_TT_CASES[:2],
    ],
)
def test_time_gcrs_to_cirs_matrix_matches_low_level_function(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    tt_jc_j2000 = ((actual.tt.jd1 - J2000) + actual.tt.jd2) / JULIAN_CENTURY
    expected = gcrs_to_cirs_matrix(tt_jc_j2000, actual.cor_delta_obliquity, actual.cor_delta_longitude)

    assert_allclose(actual.gcrs_to_cirs_matrix, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("tt_jd1", "tt_jd2", "label"),
    [
        (2451545.0, 0.0, "J2000"),
        *MODERN_EOP_TT_CASES,
        *HISTORICAL_TT_CASES[:2],
    ],
)
def test_time_cirs_to_gcrs_matrix_matches_low_level_function(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    tt_jc_j2000 = ((actual.tt.jd1 - J2000) + actual.tt.jd2) / JULIAN_CENTURY
    expected = cirs_to_gcrs_matrix(tt_jc_j2000, actual.cor_delta_obliquity, actual.cor_delta_longitude)

    assert_allclose(actual.cirs_to_gcrs_matrix, expected, atol=0.0, rtol=0.0)


def test_time_gcrs_and_cirs_matrix_shapes():
    tt_jd1 = jnp.asarray([case[0] for case in MODERN_EOP_TT_CASES], dtype=float)
    tt_jd2 = jnp.asarray([case[1] for case in MODERN_EOP_TT_CASES], dtype=float)
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert actual.shape == tt_jd1.shape
    assert actual.gcrs_to_cirs_matrix.shape == actual.shape + (3, 3)
    assert actual.cirs_to_gcrs_matrix.shape == actual.shape + (3, 3)


@pytest.mark.parametrize(
    ("tt_jd1", "tt_jd2", "label"),
    [
        (2451545.0, 0.0, "J2000"),
        *MODERN_EOP_TT_CASES,
        *HISTORICAL_TT_CASES[:2],
    ],
)
def test_time_gcrs_and_cirs_matrices_are_inverses(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    identity = jnp.eye(3, dtype=actual.gcrs_to_cirs_matrix.dtype)

    assert_allclose(actual.gcrs_to_cirs_matrix @ actual.cirs_to_gcrs_matrix, identity, atol=1e-12, rtol=0.0)


# =========================================================================
# ``UTC`` Constructor And View Tests
# =========================================================================

@pytest.mark.parametrize("label", UTC_TEST_LABELS)
def test_time_from_utc_jd_builds_expected_tt(label):
    utc = AstroTime(label, scale="utc")
    utc_jd1 = jnp.asarray(utc.jd1)
    utc_jd2 = jnp.asarray(utc.jd2)

    expected_tai_jd1, expected_tai_jd2 = utc_to_tai(utc_jd1, utc_jd2)
    expected_tt_jd1, expected_tt_jd2 = tai_to_tt(expected_tai_jd1, expected_tai_jd2)
    expected_tt_jd1, expected_tt_jd2 = renormalize_split_jd(expected_tt_jd1, expected_tt_jd2)

    actual = Time.from_utc_jd(utc_jd1, utc_jd2, eop=eop)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, expected_tt_jd1 + expected_tt_jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("label", UTC_TEST_LABELS)
def test_time_from_utc_jd_roundtrip_utc(label):
    utc = AstroTime(label, scale="utc")
    utc_jd1, utc_jd2 = renormalize_split_jd(jnp.asarray(utc.jd1), jnp.asarray(utc.jd2))

    actual = Time.from_utc_jd(utc_jd1, utc_jd2, eop=eop)
    actual_utc_jd1, actual_utc_jd2 = renormalize_split_jd(actual.utc.jd1, actual.utc.jd2)

    assert_allclose(actual_utc_jd1 + actual_utc_jd2, utc_jd1 + utc_jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("label", UTC_TEST_LABELS)
def test_time_from_utc_date_matches_from_utc_jd(label):
    date_part, *time_part = label.split()
    year_s, month_s, day_s = date_part.split("-")
    year = int(year_s)
    month = int(month_s)
    day = int(day_s)
    hour = 0
    minute = 0
    second = 0.0
    if time_part:
        hour_s, minute_s, second_s = time_part[0].split(":")
        hour = int(hour_s)
        minute = int(minute_s)
        second = float(second_s)

    actual = Time.from_utc_date(year, month, day, hour, minute, second, eop=eop)

    utc = AstroTime(label, scale="utc")
    expected = Time.from_utc_jd(jnp.asarray(utc.jd1), jnp.asarray(utc.jd2), eop=eop)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, expected.tt.jd1 + expected.tt.jd2, atol=0.0, rtol=0.0)


def test_time_from_utc_jd_rejects_pre_1962():
    with pytest.raises(eqx.EquinoxRuntimeError):
        Time.from_utc_jd(jnp.asarray(2437664.0), jnp.asarray(0.5), eop=eop)


@pytest.mark.parametrize("label", UTC_TEST_LABELS)
def test_time_from_utc_date_roundtrip_utc_fields(label):
    date_part, *time_part = label.split()
    year_s, month_s, day_s = date_part.split("-")
    year = int(year_s)
    month = int(month_s)
    day = int(day_s)
    hour = 0
    minute = 0
    second = 0.0
    if time_part:
        hour_s, minute_s, second_s = time_part[0].split(":")
        hour = int(hour_s)
        minute = int(minute_s)
        second = float(second_s)

    actual = Time.from_utc_date(year, month, day, hour, minute, second, eop=eop)
    y2, m2, d2, h2, min2, s2 = actual.utc.ymdhms

    assert int(y2) == year
    assert int(m2) == month
    assert int(d2) == day
    assert int(h2) == hour
    assert int(min2) == minute
    assert_allclose(s2, second, atol=0.0, rtol=0.0)


def test_time_from_utc_date_normalizes_overflow_fields():
    actual = Time.from_utc_date(1962.0, 1.0, 1.0, 25.0, 1.0, -1.0, eop=eop)
    expected = Time.from_utc_date(1962.0, 1.0, 2.0, 1.0, 0.0, 59.0, eop=eop)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, expected.tt.jd1 + expected.tt.jd2, atol=0.0, rtol=0.0)


def test_time_from_utc_date_normalizes_negative_overflow_fields():
    actual = Time.from_utc_date(1962.0, 1.0, 2.0, -1.0, 0.0, 0.0, eop=eop)
    expected = Time.from_utc_date(1962.0, 1.0, 1.0, 23.0, 0.0, 0.0, eop=eop)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, expected.tt.jd1 + expected.tt.jd2, atol=0.0, rtol=0.0)


def test_time_from_utc_date_rejects_pre_1962():
    with pytest.raises(eqx.EquinoxRuntimeError):
        Time.from_utc_date(1961, 12, 31, 23, 59, 59.0, eop=eop)


@pytest.mark.parametrize(
    "label",
    [
        "1963-10-31 23:59:59",
        "1963-11-01",
        "1972-06-30 23:59:60",
        "2000-01-01 12:34:56",
        "2016-12-31 23:59:60.5",
    ],
)
def test_time_utc_view_keeps_canonical_quasi_jd_split(label):
    utc = AstroTime(label, scale="utc")
    actual = Time.from_utc_jd(jnp.asarray(utc.jd1), jnp.asarray(utc.jd2), eop=eop)

    actual_utc_jd1 = actual.utc.jd1
    actual_utc_jd2 = actual.utc.jd2
    split_jd1, split_jd2 = split_utc_quasi_jd(actual_utc_jd1, actual_utc_jd2)

    assert_allclose(actual_utc_jd1, split_jd1, atol=0.0, rtol=0.0)
    assert_allclose(actual_utc_jd2, split_jd2, atol=0.0, rtol=0.0)


def test_time_from_utc_jd_shapes():
    labels = ["1962-01-01", "2016-12-31 23:59:60.5"]
    utc_jd1 = jnp.array([AstroTime(label, scale="utc").jd1 for label in labels])
    utc_jd2 = jnp.array([AstroTime(label, scale="utc").jd2 for label in labels])

    actual = Time.from_utc_jd(utc_jd1, utc_jd2, eop=eop)

    assert actual.shape == (2,)
    assert actual.tt.jd1.shape == (2,)
    assert actual.tt.jd2.shape == (2,)
    assert actual.utc.jd1.shape == (2,)
    assert actual.utc.jd2.shape == (2,)


def test_time_from_utc_date_shapes():
    year = jnp.array([1962.0, 2016.0])
    month = jnp.array([1.0, 12.0])
    day = jnp.array([1.0, 31.0])
    hour = jnp.array([0.0, 23.0])
    minute = jnp.array([0.0, 59.0])
    second = jnp.array([0.0, 60.5])

    actual = Time.from_utc_date(year, month, day, hour, minute, second, eop=eop)

    assert actual.shape == (2,)
    assert actual.tt.jd1.shape == (2,)
    assert actual.tt.jd2.shape == (2,)
    assert actual.utc.jd1.shape == (2,)
    assert actual.utc.jd2.shape == (2,)


# =========================================================================
# Mixed ``UT`` Constructor Tests
# =========================================================================

PRE_1962_UT_TT_CASES = HISTORICAL_TT_CASES + [UT1_BOUNDARY_TT_CASES[0]]
UT_DATE_CASES_PRE_1962 = [
    (1960.0, 1.0, 1.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD),
    (1961.0, 12.0, 31.0, 23.0, 0.0, 0.0, GREGORIAN_START_JD),
    (1582.0, 10.0, 4.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD),
    (1582.0, 10.0, 15.0, 0.0, 0.0, 0.0, jnp.inf),
]
UT_DATE_CASES_POST_1962 = [
    (1962.0, 1.0, 1.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD),
    (1967.0, 1.0, 16.0, 12.0, 0.0, 0.0, GREGORIAN_START_JD),
    (1972.0, 6.0, 30.0, 23.0, 59.0, 60.5, GREGORIAN_START_JD),
    (2000.0, 1.0, 1.0, 12.0, 34.0, 56.0, GREGORIAN_START_JD),
    (2016.0, 12.0, 31.0, 23.0, 59.0, 60.5, GREGORIAN_START_JD),
]


@pytest.mark.parametrize(("tt_jd1", "tt_jd2", "label"), PRE_1962_UT_TT_CASES)
def test_time_from_ut_jd_builds_expected_tt_pre_1962(tt_jd1, tt_jd2, label):
    ut_jd1, ut_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)

    actual = Time.from_ut_jd(ut_jd1, ut_jd2, eop=eop)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, tt_jd1 + tt_jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("label", UTC_TEST_LABELS)
def test_time_from_ut_jd_builds_expected_tt_post_1962(label):
    utc = AstroTime(label, scale="utc")
    ut_jd1 = jnp.asarray(utc.jd1)
    ut_jd2 = jnp.asarray(utc.jd2)

    actual = Time.from_ut_jd(ut_jd1, ut_jd2, eop=eop)
    expected = Time.from_utc_jd(ut_jd1, ut_jd2, eop=eop)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, expected.tt.jd1 + expected.tt.jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(("tt_jd1", "tt_jd2", "label"), PRE_1962_UT_TT_CASES)
def test_time_from_ut_jd_roundtrip_ut_pre_1962(tt_jd1, tt_jd2, label):
    ut_jd1, ut_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)

    actual = Time.from_ut_jd(ut_jd1, ut_jd2, eop=eop)

    assert_allclose(actual.ut.jd1 + actual.ut.jd2, ut_jd1 + ut_jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("label", UTC_TEST_LABELS)
def test_time_from_ut_jd_roundtrip_ut_post_1962(label):
    utc = AstroTime(label, scale="utc")
    ut_jd1, ut_jd2 = renormalize_split_jd(jnp.asarray(utc.jd1), jnp.asarray(utc.jd2))

    actual = Time.from_ut_jd(ut_jd1, ut_jd2, eop=eop)
    actual_ut_jd1, actual_ut_jd2 = renormalize_split_jd(actual.ut.jd1, actual.ut.jd2)

    assert_allclose(actual_ut_jd1 + actual_ut_jd2, ut_jd1 + ut_jd2, atol=0.0, rtol=0.0)


def test_time_from_ut_jd_mixed_batch_builds_expected_tt():
    ut1_input_jd1, ut1_input_jd2 = tt_to_ut1_single(2437665.0, 0.499999999, eop)
    utc_input = AstroTime("1962-01-01", scale="utc")

    ut_jd1 = jnp.array([ut1_input_jd1, utc_input.jd1])
    ut_jd2 = jnp.array([ut1_input_jd2, utc_input.jd2])

    actual = Time.from_ut_jd(ut_jd1, ut_jd2, eop=eop)

    expected_tt_pre = Time.from_ut1_jd(ut1_input_jd1, ut1_input_jd2, eop=eop).tt.jd1 + \
                      Time.from_ut1_jd(ut1_input_jd1, ut1_input_jd2, eop=eop).tt.jd2
    expected_tt_post = Time.from_utc_jd(jnp.asarray(utc_input.jd1), jnp.asarray(utc_input.jd2), eop=eop).tt.jd1 + \
                       Time.from_utc_jd(jnp.asarray(utc_input.jd1), jnp.asarray(utc_input.jd2), eop=eop).tt.jd2

    assert_allclose(actual.tt.jd1[0] + actual.tt.jd2[0], expected_tt_pre, atol=0.0, rtol=0.0)
    assert_allclose(actual.tt.jd1[1] + actual.tt.jd2[1], expected_tt_post, atol=0.0, rtol=0.0)


def test_time_from_ut_jd_shapes():
    ut1_input_jd1, ut1_input_jd2 = tt_to_ut1_single(2437665.0, 0.499999999, eop)
    utc_input = AstroTime("1962-01-01", scale="utc")

    ut_jd1 = jnp.array([ut1_input_jd1, utc_input.jd1])
    ut_jd2 = jnp.array([ut1_input_jd2, utc_input.jd2])

    actual = Time.from_ut_jd(ut_jd1, ut_jd2, eop=eop)

    assert actual.shape == (2,)
    assert actual.tt.jd1.shape == (2,)
    assert actual.tt.jd2.shape == (2,)
    assert actual.ut.jd1.shape == (2,)
    assert actual.ut.jd2.shape == (2,)


@pytest.mark.parametrize(
    ("year", "month", "day", "hour", "minute", "second", "gregorian_start"),
    UT_DATE_CASES_PRE_1962,
)
def test_time_from_ut_date_matches_from_ut_jd_pre_1962(year, month, day, hour, minute, second, gregorian_start):
    actual = Time.from_ut_date(year, month, day, hour, minute, second, eop=eop, gregorian_start=gregorian_start)

    ut_jd1, ut_jd2 = julian_date(year, month, day, hour, minute, second, gregorian_start)
    expected = Time.from_ut_jd(ut_jd1, ut_jd2, eop=eop, gregorian_start=gregorian_start)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, expected.tt.jd1 + expected.tt.jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("year", "month", "day", "hour", "minute", "second", "gregorian_start"),
    UT_DATE_CASES_POST_1962,
)
def test_time_from_ut_date_matches_from_ut_jd_post_1962(year, month, day, hour, minute, second, gregorian_start):
    actual = Time.from_ut_date(year, month, day, hour, minute, second, eop=eop, gregorian_start=gregorian_start)

    ut_jd1, ut_jd2 = julian_date_for_utc(year, month, day, hour, minute, second)
    expected = Time.from_ut_jd(ut_jd1, ut_jd2, eop=eop, gregorian_start=gregorian_start)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, expected.tt.jd1 + expected.tt.jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("year", "month", "day", "hour", "minute", "second", "gregorian_start"),
    UT_DATE_CASES_PRE_1962 + UT_DATE_CASES_POST_1962,
)
def test_time_from_ut_date_roundtrip_ut_fields(year, month, day, hour, minute, second, gregorian_start):
    actual = Time.from_ut_date(year, month, day, hour, minute, second, eop=eop, gregorian_start=gregorian_start)
    y2, m2, d2, h2, min2, s2 = actual.ut.ymdhms

    assert_allclose(y2, year, atol=0.0, rtol=0.0)
    assert_allclose(m2, month, atol=0.0, rtol=0.0)
    assert_allclose(d2, day, atol=0.0, rtol=0.0)
    assert_allclose(h2, hour, atol=0.0, rtol=0.0)
    assert_allclose(min2, minute, atol=0.0, rtol=0.0)
    assert_allclose(s2, second, atol=0.0, rtol=0.0)


def test_time_from_ut_date_mixed_batch_roundtrip_ut_fields():
    year = jnp.array([1961.0, 1962.0])
    month = jnp.array([12.0, 1.0])
    day = jnp.array([31.0, 1.0])
    hour = jnp.array([23.0, 0.0])
    minute = jnp.array([0.0, 0.0])
    second = jnp.array([0.0, 0.0])

    actual = Time.from_ut_date(year, month, day, hour, minute, second, eop=eop)
    y2, m2, d2, h2, min2, s2 = actual.ut.ymdhms

    assert_allclose(y2, year, atol=0.0, rtol=0.0)
    assert_allclose(m2, month, atol=0.0, rtol=0.0)
    assert_allclose(d2, day, atol=0.0, rtol=0.0)
    assert_allclose(h2, hour, atol=0.0, rtol=0.0)
    assert_allclose(min2, minute, atol=0.0, rtol=0.0)
    assert_allclose(s2, second, atol=0.0, rtol=0.0)


def test_time_from_ut_date_shapes():
    year = jnp.array([1961.0, 1962.0])
    month = jnp.array([12.0, 1.0])
    day = jnp.array([31.0, 1.0])
    hour = jnp.array([23.0, 0.0])
    minute = jnp.array([0.0, 0.0])
    second = jnp.array([0.0, 0.0])

    actual = Time.from_ut_date(year, month, day, hour, minute, second, eop=eop)

    assert actual.shape == (2,)
    assert actual.tt.jd1.shape == (2,)
    assert actual.tt.jd2.shape == (2,)
    assert actual.ut.jd1.shape == (2,)
    assert actual.ut.jd2.shape == (2,)


# =========================================================================
# Mixed ``UT`` View Tests
# =========================================================================

@pytest.mark.parametrize(("tt_jd1", "tt_jd2", "label"), PRE_1962_UT_TT_CASES)
def test_time_ut_view_matches_ut1_before_1962(tt_jd1, tt_jd2, label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert_allclose(actual.ut.jd1 + actual.ut.jd2, actual.ut1.jd1 + actual.ut1.jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("label", UTC_TEST_LABELS)
def test_time_ut_view_matches_utc_on_and_after_1962(label):
    utc = AstroTime(label, scale="utc")
    actual = Time.from_utc_jd(jnp.asarray(utc.jd1), jnp.asarray(utc.jd2), eop=eop)

    assert_allclose(actual.ut.jd1 + actual.ut.jd2, actual.utc.jd1 + actual.utc.jd2, atol=0.0, rtol=0.0)


def test_time_ut_view_ymdhms_switches_semantics_at_1962_boundary():
    year = jnp.array([1961.0, 1962.0])
    month = jnp.array([12.0, 1.0])
    day = jnp.array([31.0, 1.0])
    hour = jnp.array([23.0, 0.0])
    minute = jnp.array([0.0, 0.0])
    second = jnp.array([0.0, 0.0])

    actual = Time.from_ut_date(year, month, day, hour, minute, second, eop=eop)
    y2, m2, d2, h2, min2, s2 = actual.ut.ymdhms

    assert_allclose(y2, year, atol=0.0, rtol=0.0)
    assert_allclose(m2, month, atol=0.0, rtol=0.0)
    assert_allclose(d2, day, atol=0.0, rtol=0.0)
    assert_allclose(h2, hour, atol=0.0, rtol=0.0)
    assert_allclose(min2, minute, atol=0.0, rtol=0.0)
    assert_allclose(s2, second, atol=0.0, rtol=0.0)


def test_time_ut_view_shapes():
    year = jnp.array([1961.0, 1962.0])
    month = jnp.array([12.0, 1.0])
    day = jnp.array([31.0, 1.0])
    hour = jnp.array([23.0, 0.0])
    minute = jnp.array([0.0, 0.0])
    second = jnp.array([0.0, 0.0])

    actual = Time.from_ut_date(year, month, day, hour, minute, second, eop=eop)

    assert actual.ut.shape == (2,)
    assert actual.ut.jd1.shape == (2,)
    assert actual.ut.jd2.shape == (2,)


# =========================================================================
# ``TAI`` Constructor And View Tests
# =========================================================================

@pytest.mark.parametrize("tai_jd1, tai_jd2, time_label", TAI_TT_CASES)
def test_time_from_tai_jd_builds_expected_tt(tai_jd1, tai_jd2, time_label):
    expected_tt_jd1, expected_tt_jd2 = tai_to_tt(tai_jd1, tai_jd2)
    actual = Time.from_tai_jd(tai_jd1, tai_jd2, eop=eop)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, expected_tt_jd1 + expected_tt_jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("tai_jd1, tai_jd2, time_label", TAI_TT_CASES)
def test_time_from_tai_jd_roundtrips_tai(tai_jd1, tai_jd2, time_label):
    actual = Time.from_tai_jd(tai_jd1, tai_jd2, eop=eop)

    assert_allclose(actual.tai.jd1 + actual.tai.jd2, tai_jd1 + tai_jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TAI_TT_CASES)
def test_time_tai_view_matches_tt_to_tai(tt_jd1, tt_jd2, time_label):
    expected_tai_jd1, expected_tai_jd2 = tt_to_tai(tt_jd1, tt_jd2)
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert_allclose(actual.tai.jd1 + actual.tai.jd2, expected_tai_jd1 + expected_tai_jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("year", "month", "day", "hour", "minute", "second", "gregorian_start"),
    TT_DATE_CASES,
)
def test_time_from_tai_date_matches_from_tai_jd(year, month, day, hour, minute, second, gregorian_start):
    actual = Time.from_tai_date(year, month, day, hour, minute, second, eop=eop, gregorian_start=gregorian_start)

    tai_jd1, tai_jd2 = julian_date(year, month, day, hour, minute, second, gregorian_start)
    expected = Time.from_tai_jd(tai_jd1, tai_jd2, eop=eop, gregorian_start=gregorian_start)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, expected.tt.jd1 + expected.tt.jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("year", "month", "day", "hour", "minute", "second", "gregorian_start"),
    TT_DATE_CASES,
)
def test_time_from_tai_date_roundtrip_tai_fields(year, month, day, hour, minute, second, gregorian_start):
    actual = Time.from_tai_date(year, month, day, hour, minute, second, eop=eop, gregorian_start=gregorian_start)
    y2, m2, d2, h2, min2, s2 = actual.tai.ymdhms

    assert_allclose(y2, year, atol=0.0, rtol=0.0)
    assert_allclose(m2, month, atol=0.0, rtol=0.0)
    assert_allclose(d2, day, atol=0.0, rtol=0.0)
    assert_allclose(h2, hour, atol=0.0, rtol=0.0)
    assert_allclose(min2, minute, atol=0.0, rtol=0.0)
    assert_allclose(s2, second, atol=1e-10, rtol=0.0)


def test_time_from_tai_jd_shapes():
    actual = Time.from_tai_jd(
        tai_jd1=jnp.array([2451545.0, 2451727.0], dtype=float),
        tai_jd2=jnp.array([0.0, 0.625], dtype=float),
        eop=eop,
    )

    assert actual.shape == (2,)
    assert actual.tt.jd1.shape == (2,)
    assert actual.tt.jd2.shape == (2,)
    assert actual.tai.jd1.shape == (2,)
    assert actual.tai.jd2.shape == (2,)


def test_time_from_tai_date_shapes():
    actual = Time.from_tai_date(
        jnp.array([2000.0, 2000.0], dtype=float),
        1.0,
        jnp.array([1.0, 1.0], dtype=float),
        jnp.array([12.0, 0.0], dtype=float),
        jnp.array([34.0, 0.0], dtype=float),
        jnp.array([56.0, 0.123456789], dtype=float),
        eop=eop,
        gregorian_start=GREGORIAN_START_JD,
    )

    assert actual.shape == (2,)
    assert actual.tt.jd1.shape == (2,)
    assert actual.tt.jd2.shape == (2,)
    assert actual.tai.jd1.shape == (2,)
    assert actual.tai.jd2.shape == (2,)


# =========================================================================
# ``TT`` Constructor Tests
# =========================================================================


@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TDB_TT_CASES)
def test_time_from_tt_jd_roundtrips_tt(tt_jd1, tt_jd2, time_label):
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, tt_jd1 + tt_jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("year", "month", "day", "hour", "minute", "second", "gregorian_start"),
    TT_DATE_CASES,
)
def test_time_from_tt_date_matches_from_tt_jd(year, month, day, hour, minute, second, gregorian_start):
    actual = Time.from_tt_date(year, month, day, hour, minute, second, eop=eop, gregorian_start=gregorian_start)

    tt_jd1, tt_jd2 = julian_date(year, month, day, hour, minute, second, gregorian_start)
    expected = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop, gregorian_start=gregorian_start)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, expected.tt.jd1 + expected.tt.jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("year", "month", "day", "hour", "minute", "second", "gregorian_start"),
    TT_DATE_CASES,
)
def test_time_from_tt_date_roundtrip_tt_fields(year, month, day, hour, minute, second, gregorian_start):
    actual = Time.from_tt_date(year, month, day, hour, minute, second, eop=eop, gregorian_start=gregorian_start)
    y2, m2, d2, h2, min2, s2 = actual.tt.ymdhms

    assert_allclose(y2, year, atol=0.0, rtol=0.0)
    assert_allclose(m2, month, atol=0.0, rtol=0.0)
    assert_allclose(d2, day, atol=0.0, rtol=0.0)
    assert_allclose(h2, hour, atol=0.0, rtol=0.0)
    assert_allclose(min2, minute, atol=0.0, rtol=0.0)
    assert_allclose(s2, second, atol=1e-10, rtol=0.0)


def test_time_from_tt_jd_shapes():
    actual = Time.from_tt_jd(
        tt_jd1=jnp.array([2451545.0, 2451727.0], dtype=float),
        tt_jd2=jnp.array([0.0, 0.625], dtype=float),
        eop=eop,
    )

    assert actual.shape == (2,)
    assert actual.tt.jd1.shape == (2,)
    assert actual.tt.jd2.shape == (2,)


def test_time_from_tt_date_shapes():
    actual = Time.from_tt_date(
        jnp.array([2000.0, 2000.0], dtype=float),
        1.0,
        jnp.array([1.0, 1.0], dtype=float),
        jnp.array([12.0, 0.0], dtype=float),
        jnp.array([34.0, 0.0], dtype=float),
        jnp.array([56.0, 0.123456789], dtype=float),
        eop=eop,
        gregorian_start=GREGORIAN_START_JD,
    )

    assert actual.shape == (2,)
    assert actual.tt.jd1.shape == (2,)
    assert actual.tt.jd2.shape == (2,)


# =========================================================================
# ``UT1`` Constructor Tests
# =========================================================================

@pytest.mark.parametrize(
    ("tt_jd1", "tt_jd2", "label"),
    HISTORICAL_TT_CASES + UT1_BOUNDARY_TT_CASES + MODERN_EOP_TT_CASES,
)
def test_time_from_ut1_jd_builds_expected_tt(tt_jd1, tt_jd2, label):
    ut1_jd1, ut1_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)
    actual = Time.from_ut1_jd(ut1_jd1, ut1_jd2, eop=eop)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, tt_jd1 + tt_jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("tt_jd1", "tt_jd2", "label"),
    HISTORICAL_TT_CASES + UT1_BOUNDARY_TT_CASES + MODERN_EOP_TT_CASES,
)
def test_time_from_ut1_jd_roundtrip_ut1(tt_jd1, tt_jd2, label):
    ut1_jd1, ut1_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)
    actual = Time.from_ut1_jd(ut1_jd1, ut1_jd2, eop=eop)

    assert_allclose(actual.ut1.jd1 + actual.ut1.jd2, ut1_jd1 + ut1_jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("year", "month", "day", "hour", "minute", "second", "gregorian_start"),
    [
        (1960.0, 1.0, 1.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD),
        (1972.0, 1.0, 1.0, 12.0, 0.0, 0.0, GREGORIAN_START_JD),
        (2000.0, 1.0, 1.0, 12.0, 34.0, 56.0, GREGORIAN_START_JD),
        (2000.0, 1.0, 1.0, 0.0, 0.0, 0.123456789, GREGORIAN_START_JD),
        (1582.0, 10.0, 4.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD),
        (1582.0, 10.0, 15.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD),
        (1582.0, 10.0, 15.0, 0.0, 0.0, 0.0, jnp.inf),
        (1582.0, 10.0, 4.0, 0.0, 0.0, 0.0, -jnp.inf),
    ],
)
def test_time_from_ut1_date_matches_from_ut1_jd(year, month, day, hour, minute, second, gregorian_start):
    actual = Time.from_ut1_date(year, month, day, hour, minute, second, eop=eop, gregorian_start=gregorian_start)

    ut1_jd1, ut1_jd2 = julian_date(year, month, day, hour, minute, second, gregorian_start)
    expected = Time.from_ut1_jd(ut1_jd1, ut1_jd2, eop=eop, gregorian_start=gregorian_start)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, expected.tt.jd1 + expected.tt.jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("year", "month", "day", "hour", "minute", "second", "gregorian_start"),
    [
        (1960.0, 1.0, 1.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD),
        (1972.0, 1.0, 1.0, 12.0, 0.0, 0.0, GREGORIAN_START_JD),
        (2000.0, 1.0, 1.0, 12.0, 34.0, 56.0, GREGORIAN_START_JD),
        (2000.0, 1.0, 1.0, 0.0, 0.0, 0.123456789, GREGORIAN_START_JD),
        (1582.0, 10.0, 4.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD),
        (1582.0, 10.0, 15.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD),
        (1582.0, 10.0, 15.0, 0.0, 0.0, 0.0, jnp.inf),
        (1582.0, 10.0, 4.0, 0.0, 0.0, 0.0, -jnp.inf),
    ],
)
def test_time_from_ut1_date_roundtrip_ut1_fields(year, month, day, hour, minute, second, gregorian_start):
    actual = Time.from_ut1_date(year, month, day, hour, minute, second, eop=eop, gregorian_start=gregorian_start)
    y2, m2, d2, h2, min2, s2 = actual.ut1.ymdhms

    assert_allclose(y2, year, atol=0.0, rtol=0.0)
    assert_allclose(m2, month, atol=0.0, rtol=0.0)
    assert_allclose(d2, day, atol=0.0, rtol=0.0)
    assert_allclose(h2, hour, atol=0.0, rtol=0.0)
    assert_allclose(min2, minute, atol=0.0, rtol=0.0)
    assert_allclose(s2, second, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize(
    ("tt_jd1", "tt_jd2", "label"),
    HISTORICAL_TT_CASES + UT1_BOUNDARY_TT_CASES + MODERN_EOP_TT_CASES,
)
def test_time_ut1_view_matches_tt_to_ut1(tt_jd1, tt_jd2, label):
    expected_ut1_jd1, expected_ut1_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)
    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert_allclose(actual.ut1.jd1 + actual.ut1.jd2, expected_ut1_jd1 + expected_ut1_jd2, atol=0.0, rtol=0.0)


def test_time_from_ut1_jd_shapes():
    tt_jd1 = jnp.array([2436934.0, eop.tt_jds[10]])
    tt_jd2 = jnp.array([0.5, 0.0])
    ut1_jd1, ut1_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)

    actual = Time.from_ut1_jd(ut1_jd1, ut1_jd2, eop=eop)

    assert actual.shape == (2,)
    assert actual.tt.jd1.shape == (2,)
    assert actual.tt.jd2.shape == (2,)
    assert actual.ut1.jd1.shape == (2,)
    assert actual.ut1.jd2.shape == (2,)


def test_time_from_ut1_date_shapes():
    year = jnp.array([1960.0, 2000.0])
    month = 1.0
    day = jnp.array([1.0, 1.0])
    hour = jnp.array([0.0, 12.0])
    minute = 0.0
    second = jnp.array([0.0, 56.0])

    actual = Time.from_ut1_date(year, month, day, hour, minute, second, eop=eop)

    assert actual.shape == (2,)
    assert actual.tt.jd1.shape == (2,)
    assert actual.tt.jd2.shape == (2,)
    assert actual.ut1.jd1.shape == (2,)
    assert actual.ut1.jd2.shape == (2,)


# =========================================================================
# ``TDB`` Constructor Tests
# =========================================================================

@pytest.mark.parametrize("use_geocenter", [False, True])
@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TDB_TT_CASES)
@pytest.mark.parametrize(("lon_deg", "u", "v", "loc_label"), TDB_LOCATION_CASES)
def test_time_from_tdb_jd_builds_expected_tt(tt_jd1, tt_jd2, time_label, lon_deg, u, v, loc_label, use_geocenter):
    if use_geocenter and loc_label != "Geocenter":
        pytest.skip("Geocenter branch is only applicable to the geocenter location case.")

    lon_rad = jnp.deg2rad(lon_deg)
    if use_geocenter:
        location = None
    else:
        location = ITRS(
            pos=jnp.array([
                u * 1000.0 * jnp.cos(lon_rad),
                u * 1000.0 * jnp.sin(lon_rad),
                v * 1000.0,
            ], dtype=float),
            lon=lon_rad,
        )

    ut1_jd1, ut1_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)
    ut1_frac = ut1_fraction(ut1_jd1, ut1_jd2)
    tdb_jd1, tdb_jd2 = tt_to_tdb_single(lon_rad, u, v, tt_jd1, tt_jd2, ut1_frac)

    actual = Time.from_tdb_jd(tdb_jd1, tdb_jd2, eop=eop, location=location)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, tt_jd1 + tt_jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("use_geocenter", [False, True])
@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TDB_TT_CASES)
@pytest.mark.parametrize(("lon_deg", "u", "v", "loc_label"), TDB_LOCATION_CASES)
def test_time_from_tdb_jd_roundtrip_tdb(tt_jd1, tt_jd2, time_label, lon_deg, u, v, loc_label, use_geocenter):
    if use_geocenter and loc_label != "Geocenter":
        pytest.skip("Geocenter branch is only applicable to the geocenter location case.")

    lon_rad = jnp.deg2rad(lon_deg)
    if use_geocenter:
        location = None
    else:
        location = ITRS(
            pos=jnp.array([
                u * 1000.0 * jnp.cos(lon_rad),
                u * 1000.0 * jnp.sin(lon_rad),
                v * 1000.0,
            ], dtype=float),
            lon=lon_rad,
        )

    ut1_jd1, ut1_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)
    ut1_frac = ut1_fraction(ut1_jd1, ut1_jd2)
    tdb_jd1, tdb_jd2 = tt_to_tdb_single(lon_rad, u, v, tt_jd1, tt_jd2, ut1_frac)

    actual = Time.from_tdb_jd(tdb_jd1, tdb_jd2, eop=eop, location=location)
    actual_tdb = actual.tdb(location)

    assert_allclose(actual_tdb.jd1 + actual_tdb.jd2, tdb_jd1 + tdb_jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("use_geocenter", [False, True])
@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TDB_TT_CASES)
@pytest.mark.parametrize(("lon_deg", "u", "v", "loc_label"), TDB_LOCATION_CASES)
def test_time_tdb_view_matches_tt_to_tdb(tt_jd1, tt_jd2, time_label, lon_deg, u, v, loc_label, use_geocenter):
    if use_geocenter and loc_label != "Geocenter":
        pytest.skip("Geocenter branch is only applicable to the geocenter location case.")

    lon_rad = jnp.deg2rad(lon_deg)
    if use_geocenter:
        location = None
    else:
        location = ITRS(
            pos=jnp.array([
                u * 1000.0 * jnp.cos(lon_rad),
                u * 1000.0 * jnp.sin(lon_rad),
                v * 1000.0,
            ], dtype=float),
            lon=lon_rad,
        )

    ut1_jd1, ut1_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)
    ut1_frac = ut1_fraction(ut1_jd1, ut1_jd2)
    expected_tdb_jd1, expected_tdb_jd2 = tt_to_tdb_single(lon_rad, u, v, tt_jd1, tt_jd2, ut1_frac)

    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    actual_tdb = actual.tdb(location)

    assert_allclose(actual_tdb.jd1 + actual_tdb.jd2, expected_tdb_jd1 + expected_tdb_jd2, atol=0.0, rtol=0.0)


def test_time_tdb_view_uses_tt_to_ut1_for_historical_epochs(monkeypatch):
    """Guard the ``Time.tdb`` call path.

    This regression test patches ``difforb.core.time.timescale.tt_to_ut1``
    with a recognizable fake implementation and checks that ``Time.tdb``
    follows that hook. If ``Time.tdb`` is changed to rebuild ``UT1`` directly
    from ``eop.ut1dtt`` instead of calling the shared conversion entry point,
    this test must fail.
    """
    tt_jd1, tt_jd2 = 2436934.0, 0.5  # 1960-01-01 TT
    lon_deg, u, v = 1.2, 3800.0, 4200.0
    lon_rad = jnp.deg2rad(lon_deg)
    location = ITRS(
        pos=jnp.array([
            u * 1000.0 * jnp.cos(lon_rad),
            u * 1000.0 * jnp.sin(lon_rad),
            v * 1000.0,
        ], dtype=float),
        lon=lon_rad,
    )

    def fake_tt_to_ut1(tt_jd1, tt_jd2, eop):
        return renormalize_split_jd(tt_jd1, tt_jd2 - 0.375)

    expected_ut1_jd1, expected_ut1_jd2 = fake_tt_to_ut1(tt_jd1, tt_jd2, eop)
    expected_ut1_frac = ut1_fraction(expected_ut1_jd1, expected_ut1_jd2)
    expected_tdb_jd1, expected_tdb_jd2 = tt_to_tdb_single(lon_rad, u, v, tt_jd1, tt_jd2, expected_ut1_frac)
    expected_tdb_jd1, expected_tdb_jd2 = renormalize_split_jd(expected_tdb_jd1, expected_tdb_jd2)

    monkeypatch.setattr(time2_timescale, "tt_to_ut1", fake_tt_to_ut1)

    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    actual_tdb = actual.tdb(location)

    assert_allclose(actual_tdb.jd1, expected_tdb_jd1, atol=0.0, rtol=0.0)
    assert_allclose(actual_tdb.jd2, expected_tdb_jd2, atol=0.0, rtol=0.0)


def test_time_era_uses_tt_to_ut1_for_historical_epochs(monkeypatch):
    """Guard the ``Time.ERA`` call path.

    This regression test patches ``difforb.core.time.timescale.tt_to_ut1``
    with a recognizable fake implementation and checks that ``Time.ERA``
    derives ``UT1`` through that shared hook. If ``ERA`` is changed to rebuild
    ``UT1`` directly from ``eop.ut1dtt`` instead, this test must fail.
    """
    tt_jd1, tt_jd2 = 2436934.0, 0.5  # 1960-01-01 TT

    def fake_tt_to_ut1(tt_jd1, tt_jd2, eop):
        return renormalize_split_jd(tt_jd1, tt_jd2 - 0.375)

    expected_ut1_jd1, expected_ut1_jd2 = fake_tt_to_ut1(tt_jd1, tt_jd2, eop)
    expected_era = earth_rotation_angle(expected_ut1_jd1, expected_ut1_jd2)

    monkeypatch.setattr(time2_timescale, "tt_to_ut1", fake_tt_to_ut1)

    actual = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    assert_allclose(actual.ERA, expected_era, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("use_geocenter", [False, True])
@pytest.mark.parametrize(
    ("year", "month", "day", "hour", "minute", "second", "gregorian_start"),
    [
        (2000.0, 1.0, 1.0, 12.0, 34.0, 56.0, GREGORIAN_START_JD),
        (2000.0, 1.0, 1.0, 0.0, 0.0, 0.123456789, GREGORIAN_START_JD),
        (1582.0, 10.0, 4.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD),
        (1582.0, 10.0, 15.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD),
        (1582.0, 10.0, 15.0, 0.0, 0.0, 0.0, jnp.inf),
        (1582.0, 10.0, 4.0, 0.0, 0.0, 0.0, -jnp.inf),
    ],
)
def test_time_from_tdb_date_matches_from_tdb_jd(year, month, day, hour, minute, second, gregorian_start, use_geocenter):
    if use_geocenter:
        location = None
    else:
        lon_rad = jnp.deg2rad(1.2)
        location = ITRS(
            pos=jnp.array([
                3800.0 * 1000.0 * jnp.cos(lon_rad),
                3800.0 * 1000.0 * jnp.sin(lon_rad),
                4200.0 * 1000.0,
            ], dtype=float),
            lon=lon_rad,
        )

    actual = Time.from_tdb_date(
        year, month, day, hour, minute, second,
        eop=eop, location=location, gregorian_start=gregorian_start
    )

    tdb_jd1, tdb_jd2 = julian_date(year, month, day, hour, minute, second, gregorian_start)
    expected = Time.from_tdb_jd(tdb_jd1, tdb_jd2, eop=eop, location=location, gregorian_start=gregorian_start)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, expected.tt.jd1 + expected.tt.jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("use_geocenter", [False, True])
@pytest.mark.parametrize(
    ("year", "month", "day", "hour", "minute", "second", "gregorian_start"),
    [
        (2000.0, 1.0, 1.0, 12.0, 34.0, 56.0, GREGORIAN_START_JD),
        (2000.0, 1.0, 1.0, 0.0, 0.0, 0.123456789, GREGORIAN_START_JD),
        (1582.0, 10.0, 4.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD),
        (1582.0, 10.0, 15.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD),
        (1582.0, 10.0, 15.0, 0.0, 0.0, 0.0, jnp.inf),
        (1582.0, 10.0, 4.0, 0.0, 0.0, 0.0, -jnp.inf),
    ],
)
def test_time_from_tdb_date_roundtrip_tdb_fields(year, month, day, hour, minute, second, gregorian_start, use_geocenter):
    if use_geocenter:
        location = None
    else:
        lon_rad = jnp.deg2rad(1.2)
        location = ITRS(
            pos=jnp.array([
                3800.0 * 1000.0 * jnp.cos(lon_rad),
                3800.0 * 1000.0 * jnp.sin(lon_rad),
                4200.0 * 1000.0,
            ], dtype=float),
            lon=lon_rad,
        )

    actual = Time.from_tdb_date(
        year, month, day, hour, minute, second,
        eop=eop, location=location, gregorian_start=gregorian_start
    )
    y2, m2, d2, h2, min2, s2 = actual.tdb(location).ymdhms

    assert_allclose(y2, year, atol=0.0, rtol=0.0)
    assert_allclose(m2, month, atol=0.0, rtol=0.0)
    assert_allclose(d2, day, atol=0.0, rtol=0.0)
    assert_allclose(h2, hour, atol=0.0, rtol=0.0)
    assert_allclose(min2, minute, atol=0.0, rtol=0.0)
    sec_atol = 1e-10 if use_geocenter else 1e-8
    assert_allclose(s2, second, atol=sec_atol, rtol=0.0)


@pytest.mark.parametrize("use_geocenter", [False, True])
def test_time_tdb_view_shapes(use_geocenter):
    if use_geocenter:
        location = None
    else:
        lon_rad = jnp.deg2rad(1.2)
        location = ITRS(
            pos=jnp.array([
                3800.0 * 1000.0 * jnp.cos(lon_rad),
                3800.0 * 1000.0 * jnp.sin(lon_rad),
                4200.0 * 1000.0,
            ], dtype=float),
            lon=lon_rad,
        )

    actual = Time.from_tt_jd(
        tt_jd1=jnp.array([2451545.0, 2451727.0], dtype=float),
        tt_jd2=jnp.array([0.0, 0.625], dtype=float),
        eop=eop,
    )
    actual_tdb = actual.tdb(location)

    assert actual.shape == (2,)
    assert actual_tdb.jd1.shape == (2,)
    assert actual_tdb.jd2.shape == (2,)


@pytest.mark.parametrize("use_geocenter", [False, True])
def test_time_from_tdb_jd_shapes(use_geocenter):
    if use_geocenter:
        location = None
        lon_deg, u, v = 0.0, 0.0, 0.0
    else:
        lon_deg, u, v = 1.2, 3800.0, 4200.0
        lon_rad = jnp.deg2rad(lon_deg)
        location = ITRS(
            pos=jnp.array([
                u * 1000.0 * jnp.cos(lon_rad),
                u * 1000.0 * jnp.sin(lon_rad),
                v * 1000.0,
            ], dtype=float),
            lon=lon_rad,
        )

    tt_jd1 = jnp.array([2451545.0, 2451727.0], dtype=float)
    tt_jd2 = jnp.array([0.0, 0.625], dtype=float)
    ut1_jd1, ut1_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)
    ut1_frac = ut1_fraction(ut1_jd1, ut1_jd2)
    lon_rad = jnp.deg2rad(lon_deg)
    tdb_jd1, tdb_jd2 = tt_to_tdb(lon_rad, u, v, tt_jd1, tt_jd2, ut1_frac, False)

    actual = Time.from_tdb_jd(tdb_jd1, tdb_jd2, eop=eop, location=location)
    actual_tdb = actual.tdb(location)

    assert actual.shape == (2,)
    assert actual.tt.jd1.shape == (2,)
    assert actual.tt.jd2.shape == (2,)
    assert actual_tdb.jd1.shape == (2,)
    assert actual_tdb.jd2.shape == (2,)


@pytest.mark.parametrize("use_geocenter", [False, True])
def test_time_from_tdb_date_shapes(use_geocenter):
    if use_geocenter:
        location = None
    else:
        lon_rad = jnp.deg2rad(1.2)
        location = ITRS(
            pos=jnp.array([
                3800.0 * 1000.0 * jnp.cos(lon_rad),
                3800.0 * 1000.0 * jnp.sin(lon_rad),
                4200.0 * 1000.0,
            ], dtype=float),
            lon=lon_rad,
        )

    actual = Time.from_tdb_date(
        year=jnp.array([2000.0, 2000.0], dtype=float),
        month=1.0,
        day=jnp.array([1.0, 1.0], dtype=float),
        hour=jnp.array([12.0, 0.0], dtype=float),
        min=jnp.array([34.0, 0.0], dtype=float),
        sec=jnp.array([56.0, 0.123456789], dtype=float),
        eop=eop,
        location=location,
        gregorian_start=GREGORIAN_START_JD,
    )
    actual_tdb = actual.tdb(location)

    assert actual.shape == (2,)
    assert actual.tt.jd1.shape == (2,)
    assert actual.tt.jd2.shape == (2,)
    assert actual_tdb.jd1.shape == (2,)
    assert actual_tdb.jd2.shape == (2,)


def test_time_tdb_view_shapes_grid():
    location = ITRS(
        pos=jnp.array([
            [6378.1 * 1000.0, 0.0, 0.0],
            [3800.0 * 1000.0 * jnp.cos(jnp.deg2rad(1.2)), 3800.0 * 1000.0 * jnp.sin(jnp.deg2rad(1.2)), 4200.0 * 1000.0],
        ], dtype=float),
        lon=jnp.array([0.0, jnp.deg2rad(1.2)], dtype=float),
    )
    actual = Time.from_tt_jd(
        tt_jd1=jnp.array([2451545.0, 2451727.0, 2816787.0], dtype=float),
        tt_jd2=jnp.array([0.0, 0.625, 0.5], dtype=float),
        eop=eop,
    )
    actual_tdb = actual.tdb(location, grid=True)

    assert actual.shape == (3,)
    assert actual_tdb.shape == (2, 3)
    assert actual_tdb.jd1.shape == (2, 3)
    assert actual_tdb.jd2.shape == (2, 3)


def test_time_from_tdb_jd_shapes_grid():
    location = ITRS(
        pos=jnp.array([
            [6378.1 * 1000.0, 0.0, 0.0],
            [3800.0 * 1000.0 * jnp.cos(jnp.deg2rad(1.2)), 3800.0 * 1000.0 * jnp.sin(jnp.deg2rad(1.2)), 4200.0 * 1000.0],
        ], dtype=float),
        lon=jnp.array([0.0, jnp.deg2rad(1.2)], dtype=float),
    )
    tdb_jd1 = jnp.array([2451545.0, 2451727.0, 2816787.0], dtype=float)
    tdb_jd2 = jnp.array([0.0, 0.625, 0.5], dtype=float)

    actual = Time.from_tdb_jd(tdb_jd1, tdb_jd2, eop=eop, location=location, grid=True)
    actual_tdb = actual.tdb(location, grid=True)

    assert actual.shape == (2, 3)
    assert actual.tt.jd1.shape == (2, 3)
    assert actual.tt.jd2.shape == (2, 3)
    assert actual_tdb.jd1.shape == (2, 2, 3)
    assert actual_tdb.jd2.shape == (2, 2, 3)


def test_time_from_tdb_date_shapes_grid():
    location = ITRS(
        pos=jnp.array([
            [6378.1 * 1000.0, 0.0, 0.0],
            [3800.0 * 1000.0 * jnp.cos(jnp.deg2rad(1.2)), 3800.0 * 1000.0 * jnp.sin(jnp.deg2rad(1.2)), 4200.0 * 1000.0],
        ], dtype=float),
        lon=jnp.array([0.0, jnp.deg2rad(1.2)], dtype=float),
    )

    actual = Time.from_tdb_date(
        year=jnp.array([2000.0, 2000.0, 3000.0], dtype=float),
        month=1.0,
        day=jnp.array([1.0, 1.0, 1.0], dtype=float),
        hour=jnp.array([12.0, 0.0, 0.0], dtype=float),
        min=jnp.array([34.0, 0.0, 0.0], dtype=float),
        sec=jnp.array([56.0, 0.123456789, 0.0], dtype=float),
        eop=eop,
        location=location,
        grid=True,
        gregorian_start=GREGORIAN_START_JD,
    )
    actual_tdb = actual.tdb(location, grid=True)

    assert actual.shape == (2, 3)
    assert actual.tt.jd1.shape == (2, 3)
    assert actual.tt.jd2.shape == (2, 3)
    assert actual_tdb.jd1.shape == (2, 2, 3)
    assert actual_tdb.jd2.shape == (2, 2, 3)


# =========================================================================
# ``Time`` Arithmetic Tests
# =========================================================================


@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TDB_TT_CASES)
def test_time_add_timedelta_shifts_tt(tt_jd1, tt_jd2, time_label):
    time = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    delta = TimeDelta.from_days(1.25)

    actual = time + delta

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, tt_jd1 + tt_jd2 + 1.25, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TDB_TT_CASES)
def test_time_add_float_matches_timedelta_days(tt_jd1, tt_jd2, time_label):
    time = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    actual = time + 1.25
    expected = time + TimeDelta.from_days(1.25)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, expected.tt.jd1 + expected.tt.jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TDB_TT_CASES)
def test_time_radd_float_matches_add(tt_jd1, tt_jd2, time_label):
    time = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    actual = 1.25 + time
    expected = time + 1.25

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, expected.tt.jd1 + expected.tt.jd2, atol=0.0, rtol=0.0)


def test_time_add_shapes():
    time = Time.from_tt_jd(
        tt_jd1=jnp.array([2451545.0, 2451727.0], dtype=float),
        tt_jd2=jnp.array([0.0, 0.625], dtype=float),
        eop=eop,
    )
    delta = TimeDelta.from_seconds(jnp.array([60.0, 120.0], dtype=float))

    actual = time + delta

    assert actual.shape == (2,)
    assert actual.tt.jd1.shape == (2,)
    assert actual.tt.jd2.shape == (2,)


def test_time_add_uses_uniform_tt_duration_across_leap_second():
    time = Time.from_utc_date(2016, 12, 31, 23, 59, 30.0, eop=eop)
    delta = TimeDelta.from_seconds(60.0)

    actual = time + delta

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, time.tt.jd1 + time.tt.jd2 + 60.0 / 86400.0, atol=1e-9, rtol=0.0)
    assert actual.utc.format_string() == "2017-01-01 00:00:29.000"


@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TDB_TT_CASES)
def test_time_sub_timedelta_shifts_tt(tt_jd1, tt_jd2, time_label):
    time = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    delta = TimeDelta.from_days(1.25)

    actual = time - delta

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, tt_jd1 + tt_jd2 - 1.25, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TDB_TT_CASES)
def test_time_sub_float_matches_timedelta_days(tt_jd1, tt_jd2, time_label):
    time = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)

    actual = time - 1.25
    expected = time - TimeDelta.from_days(1.25)

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, expected.tt.jd1 + expected.tt.jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TDB_TT_CASES)
def test_time_sub_time_returns_timedelta(tt_jd1, tt_jd2, time_label):
    left = Time.from_tt_jd(tt_jd1, tt_jd2, eop=eop)
    right = left + TimeDelta.from_seconds(90.0)

    actual = right - left

    assert isinstance(actual, TimeDelta)
    assert_allclose(actual.seconds, 90.0, atol=1e-11, rtol=0.0)


def test_time_sub_shapes():
    time = Time.from_tt_jd(
        tt_jd1=jnp.array([2451545.0, 2451727.0], dtype=float),
        tt_jd2=jnp.array([0.0, 0.625], dtype=float),
        eop=eop,
    )
    delta = TimeDelta.from_seconds(jnp.array([60.0, 120.0], dtype=float))

    actual = time - delta

    assert actual.shape == (2,)
    assert actual.tt.jd1.shape == (2,)
    assert actual.tt.jd2.shape == (2,)


def test_time_sub_uses_uniform_tt_duration_across_leap_second():
    time = Time.from_utc_date(2017, 1, 1, 0, 0, 29.0, eop=eop)
    delta = TimeDelta.from_seconds(60.0)

    actual = time - delta

    assert_allclose(actual.tt.jd1 + actual.tt.jd2, time.tt.jd1 + time.tt.jd2 - 60.0 / 86400.0, atol=1e-9, rtol=0.0)
    assert actual.utc.format_string() == "2016-12-31 23:59:30.000"


# =========================================================================
# ``Time`` Comparison Tests
# =========================================================================


def test_time_eq_same_epoch():
    left = Time.from_tt_jd(2451545.0, 0.0, eop=eop)
    right = Time.from_tt_jd(2451545.0, 0.0, eop=eop)

    assert bool(left == right)
    assert not bool(left != right)


def test_time_eq_different_constructors_same_epoch():
    utc = AstroTime("2017-01-01 00:00:00", scale="utc")
    tt = utc.tt
    left = Time.from_utc_jd(jnp.asarray(utc.jd1), jnp.asarray(utc.jd2), eop=eop)
    right = Time.from_tt_jd(jnp.asarray(tt.jd1), jnp.asarray(tt.jd2), eop=eop)

    assert bool(left == right)
    assert not bool(left != right)


def test_time_lt_gt_ordering():
    left = Time.from_tt_jd(2451545.0, 0.0, eop=eop)
    right = left + TimeDelta.from_seconds(60.0)

    assert bool(left < right)
    assert bool(left <= right)
    assert bool(right > left)
    assert bool(right >= left)
    assert not bool(right < left)


def test_time_comparison_shapes():
    left = Time.from_tt_jd(
        tt_jd1=jnp.array([2451545.0, 2451545.0], dtype=float),
        tt_jd2=jnp.array([0.0, 0.5], dtype=float),
        eop=eop,
    )
    right = Time.from_tt_jd(
        tt_jd1=jnp.array([2451545.0, 2451545.0], dtype=float),
        tt_jd2=jnp.array([0.25, 0.25], dtype=float),
        eop=eop,
    )

    actual_lt = left < right
    actual_eq = left == right

    assert actual_lt.shape == (2,)
    assert actual_eq.shape == (2,)
    assert_allclose(actual_lt.astype(jnp.int32), jnp.array([1, 0], dtype=jnp.int32), atol=0.0, rtol=0.0)
    assert_allclose(actual_eq.astype(jnp.int32), jnp.array([0, 0], dtype=jnp.int32), atol=0.0, rtol=0.0)


def test_time_comparison_rejects_non_time():
    actual = Time.from_tt_jd(2451545.0, 0.0, eop=eop)

    with pytest.raises(TypeError):
        _ = actual < 1.0

    with pytest.raises(TypeError):
        _ = actual > 1.0

    assert not bool(actual == 1.0)
    assert bool(actual != 1.0)


# =========================================================================
# ``TimeView`` Formatting Tests
# =========================================================================

def test_timeview_iso_string_matches_default_format():
    actual = Time.from_tt_date(2000.0, 1.0, 1.0, 12.0, 34.0, 56.987654321, eop=eop)

    assert actual.tt.iso_string == actual.tt.format_string()


def test_timeview_format_string_renders_supported_placeholders():
    actual = Time.from_tt_date(2000.0, 1.0, 2.0, 3.0, 4.0, 5.6789, eop=eop)

    formatted = actual.tt.format_string(
        "{YYYY}|{Y}|{MM}|{M}|{DD}|{D}|{hh}|{h}|{mm}|{m}|{ss}|{s}|{ss.3}|{s.3}"
    )

    assert formatted == "2000|2000|01|1|02|2|03|3|04|4|05|5|05.678|5.678"


def test_timeview_format_string_truncates_fractional_seconds():
    actual = Time.from_tt_date(2000.0, 1.0, 1.0, 12.0, 34.0, 56.987654321, eop=eop)

    assert actual.tt.format_string("{ss.3}") == "56.987"
    assert actual.tt.format_string("{ss.6}") == "56.987654"


def test_timeview_format_string_returns_nested_lists_for_batches():
    actual = Time.from_tt_date(
        year=jnp.array([[2000.0], [2001.0]], dtype=float),
        month=1.0,
        day=jnp.array([[1.0], [2.0]], dtype=float),
        hour=jnp.array([[12.0], [3.0]], dtype=float),
        min=jnp.array([[34.0], [4.0]], dtype=float),
        sec=jnp.array([[56.0], [5.5]], dtype=float),
        eop=eop,
        gregorian_start=GREGORIAN_START_JD,
    )

    formatted = actual.tt.format_string("{YYYY}-{MM}-{DD} {hh}:{mm}:{ss.1}")

    assert formatted == [
        ["2000-01-01 12:34:56.0"],
        ["2001-01-02 03:04:05.5"],
    ]


@pytest.mark.parametrize(
    "template",
    [
        "{YYYY",
        "{foo}",
        "{ss.0}",
        "{ss.x}",
    ],
)
def test_timeview_format_string_rejects_invalid_templates(template):
    actual = Time.from_tt_date(2000.0, 1.0, 1.0, 12.0, 34.0, 56.0, eop=eop)

    with pytest.raises(ValueError):
        actual.tt.format_string(template)


def test_utc_view_format_string_preserves_leap_second():
    actual = Time.from_utc_date(2016.0, 12.0, 31.0, 23.0, 59.0, 60.5, eop=eop)

    assert actual.utc.format_string("{YYYY}-{MM}-{DD} {hh}:{mm}:{ss}") == "2016-12-31 23:59:60"
    assert actual.utc.format_string("{YYYY}-{MM}-{DD} {hh}:{mm}:{ss.1}") == "2016-12-31 23:59:60.5"


def test_utc_view_format_string_preserves_pre_1972_civil_label():
    actual = Time.from_utc_date(1963.0, 10.0, 31.0, 23.0, 59.0, 59.0, eop=eop)

    assert actual.utc.format_string("{YYYY}-{MM}-{DD} {hh}:{mm}:{ss}") == "1963-10-31 23:59:59"


def test_timeview_shape_matches_underlying_batch_shape():
    lon_rad = jnp.deg2rad(1.2)
    location = ITRS(
        pos=jnp.array([
            3800.0 * 1000.0 * jnp.cos(lon_rad),
            3800.0 * 1000.0 * jnp.sin(lon_rad),
            4200.0 * 1000.0,
        ], dtype=float),
        lon=lon_rad,
    )
    actual = Time.from_tt_date(
        year=jnp.array([2000.0, 2001.0], dtype=float),
        month=1.0,
        day=jnp.array([1.0, 2.0], dtype=float),
        hour=jnp.array([12.0, 3.0], dtype=float),
        min=jnp.array([34.0, 4.0], dtype=float),
        sec=jnp.array([56.0, 5.5], dtype=float),
        eop=eop,
        gregorian_start=GREGORIAN_START_JD,
    )

    assert actual.shape == (2,)
    assert actual.tt.shape == (2,)
    assert actual.tai.shape == (2,)
    assert actual.ut1.shape == (2,)
    assert actual.utc.shape == (2,)
    assert actual.tdb(None).shape == (2,)
    assert actual.tdb(location).shape == (2,)
