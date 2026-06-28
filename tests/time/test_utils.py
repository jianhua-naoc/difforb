import jax
import jax.numpy as jnp
import pytest

from difforb.core.time.utils import (
    GREGORIAN_START_JD,
    julian_date_core,
    calendar_date_single,
    julian_date,
    calendar_date,
    renormalize_split_jd,
)
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("year", "month", "day", "expected_jd", "label"),
    [
        (2000.0, 1.0, 1.0, 2451544.5, "J2000 midnight"),
        (2000.0, 1.0, 1.5, 2451545.0, "J2000 noon"),
        (1970.0, 1.0, 1.0, 2440587.5, "Unix epoch midnight"),
        (1582.0, 10.0, 4.0, 2299159.5, "Last Julian day before cutoff"),
        (1582.0, 10.0, 15.0, 2299160.5, "First Gregorian day after cutoff"),
    ],
)
def test_julian_date_core_known_values(year, month, day, expected_jd, label):
    actual_jd = julian_date_core(year, month, day)

    print(
        "[julian_date_core] "
        f"label={label:<32} "
        f"diff={float(actual_jd - expected_jd):+.12e} day "
    )

    assert_allclose(actual_jd, expected_jd, atol=0.0, rtol=0.0)


def test_julian_date_core_cutoff_gap_is_one_day():
    jd_julian = julian_date_core(1582.0, 10.0, 4.0)
    jd_gregorian = julian_date_core(1582.0, 10.0, 15.0)

    assert_allclose(jd_gregorian - jd_julian, 1.0, atol=0.0, rtol=0.0)


def test_julian_date_core_custom_cutoff_can_force_julian():
    actual_jd = julian_date_core(1582.0, 10.0, 15.0, gregorian_start=jnp.inf)

    assert_allclose(actual_jd, 2299170.5, atol=0.0, rtol=0.0)


def test_julian_date_core_custom_cutoff_can_force_gregorian():
    actual_jd = julian_date_core(1582.0, 10.0, 4.0, gregorian_start=-jnp.inf)

    assert_allclose(actual_jd, 2299149.5, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("jd1", "jd2", "gregorian_start", "exp_year", "exp_month", "exp_day", "exp_hour", "exp_minute", "exp_second", "label"),
    [
        (2451544.5, 0.0, GREGORIAN_START_JD, 2000.0, 1.0, 1.0, 0.0, 0.0, 0.0, "J2000 midnight"),
        (2451545.0, 0.0, GREGORIAN_START_JD, 2000.0, 1.0, 1.0, 12.0, 0.0, 0.0, "J2000 noon"),
        (2451544.0, 0.75, GREGORIAN_START_JD, 2000.0, 1.0, 1.0, 6.0, 0.0, 0.0, "J2000 split input"),
        (2451544.5, 1.4288980208333335e-06, GREGORIAN_START_JD, 2000.0, 1.0, 1.0, 0.0, 0.0, 0.123456789,
         "J2000 fractional second"),
        (2440587.5, 0.0, GREGORIAN_START_JD, 1970.0, 1.0, 1.0, 0.0, 0.0, 0.0, "Unix epoch midnight"),
        (2299159.5, 0.0, GREGORIAN_START_JD, 1582.0, 10.0, 4.0, 0.0, 0.0, 0.0, "Last Julian day before cutoff"),
        (2299160.5, 0.0, GREGORIAN_START_JD, 1582.0, 10.0, 15.0, 0.0, 0.0, 0.0, "First Gregorian day after cutoff"),
    ],
)
def test_calendar_date_single_known_values(
        jd1, jd2, gregorian_start, exp_year, exp_month, exp_day, exp_hour, exp_minute, exp_second, label
):
    actual_year, actual_month, actual_day, actual_hour, actual_minute, actual_second = calendar_date_single(
        jd1, jd2, gregorian_start=gregorian_start
    )

    assert_allclose(actual_year, exp_year, atol=0.0, rtol=0.0)
    assert_allclose(actual_month, exp_month, atol=0.0, rtol=0.0)
    assert_allclose(actual_day, exp_day, atol=0.0, rtol=0.0)
    assert_allclose(actual_hour, exp_hour, atol=0.0, rtol=0.0)
    assert_allclose(actual_minute, exp_minute, atol=0.0, rtol=0.0)
    assert_allclose(actual_second, exp_second, atol=0.0, rtol=0.0)


def test_calendar_date_single_custom_cutoff_can_force_julian():
    actual_year, actual_month, actual_day, actual_hour, actual_minute, actual_second = calendar_date_single(
        2299170.5, 0.0, gregorian_start=jnp.inf
    )

    assert_allclose(actual_year, 1582.0, atol=0.0, rtol=0.0)
    assert_allclose(actual_month, 10.0, atol=0.0, rtol=0.0)
    assert_allclose(actual_day, 15.0, atol=0.0, rtol=0.0)
    assert_allclose(actual_hour, 0.0, atol=0.0, rtol=0.0)
    assert_allclose(actual_minute, 0.0, atol=0.0, rtol=0.0)
    assert_allclose(actual_second, 0.0, atol=0.0, rtol=0.0)


def test_calendar_date_single_custom_cutoff_can_force_gregorian():
    actual_year, actual_month, actual_day, actual_hour, actual_minute, actual_second = calendar_date_single(
        2299149.5, 0.0, gregorian_start=-jnp.inf
    )

    assert_allclose(actual_year, 1582.0, atol=0.0, rtol=0.0)
    assert_allclose(actual_month, 10.0, atol=0.0, rtol=0.0)
    assert_allclose(actual_day, 4.0, atol=0.0, rtol=0.0)
    assert_allclose(actual_hour, 0.0, atol=0.0, rtol=0.0)
    assert_allclose(actual_minute, 0.0, atol=0.0, rtol=0.0)
    assert_allclose(actual_second, 0.0, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("year", "month", "day", "hour", "minute", "second", "gregorian_start", "label"),
    [
        (2000.0, 1.0, 1.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD, "J2000 midnight"),
        (2000.0, 1.0, 1.0, 12.0, 0.0, 0.0, GREGORIAN_START_JD, "J2000 noon"),
        (2000.0, 1.0, 1.0, 0.0, 0.0, 0.123456789, GREGORIAN_START_JD, "J2000 fractional second"),
        (1970.0, 1.0, 1.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD, "Unix epoch midnight"),
        (1582.0, 10.0, 4.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD, "Last Julian day before cutoff"),
        (1582.0, 10.0, 15.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD, "First Gregorian day after cutoff"),
        (1582.0, 10.0, 15.0, 0.0, 0.0, 0.0, jnp.inf, "Forced Julian cutoff"),
        (1582.0, 10.0, 4.0, 0.0, 0.0, 0.0, -jnp.inf, "Forced Gregorian cutoff"),
    ],
)
def test_julian_date_calendar_roundtrip(year, month, day, hour, minute, second, gregorian_start, label):
    jd1, jd2 = julian_date(year, month, day, hour, minute, second, gregorian_start=gregorian_start)
    actual_year, actual_month, actual_day, actual_hour, actual_minute, actual_second = calendar_date(
        jd1, jd2, gregorian_start=gregorian_start
    )

    print(
        "[julian_calendar_roundtrip] "
        f"label={label:<32} "
        f"actual=({float(actual_year):04.0f}-{float(actual_month):02.0f}-{float(actual_day):02.0f} "
        f"{float(actual_hour):02.0f}:{float(actual_minute):02.0f}:{float(actual_second):012.9f}) "
    )

    assert_allclose(actual_year, year, atol=0.0, rtol=0.0)
    assert_allclose(actual_month, month, atol=0.0, rtol=0.0)
    assert_allclose(actual_day, day, atol=0.0, rtol=0.0)
    assert_allclose(actual_hour, hour, atol=0.0, rtol=0.0)
    assert_allclose(actual_minute, minute, atol=0.0, rtol=0.0)
    assert_allclose(actual_second, second, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize(
    (
            "year", "month", "day", "hour", "minute", "second", "gregorian_start",
            "exp_year", "exp_month", "exp_day", "exp_hour", "exp_minute", "exp_second",
    ),
    [
        pytest.param(2000.0, 1.0, 1.0, 25.0, 1.0, -1.0, GREGORIAN_START_JD, 2000.0, 1.0, 2.0, 1.0, 0.0, 59.0,
                     id="positive_overflow"),
        pytest.param(2000.0, 1.0, 2.0, -1.0, 0.0, 0.0, GREGORIAN_START_JD, 2000.0, 1.0, 1.0, 23.0, 0.0, 0.0,
                     id="negative_hour"),
        pytest.param(2000.0, 1.0, 1.0, 0.0, 59.0, 61.0, GREGORIAN_START_JD, 2000.0, 1.0, 1.0, 1.0, 0.0, 1.0,
                     id="second_overflow"),
        pytest.param(2000.0, 1.0, 1.5, 12.0, 0.0, 0.0, GREGORIAN_START_JD, 2000.0, 1.0, 2.0, 0.0, 0.0, 0.0,
                     id="fractional_day_overflow"),
        pytest.param(2000.0, 13.0, 1.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD, 2001.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                     id="month_overflow_forward"),
        pytest.param(2000.0, 0.0, 1.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD, 1999.0, 12.0, 1.0, 0.0, 0.0, 0.0,
                     id="month_overflow_zero"),
        pytest.param(2000.0, -1.0, 1.0, 0.0, 0.0, 0.0, GREGORIAN_START_JD, 1999.0, 11.0, 1.0, 0.0, 0.0, 0.0,
                     id="month_overflow_negative"),
        pytest.param(2000.0, 13.0, 32.0, 25.0, 61.0, 61.0, GREGORIAN_START_JD, 2001.0, 2.0, 2.0, 2.0, 2.0, 1.0,
                     id="compound_month_day_time_overflow"),
        pytest.param(1582.0, 10.0, 4.0, 25.0, 0.0, 0.0, GREGORIAN_START_JD, 1582.0, 10.0, 15.0, 1.0, 0.0, 0.0,
                     id="cutover_forward_overflow"),
        pytest.param(1582.0, 10.0, 15.0, -1.0, 0.0, 0.0, GREGORIAN_START_JD, 1582.0, 10.0, 4.0, 23.0, 0.0, 0.0,
                     id="cutover_backward_overflow"),
    ],
)
def test_julian_date_normalizes_overflow_fields(
        year, month, day, hour, minute, second, gregorian_start,
        exp_year, exp_month, exp_day, exp_hour, exp_minute, exp_second):
    actual_jd1, actual_jd2 = julian_date(
        year, month, day, hour, minute, second, gregorian_start=gregorian_start
    )
    expected_jd1, expected_jd2 = julian_date(
        exp_year, exp_month, exp_day, exp_hour, exp_minute, exp_second, gregorian_start=gregorian_start
    )
    rt_year, rt_month, rt_day, rt_hour, rt_minute, rt_second = calendar_date(
        actual_jd1, actual_jd2, gregorian_start=gregorian_start
    )

    assert_allclose(actual_jd1 + actual_jd2, expected_jd1 + expected_jd2, atol=0.0, rtol=0.0)
    assert_allclose(rt_year, exp_year, atol=0.0, rtol=0.0)
    assert_allclose(rt_month, exp_month, atol=0.0, rtol=0.0)
    assert_allclose(rt_day, exp_day, atol=0.0, rtol=0.0)
    assert_allclose(rt_hour, exp_hour, atol=0.0, rtol=0.0)
    assert_allclose(rt_minute, exp_minute, atol=0.0, rtol=0.0)
    assert_allclose(rt_second, exp_second, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize(
    ("jd1", "jd2"),
    [
        (2451545.3, 0.4),
        (2451545.8, -0.9),
        (2451545.0, 1.2),
        (2451545.49, 0.49),
        (-10.7, 0.9),
    ],
)
def test_renormalize_split_jd_preserves_total(jd1, jd2):
    actual_jd1, actual_jd2 = renormalize_split_jd(jd1, jd2)

    assert_allclose(actual_jd1 + actual_jd2, jd1 + jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("jd1", "jd2"),
    [
        (2451545.3, 0.4),
        (2451545.8, -0.9),
        (2451545.0, 1.2),
        (2451545.49, 0.49),
        (-10.7, 0.9),
    ],
)
def test_renormalize_split_jd_centers_residual(jd1, jd2):
    _, actual_jd2 = renormalize_split_jd(jd1, jd2)

    assert abs(float(actual_jd2)) <= 0.5


@pytest.mark.parametrize(
    ("jd1", "jd2"),
    [
        (2451545.3, 0.4),
        (2451545.8, -0.9),
        (2451545.0, 1.2),
        (2451545.49, 0.49),
        (-10.7, 0.9),
    ],
)
def test_renormalize_split_jd_is_idempotent(jd1, jd2):
    jd1_first, jd2_first = renormalize_split_jd(jd1, jd2)
    jd1_second, jd2_second = renormalize_split_jd(jd1_first, jd2_first)

    assert_allclose(jd1_second, jd1_first, atol=0.0, rtol=0.0)
    assert_allclose(jd2_second, jd2_first, atol=0.0, rtol=0.0)


def test_julian_date_shapes():
    year = jnp.array([2000.0, 1970.0])
    month = jnp.array([1.0, 1.0])
    day = jnp.array([1.0, 1.0])
    hour = jnp.array([12.0, 0.0])
    minute = 0.0
    second = jnp.array([0.0, 0.123456789])

    jd1, jd2 = julian_date(year, month, day, hour, minute, second)

    assert jd1.shape == (2,)
    assert jd2.shape == (2,)


def test_calendar_date_shapes():
    jd1 = jnp.array([2451544.5, 2440587.5])
    jd2 = jnp.array([0.5, 1.4288980208333335e-06])

    year, month, day, hour, minute, second = calendar_date(jd1, jd2)

    assert year.shape == (2,)
    assert month.shape == (2,)
    assert day.shape == (2,)
    assert hour.shape == (2,)
    assert minute.shape == (2,)
    assert second.shape == (2,)


def test_renormalize_split_jd_shapes():
    jd1 = jnp.array([2451545.3, 2451545.8])
    jd2 = jnp.array([0.4, -0.9])

    actual_jd1, actual_jd2 = renormalize_split_jd(jd1, jd2)

    assert actual_jd1.shape == (2,)
    assert actual_jd2.shape == (2,)
