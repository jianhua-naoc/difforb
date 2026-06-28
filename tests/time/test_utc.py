import pytest
from astropy.time import Time
import erfa
import jax
import jax.numpy as jnp

from difforb.core.constants import DAY_S
from difforb.core.time.utc import utc_to_tai_single, tai_to_utc_single, julian_date_for_utc_single, calendar_date_for_utc_single, \
    utc_to_tai, tai_to_utc, julian_date_for_utc, calendar_date_for_utc
from difforb.core.time.utils import renormalize_split_jd
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)

UTC_TEST_LABELS = [
    "1962-01-01",
    "1962-12-01 12:00:00",
    "1963-10-31 23:59:59",
    "1963-11-01",
    "1963-12-01 12:00:00",
    "1963-12-31 23:59:59",
    "1964-01-01",
    "1964-02-15 12:00:00",
    "1964-03-31 23:59:59",
    "1964-04-01",
    "1964-06-16 12:00:00",
    "1964-08-31 23:59:59",
    "1964-09-01",
    "1964-11-01",
    "1964-12-31 23:59:59",
    "1965-01-01",
    "1965-01-30 12:00:00",
    "1965-02-28 23:59:59",
    "1965-03-01",
    "1965-05-01",
    "1965-06-30 23:59:59",
    "1965-07-01",
    "1965-08-01",
    "1965-08-31 23:59:59",
    "1965-09-01",
    "1965-11-01",
    "1965-12-31 23:59:59",
    "1966-01-01",
    "1967-01-16 12:00:00",
    "1968-01-31 23:59:59",
    "1968-02-01",
    "1970-01-16",
    "1971-12-31 23:59:59",
    "1972-01-01",
    "1972-01-01 12:00:00",
    "1972-06-30 23:59:59",
    "1972-06-30 23:59:60",
    "1972-06-30 23:59:60.5",
    "1972-07-01",
    "1972-12-31 23:59:59",
    "1972-12-31 23:59:60",
    "1973-01-01",
    "1973-12-31 23:59:59",
    "1973-12-31 23:59:60",
    "1974-01-01",
    "1974-12-31 23:59:59",
    "1974-12-31 23:59:60",
    "1975-01-01",
    "1975-12-31 23:59:59",
    "1975-12-31 23:59:60",
    "1976-01-01",
    "1976-12-31 23:59:59",
    "1976-12-31 23:59:60",
    "1977-01-01",
    "1977-12-31 23:59:59",
    "1977-12-31 23:59:60",
    "1978-01-01",
    "1978-12-31 23:59:59",
    "1978-12-31 23:59:60",
    "1979-01-01",
    "1979-12-31 23:59:59",
    "1979-12-31 23:59:60",
    "1980-01-01",
    "1981-06-30 23:59:59",
    "1981-06-30 23:59:60",
    "1981-07-01",
    "1982-06-30 23:59:59",
    "1982-06-30 23:59:60",
    "1982-07-01",
    "1983-06-30 23:59:59",
    "1983-06-30 23:59:60",
    "1983-07-01",
    "1985-06-30 23:59:59",
    "1985-06-30 23:59:60",
    "1985-07-01",
    "1987-12-31 23:59:59",
    "1987-12-31 23:59:60",
    "1988-01-01",
    "1989-12-31 23:59:59",
    "1989-12-31 23:59:60",
    "1990-01-01",
    "1990-12-31 23:59:59",
    "1990-12-31 23:59:60",
    "1991-01-01",
    "1992-06-30 23:59:59",
    "1992-06-30 23:59:60",
    "1992-07-01",
    "1993-06-30 23:59:59",
    "1993-06-30 23:59:60",
    "1993-07-01",
    "1994-06-30 23:59:59",
    "1994-06-30 23:59:60",
    "1994-07-01",
    "1995-12-31 23:59:59",
    "1995-12-31 23:59:60",
    "1996-01-01",
    "1997-06-30 23:59:59",
    "1997-06-30 23:59:60",
    "1997-07-01",
    "1998-12-31 23:59:59",
    "1998-12-31 23:59:60",
    "1999-01-01",
    "2000-01-01 12:34:56",
    "2005-12-31 23:59:59",
    "2005-12-31 23:59:60",
    "2006-01-01",
    "2008-12-31 23:59:59",
    "2008-12-31 23:59:60",
    "2009-01-01",
    "2012-06-30 23:59:59",
    "2012-06-30 23:59:60",
    "2012-07-01",
    "2015-06-30 23:59:59",
    "2015-06-30 23:59:60",
    "2015-07-01",
    "2016-12-31 23:59:59",
    "2016-12-31 23:59:60",
    "2016-12-31 23:59:60.5",
    "2017-01-01",
]

POST_1972_UTC_TEST_LABELS = [label for label in UTC_TEST_LABELS if int(label[:4]) >= 1972]


@pytest.mark.parametrize("label", UTC_TEST_LABELS)
def test_utc_to_tai_single_against_astropy(label):
    utc = Time(label, scale="utc")
    utc_jd1, utc_jd2 = utc.jd1, utc.jd2
    actual_tai_jd1, actual_tai_jd2 = utc_to_tai_single(utc_jd1, utc_jd2)
    actual_tai_jd1, actual_tai_jd2 = renormalize_split_jd(actual_tai_jd1, actual_tai_jd2)
    exp_tai = utc.tai
    exp_tai_jd1, exp_tai_jd2 = exp_tai.jd1, exp_tai.jd2
    print(
        "[utc_to_tai_single] "
        f"label={label:<22} "
        f"diff=({float((actual_tai_jd1 - exp_tai_jd1) * DAY_S):+.12e}, "
        f"{float((actual_tai_jd2 - exp_tai_jd2) * DAY_S):+.12e}) s "
    )
    assert_allclose(actual_tai_jd1, exp_tai_jd1, atol=0., rtol=0.)
    assert_allclose(actual_tai_jd2, exp_tai_jd2, atol=1e-13, rtol=0.)


@pytest.mark.parametrize("label", UTC_TEST_LABELS)
def test_tai_to_utc_single_against_astropy(label):
    t_utc = Time(label, scale="utc")
    t_tai = t_utc.tai
    tai_jd1, tai_jd2 = t_tai.jd1, t_tai.jd2
    exp_utc_jd1, exp_utc_jd2 = t_utc.jd1, t_utc.jd2
    actual_utc_jd1, actual_utc_jd2, _ = tai_to_utc_single(tai_jd1, tai_jd2)
    actual_utc_jd1, actual_utc_jd2 = renormalize_split_jd(actual_utc_jd1, actual_utc_jd2)
    print(
        "[tai_to_utc_single] "
        f"label={label:<22} "
        f"diff={float(((actual_utc_jd1 + actual_utc_jd2) - (exp_utc_jd1 + exp_utc_jd2)) * DAY_S):+.12e} s "
    )
    assert_allclose(actual_utc_jd1 + actual_utc_jd2, exp_utc_jd1 + exp_utc_jd2, atol=1e-15, rtol=0)


@pytest.mark.parametrize(
    "label, exp_valid",
    [
        ("1961-12-31 23:59:59.9", False),
        ("1962-01-01", True),
    ]
)
def test_tai_to_utc_single_ret_is_valid(label, exp_valid):
    t_utc = Time(label, scale="utc")
    t_tai = t_utc.tai
    tai_jd1, tai_jd2 = t_tai.jd1, t_tai.jd2
    _, _, is_valid = tai_to_utc_single(tai_jd1, tai_jd2)
    assert is_valid == exp_valid


def test_astropy_pre_1972_utc_string_is_not_civil_roundtrip():
    """Record Astropy/ERFA pre-1972 UTC formatting behavior.

    For the rate-adjusted UTC era before 1972-01-01, Astropy follows the
    ERFA/SOFA quasi-JD convention for UTC. In that convention, converting a
    physical instant from TT/TAI to a UTC object is internally self-consistent,
    but the formatted UTC clock string is not guaranteed to be a civil-UTC
    label that can be reparsed to recover the same instant.

    This test documents that behavior with the TT epoch 1972-01-01 00:00:00.
    Astropy returns a UTC string near 1971-12-31 23:59:17, and the UTC object
    itself converts back to the original TAI instant. However, reparsing the
    displayed string as a fresh UTC input produces a different TAI instant by
    about 0.1 s. The goal here is to pin down the external-library behavior,
    not to use it as the correctness criterion for this package's civil-UTC
    semantics.
    """
    tt = Time("1972-01-01", scale="tt")
    utc = tt.utc
    tai = tt.tai

    assert utc.iso.startswith("1971-12-31 23:59:17.")
    assert_allclose(utc.tai.jd, tai.jd, atol=1e-15, rtol=0.0)

    reparsed = Time(utc.iso, scale="utc")
    diff_sec = float((reparsed.tai.jd - tai.jd) * DAY_S)

    assert abs(diff_sec) > 0.05
    assert abs(diff_sec) < 0.2


@pytest.mark.parametrize("label", UTC_TEST_LABELS)
def test_utc_calendar_roundtrip(label):
    parts = label.split()
    year, month, day = [int(field) for field in parts[0].split("-")]
    if len(parts) == 1:
        hour, minute, second = 0, 0, 0.0
    else:
        hms = parts[1].split(":")
        hour = int(hms[0])
        minute = int(hms[1])
        second = float(hms[2])

    actual_jd1, actual_jd2 = julian_date_for_utc_single(year, month, day, hour, minute, second)
    rt_year, rt_month, rt_day, rt_hour, rt_minute, rt_second = calendar_date_for_utc_single(actual_jd1, actual_jd2)

    print(
        "[julian_calendar_roundtrip] "
        f"label={label:<22} "
        f"roundtrip=({int(rt_year):04d}-{int(rt_month):02d}-{int(rt_day):02d} "
        f"{int(rt_hour):02d}:{int(rt_minute):02d}:{float(rt_second):09.6f}) "
    )
    assert int(rt_year) == year
    assert int(rt_month) == month
    assert int(rt_day) == day
    assert int(rt_hour) == hour
    assert int(rt_minute) == minute
    assert_allclose(rt_second, second, atol=1e-9, rtol=0.0)


@pytest.mark.parametrize(
    (
            "year", "month", "day", "hour", "minute", "second",
            "exp_year", "exp_month", "exp_day", "exp_hour", "exp_minute", "exp_second",
    ),
    [
        pytest.param(1962, 1, 1.0, 25, 1, -1.0, 1962, 1, 2.0, 1, 0, 59.0, id="positive_overflow"),
        pytest.param(1962, 1, 2.0, -1, 0, 0.0, 1962, 1, 1.0, 23, 0, 0.0, id="negative_hour"),
        pytest.param(1962, 1, 1.0, 0, 59, 61.0, 1962, 1, 1.0, 1, 0, 1.0, id="second_overflow"),
        pytest.param(1962, 1, 1.5, 12, 0, 0.0, 1962, 1, 2.0, 0, 0, 0.0, id="fractional_day_overflow"),
        pytest.param(1962, 13, 1.0, 0, 0, 0.0, 1963, 1, 1.0, 0, 0, 0.0, id="month_overflow_forward"),
        pytest.param(1962, 0, 1.0, 0, 0, 0.0, 1961, 12, 1.0, 0, 0, 0.0, id="month_overflow_zero"),
        pytest.param(1962, -1, 1.0, 0, 0, 0.0, 1961, 11, 1.0, 0, 0, 0.0, id="month_overflow_negative"),
        pytest.param(1962, 13, 32.0, 25, 61, 61.0, 1963, 2, 2.0, 2, 2, 1.0, id="compound_month_day_time_overflow"),
    ],
)
def test_julian_date_for_utc_single_normalizes_overflow_fields(
        year, month, day, hour, minute, second,
        exp_year, exp_month, exp_day, exp_hour, exp_minute, exp_second):
    actual_jd1, actual_jd2 = julian_date_for_utc_single(year, month, day, hour, minute, second)
    expected_jd1, expected_jd2 = julian_date_for_utc_single(
        exp_year, exp_month, exp_day, exp_hour, exp_minute, exp_second
    )
    rt_year, rt_month, rt_day, rt_hour, rt_minute, rt_second = calendar_date_for_utc_single(actual_jd1, actual_jd2)

    assert_allclose(actual_jd1 + actual_jd2, expected_jd1 + expected_jd2, atol=0.0, rtol=0.0)
    assert int(rt_year) == exp_year
    assert int(rt_month) == exp_month
    assert int(rt_day) == int(exp_day)
    assert int(rt_hour) == exp_hour
    assert int(rt_minute) == exp_minute
    assert_allclose(rt_second, exp_second, atol=1e-9, rtol=0.0)


@pytest.mark.parametrize("label", POST_1972_UTC_TEST_LABELS)
def test_julian_date_for_utc_single_against_erfa_post_1972(label):
    parts = label.split()
    year, month, day = [int(field) for field in parts[0].split("-")]
    if len(parts) == 1:
        hour, minute, second = 0, 0, 0.0
    else:
        hms = parts[1].split(":")
        hour = int(hms[0])
        minute = int(hms[1])
        second = float(hms[2])

    actual_jd1, actual_jd2 = julian_date_for_utc_single(year, month, day, hour, minute, second)
    actual_jd1, actual_jd2 = renormalize_split_jd(actual_jd1, actual_jd2)
    exp_jd1, exp_jd2 = erfa.dtf2d("UTC", year, month, day, hour, minute, second)
    actual_jd = actual_jd1 + actual_jd2
    exp_jd = exp_jd1 + exp_jd2

    print(
        "[julian_date_for_utc_single/erfa] "
        f"label={label:<22} "
        f"diff={float((actual_jd - exp_jd) * DAY_S):+.12e} s "
    )
    assert_allclose(actual_jd, exp_jd, atol=1e-15, rtol=0.0)


@pytest.mark.parametrize("label", POST_1972_UTC_TEST_LABELS)
def test_calendar_date_for_utc_single_against_erfa_post_1972(label):
    utc = Time(label, scale="utc")
    year, month, day, hour, minute, second = calendar_date_for_utc_single(utc.jd1, utc.jd2)
    exp_year, exp_month, exp_day, ihmsf = erfa.d2dtf("UTC", 9, utc.jd1, utc.jd2)
    exp_hour = ihmsf["h"]
    exp_minute = ihmsf["m"]
    exp_second = ihmsf["s"] + ihmsf["f"] * 1e-9

    print(
        "[calendar_date_for_utc_single/erfa] "
        f"label={label:<22} "
        f"actual=({int(year):04d}-{int(month):02d}-{int(day):02d} "
        f"{int(hour):02d}:{int(minute):02d}:{float(second):09.6f}) "
        f"expected=({int(exp_year):04d}-{int(exp_month):02d}-{int(exp_day):02d} "
        f"{int(exp_hour):02d}:{int(exp_minute):02d}:{float(exp_second):09.6f}) "
    )
    assert int(year) == int(exp_year)
    assert int(month) == int(exp_month)
    assert int(day) == int(exp_day)
    assert int(hour) == int(exp_hour)
    assert int(minute) == int(exp_minute)
    assert_allclose(second, exp_second, atol=1e-9, rtol=0.0)


def test_utc_to_tai_shapes():
    utc_jd1 = jnp.array([2441499.0, 2457754.0])
    utc_jd2 = jnp.array([0.4999884260598836, 0.49999421302994174])

    tai_jd1, tai_jd2 = utc_to_tai(utc_jd1, utc_jd2)

    assert tai_jd1.shape == (2,)
    assert tai_jd2.shape == (2,)


def test_tai_to_utc_shapes():
    tai_jd1 = jnp.array([2441499.0, 2457754.0])
    tai_jd2 = jnp.array([0.5001157407407407, 0.5004282407407407])

    utc_jd1, utc_jd2, is_valid = tai_to_utc(tai_jd1, tai_jd2)

    assert utc_jd1.shape == (2,)
    assert utc_jd2.shape == (2,)
    assert is_valid.shape == (2,)


def test_julian_date_for_utc_shapes():
    year = jnp.array([1972.0, 2016.0])
    month = jnp.array([6.0, 12.0])
    day = jnp.array([30.0, 31.0])
    hour = 23.0
    minute = 59.0
    second = jnp.array([60.0, 60.5])

    jd1, jd2 = julian_date_for_utc(year, month, day, hour, minute, second)

    assert jd1.shape == (2,)
    assert jd2.shape == (2,)


def test_calendar_date_for_utc_shapes():
    jd1 = jnp.array([2441499.0, 2457754.0])
    jd2 = jnp.array([0.4999884260598836, 0.49999421302994174])

    year, month, day, hour, minute, second = calendar_date_for_utc(jd1, jd2)

    assert year.shape == (2,)
    assert month.shape == (2,)
    assert day.shape == (2,)
    assert hour.shape == (2,)
    assert minute.shape == (2,)
    assert second.shape == (2,)
