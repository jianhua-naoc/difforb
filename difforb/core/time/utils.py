"""Calendar helpers for split-Julian-date time classes.

This module provides scalar and vectorized helpers that convert between calendar dates, Julian dates, and day-time fields. The ``time`` classes in :mod:`difforb.core.time.timescale` use these helpers to build and display epochs.
"""

from functools import partial
from typing import Tuple

import jax
from jax import Array, numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float

from difforb.core.batch import safe_dispatch

jax.config.update("jax_enable_x64", True)

# =========================================================================
# Calendar Conversion Algorithms (Hybrid Julian/Gregorian)
# =========================================================================
# Default Julian Date for the start of the Gregorian calendar: 1582-10-15
GREGORIAN_START_JD = jnp.array(2299160.5, dtype=float)
NS_PER_SECOND = 1_000_000_000
NS_PER_MINUTE = 60 * NS_PER_SECOND
NS_PER_HOUR = 60 * NS_PER_MINUTE
NS_PER_DAY = 24 * NS_PER_HOUR


def julian_date_core(year: Float[Array, "..."], month: Float[Array, "..."], day: Float[Array, "..."],
                     gregorian_start: Float[Array, ""] = GREGORIAN_START_JD) -> Float[Array, "..."]:
    """
    Convert a calendar date to Julian date with a hybrid Julian/Gregorian calendar.

    Parameters
    ----------
    year : Float[Array, "..."]
        Calendar year (e.g., 2023.0).
    month : Float[Array, "..."]
        Calendar month (1-12).
    day : Float[Array, ""]
        Calendar day (can include fractional part).
    gregorian_start : Float[Array, ""], default=GREGORIAN_START_JD
        Julian date at which the Gregorian calendar begins.

    Returns
    -------
    Float[Array, ""]
        Julian date that matches the input calendar date.

    Notes
    -----
    This function follows Algorithm 3 from the reference. Dates before ``gregorian_start`` use the Julian calendar. Dates on or after ``gregorian_start`` use the Gregorian calendar.

    References
    ----------
    Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. 3rd ed., Algorithm 3, p. 618.
    """
    # Parameters from Table 15.14 for Julian/Gregorian
    y, j, m, n, r, p, q, v, u, s, t, w, A, C = 4716, 1401, 2, 12, 4, 1461, 0, 3, 5, 153, 2, 2, 184, -38

    h = month - m
    g = year + y - (n - h) // n
    f = (h - 1 + n) % n
    e = (p * g + q) // r + day - 1 - j

    # Algorithm 3, Step 5: Initial Julian Day computation
    jd_base = e + (s * f + t) // u

    # Algorithm 3, Step 6: Apply Gregorian leap year correction
    # Note: Step 6 is only performed for Gregorian type calendars
    jd_gregorian = jd_base - (3 * ((g + A) // 100)) // 4 - C
    jd_julian = jd_base

    # Compare with start threshold. We subtract 0.5 because the formula computes JD
    # for 12h (noon), but calendar dates begin at 0h (midnight).
    res_jd = jnp.where(jd_gregorian - 0.5 >= gregorian_start, jd_gregorian, jd_julian)

    return res_jd - 0.5


def calendar_date_single(jd1: Float[Array, ""], jd2: Float[Array, ""],
                         gregorian_start: Float[Array, ""] = GREGORIAN_START_JD) -> Tuple[
    Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """
    Convert a Julian date to a calendar date.

    Parameters
    ----------
    jd1, jd2 : Float[Array, ""]
        Split Julian date to convert.
    gregorian_start : Float[Array, ""], default=GREGORIAN_START_JD
        Julian date at which the Gregorian calendar begins.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]]
        Calendar fields ``(year, month, day, hour, minute, second)``.

    Notes
    -----
    This function follows Algorithm 4 from the reference. Dates before ``gregorian_start`` are returned in the Julian calendar. Later dates are returned in the Gregorian calendar. The returned ``day`` is the integer day of month. Any fractional day is split into ``hour``, ``minute``, and ``second``.

    References
    ----------
    Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. 3rd ed., Algorithm 4, p. 619.
    """
    jd = jd1 + jd2
    frac1, whole1 = jnp.modf(jd)
    frac, whole2 = jnp.modf(frac1 + 0.5)
    whole = whole1 + whole2

    # Parameters from Table 15.14
    y, j, m, n, r, p, q, v, u, s, t, w, A, B, C = 4716, 1401, 2, 12, 4, 1461, 0, 3, 5, 153, 2, 2, 184, 274277, -38

    # Algorithm 4, Step 1: Base value f
    f_julian = whole + j

    # Algorithm 4, Step 1a: Gregorian correction
    f_gregorian = f_julian + (((4 * whole + B) // 146097) * 3) // 4 + C

    # Switch between Julian and Gregorian algorithm based on the JD
    f = jnp.where(jd >= gregorian_start, f_gregorian, f_julian)

    # Follow-up steps 2 to 7
    e = r * f + v
    g = (e % p) // r
    h = u * g + w
    day_int = (h % s) // u + 1
    month = ((h // s + m) % n) + 1
    year = e // p - y + (n + m - month) // n

    jd_midnight = julian_date_core(year, month, day_int, gregorian_start)
    sec_total = ((jd1 - jd_midnight) + jd2) * 86400.0
    total_ns = jnp.rint(sec_total * NS_PER_SECOND).astype(jnp.int64)
    hour = total_ns // NS_PER_HOUR
    rem_ns = total_ns - hour * NS_PER_HOUR
    minute = rem_ns // NS_PER_MINUTE
    second_ns = rem_ns - minute * NS_PER_MINUTE
    second_whole = second_ns // NS_PER_SECOND
    second_frac_ns = second_ns - second_whole * NS_PER_SECOND
    second = second_whole.astype(jnp.float64) + second_frac_ns.astype(jnp.float64) / NS_PER_SECOND

    return year, month, day_int, hour.astype(jnp.float64), minute.astype(jnp.float64), second


# =========================================================================
# Vectorized Public API
# =========================================================================

@jax.jit
def julian_date(year: Float[ArrayLike, "..."], month: Float[ArrayLike, "..."], day: Float[ArrayLike, "..."],
                hour: Float[ArrayLike, "..."], min: Float[ArrayLike, "..."], sec: Float[ArrayLike, "..."],
                gregorian_start: Float[Array, ""] = GREGORIAN_START_JD) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Convert calendar dates to split Julian dates.

    Parameters
    ----------
    year, month, day, hour, min, sec : Float[ArrayLike, "..."]
        Calendar fields of the requested epochs.
    gregorian_start : Float[Array, ""], default=GREGORIAN_START_JD
        Julian date at which the Gregorian calendar begins.

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."]]
        Split Julian date of the requested epochs.

    Notes
    -----
    Vectorize :func:`julian_date_core`.
    """
    year = jnp.asarray(year, dtype=jnp.float64)
    month = jnp.asarray(month, dtype=jnp.float64)
    day = jnp.asarray(day, dtype=jnp.float64)
    hour = jnp.asarray(hour, dtype=jnp.float64)
    min = jnp.asarray(min, dtype=jnp.float64)
    sec = jnp.asarray(sec, dtype=jnp.float64)
    gregorian_start = jnp.asarray(gregorian_start, dtype=jnp.float64)
    year, month, day, hour, min, sec = jnp.broadcast_arrays(year, month, day, hour, min, sec)

    day_int = jnp.floor(day)
    day_frac = day - day_int
    jd_midnight = julian_date_core(year, month, day_int, gregorian_start)
    jd_offset_from_midnight = (day_frac * 86400.0 + hour * 3600.0 + min * 60.0 + sec) / 86400.0
    return renormalize_split_jd(jd_midnight, jd_offset_from_midnight)


@jax.jit
def calendar_date(jd1: Float[Array, "..."], jd2: Float[Array, "..."],
                  gregorian_start: Float[Array, ""] = GREGORIAN_START_JD) -> Tuple[
    Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """Convert a Julian date to a calendar date.

    Parameters
    ----------
    jd1, jd2 : Float[Array, "..."]
        Split Julian date of the epochs to convert.
    gregorian_start : Float[Array, ""], default=GREGORIAN_START_JD
        Julian date at which the Gregorian calendar begins.

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]
        Calendar fields ``(year, month, day, hour, minute, second)``.

    Notes
    -----
    Vectorize :func:`calendar_date_single`.
    """
    wrapper = partial(calendar_date_single, gregorian_start=gregorian_start)
    return safe_dispatch(wrapper, (0, 0), jd1, jd2)


@jax.jit
def renormalize_split_jd(
        jd1: Float[Array, "..."], jd2: Float[Array, "..."]
) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Renormalize a split Julian date pair.

    Parameters
    ----------
    jd1 : Float[Array, "..."]
        Large component of the Julian date.
    jd2 : Float[Array, "..."]
        Small remainder component of the Julian date.

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."]]
        Renormalized ``(jd1, jd2)`` with ``jd2`` kept close to zero.

    Notes
    -----
    ``jnp.round`` is used instead of ``floor`` so that the residual stays near the symmetric interval ``[-0.5, 0.5]``. This minimizes the
    magnitude of ``jd2`` and preserves precision in later arithmetic.
    """
    # Keep the residual centered near zero rather than forcing a positive
    # fractional part. This is more stable for repeated time arithmetic.
    jd1_int = jnp.round(jd1)
    jd1_frac = jd1 - jd1_int

    jd2_total = jd2 + jd1_frac

    jd2_int = jnp.round(jd2_total)
    jd2_frac = jd2_total - jd2_int

    return jd1_int + jd2_int, jd2_frac


@jax.jit
def ut1_fraction(ut1_jd1: Float[Array, "..."], ut1_jd2: Float[Array, "..."]) -> Float[Array, "..."]:
    """Return the fraction of one ``UT1`` day.

    Parameters
    ----------
    ut1_jd1, ut1_jd2 : Float[Array, "..."]
        Split Julian date of the ``UT1`` epoch.

    Returns
    -------
    Float[Array, "..."]
        Fractional part of the ``UT1`` day in the range ``[0, 1)``.
    """
    return ((ut1_jd1 - 0.5) % 1.0 + ut1_jd2) % 1.0
