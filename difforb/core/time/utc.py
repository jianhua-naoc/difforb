"""Low-level ``UTC`` kernels for ``time``.

This module implements split-Julian-date helpers for ``UTC`` handling after 1962-01-01. It covers the pre-1972 linear ``TAI - UTC`` model, post-1972 leap seconds, and the quasi-Julian-date convention used for ``UTC`` calendar conversion.

Calendar conversion depends on :mod:`difforb.core.time.utils`. These kernels are used by :mod:`difforb.core.time.timescale`.
"""

from typing import Tuple

import equinox as eqx
import jax
from jax import numpy as jnp, Array
from jax.typing import ArrayLike
from jaxtyping import Float, Bool

from difforb.core.batch import safe_dispatch
from difforb.core.constants import DAY_S
from difforb.core.time.utils import renormalize_split_jd, julian_date_core, calendar_date_single, NS_PER_SECOND, \
    NS_PER_MINUTE, NS_PER_HOUR, NS_PER_DAY

jax.config.update("jax_enable_x64", True)

# ``UTC`` begins on 1962-01-01 here.
UTC_START_JD = 2437665.5
UTC_ZERO_SNAP_NS = 1
UTC_INTEGER_SECOND_SNAP_NS = 3

# ``TAI - UTC`` table. Columns are ``UTC`` JD, offset base (in seconds), linear rate (in seconds), and reference JD,
# matching the pre-1972 linear definitions and post-1972 steps.
# Given that EOP file only covers times after 1962-01-01, table starts from 1962-01-01.
TAIDUTC_TABLE = jnp.array([
    [2437665.5, 1.8458580, 0.0011232, 2437665.5],
    [2438334.5, 1.9458580, 0.0011232, 2437665.5],
    [2438395.5, 3.2401300, 0.001296, 2438761.5],
    [2438486.5, 3.3401300, 0.001296, 2438761.5],
    [2438639.5, 3.4401300, 0.001296, 2438761.5],
    [2438761.5, 3.5401300, 0.001296, 2438761.5],
    [2438820.5, 3.6401300, 0.001296, 2438761.5],
    [2438942.5, 3.7401300, 0.001296, 2438761.5],
    [2439004.5, 3.8401300, 0.001296, 2438761.5],
    [2439126.5, 4.3131700, 0.002592, 2439126.5],
    [2439887.5, 4.2131700, 0.002592, 2439126.5],
    [2441317.5, 10.0, 0.0, 0.0],
    [2441499.5, 11.0, 0.0, 0.0],
    [2441683.5, 12.0, 0.0, 0.0],
    [2442048.5, 13.0, 0.0, 0.0],
    [2442413.5, 14.0, 0.0, 0.0],
    [2442778.5, 15.0, 0.0, 0.0],
    [2443144.5, 16.0, 0.0, 0.0],
    [2443509.5, 17.0, 0.0, 0.0],
    [2443874.5, 18.0, 0.0, 0.0],
    [2444239.5, 19.0, 0.0, 0.0],
    [2444786.5, 20.0, 0.0, 0.0],
    [2445151.5, 21.0, 0.0, 0.0],
    [2445516.5, 22.0, 0.0, 0.0],
    [2446247.5, 23.0, 0.0, 0.0],
    [2447161.5, 24.0, 0.0, 0.0],
    [2447892.5, 25.0, 0.0, 0.0],
    [2448257.5, 26.0, 0.0, 0.0],
    [2448804.5, 27.0, 0.0, 0.0],
    [2449169.5, 28.0, 0.0, 0.0],
    [2449534.5, 29.0, 0.0, 0.0],
    [2450083.5, 30.0, 0.0, 0.0],
    [2450630.5, 31.0, 0.0, 0.0],
    [2451179.5, 32.0, 0.0, 0.0],
    [2453736.5, 33.0, 0.0, 0.0],
    [2454832.5, 34.0, 0.0, 0.0],
    [2456109.5, 35.0, 0.0, 0.0],
    [2457204.5, 36.0, 0.0, 0.0],
    [2457754.5, 37.0, 0.0, 0.0]
], dtype=float)


def get_tai_boundaries():
    """Build the ``TAI`` boundary epochs of the ``TAI - UTC`` table.

    Returns
    -------
    Float[Array, "n"]
        Start epochs of each table segment, expressed in ``TAI`` Julian Date.

    Notes
    -----
    The source table is indexed by ``UTC``. This helper maps each segment start to the matching ``TAI`` epoch so that ``TAI -> UTC`` lookup can use the same piecewise model.
    """
    utc_jd = TAIDUTC_TABLE[:, 0]
    base = TAIDUTC_TABLE[:, 1] / DAY_S
    rate = TAIDUTC_TABLE[:, 2] / DAY_S
    ref_jd = TAIDUTC_TABLE[:, 3]

    tai_jd = (1. + rate) * utc_jd + base - ref_jd * rate
    return tai_jd


# ``TAI - UTC`` table indexed by ``TAI``.
TAI_BOUNDARIES = get_tai_boundaries()


def select_taidutc_table_single(epoch_jd: Float[Array, ""], boundaries: Float[Array, "N"]) -> tuple[
    Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Select one ``TAI - UTC`` table row from boundary epochs.

    Parameters
    ----------
    epoch_jd : Float[Array, ""]
        Epoch in Julian Date.
    boundaries : Float[Array, "N"]
        Boundary epochs for the lookup table, expressed in the same time scale as ``epoch_jd``.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]
        Table parameters ``(base, rate, ref_jd)`` for the matched segment.
    """
    idx = jnp.searchsorted(boundaries, epoch_jd, side='right') - 1
    idx = jnp.clip(idx, 0, len(boundaries) - 1)
    params = TAIDUTC_TABLE[idx]
    base = params[1]
    rate = params[2]
    ref_jd = params[3]
    return base, rate, ref_jd


def select_taidutc_table_from_utc_single(utc_quasi_jd1: Float[Array, ""], utc_quasi_jd2: Float[Array, ""]) -> tuple[
    Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Select one ``TAI - UTC`` table row from a ``UTC`` epoch."""
    return select_taidutc_table_single(utc_quasi_jd1 + utc_quasi_jd2, TAIDUTC_TABLE[:, 0])


def select_taidutc_table_from_tai_single(tai_jd1: Float[Array, ""], tai_jd2: Float[Array, ""]) -> tuple[
    Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Select one ``TAI - UTC`` table row from a ``TAI`` epoch."""
    return select_taidutc_table_single(tai_jd1 + tai_jd2, TAI_BOUNDARIES)


def compute_taidutc_from_utc_single(utc_quasi_jd1: Float[Array, ""], utc_quasi_jd2: Float[Array, ""]) -> Float[Array, ""]:
    """Evaluate ``TAI - UTC`` in seconds for one ``UTC`` epoch.

    Parameters
    ----------
    utc_quasi_jd1, utc_quasi_jd2 : Float[Array, ""]
        Split quasi-Julian date of the ``UTC`` epoch.

    Returns
    -------
    Float[Array, ""]
        ``TAI - UTC`` in seconds.
    """
    base, rate, ref_jd = select_taidutc_table_from_utc_single(utc_quasi_jd1, utc_quasi_jd2)
    return base + (utc_quasi_jd1 - ref_jd + utc_quasi_jd2) * rate


def compute_utc_actual_jd_from_tai_single(tai_jd1: Float[Array, ""], tai_jd2: Float[Array, ""]) -> tuple[Float[Array, ""],
Float[Array, ""]]:
    """Recover the real ``UTC`` Julian date from one ``TAI`` epoch.

    Parameters
    ----------
    tai_jd1, tai_jd2 : Float[Array, ""]
        Split Julian date of the ``TAI`` epoch.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""]]
        Split actual Julian date of the matching ``UTC`` epoch, before quasi-Julian-date handling for leap seconds.
    """
    base, rate, ref_jd = select_taidutc_table_from_tai_single(tai_jd1, tai_jd2)
    base_day, rate_day = base / DAY_S, rate / DAY_S
    utc_actual_jd1 = tai_jd1
    utc_actual_jd2 = (tai_jd2 + (ref_jd - tai_jd1) * rate_day - base_day) / (1.0 + rate_day)
    utc_actual_jd1, utc_actual_jd2 = renormalize_split_jd(utc_actual_jd1, utc_actual_jd2)
    return utc_actual_jd1, utc_actual_jd2


def utc_day_length_single(utc_jd_midnight: Float[Array, ""]) -> tuple[Float[Array, ""], Bool[Array, ""]]:
    """Return the actual length of one ``UTC`` day.

    Parameters
    ----------
    utc_jd_midnight : Float[Array, ""]
        Midnight epoch of the ``UTC`` day, in Julian Date.

    Returns
    -------
    tuple[Float[Array, ""], Bool[Array, ""]]
        Actual day length in seconds and a flag that marks whether the day contains a positive leap second.
    """
    taidutc_0 = compute_taidutc_from_utc_single(utc_jd_midnight, 0.0)
    taidutc_12 = compute_taidutc_from_utc_single(utc_jd_midnight, 0.5)
    taidutc_24 = compute_taidutc_from_utc_single(utc_jd_midnight + 1.0, 0.0)
    diff_taidutc = taidutc_24 - (2.0 * taidutc_12 - taidutc_0)
    actual_day_s_in_utc = DAY_S + diff_taidutc
    is_leap = diff_taidutc > 0.5
    return actual_day_s_in_utc, is_leap


def utc_to_tai_single(utc_quasi_jd1: Float[Array, ""], utc_quasi_jd2: Float[Array, ""]) -> tuple[
    Float[Array, ""], Float[Array, ""]]:
    """
    Transform ``UTC`` epoch to ``TAI`` epoch.

    Parameters
    ----------
    utc_quasi_jd1, utc_quasi_jd2 : Float[Array, ""]
        Split quasi-Julian date of the ``UTC`` epoch.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""]]
        Split Julian date of the ``TAI`` epoch.

    Notes
    -----
    ``utc_quasi_jd1`` and ``utc_quasi_jd2`` use the quasi-Julian-date convention for ``UTC``. The day fraction is normalized by the actual length of that ``UTC`` day. ``utc_actual_jd1`` and ``utc_actual_jd2`` would instead mean the matching physical ``UTC`` Julian date measured from actual elapsed seconds in the day.

    This function takes the quasi-Julian-date form as input, recovers the elapsed seconds within the actual ``UTC`` day, and then builds the matching ``TAI`` epoch. Before 1972, the transform is through piecewise-linear rate changes rather than integer leap seconds.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Table 3.1.
    """
    taidutc = compute_taidutc_from_utc_single(utc_quasi_jd1, utc_quasi_jd2)

    utc_quasi_midnight_jd, utc_quasi_jd_frac = split_utc_quasi_jd(utc_quasi_jd1, utc_quasi_jd2)

    utc_day_s, _ = utc_day_length_single(utc_quasi_midnight_jd)
    elapsed_seconds = utc_quasi_jd_frac * utc_day_s

    tai_jd1 = utc_quasi_midnight_jd
    tai_jd2 = (elapsed_seconds + taidutc) / DAY_S
    return tai_jd1, tai_jd2


def tai_to_utc_single(tai_jd1: Float[Array, ""], tai_jd2: Float[Array, ""]) -> tuple[Float[Array, ""],
Float[Array, ""], Bool[Array, ""]]:
    """Transform ``TAI`` epoch to ``UTC`` epoch.

    Parameters
    ----------
    tai_jd1, tai_jd2 : Float[Array, ""]
        Split Julian date of the ``TAI`` epoch.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""], Bool[Array, ""]]
        Split quasi-Julian date of the ``UTC`` epoch and a flag that marks whether the result is a valid ``UTC`` epoch on or after 1962-01-01.

    Notes
    -----
    ``utc_actual_jd1`` and ``utc_actual_jd2`` mean the physical ``UTC`` Julian date recovered from ``TAI``. ``utc_quasi_jd1`` and ``utc_quasi_jd2`` mean the quasi-Julian-date form used by this module for ``UTC`` storage and calendar conversion.

    This function first recovers the actual ``UTC`` Julian date from ``TAI`` and then converts it to the output quasi-Julian-date form. Values before 1962-01-01 do not belong to the mixed-``UT`` / ``UTC`` convention used by this module, so the returned validity flag is ``False`` for those epochs.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Table 3.1.
    """
    tai_jd = tai_jd1 + tai_jd2
    is_valid = tai_jd >= TAI_BOUNDARIES[0]
    utc_actual_jd1, utc_actual_jd2 = compute_utc_actual_jd_from_tai_single(tai_jd1, tai_jd2)
    utc_actual_jd = utc_actual_jd1 + utc_actual_jd2
    utc_actual_nominal_midnight_jd = jnp.floor(utc_actual_jd + 0.5) - 0.5

    # Get the next segment boundary in ``UTC``.
    idx = jnp.searchsorted(TAI_BOUNDARIES, tai_jd, side='right') - 1
    next_utc_boundary = jnp.where(
        idx + 1 < TAIDUTC_TABLE.shape[0],
        TAIDUTC_TABLE[jnp.minimum(idx + 1, TAIDUTC_TABLE.shape[0] - 1), 0],
        jnp.inf
    )

    # If the nominal midnight reaches the next segment, the epoch is inside the leap-second part of the previous day.
    in_leap_second = utc_actual_nominal_midnight_jd >= next_utc_boundary
    utc_quasi_midnight_jd = jnp.where(in_leap_second, utc_actual_nominal_midnight_jd - 1.0, utc_actual_nominal_midnight_jd)

    # Get the actual length of that ``UTC`` day.
    utc_day_s, _ = utc_day_length_single(utc_quasi_midnight_jd)

    # Build the quasi-Julian-date fraction.
    elapsed_ns = jnp.rint(((utc_actual_jd1 - utc_quasi_midnight_jd) + utc_actual_jd2) * DAY_S * NS_PER_SECOND).astype(jnp.int64)
    utc_day_ns = jnp.rint(utc_day_s * NS_PER_SECOND).astype(jnp.int64)

    carry_next_day = utc_day_ns - elapsed_ns <= UTC_ZERO_SNAP_NS
    elapsed_ns = jnp.where(carry_next_day, 0, elapsed_ns)
    utc_quasi_midnight_jd = jnp.where(carry_next_day, utc_quasi_midnight_jd + 1.0, utc_quasi_midnight_jd)
    elapsed_ns = jnp.where(jnp.abs(elapsed_ns) <= UTC_ZERO_SNAP_NS, 0, elapsed_ns)

    utc_quasi_jd_frac = elapsed_ns.astype(jnp.float64) / utc_day_ns.astype(jnp.float64)

    utc_quasi_jd1 = utc_quasi_midnight_jd
    utc_quasi_jd2 = utc_quasi_jd_frac
    return utc_quasi_jd1, utc_quasi_jd2, is_valid


def julian_date_for_utc_single(
        year: Float[Array, ""], month: Float[Array, ""], day: Float[Array, ""],
        hour: Float[Array, ""], minute: Float[Array, ""], second: Float[Array, ""]
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Convert one ``UTC`` calendar date to a quasi-Julian date.

    Parameters
    ----------
    year, month, day, hour, minute, second : Float[Array, ""]
        Calendar fields of the ``UTC`` epoch.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""]]
        Split quasi-Julian date of the ``UTC`` epoch.

    Notes
    -----
    All valid ``UTC`` epochs handled by this module are on or after 1962-01-01, so the calendar date is always interpreted in the Gregorian calendar.

    This follows the same quasi-Julian-date convention as SOFA ``iauDtf2d``. On leap-second days, the day fraction is normalized by the actual ``UTC`` day length rather than a fixed 86400-second day.

    Before 1972, ``UTC`` used rate adjustments rather than integer leap seconds. In that era, this function keeps the reported civil ``UTC`` label unchanged when it is converted to the internal quasi-Julian-date form and later converted back by :func:`calendar_date_for_utc_single`.

    This is different from low-level ``SOFA`` / ``ERFA`` ``dtf2d`` / ``d2dtf`` behavior. Those routines follow their own historical quasi-Julian-date convention and can return second values near ``58.9`` for some timestamps that are commonly written as ``23:59:59`` in external data files.

    For example, the civil label ``1963-10-31 23:59:59`` maps to the same physical instant in both conventions. The difference is only in how that instant is written back as ``UTC`` calendar fields: this module returns the original label, while low-level ``SOFA`` / ``ERFA`` formatting can return a second value near ``58.9`` for the same instant.

    The convention used here is meant for orbit-determination workflows. Observation files and ``EOP`` files are usually written with civil ``UTC`` date labels, and this function keeps those labels stable through internal time-scale conversions.
    """

    # -------------------------------------------------------------------------
    # Step 1: Normalize the civil day and the nominal 24-hour clock part
    # -------------------------------------------------------------------------
    day_int = jnp.floor(day)
    day_frac_sec = (day - day_int) * 86400.0
    nominal_time = minute * 60.0 + hour * 3600.0 + day_frac_sec
    day_offset = jnp.floor(nominal_time / 86400.0)
    sec_in_nominal_day = nominal_time - day_offset * 86400.0

    jd_midnight = julian_date_core(year, month, day_int + day_offset)
    time_in_day = sec_in_nominal_day + second

    # -------------------------------------------------------------------------
    # Step 2: Rebalance against the actual ``UTC`` day length
    # -------------------------------------------------------------------------
    def neg_cond(state):
        _, cur_time_in_day = state
        return cur_time_in_day < 0.0

    def neg_body(state):
        cur_midnight, cur_time_in_day = state
        prev_midnight = cur_midnight - 1.0
        prev_day_s, _ = utc_day_length_single(prev_midnight)
        return prev_midnight, cur_time_in_day + prev_day_s

    jd_midnight, time_in_day = jax.lax.while_loop(neg_cond, neg_body, (jd_midnight, time_in_day))

    def pos_cond(state):
        cur_midnight, cur_time_in_day = state
        cur_day_s, _ = utc_day_length_single(cur_midnight)
        return cur_time_in_day >= cur_day_s

    def pos_body(state):
        cur_midnight, cur_time_in_day = state
        cur_day_s, _ = utc_day_length_single(cur_midnight)
        return cur_midnight + 1.0, cur_time_in_day - cur_day_s

    jd_midnight, time_in_day = jax.lax.while_loop(pos_cond, pos_body, (jd_midnight, time_in_day))

    # -------------------------------------------------------------------------
    # Step 3: Convert the normalized time-of-day to a quasi-day fraction
    # -------------------------------------------------------------------------
    day_s, _ = utc_day_length_single(jd_midnight)
    jd_frac = time_in_day / day_s

    return jd_midnight, jd_frac


def calendar_date_for_utc_single(
        utc_quasi_jd1: Float[Array, ""], utc_quasi_jd2: Float[Array, ""]
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Convert one ``UTC`` quasi-Julian date to calendar fields.

    Parameters
    ----------
    utc_quasi_jd1, utc_quasi_jd2 : Float[Array, ""]
        Split quasi-Julian date of the ``UTC`` epoch.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]]
        Calendar fields ``(year, month, day, hour, minute, second)``.

    Notes
    -----
    All valid ``UTC`` epochs handled by this module are on or after 1962-01-01, so the calendar date is always returned in the Gregorian calendar.

    This follows the same quasi-Julian-date convention as SOFA ``iauD2dtf``. Leap seconds are preserved as ``23:59:60`` instead of being carried into the next day.

    Before 1972, ``UTC`` used rate adjustments rather than integer leap seconds. In that era, this function is designed to work with :func:`julian_date_for_utc_single` so that a civil ``UTC`` label can round-trip through the internal quasi-Julian-date form without changing the reported timestamp.

    This is different from low-level ``SOFA`` / ``ERFA`` ``d2dtf`` behavior. Those routines follow their own historical quasi-Julian-date convention and can return second values near ``58.9`` for some epochs that are commonly written as ``23:59:59`` in external data files.

    For example, the civil label ``1963-10-31 23:59:59`` corresponds to the same physical instant in both conventions. The difference is only in the formatting rule used when that instant is written back as ``UTC`` calendar fields: this module returns ``23:59:59``, while low-level ``SOFA`` / ``ERFA`` formatting can return a second value near ``58.9`` for the same instant.

    The convention used here is meant for orbit-determination workflows. Observation files and ``EOP`` files are usually written with civil ``UTC`` date labels, and this function returns those labels in the same form after internal conversion.
    """

    utc_quasi_midnight_jd, utc_quasi_jd_frac = split_utc_quasi_jd(utc_quasi_jd1, utc_quasi_jd2)

    # -------------------------------------------------------------------------
    # Step 1: Recover the base calendar day from midnight
    # -------------------------------------------------------------------------
    year, month, day_int, _, _, _ = calendar_date_single(utc_quasi_midnight_jd, 0.0)

    # -------------------------------------------------------------------------
    # Step 2: Get the actual length of the ``UTC`` day
    # -------------------------------------------------------------------------
    utc_day_s, is_leap = utc_day_length_single(utc_quasi_midnight_jd)

    # -------------------------------------------------------------------------
    # Step 3: Convert the quasi-day fraction back to elapsed seconds
    # -------------------------------------------------------------------------
    total_ns = jnp.rint(utc_quasi_jd_frac * utc_day_s * NS_PER_SECOND).astype(jnp.int64)
    total_ns = jnp.where(jnp.abs(total_ns) <= UTC_ZERO_SNAP_NS, 0, total_ns)

    # -------------------------------------------------------------------------
    # Step 4: Handle day rollover and split hour, minute, second
    # -------------------------------------------------------------------------
    max_ns = jnp.where(is_leap, NS_PER_DAY + NS_PER_SECOND, NS_PER_DAY)
    rollover = total_ns >= max_ns

    utc_quasi_midnight_jd_final = jnp.where(rollover, utc_quasi_midnight_jd + 1.0, utc_quasi_midnight_jd)
    total_ns_final = jnp.where(rollover, total_ns - max_ns, total_ns)
    total_ns_final = jnp.where(jnp.abs(total_ns_final) <= UTC_ZERO_SNAP_NS, 0, total_ns_final)

    # Rebuild the date after rollover, if needed.
    year, month, day_int, _, _, _ = calendar_date_single(utc_quasi_midnight_jd_final, 0.0)

    hour = total_ns_final // NS_PER_HOUR
    rem_ns = total_ns_final - hour * NS_PER_HOUR
    minute = rem_ns // NS_PER_MINUTE
    second_ns = rem_ns - minute * NS_PER_MINUTE
    second_down = (second_ns // NS_PER_SECOND) * NS_PER_SECOND
    second_up = second_down + NS_PER_SECOND
    down_diff = second_ns - second_down
    up_diff = second_up - second_ns
    second_ns = jnp.where(down_diff <= UTC_INTEGER_SECOND_SNAP_NS, second_down, second_ns)
    second_ns = jnp.where(up_diff <= UTC_INTEGER_SECOND_SNAP_NS, second_up, second_ns)
    second_whole = second_ns // NS_PER_SECOND
    second_frac_ns = second_ns - second_whole * NS_PER_SECOND
    second = second_whole.astype(jnp.float64) + second_frac_ns.astype(jnp.float64) / NS_PER_SECOND

    # Keep leap-second epochs as ``23:59:60`` instead of rolling into the next day.
    is_leap_second_range = total_ns_final >= NS_PER_DAY

    hour = jnp.where(is_leap_second_range, 23.0, hour.astype(jnp.float64))
    minute = jnp.where(is_leap_second_range, 59.0, minute.astype(jnp.float64))
    leap_second_ns = total_ns_final - NS_PER_DAY
    leap_second_whole = leap_second_ns // NS_PER_SECOND
    leap_second_frac_ns = leap_second_ns - leap_second_whole * NS_PER_SECOND
    leap_second = 60.0 + leap_second_whole.astype(jnp.float64) + leap_second_frac_ns.astype(jnp.float64) / NS_PER_SECOND
    second = jnp.where(is_leap_second_range, leap_second, second)

    return year, month, jnp.floor(day_int), hour, minute, second


# =========================================================================
# Vectorized Public API
# =========================================================================

@eqx.filter_jit
def split_utc_quasi_jd(utc_quasi_jd1: Float[Array, "..."], utc_quasi_jd2: Float[Array, "..."]) -> tuple[
    Float[Array, "..."], Float[Array, "..."]]:
    """Split one ``UTC`` quasi-Julian date into midnight and day fraction.

    Parameters
    ----------
    utc_quasi_jd1, utc_quasi_jd2 : Float[Array, "..."]
        Split quasi-Julian date of the ``UTC`` epoch.

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."]]
        Midnight Julian Date and the quasi-day fraction measured from that midnight.
    """
    utc_quasi_jd_sum = utc_quasi_jd1 + utc_quasi_jd2
    utc_quasi_midnight_jd = jnp.floor(utc_quasi_jd_sum + 0.5) - 0.5
    utc_quasi_jd_delta = jax.lax.optimization_barrier(utc_quasi_jd1 - utc_quasi_midnight_jd)
    utc_quasi_jd_frac = utc_quasi_jd_delta + utc_quasi_jd2

    carry_pos = utc_quasi_jd_frac >= 1.0
    utc_quasi_midnight_jd = jnp.where(carry_pos, utc_quasi_midnight_jd + 1.0, utc_quasi_midnight_jd)
    utc_quasi_jd_frac = jnp.where(carry_pos, utc_quasi_jd_frac - 1.0, utc_quasi_jd_frac)

    carry_neg = utc_quasi_jd_frac < 0.0
    utc_quasi_midnight_jd = jnp.where(carry_neg, utc_quasi_midnight_jd - 1.0, utc_quasi_midnight_jd)
    utc_quasi_jd_frac = jnp.where(carry_neg, utc_quasi_jd_frac + 1.0, utc_quasi_jd_frac)

    return utc_quasi_midnight_jd, utc_quasi_jd_frac


@eqx.filter_jit
def utc_to_tai(utc_quasi_jd1: Float[Array, "..."], utc_quasi_jd2: Float[Array, "..."]) -> tuple[
    Float[Array, "..."], Float[Array, "..."]]:
    """Transform ``UTC`` epoch to ``TAI`` epoch.

    Parameters
    ----------
    utc_quasi_jd1, utc_quasi_jd2 : Float[Array, "..."]
        Split quasi-Julian date of the ``UTC`` epoch.

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."]]
        Split Julian date of the ``TAI`` epoch.

    Notes
    -----
    ``utc_quasi_jd1`` and ``utc_quasi_jd2`` use the quasi-Julian-date convention for ``UTC`` storage. ``utc_actual_jd1`` and ``utc_actual_jd2`` would mean the matching physical ``UTC`` Julian date measured from actual elapsed seconds in the day.

    Before 1972, the transform is through piecewise-linear rate changes rather than integer leap seconds.

    Vectorize :func:`utc_to_tai_single`.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Table 3.1.
    """
    return safe_dispatch(utc_to_tai_single, (0, 0), utc_quasi_jd1, utc_quasi_jd2)


@eqx.filter_jit
def tai_to_utc(tai_jd1: Float[Array, "..."], tai_jd2: Float[Array, "..."]) -> tuple[Float[Array, "..."],
Float[Array, "..."], Bool[Array, "..."]]:
    """Transform ``TAI`` epoch to ``UTC`` epoch.

    Parameters
    ----------
    tai_jd1, tai_jd2 : Float[Array, "..."]
        Split Julian date of the ``TAI`` epoch.

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."], Bool[Array, "..."]]
        Split quasi-Julian date of the ``UTC`` epoch and a flag that marks whether each result is a valid ``UTC`` epoch on or after 1962-01-01.

    Notes
    -----
    This function first recovers ``utc_actual_jd1`` and ``utc_actual_jd2`` from ``TAI`` and then converts them to output ``utc_quasi_jd1`` and ``utc_quasi_jd2``. The quasi-Julian-date form is the one used by this module for ``UTC`` storage and calendar conversion.

    Vectorize :func:`tai_to_utc_single`.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Table 3.1.
    """
    return safe_dispatch(tai_to_utc_single, (0, 0), tai_jd1, tai_jd2)


@eqx.filter_jit
def julian_date_for_utc(year: Float[ArrayLike, "..."], month: Float[ArrayLike, "..."], day: Float[ArrayLike, "..."],
                        hour: Float[ArrayLike, "..."], min: Float[ArrayLike, "..."], sec: Float[ArrayLike, "..."]) -> tuple[
    Float[Array, "..."], Float[Array, "..."]]:
    """Convert ``UTC`` calendar dates to quasi-Julian dates.

    Parameters
    ----------
    year, month, day, hour, min, sec : Float[ArrayLike, "..."]
        Calendar fields of the ``UTC`` epochs.

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."]]
        Split quasi-Julian date of the ``UTC`` epochs.

    Notes
    -----
    All valid ``UTC`` epochs handled by this module are on or after 1962-01-01, so the calendar date is always interpreted in the Gregorian calendar.

    Vectorize :func:`julian_date_for_utc_single`.
    """
    year = jnp.asarray(year, dtype=jnp.float64)
    month = jnp.asarray(month, dtype=jnp.float64)
    day = jnp.asarray(day, dtype=jnp.float64)
    hour = jnp.asarray(hour, dtype=jnp.float64)
    min = jnp.asarray(min, dtype=jnp.float64)
    sec = jnp.asarray(sec, dtype=jnp.float64)
    year, month, day, hour, min, sec = jnp.broadcast_arrays(year, month, day, hour, min, sec)
    return safe_dispatch(julian_date_for_utc_single, (0, 0, 0, 0, 0, 0), year, month, day, hour, min, sec)


@eqx.filter_jit
def calendar_date_for_utc(jd1: Float[Array, "..."], jd2: Float[Array, "..."]) -> Tuple[
    Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """Convert ``UTC`` quasi-Julian dates to calendar fields.

    Parameters
    ----------
    jd1, jd2 : Float[Array, "..."]
        Split quasi-Julian date of the ``UTC`` epochs.

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]
        Calendar fields ``(year, month, day, hour, minute, second)``.

    Notes
    -----
    All valid ``UTC`` epochs handled by this module are on or after 1962-01-01, so the calendar date is always returned in the Gregorian calendar.

    Vectorize :func:`calendar_date_for_utc_single`.
    """
    return safe_dispatch(calendar_date_for_utc_single, (0, 0), jd1, jd2)
