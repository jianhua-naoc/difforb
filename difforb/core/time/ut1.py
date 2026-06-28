"""Low-level ``TT`` / ``UT1`` transforms for ``time``.

This module implements split-Julian-date transforms between ``TT`` and ``UT1``. It combines the modern ``EOP``-based correction with the historical Morrison et al. 2021 Addendum ``Delta T`` spline model before the covered ``EOP`` range.
"""

from functools import partial

import equinox as eqx
import jax
from jax import numpy as jnp, Array
from jaxtyping import Float

from difforb.core.batch import safe_dispatch
from difforb.core.constants import DAY_S
from difforb.core.eop.container import EarthOrientationData

jax.config.update("jax_enable_x64", True)

# Polynomial segments for historical ``Delta T = TT - UT1`` before the covered ``EOP`` range.
DELTA_T_TABLE = jnp.array([
    # Table structure: start Julian year[0], end Julian year[1], coefficients[2:]
    # Morrison et al. 2021 Addendum Table S15 v2020.  For Julian year Y,
    # t = (Y - start) / (end - start), and Delta T = a0 + a1 t + a2 t^2 + a3 t^3.
    [-720.0, -100.0, 20371.848, -9999.586, 776.247, 409.160],
    [-100.0, 400.0, 11557.668, -5822.270, 1303.151, -503.433],
    [400.0, 1000.0, 6535.116, -5671.519, -298.291, 1085.087],
    [1000.0, 1150.0, 1650.393, -753.210, 184.811, -25.346],
    [1150.0, 1300.0, 1056.647, -459.628, 108.771, -24.641],
    [1300.0, 1500.0, 681.149, -421.345, 61.953, -29.414],
    [1500.0, 1600.0, 292.343, -192.841, -6.572, 16.197],
    [1600.0, 1650.0, 109.127, -78.697, 10.505, 3.018],
    [1650.0, 1720.0, 43.952, -68.089, 38.333, -2.127],
    [1720.0, 1800.0, 12.068, 2.507, 41.731, -37.939],
    [1800.0, 1810.0, 18.367, -3.481, -1.126, 1.918],
    [1810.0, 1820.0, 15.678, 0.021, 4.629, -3.812],
    [1820.0, 1830.0, 16.516, -2.157, -6.806, 3.250],
    [1830.0, 1840.0, 10.804, -6.018, 2.944, -0.096],
    [1840.0, 1850.0, 7.634, -0.416, 2.658, -0.539],
    [1850.0, 1855.0, 9.338, 1.642, 0.261, -0.883],
    [1855.0, 1860.0, 10.357, -0.486, -2.389, 1.558],
    [1860.0, 1865.0, 9.040, -0.591, 2.284, -2.477],
    [1865.0, 1870.0, 8.255, -3.456, -5.148, 2.720],
    [1870.0, 1875.0, 2.371, -5.593, 3.011, -0.914],
    [1875.0, 1880.0, -1.126, -2.314, 0.269, -0.039],
    [1880.0, 1885.0, -3.210, -1.893, 0.152, 0.563],
    [1885.0, 1890.0, -4.388, 0.101, 1.842, -1.438],
    [1890.0, 1895.0, -3.884, -0.531, -2.474, 1.871],
    [1895.0, 1900.0, -5.017, 0.134, 3.138, -0.232],
    [1900.0, 1905.0, -1.977, 5.715, 2.443, -1.257],
    [1905.0, 1910.0, 4.923, 6.828, -1.329, 0.720],
    [1910.0, 1915.0, 11.142, 6.330, 0.831, -0.825],
    [1915.0, 1920.0, 17.479, 5.518, -1.643, 0.262],
    [1920.0, 1925.0, 21.617, 3.020, -0.856, 0.008],
    [1925.0, 1930.0, 23.789, 1.333, -0.831, 0.127],
    [1930.0, 1935.0, 24.418, 0.052, -0.449, 0.142],
    [1935.0, 1940.0, 24.164, -0.419, -0.022, 0.702],
    [1940.0, 1945.0, 24.426, 1.645, 2.086, -1.106],
    [1945.0, 1950.0, 27.050, 2.499, -1.232, 0.614],
    [1950.0, 1953.0, 28.932, 1.127, 0.220, -0.277],
    [1953.0, 1956.0, 30.002, 0.737, -0.610, 0.631],
    [1956.0, 1959.0, 30.760, 1.409, 1.282, -0.799],
    [1959.0, 1962.0, 32.652, 1.577, -1.115, 0.507],
    [1962.0, 1965.0, 33.621, 0.868, 0.406, 0.199],
    [1965.0, 1968.0, 35.093, 2.275, 1.002, -0.414],
    [1968.0, 1971.0, 37.956, 3.035, -0.242, 0.202],
    [1971.0, 1974.0, 40.951, 3.157, 0.364, -0.229],
    [1974.0, 1977.0, 44.244, 3.199, -0.323, 0.172],
    [1977.0, 1980.0, 47.291, 3.069, 0.193, -0.192],
    [1980.0, 1983.0, 50.361, 2.878, -0.384, 0.081],
    [1983.0, 1986.0, 52.936, 2.354, -0.140, -0.165],
    [1986.0, 1989.0, 54.984, 1.577, -0.637, 0.448],
    [1989.0, 1992.0, 56.373, 1.648, 0.708, -0.276],
    [1992.0, 1995.0, 58.453, 2.235, -0.121, 0.110],
    [1995.0, 1998.0, 60.678, 2.324, 0.210, -0.313],
    [1998.0, 2001.0, 62.898, 1.804, -0.729, 0.109],
    [2001.0, 2004.0, 64.083, 0.674, -0.402, 0.199],
    [2004.0, 2007.0, 64.553, 0.466, 0.194, -0.017],
    [2007.0, 2010.0, 65.197, 0.804, 0.144, -0.084],
    [2010.0, 2013.0, 66.061, 0.839, -0.109, 0.128],
    [2013.0, 2016.0, 66.920, 1.007, 0.277, -0.095],
    [2016.0, 2019.0, 68.109, 1.277, -0.007, -0.139],
], dtype=float)


def _historical_delta_t_single(tt_jd1: Float[Array, ""], tt_jd2: Float[Array, ""]) -> Float[Array, ""]:
    """Evaluate the historical ``Delta T = TT - UT1`` Morrison et al. 2021 Addendum model."""
    y = ((tt_jd1 + tt_jd2) - 1721045.0) / 365.25

    boundaries = DELTA_T_TABLE[:, 0]
    idx = jnp.searchsorted(boundaries, y, side='right') - 1
    idx = jnp.clip(idx, 0, DELTA_T_TABLE.shape[0] - 1)
    params = DELTA_T_TABLE[idx]
    start = params[..., 0]
    end = params[..., 1]
    coeffs = params[..., 2:]
    t = (y - start) / (end - start)

    ttdut1_addendum = coeffs[..., 3]
    ttdut1_addendum = ttdut1_addendum * t + coeffs[..., 2]
    ttdut1_addendum = ttdut1_addendum * t + coeffs[..., 1]
    ttdut1_addendum = ttdut1_addendum * t + coeffs[..., 0]

    # Table S15 is not valid before Julian year -720.  Keep the transform defined for legacy callers with
    # the long-term Stephenson-Morrison-Hohenkerk parabola used for dates before the Addendum spline.
    t_long = (y - 1825.0) / 100.0
    ttdut1_long = 32.5 * t_long * t_long - 320.0
    is_addendum = y >= DELTA_T_TABLE[0, 0]
    return jnp.where(is_addendum, ttdut1_addendum, ttdut1_long)


def tt_to_ut1_single(tt_jd1: Float[Array, ""], tt_jd2: Float[Array, ""], eop: EarthOrientationData) -> tuple[
    Float[Array, ""], Float[Array, ""]]:
    """Transform ``TT`` epoch to ``UT1`` epoch.

    Parameters
    ----------
    tt_jd1, tt_jd2 : Float[Array, ""]
        Split Julian date of the ``TT`` epoch.
    eop : EarthOrientationData
        Earth orientation data used for modern ``UT1 - TT`` corrections.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""]]
        Split Julian date of the ``UT1`` epoch.

    Notes
    -----
    Before the covered time of the ``EOP`` file, the transform uses the Morrison et al. 2021 Addendum ``Delta T`` spline model. Within the covered time of the ``EOP`` file, it uses the relation

    ``UT1`` - ``TT`` = (``UT1`` - ``UTC``) - (``TAI`` - ``UTC``) - (``TT`` - ``TAI``)

    References
    ----------
    1. Morrison, L. V., Stephenson, F. R., Hohenkerk, C. Y., & Zawilski, M. Addendum 2020 to Measurement of the Earth's rotation: 720 BC to AD 2015.
       https://doi.org/10.1098/rspa.2020.0776
    """
    # -------------------------------------------------------------------------
    # Step 1: Evaluate the historical UT1
    # -------------------------------------------------------------------------
    ttdut1 = _historical_delta_t_single(tt_jd1, tt_jd2)
    ut1_jd1_historical = tt_jd1
    ut1_jd2_historical = tt_jd2 - ttdut1 / DAY_S

    # -------------------------------------------------------------------------
    # Step 2: Evaluate the current UT1
    # -------------------------------------------------------------------------
    ut1dtt = eop.ut1dtt(tt_jd1, tt_jd2)
    ut1_jd1_cur = tt_jd1
    ut1_jd2_cur = tt_jd2 + ut1dtt / DAY_S

    # -------------------------------------------------------------------------
    # Step 3: Select the final ``UT1`` model
    # -------------------------------------------------------------------------
    is_cur = (tt_jd1 + tt_jd2) >= eop.final_date_range[0]
    ut1_jd1 = jnp.where(is_cur, ut1_jd1_cur, ut1_jd1_historical)
    ut1_jd2 = jnp.where(is_cur, ut1_jd2_cur, ut1_jd2_historical)

    return ut1_jd1, ut1_jd2


def ut1_to_tt_single(ut1_jd1: Float[Array, ""], ut1_jd2: Float[Array, ""], eop: EarthOrientationData) -> tuple[
    Float[Array, ""], Float[Array, ""]]:
    """Transform ``UT1`` epoch to ``TT`` epoch.

    Parameters
    ----------
    ut1_jd1, ut1_jd2 : Float[Array, ""]
        Split Julian date of the ``UT1`` epoch.
    eop : EarthOrientationData
        Earth orientation data used by the inverse transform.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""]]
        Split Julian date of the ``TT`` epoch.

    Notes
    -----
    This function inverts :func:`tt_to_ut1_single` by building historical fixed-point candidates, the nearest historical spline-boundary candidates, one modern ``EOP`` fixed-point candidate, and the exact branch-boundary candidate, and checking them with the forward model. The final answer is the candidate with the smallest forward ``UT1`` residual. If several candidates give the same residual, the exact branch boundary is selected first, cross-branch ties are resolved toward the historical candidate, and remaining ties are resolved toward the latest ``TT`` epoch.

    References
    ----------
    1. Morrison, L. V., Stephenson, F. R., Hohenkerk, C. Y., & Zawilski, M. Addendum 2020 to Measurement of the Earth's rotation: 720 BC to AD 2015. https://doi.org/10.1098/rspa.2020.0776
    """

    # -------------------------------------------------------------------------
    # Step 1: Build historical candidates by fixed-point iteration
    # -------------------------------------------------------------------------
    historical_seed_jd1 = jnp.full((3,), ut1_jd1)
    historical_seed_jd2 = jnp.stack([ut1_jd2 - 0.5, ut1_jd2, ut1_jd2 + 0.5])

    def historical_body_func(i, state):
        est_tt_jd1, est_tt_jd2 = state
        est_ttdut1 = _historical_delta_t_single(est_tt_jd1, est_tt_jd2)
        est_tt_jd2 = ut1_jd2 + est_ttdut1 / DAY_S
        return est_tt_jd1, est_tt_jd2

    historical_tt_jd1, historical_tt_jd2 = jax.lax.fori_loop(
        0, 6, historical_body_func, (historical_seed_jd1, historical_seed_jd2)
    )

    # -------------------------------------------------------------------------
    # Step 2: Add the nearest historical spline-boundary candidates
    # -------------------------------------------------------------------------
    historical_year = ((historical_tt_jd1[1] + historical_tt_jd2[1]) - 1721045.0) / 365.25
    historical_idx = jnp.searchsorted(DELTA_T_TABLE[:, 0], historical_year, side="right") - 1
    historical_idx = jnp.clip(historical_idx, 0, DELTA_T_TABLE.shape[0] - 1)
    spline_boundary_years = jnp.stack([DELTA_T_TABLE[historical_idx, 0], DELTA_T_TABLE[historical_idx, 1]])
    spline_boundary_jds = 1721045.0 + 365.25 * spline_boundary_years
    spline_boundary_jd1 = jnp.floor(spline_boundary_jds)
    spline_boundary_jd2 = spline_boundary_jds - spline_boundary_jd1

    # -------------------------------------------------------------------------
    # Step 3: Build the modern ``EOP`` candidate by fixed-point iteration
    # -------------------------------------------------------------------------
    def modern_body_func(i, state):
        est_tt_jd1, est_tt_jd2 = state
        est_ut1dtt = eop.ut1dtt(est_tt_jd1, est_tt_jd2)
        est_tt_jd2 = ut1_jd2 - est_ut1dtt / DAY_S
        return est_tt_jd1, est_tt_jd2

    modern_tt_jd1, modern_tt_jd2 = jax.lax.fori_loop(0, 6, modern_body_func, (ut1_jd1, ut1_jd2))

    # -------------------------------------------------------------------------
    # Step 4: Add the exact branch-boundary candidate
    # -------------------------------------------------------------------------
    boundary_tt_jd = eop.final_date_range[0]
    boundary_tt_jd1 = jnp.floor(boundary_tt_jd)
    boundary_tt_jd2 = boundary_tt_jd - boundary_tt_jd1

    # -------------------------------------------------------------------------
    # Step 5: Check all candidates with the forward model and pick the best
    # -------------------------------------------------------------------------
    cand_tt_jd1 = jnp.concatenate(
        [historical_tt_jd1, spline_boundary_jd1, modern_tt_jd1[None], boundary_tt_jd1[None]], axis=0
    )
    cand_tt_jd2 = jnp.concatenate(
        [historical_tt_jd2, spline_boundary_jd2, modern_tt_jd2[None], boundary_tt_jd2[None]], axis=0
    )

    forward_ut1_jd1, forward_ut1_jd2 = jax.vmap(partial(tt_to_ut1_single, eop=eop))(cand_tt_jd1, cand_tt_jd2)
    residuals = jnp.abs((forward_ut1_jd1 - ut1_jd1) + (forward_ut1_jd2 - ut1_jd2))
    min_residual = jnp.min(residuals)

    candidate_tt_jds = cand_tt_jd1 + cand_tt_jd2
    candidate_tt_order = (cand_tt_jd1 - ut1_jd1) + cand_tt_jd2
    is_best = residuals <= (min_residual + 1e-15 / DAY_S)

    has_historical_best = jnp.any(is_best & (candidate_tt_jds < boundary_tt_jd))
    has_modern_best = jnp.any(is_best & (candidate_tt_jds >= boundary_tt_jd))
    is_cross_branch = has_historical_best & has_modern_best
    boundary_idx = cand_tt_jd1.shape[0] - 1

    latest_idx = jnp.argmax(jnp.where(is_best, candidate_tt_order, -jnp.inf))
    earliest_idx = jnp.argmin(jnp.where(is_best, candidate_tt_order, jnp.inf))
    best_idx = jnp.where(is_best[boundary_idx], boundary_idx, jnp.where(is_cross_branch, earliest_idx, latest_idx))

    return cand_tt_jd1[best_idx], cand_tt_jd2[best_idx]


# =========================================================================
# Vectorized Public API
# =========================================================================

@eqx.filter_jit
def tt_to_ut1(tt_jd1: Float[Array, "..."], tt_jd2: Float[Array, "..."], eop: EarthOrientationData) -> tuple[
    Float[Array, "..."], Float[Array, "..."]]:
    """Transform ``TT`` epoch to ``UT1`` epoch.

    Parameters
    ----------
    tt_jd1, tt_jd2 : Float[Array, "..."]
        Split Julian date of the ``TT`` epoch.
    eop : EarthOrientationData
        Earth orientation data used for modern ``UT1 - TT`` corrections.

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."]]
        Split Julian date of the ``UT1`` epoch.

    Notes
    -----
    Before the covered time of the ``EOP`` file, the transform uses the Morrison et al. 2021 Addendum ``Delta T`` spline model. Within the covered time of the ``EOP`` file, it uses the relation

    ``UT1`` - ``TT`` = (``UT1`` - ``UTC``) - (``TAI`` - ``UTC``) - (``TT`` - ``TAI``)

    Vectorize :func:`tt_to_ut1_single`.

    References
    ----------
    1. Morrison, L. V., Stephenson, F. R., Hohenkerk, C. Y., & Zawilski, M. Addendum 2020 to Measurement of the Earth's rotation: 720 BC to AD 2015.
       https://doi.org/10.1098/rspa.2020.0776
    """
    wrapper = partial(tt_to_ut1_single, eop=eop)
    return safe_dispatch(wrapper, (0, 0), tt_jd1, tt_jd2)


@eqx.filter_jit
def ut1_to_tt(ut1_jd1: Float[Array, "..."], ut1_jd2: Float[Array, "..."], eop: EarthOrientationData) -> tuple[
    Float[Array, "..."], Float[Array, "..."]]:
    """Transform ``UT1`` epoch to ``TT`` epoch.

    Parameters
    ----------
    ut1_jd1, ut1_jd2 : Float[Array, "..."]
        Split Julian date of the ``UT1`` epoch.
    eop : EarthOrientationData
        Earth orientation data used by the inverse transform.

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."]]
        Split Julian date of the ``TT`` epoch.

    Notes
    -----
    Vectorize :func:`ut1_to_tt_single`.

    References
    ----------
    1. Morrison, L. V., Stephenson, F. R., Hohenkerk, C. Y., & Zawilski, M. Addendum 2020 to Measurement of the Earth's rotation: 720 BC to AD 2015. https://doi.org/10.1098/rspa.2020.0776
    """
    wrapper = partial(ut1_to_tt_single, eop=eop)
    return safe_dispatch(wrapper, (0, 0), ut1_jd1, ut1_jd2)
