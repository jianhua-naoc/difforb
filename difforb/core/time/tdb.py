"""Low-level ``TT`` / ``TDB`` transforms for ``time``.

This module implements split-Julian-date transforms between ``TT`` and ``TDB``. It uses the Fairhead-Bretagnon geocentric series, the SOFA-style topocentric correction, and a fixed-point inversion for the reverse transform.
"""

from functools import partial

import equinox as eqx
import jax
from jax import numpy as jnp, Array
from jaxtyping import Float

from difforb.core.batch import safe_dispatch, safe_cartesian_dispatch
from difforb.core.config import get_data_path
from difforb.core.constants import J2000, JULIAN_CENTURY, DAY_S
from difforb.core.eop.container import EarthOrientationData
from difforb.core.time.ut1 import tt_to_ut1_single
from difforb.core.time.utils import ut1_fraction
from difforb.spk.spk import TTMinusTDBKernel

jax.config.update("jax_enable_x64", True)

# Fairhead-Bretagnon coefficient blocks for the geocentric ``TDB - TT`` series.
coff_dtdb = jnp.load(str(get_data_path("dtdb_series/coff_dtdb.npy", dataset="dtdb-series")))
a = coff_dtdb[:474, 0]
omega_a = coff_dtdb[:474, 1]
phi_a = coff_dtdb[:474, 2]
b = coff_dtdb[474:679, 0]
omega_b = coff_dtdb[474:679, 1]
phi_b = coff_dtdb[474:679, 2]
c = coff_dtdb[679:764, 0]
omega_c = coff_dtdb[679:764, 1]
phi_c = coff_dtdb[679:764, 2]
d = coff_dtdb[764:784, 0]
omega_d = coff_dtdb[764:784, 1]
phi_d = coff_dtdb[764:784, 2]
e = coff_dtdb[784:, 0]
omega_e = coff_dtdb[784:, 1]
phi_e = coff_dtdb[784:, 2]


def tt_to_tdb_single(lon: Float[Array, ""], u: Float[Array, ""], v: Float[Array, ""], tt_jd1: Float[Array, ""],
                     tt_jd2: Float[Array, ""], ut1_fraction: Float[Array, ""]) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Transform ``TT`` epoch to ``TDB`` epoch.

    Parameters
    ----------
    lon : Float[Array, ""]
        Observer east longitude in radians.
    u : Float[Array, ""]
        Observer distance from the Earth rotation axis in kilometers.
    v : Float[Array, ""]
        Observer distance north of the terrestrial equatorial plane in kilometers.
    tt_jd1, tt_jd2 : Float[Array, ""]
        Split Julian date of the ``TT`` epoch. In the periodic terms used here, SOFA notes that substituting ``TT`` for ``TDB`` in the time argument has no practical effect on the prediction accuracy.
    ut1_fraction : Float[Array, ""]
        Fraction of one day in ``UT1``.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""]]
        Split Julian date of the ``TDB`` epoch.

    Notes
    -----
    The result is the sum of three contributions:

    - a geocentric Fairhead-Bretagnon trigonometric series,
    - a topocentric correction driven by local solar time and observer geometry,
    - small mass-adjustment terms aligned with the SOFA ``iauDtdb`` routine.

    References
    ----------
    1. Fairhead, L., & Bretagnon, P. (1990). An analytical formula for the time transformation ``TDB``-``TT``.
    2. Simon, J. L., Bretagnon, P., Chapront, J., Chapront-Touze, M., Francou, G., & Laskar, J. (1994). Numerical expressions for precession formulae and mean elements for the Moon and the planets.
    3. SOFA ``iauDtdb``.
    """
    tt_jd_j2000 = ((tt_jd1 - J2000) + tt_jd2) / JULIAN_CENTURY
    t = tt_jd_j2000 / 10.  # Julian kyrs, not Julian centuries

    # Geocentric part
    delta_g1 = jnp.sum(a * jnp.sin(omega_a * t + phi_a))
    delta_g2 = jnp.sum(b * jnp.sin(omega_b * t + phi_b))
    delta_g3 = jnp.sum(c * jnp.sin(omega_c * t + phi_c))
    delta_g4 = jnp.sum(d * jnp.sin(omega_d * t + phi_d))
    delta_g5 = jnp.sum(e * jnp.sin(omega_e * t + phi_e))
    delta_geocentric = delta_g1 + t * (t * (t * (t * delta_g5 + delta_g4) + delta_g3) + delta_g2)

    # Topocentric part
    local_solar_time = ut1_fraction * 2 * jnp.pi + lon
    w = t / 3600.
    # Mean longitude of sun
    L = jnp.deg2rad((280.46645683 + 1296027711.03429 * w) % 360.)
    # Mean anomaly of sun
    M = jnp.deg2rad((357.52910918 + 1295965810.481 * w) % 360.)
    # Mean elong of moon from sun, Simon Sec 3.5(b), SOFA abandoned high-order terms
    """
    D = jnp.deg2rad(297.85019547 + (
            (16029616012.090 * t - 6.3706 * t * t + 0.006593 * t * t * t - 0.00003169 * t * t * t * t) / 3600.))
    """
    D = jnp.deg2rad((297.85019547 + 16029616012.090 * w) % 360.)
    # Mean longitude of jupiter, Simon Sec 5.9.5, SOFA abandoned high-order terms
    """
    L_j = jnp.deg2rad(34.35151874 + (
            109306899.89453 * t + 80.38700 * t * t + 0.13327 * t * t * t - 0.18850 * t * t * t * t + 0.00411 * t ** 5 - 0.00014 * t ** 6) / 3600.)
    """
    L_j = jnp.deg2rad((34.35151874 + 109306899.89453 * w) % 360.)
    # Mean longitude of saturn, Simon Sec 5.9.6, SOFA abandoned high-order terms
    """
    L_sa = jnp.deg2rad(50.07744430 + (
            44046398.47038 * t + 186.86817 * t ** 2 - 0.10748 * t ** 3 - 0.35004 * t ** 4 - 0.01630 * t ** 5 + 0.00103 * t ** 6) / 3600.)
    """
    L_sa = jnp.deg2rad((50.07744430 + 44046398.47038 * w) % 360.)

    delta_topcentric = (
            3.17679e-10 * u * jnp.sin(local_solar_time)
            + 5.312e-12 * u * jnp.sin(local_solar_time - M)
            + 1e-13 * u * jnp.sin(local_solar_time - 2 * M)
            - 1.3677e-11 * u * jnp.sin(local_solar_time + 2 * L)
            - 2.29e-13 * u * jnp.sin(local_solar_time + 2 * L + M)
            + 1.33e-13 * u * jnp.sin(local_solar_time - D)
            - 1.3184e-10 * v * jnp.cos(L)
            + 1.33e-13 * u * jnp.sin(local_solar_time + L - L_j)
            + 2.9e-14 * u * jnp.sin(local_solar_time + L - L_sa)
            - 2.2e-12 * v * jnp.cos(L + M))

    # Adjustments to used JPL masses instead of IAU (from SOFA 'iauDtdb' func)
    delta_aj = 0.00065e-6 * jnp.sin(6069.776754 * t + 4.021194) + 0.00033e-6 * jnp.sin(
        213.299095 * t + 5.543132) + (
                       -0.00196e-6 * jnp.sin(6208.294251 * t + 5.696701)) + (
                       -0.00173e-6 * jnp.sin(74.781599 * t + 2.435900)) + 0.03638e-6 * t * t
    delta = (delta_geocentric + delta_topcentric + delta_aj)

    tdb_jd1 = tt_jd1
    tdb_jd2 = tt_jd2 + delta / DAY_S

    return tdb_jd1, tdb_jd2


# def tt_to_tdb_jpl_geo_single(
#         tt_jd1: Float[Array, ""],
#         tt_jd2: Float[Array, ""],
#         tt_minus_tdb_kernel: TTMinusTDBKernel,
# ) -> tuple[Float[Array, ""], Float[Array, ""]]:
#     tdb_jd1 = tt_jd1
#     est_delta = tt_minus_tdb_kernel.tt_minus_tdb(tt_jd1, tt_jd2)
#     est_tdb_jd2 = tt_jd2 - est_delta / DAY_S
#
#     def body_func(_, est_tdb_jd2):
#         delta_tt_tdb = tt_minus_tdb_kernel.tt_minus_tdb(tdb_jd1, est_tdb_jd2)
#         rate = tt_minus_tdb_kernel.dtt_minus_tdb_dtdb(tdb_jd1, est_tdb_jd2)
#
#         residual = est_tdb_jd2 + delta_tt_tdb / DAY_S - tt_jd2
#         return est_tdb_jd2 - residual / (1.0 + rate)
#
#     est_tdb_jd2 = jax.lax.fori_loop(0, 3, body_func, est_tdb_jd2)
#     return tdb_jd1, est_tdb_jd2


def tdb_to_tt_single(lon: Float[Array, ""], u: Float[Array, ""], v: Float[Array, ""], tdb_jd1: Float[Array, ""],
                     tdb_jd2: Float[Array, ""], eop: EarthOrientationData) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Transform ``TDB`` epoch to ``TT`` epoch.

    Parameters
    ----------
    lon : Float[Array, ""]
        Observer east longitude in radians.
    u : Float[Array, ""]
        Observer distance from the Earth rotation axis in kilometers.
    v : Float[Array, ""]
        Observer distance north of the terrestrial equatorial plane in kilometers.
    tdb_jd1, tdb_jd2 : Float[Array, ""]
        Split Julian date of the ``TDB`` epoch.
    eop : EarthOrientationData
        Earth orientation data used to recover the matching ``UT1`` fraction during the inversion.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""]]
        Split Julian date of the ``TT`` epoch.

    Notes
    -----
    This function inverts :func:`tt_to_tdb_single` by fixed-point iteration. Each iteration updates the current ``TT`` estimate, recomputes the matching ``UT1`` fraction, and reevaluates the forward ``TT -> TDB`` model.

    References
    ----------
    1. Fairhead, L., & Bretagnon, P. (1990). An analytical formula for the time transformation ``TDB``-``TT``.
    2. Simon, J. L., Bretagnon, P., Chapront, J., Chapront-Touze, M., Francou, G., & Laskar, J. (1994). Numerical expressions for precession formulae and mean elements for the Moon and the planets.
    3. SOFA ``iauDtdb``.
    """

    def body_func(i, state):
        est_tt_jd1, est_tt_jd2 = state
        est_ut1_jd1, est_ut1_jd2 = tt_to_ut1_single(est_tt_jd1, est_tt_jd2, eop)
        ut1_frac = ut1_fraction(est_ut1_jd1, est_ut1_jd2)
        est_tdb_jd1, est_tdb_jd2 = tt_to_tdb_single(lon, u, v, est_tt_jd1, est_tt_jd2, ut1_frac)
        err = (tdb_jd1 - est_tdb_jd1) + (tdb_jd2 - est_tdb_jd2)
        est_tt_jd2 += err
        return est_tt_jd1, est_tt_jd2

    est_tt_jd1, est_tt_jd2 = jax.lax.fori_loop(0, 3, body_func, (tdb_jd1, tdb_jd2))
    return est_tt_jd1, est_tt_jd2


# def tdb_to_tt_jpl_geo_single(tdb_jd1: Float[Array, ""], tdb_jd2: Float[Array, ""], tt_minus_tdb_kernel: TTMinusTDBKernel) -> \
#         tuple[Float[Array, ""], Float[Array, ""]]:
#     delta_tt_tdb = tt_minus_tdb_kernel.tt_minus_tdb(tdb_jd1, tdb_jd2)
#     tt_jd1 = tdb_jd1
#     tt_jd2 = tdb_jd2 + delta_tt_tdb / DAY_S
#     return tt_jd1, tt_jd2


# =========================================================================
# Vectorized Public API
# =========================================================================

@eqx.filter_jit
def tt_to_tdb(lon: Float[Array, ""], u: Float[Array, ""], v: Float[Array, ""], tt_jd1: Float[Array, ""],
              tt_jd2: Float[Array, ""], ut1_fraction: Float[Array, ""], grid: bool = False) -> tuple[Float[Array, ""],
Float[Array, ""]]:
    """Transform ``TT`` epoch to ``TDB`` epoch.

    Parameters
    ----------
    lon : Float[Array, "..."]
        Observer east longitude in radians.
    u : Float[Array, "..."]
        Observer distance from the Earth rotation axis in kilometers.
    v : Float[Array, "..."]
        Observer distance north of the terrestrial equatorial plane in kilometers.
    tt_jd1, tt_jd2 : Float[Array, "..."]
        Split Julian date of the ``TT`` epoch.
    ut1_fraction : Float[Array, "..."]
        Fraction of one day in ``UT1``.
    grid : bool, default=False
        If ``False``, use point-wise broadcasting. If ``True``, use the Cartesian product of the location and time batches.

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."]]
        Split Julian date of the ``TDB`` epoch.

    Notes
    -----
    Vectorize :func:`tt_to_tdb_single`.

    References
    ----------
    1. Fairhead, L., & Bretagnon, P. (1990). An analytical formula for the time transformation ``TDB``-``TT``.
    2. Simon, J. L., Bretagnon, P., Chapront, J., Chapront-Touze, M., Francou, G., & Laskar, J. (1994). Numerical expressions for precession formulae and mean elements for the Moon and the planets.
    3. SOFA ``iauDtdb``.
    """
    if not grid:
        return safe_dispatch(tt_to_tdb_single, (0, 0, 0, 0, 0, 0), lon, u, v, tt_jd1, tt_jd2, ut1_fraction)
    else:
        return safe_cartesian_dispatch(tt_to_tdb_single, ((0, 0, 0), (lon, u, v)), ((0, 0, 0), (tt_jd1, tt_jd2, ut1_fraction)))


@eqx.filter_jit
def tdb_to_tt(lon: Float[Array, "..."], u: Float[Array, "..."], v: Float[Array, "..."], tdb_jd1: Float[Array, "..."],
              tdb_jd2: Float[Array, "..."], eop: EarthOrientationData, grid: bool = False) -> tuple[Float[Array, ""],
Float[Array, ""]]:
    """Transform ``TDB`` epoch to ``TT`` epoch.

    Parameters
    ----------
    lon : Float[Array, "..."]
        Observer east longitude in radians.
    u : Float[Array, "..."]
        Observer distance from the Earth rotation axis in kilometers.
    v : Float[Array, "..."]
        Observer distance north of the terrestrial equatorial plane in kilometers.
    tdb_jd1, tdb_jd2 : Float[Array, "..."]
        Split Julian date of the ``TDB`` epoch.
    eop : EarthOrientationData
        Earth orientation data used to recover the matching ``UT1`` fraction during the inversion.
    grid : bool, default=False
        If ``False``, use point-wise broadcasting. If ``True``, use the Cartesian product of the location and time batches.

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."]]
        Split Julian date of the ``TT`` epoch.

    Notes
    -----
    Vectorize :func:`tdb_to_tt_single`.

    References
    ----------
    1. Fairhead, L., & Bretagnon, P. (1990). An analytical formula for the time transformation ``TDB``-``TT``.
    2. Simon, J. L., Bretagnon, P., Chapront, J., Chapront-Touze, M., Francou, G., & Laskar, J. (1994). Numerical expressions for precession formulae and mean elements for the Moon and the planets.
    3. SOFA ``iauDtdb``.
    """
    wrapper = partial(tdb_to_tt_single, eop=eop)
    if not grid:
        return safe_dispatch(wrapper, (0, 0, 0, 0, 0), lon, u, v, tdb_jd1, tdb_jd2)
    else:
        return safe_cartesian_dispatch(wrapper, ((0, 0, 0), (lon, u, v)), ((0, 0), (tdb_jd1, tdb_jd2)))
