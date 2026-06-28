"""Low-level ``TAI`` / ``TT`` transforms for ``time``.

This module provides the constant-offset kernels between ``TAI`` and ``TT`` in split Julian Date form.
"""

import equinox as eqx
import jax
from jax import Array
from jaxtyping import Float

from difforb.core.constants import DAY_S

jax.config.update("jax_enable_x64", True)


def ttdtai():
    """Return ``TT - TAI`` in seconds.

    Returns
    -------
    float
        Constant offset from ``TAI`` to ``TT`` in seconds.
    """
    return 32.184


@eqx.filter_jit
def tai_to_tt(tai_jd1: Float[Array, "..."], tai_jd2: Float[Array, "..."]) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Transform ``TAI`` epoch to ``TT`` epoch.

    Parameters
    ----------
    tai_jd1, tai_jd2 : Float[Array, "..."]
        Split Julian date of the ``TAI`` epoch.

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."]]
        Split Julian date of the ``TT`` epoch.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq. 3.23.
    """
    return tai_jd1, tai_jd2 + 32.184 / DAY_S


@eqx.filter_jit
def tt_to_tai(tt_jd1: Float[Array, "..."], tt_jd2: Float[Array, "..."]) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Transform ``TT`` epoch to ``TAI`` epoch.

    Parameters
    ----------
    tt_jd1, tt_jd2 : Float[Array, "..."]
        Split Julian date of the ``TT`` epoch.

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."]]
        Split Julian date of the ``TAI`` epoch.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq. 3.23.
    """
    return tt_jd1, tt_jd2 - 32.184 / DAY_S
