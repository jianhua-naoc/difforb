"""Earth orientation data container for ``EOP`` tables.

This module stores Earth Orientation Parameter (EOP) tables and exposes interpolation-based queries at ``TT`` epochs. Data is usually loaded by :mod:`difforb.core.eop.loaders`, while the adjacent :mod:`difforb.core.eop.interpolate` module provides the shared 4-point Lagrange kernel.
"""

from typing import Optional

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float
import equinox as eqx

from difforb.core.constants import DAY_S, MJD_START
from difforb.core.eop.interpolate import lagrangian_interpolate, lagrangian_interpolate_single
from difforb.core.time.tai import ttdtai, tai_to_tt
from difforb.core.time.utc import utc_to_tai
from difforb.core.time.utils import renormalize_split_jd


class EarthOrientationData(eqx.Module):
    """Interpolation container for Earth Orientation Parameter (EOP) data.

    Parameters
    ----------
    tt_jds : Float[Array, "n"]
        Sample ``JD`` values in ``TT``.
    xpoles : Float[Array, "n"]
        Polar-motion coordinate ``xp`` in arcseconds.
    ypoles : Float[Array, "n"]
        Polar-motion coordinate ``yp`` in arcseconds.
    ut1dutcs : Float[Array, "n"]
        ``UT1-UTC`` offsets in seconds.
    ut1dtts : Float[Array, "n"]
        ``UT1-TT`` offsets in seconds.
    dpsis : Float[Array, "n"]
        Additive ``dPsi`` correction to the model nutation in longitude, in arcseconds.
    depss : Float[Array, "n"]
        Additive ``dEps`` correction to the model nutation in obliquity, in arcseconds.
    final_date_range : Float[Array, "2"]
        Covered time range of the final data.
    predicted_date_range : Optional[Float[Array, "2"]], optional
        Covered time range of the predicted data, if present.
    """
    tt_jds: Float[Array, "n"]
    xpoles: Float[Array, "n"]
    ypoles: Float[Array, "n"]
    ut1dutcs: Float[Array, "n"]
    _taidutcs: Float[Array, "n"]
    ut1dtts: Float[Array, "n"]
    dpsis: Float[Array, "n"]
    depss: Float[Array, "n"]
    final_date_range: Float[Array, "2"]
    predicted_date_range: Optional[Float[Array, "2"]]

    def __init__(self, mjds: Float[Array, "n"], xpoles: Float[Array, "n"], ypoles: Float[Array, "n"], ut1dutcs: Float[Array,
    "n"], dpsis: Float[Array, "n"], depss: Float[Array, "n"], final_date_range: Float[Array, "2"],
                 predicted_date_range: Optional[Float[Array, "2"]] = None) -> None:
        utc_jds = mjds + MJD_START
        utc_jd1s, utc_jd2s = renormalize_split_jd(utc_jds, jnp.zeros_like(utc_jds))
        tai_jd1s, tai_jd2s = utc_to_tai(utc_jd1s, utc_jd2s)
        tt_jd1, tt_jd2 = tai_to_tt(tai_jd1s, tai_jd2s)
        self.tt_jds = tt_jd1 + tt_jd2
        self.xpoles = xpoles
        self.ypoles = ypoles
        self.ut1dutcs = ut1dutcs
        # Store ``TAI - UTC`` in seconds so all EOP-derived offsets share one unit.
        self._taidutcs = ((tai_jd1s - utc_jd1s) + (tai_jd2s - utc_jd2s)) * DAY_S
        self.ut1dtts = ut1dutcs - self._taidutcs - ttdtai()
        self.dpsis = dpsis
        self.depss = depss
        self.final_date_range = final_date_range
        self.predicted_date_range = predicted_date_range

    def xpole(self, tt_jd1: Float[Array, "..."], tt_jd2: Float[Array, "..."]) -> Float[Array, "..."]:
        """Interpolate the polar-motion coordinate ``xp``.

        Parameters
        ----------
        tt_jd1, tt_jd2 : Float[Array, ""]
            Split Julian date of the ``UTC`` epoch.

        Returns
        -------
        Float[Array, "..."]
            Interpolated value in arcseconds.
        """
        tt_jd = tt_jd1 + tt_jd2
        xpole = lagrangian_interpolate(self.tt_jds, self.xpoles, tt_jd)
        return jnp.where(tt_jd < self.tt_jds[0], 0.0, xpole)

    def ypole(self, tt_jd1: Float[Array, "..."], tt_jd2: Float[Array, "..."]) -> Float[Array, "..."]:
        """Interpolate the polar-motion coordinate ``yp``.

        Parameters
        ----------
        tt_jd1, tt_jd2 : Float[Array, ""]
            Split Julian date of the ``UTC`` epoch.

        Returns
        -------
        Float[Array, "..."]
            Interpolated value in arcseconds.
        """
        tt_jd = tt_jd1 + tt_jd2
        ypole = lagrangian_interpolate(self.tt_jds, self.ypoles, tt_jd)
        return jnp.where(tt_jd < self.tt_jds[0], 0.0, ypole)

    def ut1dtt(self, tt_jd1: Float[Array, "..."], tt_jd2: Float[Array, "..."]) -> Float[Array, "..."]:
        """Interpolate the ``UT1-TT`` offset.

        Parameters
        ----------
        tt_jd1, tt_jd2 : Float[Array, ""]
            Split Julian date of the ``UTC`` epoch.

        Returns
        -------
        Float[Array, "..."]
            Interpolated value in seconds.
        """
        ut1dtt = lagrangian_interpolate(self.tt_jds, self.ut1dtts, tt_jd1 + tt_jd2)
        return ut1dtt

    def ut1dutc(self, tt_jd1: Float[Array, "..."], tt_jd2: Float[Array, "..."]) -> Float[Array, "..."]:
        """Interpolate the ``UT1-UTC`` offset.

        Parameters
        ----------
        tt_jd1, tt_jd2 : Float[Array, ""]
            Split Julian date of the ``UTC`` epoch.

        Returns
        -------
        Float[Array, "..."]
            Interpolated value in seconds.
        """
        # Due to the leap seconds issus in UTC, there are jumps in ut1dutc. Therefore, interpolation cannot be directly
        # performed on ut1dutcs, but rather on the continuous ut1dtts.
        tt_jd = tt_jd1 + tt_jd2
        ut1dtt = lagrangian_interpolate(self.tt_jds, self.ut1dtts, tt_jd)
        taidutc = lagrangian_interpolate(self.tt_jds, self._taidutcs, tt_jd)
        ut1dutc = ut1dtt + taidutc + ttdtai()
        return ut1dutc

    def cor_delta_longitude(self, tt_jd1: Float[Array, "..."], tt_jd2: Float[Array, "..."]) -> Float[Array, "..."]:
        """Interpolate the additive ``dPsi`` correction to nutation in longitude.

        Parameters
        ----------
        tt_jd1, tt_jd2 : Float[Array, ""]
            Split Julian date of the ``UTC`` epoch.

        Returns
        -------
        Float[Array, "..."]
            Interpolated value in arcseconds.
        """
        tt_jd = tt_jd1 + tt_jd2
        dpsi = lagrangian_interpolate(self.tt_jds, self.dpsis, tt_jd)
        return jnp.where(tt_jd < self.tt_jds[0], 0.0, dpsi)

    def cor_delta_obliquity(self, tt_jd1: Float[Array, "..."], tt_jd2: Float[Array, "..."]) -> Float[Array, "..."]:
        """Interpolate the additive ``dEps`` correction to nutation in obliquity.

        Parameters
        ----------
        tt_jd1, tt_jd2 : Float[Array, ""]
            Split Julian date of the ``UTC`` epoch.

        Returns
        -------
        Float[Array, "..."]
            Interpolated value in arcseconds.
        """
        tt_jd = tt_jd1 + tt_jd2
        deps = lagrangian_interpolate(self.tt_jds, self.depss, tt_jd)
        return jnp.where(tt_jd < self.tt_jds[0], 0.0, deps)
