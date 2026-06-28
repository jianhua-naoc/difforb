"""Ephemeris bodies based on ``SPK`` kernels.

This module defines :class:`EphemerisBody`, which evaluates its ``BCRS`` state from an ``SPK`` ephemeris.
"""

import jax.numpy as jnp
import equinox as eqx

from typing import Optional
from jax import Array
from jaxtyping import Float

from difforb.core.time.timescale import TDBView
from difforb.core.validate import validate_timeview
from difforb.spk.spk import Ephemeris
from difforb.core.constants import AU_KM
from difforb.core.state.frame import BCRS, Frame
from difforb.core.state.origins import Origin
from difforb.core.state.state import State
import difforb.spk as spk
from difforb.body.gm import gms
from difforb.report.text import build_repr, format_float_array


class EphemerisBody(eqx.Module):
    """Body based on SPK ephemeris segments.

    Parameters
    ----------
    naif_name : str
        NAIF body name.
    eph : Ephemeris, optional
        Ephemeris object. If omitted, the default ephemeris is used.
    """
    segments: tuple
    signs: tuple
    gm: float = eqx.field(static=True)
    naif_name: str = eqx.field(static=True)

    _ephem_cache = {}

    def __init__(self, naif_name: str, eph: Optional[Ephemeris] = None) -> None:
        """Initialize an ephemeris body.

        Parameters
        ----------
        naif_name : str
            NAIF body name.
        eph : Ephemeris, optional
            Ephemeris object. If omitted, the default project ephemeris is used.

        Raises
        ------
        ValueError
            If the requested body name is not available in the ephemeris.
        RuntimeError
            If the body does not have a stored gravitational parameter in :mod:`difforb.body.gm`.
        """
        _eph = eph or spk.load_default_ephemeris()
        self.naif_name = naif_name.upper()
        cache_key = (self.naif_name, id(_eph))
        if cache_key not in EphemerisBody._ephem_cache:
            EphemerisBody._ephem_cache[cache_key] = _eph.load_body(self.naif_name)
        self.segments, self.signs = EphemerisBody._ephem_cache[cache_key]
        if self.naif_name not in gms:
            raise RuntimeError(f"Invalid object name: {self.naif_name}.")
        self.gm = gms[self.naif_name]

    @eqx.filter_jit
    def _bcrs_pv_jd(self, tdb_jd1: Float[Array, "..."], tdb_jd2: Float[Array, "..."]) -> tuple[
        Float[Array, "... 3"], Float[Array, "... 3"]]:
        target_shape = tdb_jd1.shape + (3,)
        pos = jnp.zeros(target_shape)
        vel = jnp.zeros(target_shape)
        for seg, sign in zip(self.segments, self.signs):
            p, v = seg.state(tdb_jd1, tdb_jd2)
            pos = pos + sign * p
            vel = vel + sign * v
        pos = pos / AU_KM
        vel = vel / AU_KM
        return pos, vel

    @eqx.filter_jit
    def _bcrs_pos_jd(self, tdb_jd1: Float[Array, "..."], tdb_jd2: Float[Array, "..."]) -> Float[Array, "... 3"]:
        target_shape = tdb_jd1.shape + (3,)
        pos = jnp.zeros(target_shape)
        for seg, sign in zip(self.segments, self.signs):
            p = seg.pos(tdb_jd1, tdb_jd2)
            pos = pos + sign * p
        pos = pos / AU_KM
        return pos

    @eqx.filter_jit
    def _bcrs_pva_jd(self, tdb_jd1: Float[Array, "..."], tdb_jd2: Float[Array, "..."]) -> tuple[Float[Array, "... 3"],
    Float[Array, "... 3"], Float[Array, "... 3"]]:
        target_shape = tdb_jd1.shape + (3,)
        pos = jnp.zeros(target_shape)
        vel = jnp.zeros(target_shape)
        acc = jnp.zeros(target_shape)
        for seg, sign in zip(self.segments, self.signs):
            p, v, a = seg.pva(tdb_jd1, tdb_jd2)
            pos = pos + sign * p
            vel = vel + sign * v
            acc = acc + sign * a
        pos = pos / AU_KM
        vel = vel / AU_KM
        acc = acc / AU_KM
        return pos, vel, acc

    @eqx.filter_jit
    def state(
            self,
            tdb: TDBView,
            frame: Frame = BCRS,
            *,
            sun: "EphemerisBody | None" = None,
            earth: "EphemerisBody | None" = None,
    ) -> State:
        """Return the state at the given epoch in one requested frame.

        Parameters
        ----------
        tdb : TDBView
            Epoch in ``TDB``.
        frame : Frame, default=``BCRS``
            Target output frame.
        sun : EphemerisBody, optional
            Sun ephemeris body used when ``frame`` touches the ``SUN`` origin.
        earth : EphemerisBody, optional
            Earth ephemeris body used when ``frame`` touches the ``EARTH`` origin.

        Returns
        -------
        State
            State in ``frame``. Position is in ``au`` and velocity is in ``au / day``.

        Raises
        ------
        TypeError
            If ``tdb`` is not an instance of :class:`TDBView`.
        ValueError
            If converting the canonical ``BCRS`` state to ``frame`` requires the Sun or Earth and the corresponding ephemeris body is not available.

        Notes
        -----
        The native ephemeris output is canonical ``BCRS``. This method evaluates that state first and then converts it through :class:`difforb.core.state.state.State`.
        """
        validate_timeview(tdb, TDBView, 'tdb')
        pos, vel = self._bcrs_pv_jd(tdb.jd1, tdb.jd2)
        state = State(tdb=tdb, pos=pos, vel=vel, frame=BCRS)

        if frame == BCRS:
            return state
        if frame.origin is Origin.SUN and sun is None:
            sun = EphemerisBody("sun")
        if frame.origin is Origin.EARTH and earth is None:
            earth = EphemerisBody("earth")
        return state.to(frame, sun=sun, earth=earth)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return build_repr(
            self.__class__.__name__,
            [
                ("naif_name", self.naif_name),
                ("gm_au3_per_d2", format_float_array(self.gm)),
                ("segment_count", str(len(self.segments))),
            ],
        )
