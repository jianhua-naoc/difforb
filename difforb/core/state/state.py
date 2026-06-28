"""Cartesian state vectors for ``state``.

This module defines the single Cartesian state container used by the new state-vector system. Each state stores one ``TDB`` epoch, one position-velocity pair, and one explicit :class:`Frame` that combines axis orientation and origin semantics.

Frame conversion is factored into two independent steps:

- axis rotation, handled by :mod:`difforb.core.state.axes`,
- origin translation, handled by :mod:`difforb.core.state.origins`.

The current axis families are fixed rotations, so both position and velocity are rotated by the same constant matrix. Origin changes use the origin state in ``SSB`` and ``ICRS`` returned by :func:`difforb.core.state.origins.origin_in_ssb_icrs`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float

from difforb.core.batch import BatchableObject
from difforb.core.constants import C, LC
from difforb.report.text import build_repr, format_float_array, format_shape

from .axes import Axes, axes_to_icrs_rotation, icrs_to_axes_rotation
from .frame import BCRS, GCRS, HELIO_ECLIP_J2000, HELIO_ICRS, HELIO_J2000, Frame
from .origins import Origin, origin_in_ssb_icrs

if TYPE_CHECKING:
    from difforb.body.ephbody import EphemerisBody
    from difforb.core.time.timescale import TDBView

S = TypeVar("S", bound="State")


class State(BatchableObject):
    """Cartesian state vector at one ``TDB`` epoch in one explicit frame.

    Parameters
    ----------
    tdb : TDBView
        Epoch in ``TDB``.
    pos : Float[ArrayLike, "... 3"]
        Position in the canonical distance unit of the library, ``au``.
    vel : Float[ArrayLike, "... 3"]
        Velocity in the canonical speed unit of the library, ``au / day``.
    frame : Frame
        Frame that defines the axis orientation and the origin of ``pos`` and ``vel``.
    """

    tdb: "TDBView"
    pos: Float[Array, "... 3"]
    vel: Float[Array, "... 3"]
    frame: Frame = eqx.field(static=True)

    def __init__(
            self,
            tdb: "TDBView",
            pos: Float[ArrayLike, "... 3"],
            vel: Float[ArrayLike, "... 3"],
            frame: Frame,
    ) -> None:
        """Initialize a Cartesian state from epoch, position, velocity, and frame.

        Parameters
        ----------
        tdb : TDBView
            Epoch in ``TDB``.
        pos : Float[ArrayLike, "... 3"]
            Position in ``au``.
        vel : Float[ArrayLike, "... 3"]
            Velocity in ``au / day``.
        frame : Frame
            Frame attached to the state vector.
        """

        self.tdb = tdb
        p = jnp.asarray(pos, dtype=float)
        v = jnp.asarray(vel, dtype=float)
        self.pos, self.vel = jnp.broadcast_arrays(p, v)
        if self.pos.shape[-1] != 3:
            raise ValueError(f"State position last dimension must be 3, got {self.pos.shape}.")
        if self.vel.shape[-1] != 3:
            raise ValueError(f"State velocity last dimension must be 3, got {self.vel.shape}.")
        if tdb.shape != self.pos.shape[:-1]:
            raise ValueError("``tdb`` and ``pos``/``vel`` must have same shape.")
        self.frame = frame

    def __repr__(self) -> str:
        frame_label = self.frame.name or f"{self.frame.origin.value}+{self.frame.axes.value}"
        return build_repr(
            self.__class__.__name__,
            [
                ("frame", frame_label),
                ("shape", format_shape(self.shape)),
                ("epoch_jd", format_float_array(self.tdb.jd, precision=9, scientific=False, signed=False)),
                ("pos_au", format_float_array(self.pos)),
                ("vel_au_per_d", format_float_array(self.vel)),
            ],
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the batch shape."""

        return self.pos.shape[:-1]

    @property
    def array(self) -> Float[Array, "... 6"]:
        """Return the stacked Cartesian array.

        Returns
        -------
        Float[Array, "... 6"]
            Array ordered as ``[..., x, y, z, vx, vy, vz]`` in ``au`` and ``au / day``.
        """

        return jnp.concatenate([self.pos, self.vel], axis=-1)

    @classmethod
    def from_array(
            cls: type[S],
            tdb: "TDBView",
            array: Float[ArrayLike, "... 6"],
            frame: Frame,
    ) -> S:
        """Build a state from one stacked Cartesian array.

        Parameters
        ----------
        tdb : TDBView
            Epoch in ``TDB``.
        array : Float[ArrayLike, "... 6"]
            Array ordered as ``[..., x, y, z, vx, vy, vz]`` in ``au`` and ``au / day``.
        frame : Frame
            Frame attached to the state vector.

        Returns
        -------
        State
            State object of ``cls`` in ``frame``.
        """

        arr = jnp.asarray(array, dtype=float)
        if arr.shape[-1] != 6:
            raise ValueError(f"State array last dimension must be 6, got {arr.shape}.")
        return cls(tdb=tdb, pos=arr[..., :3], vel=arr[..., 3:], frame=frame)

    @property
    def dist(self) -> Float[Array, "..."]:
        """Distance from the current frame origin in ``au``."""

        return jnp.linalg.norm(self.pos, axis=-1)

    @property
    def lt(self) -> Float[Array, "..."]:
        """Light time for the current origin distance in days."""

        return self.dist / C

    def _to_icrs_axes(self) -> "State":
        """Rotate the state axes to ``ICRS`` without changing the origin.

        Returns
        -------
        State
            Same physical state expressed in ``ICRS`` axes and the original origin.
        """

        if self.frame.axes is Axes.ICRS:
            return self
        rot = axes_to_icrs_rotation(self.frame.axes)
        return State(
            tdb=self.tdb,
            pos=self.pos @ rot,
            vel=self.vel @ rot,
            frame=Frame(axes=Axes.ICRS, origin=self.frame.origin),
        )

    def _from_icrs_axes(self, target_frame: Frame) -> "State":
        """Rotate one ``ICRS``-axes state to the target axis family.

        Parameters
        ----------
        target_frame : Frame
            Target frame. Only ``target_frame.axes`` is used by this method. The origin is preserved from ``self``.

        Returns
        -------
        State
            Same physical state expressed in ``target_frame.axes`` and the original origin.

        Raises
        ------
        ValueError
            If ``self`` is not already in ``ICRS`` axes.
        """

        if self.frame.axes is not Axes.ICRS:
            raise ValueError("``_from_icrs_axes`` requires a state already expressed in ``ICRS`` axes.")
        if target_frame.axes is Axes.ICRS:
            return self
        rot = icrs_to_axes_rotation(target_frame.axes)
        return State(
            tdb=self.tdb,
            pos=self.pos @ rot,
            vel=self.vel @ rot,
            frame=Frame(axes=target_frame.axes, origin=self.frame.origin),
        )

    def _to_ssb_origin(
            self,
            *,
            sun: "EphemerisBody | None" = None,
            earth: "EphemerisBody | None" = None,
    ) -> "State":
        """Shift the origin to ``SSB`` without changing the axes.

        Parameters
        ----------
        sun : EphemerisBody, optional
            Sun ephemeris body used when the current origin is ``SUN``.
        earth : EphemerisBody, optional
            Earth ephemeris body used when the current origin is ``EARTH``.

        Returns
        -------
        State
            Same physical state expressed with the ``SSB`` origin and the original axes.
        """

        if self.frame.origin is Origin.SSB:
            return self
        origin_pos, origin_vel = origin_in_ssb_icrs(self.frame.origin, self.tdb, sun=sun, earth=earth)
        pos, vel = self.pos, self.vel
        if self.frame.origin is Origin.EARTH:
            v_earth_c = origin_vel / C
            v_dot_p = jnp.sum(v_earth_c * pos, axis=-1, keepdims=True)
            pos = pos * (1.0 - LC) - 0.5 * v_dot_p * v_earth_c
            vel = vel * (1.0 - LC)
        return State(
            tdb=self.tdb,
            pos=pos + origin_pos,
            vel=vel + origin_vel,
            frame=Frame(axes=self.frame.axes, origin=Origin.SSB),
        )

    def _from_ssb_origin(
            self,
            target_frame: Frame,
            *,
            sun: "EphemerisBody | None" = None,
            earth: "EphemerisBody | None" = None,
    ) -> "State":
        """Shift one ``SSB``-origin state to the target origin.

        Parameters
        ----------
        target_frame : Frame
            Target frame. Only ``target_frame.origin`` is used by this method. The axes are preserved from ``self``.
        sun : EphemerisBody, optional
            Sun ephemeris body used when the target origin is ``SUN``.
        earth : EphemerisBody, optional
            Earth ephemeris body used when the target origin is ``EARTH``.

        Returns
        -------
        State
            Same physical state expressed with ``target_frame.origin`` and the original axes.

        Raises
        ------
        ValueError
            If ``self`` is not already expressed with the ``SSB`` origin.
        """

        if self.frame.origin is not Origin.SSB:
            raise ValueError("``_from_ssb_origin`` requires a state already expressed with the ``SSB`` origin.")
        if target_frame.origin is Origin.SSB:
            return self
        origin_pos, origin_vel = origin_in_ssb_icrs(target_frame.origin, self.tdb, sun=sun, earth=earth)
        pos = self.pos - origin_pos
        vel = self.vel - origin_vel
        if target_frame.origin is Origin.EARTH:
            v_earth_c = origin_vel / C
            v_dot_p = jnp.sum(v_earth_c * pos, axis=-1, keepdims=True)
            pos = pos * (1.0 + LC) + 0.5 * v_dot_p * v_earth_c
            vel = vel * (1.0 + LC)
        return State(
            tdb=self.tdb,
            pos=pos,
            vel=vel,
            frame=Frame(axes=self.frame.axes, origin=target_frame.origin),
        )

    @eqx.filter_jit
    def to(
            self,
            frame: Frame,
            *,
            sun: "EphemerisBody | None" = None,
            earth: "EphemerisBody | None" = None,
    ) -> "State":
        """Convert the state to another frame.

        Parameters
        ----------
        frame : Frame
            Target frame.
        sun : EphemerisBody, optional
            Sun ephemeris body used when the source or target origin is ``SUN``.
        earth : EphemerisBody, optional
            Earth ephemeris body used when the source or target origin is ``EARTH``.

        Returns
        -------
        State
            Same physical state expressed in ``frame``.

        Raises
        ------
        ValueError
            If the conversion touches the ``SUN`` origin and ``sun`` is not provided, or touches the ``EARTH`` origin and ``earth`` is not provided.

        Notes
        -----
        The conversion always passes through the canonical intermediate representation with ``ICRS`` axes and the ``SSB`` origin.
        """

        if self.frame == frame:
            return self
        state = self._to_icrs_axes()
        state = state._to_ssb_origin(sun=sun, earth=earth)
        state = state._from_ssb_origin(frame, sun=sun, earth=earth)
        state = state._from_icrs_axes(frame)
        return State(tdb=state.tdb, pos=state.pos, vel=state.vel, frame=frame)

    def bcrs(
            self,
            *,
            sun: "EphemerisBody | None" = None,
            earth: "EphemerisBody | None" = None,
    ) -> "State":
        """Convert the state to ``BCRS``.

        Parameters
        ----------
        sun : EphemerisBody, optional
            Sun ephemeris body used when the source origin is ``SUN``.
        earth : EphemerisBody, optional
            Earth ephemeris body used when the source origin is ``EARTH``.

        Returns
        -------
        State
            Same physical state expressed in ``BCRS``.
        """

        return self.to(BCRS, sun=sun, earth=earth)

    def gcrs(
            self,
            *,
            sun: "EphemerisBody | None" = None,
            earth: "EphemerisBody | None" = None,
    ) -> "State":
        """Convert the state to ``GCRS``.

        Parameters
        ----------
        sun : EphemerisBody, optional
            Sun ephemeris body used when the source origin is ``SUN``.
        earth : EphemerisBody, optional
            Earth ephemeris body used when the source or target origin is ``EARTH``.

        Returns
        -------
        State
            Same physical state expressed in ``GCRS``.
        """

        return self.to(GCRS, sun=sun, earth=earth)

    def helio_icrs(
            self,
            *,
            sun: "EphemerisBody | None" = None,
            earth: "EphemerisBody | None" = None,
    ) -> "State":
        """Convert the state to heliocentric ``ICRS``.

        Parameters
        ----------
        sun : EphemerisBody, optional
            Sun ephemeris body used when the source or target origin is ``SUN``.
        earth : EphemerisBody, optional
            Earth ephemeris body used when the source origin is ``EARTH``.

        Returns
        -------
        State
            Same physical state expressed in heliocentric ``ICRS``.
        """

        return self.to(HELIO_ICRS, sun=sun, earth=earth)

    def helio_j2000(
            self,
            *,
            sun: "EphemerisBody | None" = None,
            earth: "EphemerisBody | None" = None,
    ) -> "State":
        """Convert the state to heliocentric J2000.

        Parameters
        ----------
        sun : EphemerisBody, optional
            Sun ephemeris body used when the source or target origin is ``SUN``.
        earth : EphemerisBody, optional
            Earth ephemeris body used when the source origin is ``EARTH``.

        Returns
        -------
        State
            Same physical state expressed in heliocentric J2000.
        """

        return self.to(HELIO_J2000, sun=sun, earth=earth)

    def helio_eclip_j2000(
            self,
            *,
            sun: "EphemerisBody | None" = None,
            earth: "EphemerisBody | None" = None,
    ) -> "State":
        """Convert the state to heliocentric ecliptic J2000.

        Parameters
        ----------
        sun : EphemerisBody, optional
            Sun ephemeris body used when the source or target origin is ``SUN``.
        earth : EphemerisBody, optional
            Earth ephemeris body used when the source origin is ``EARTH``.

        Returns
        -------
        State
            Same physical state expressed in heliocentric ecliptic J2000.
        """

        return self.to(HELIO_ECLIP_J2000, sun=sun, earth=earth)
