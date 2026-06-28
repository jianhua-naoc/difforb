"""Relative Cartesian states in fixed ``ICRS`` axes.

This module defines :class:`RelativeState`, a lightweight container for
observer-relative or endpoint-relative position-velocity vectors. Unlike
``state.State``, a relative state does not carry origin metadata and does not
participate in frame conversion. The stored vectors are always interpreted in
fixed ``ICRS`` axes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float

from difforb.core.batch import BatchableObject
from difforb.core.constants import C
from difforb.report.text import build_repr, format_float_array, format_shape

if TYPE_CHECKING:
    from difforb.core.time.timescale import TDBView

R = TypeVar("R", bound="RelativeState")


class RelativeState(BatchableObject):
    """Relative Cartesian state at one ``TDB`` epoch in fixed ``ICRS`` axes.

    Parameters
    ----------
    tdb : TDBView
        Epoch in ``TDB``.
    pos : Float[ArrayLike, "... 3"]
        Relative position vector in ``au``.
    vel : Float[ArrayLike, "... 3"]
        Relative velocity vector in ``au / day``.

    Notes
    -----
    This class is intended for products such as observer-relative geometric,
    astrometric, and apparent vectors. It does not store an origin and does not
    support frame conversion.
    """

    tdb: "TDBView"
    pos: Float[Array, "... 3"]
    vel: Float[Array, "... 3"]

    def __init__(
            self,
            tdb: "TDBView",
            pos: Float[ArrayLike, "... 3"],
            vel: Float[ArrayLike, "... 3"],
    ) -> None:
        """Initialize a relative Cartesian state from epoch, position, and velocity.

        Parameters
        ----------
        tdb : TDBView
            Epoch in ``TDB``.
        pos : Float[ArrayLike, "... 3"]
            Relative position vector in ``au``.
        vel : Float[ArrayLike, "... 3"]
            Relative velocity vector in ``au / day``.
        """
        self.tdb = tdb
        p = jnp.asarray(pos, dtype=float)
        v = jnp.asarray(vel, dtype=float)
        self.pos, self.vel = jnp.broadcast_arrays(p, v)
        if self.pos.shape[-1] != 3:
            raise ValueError(f"RelativeState position last dimension must be 3, got {self.pos.shape}.")
        if self.vel.shape[-1] != 3:
            raise ValueError(f"RelativeState velocity last dimension must be 3, got {self.vel.shape}.")
        if tdb.shape != self.pos.shape[:-1]:
            raise ValueError("``tdb`` and ``pos``/``vel`` must have same shape.")

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("axes", "ICRS"),
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
            Array ordered as ``[..., x, y, z, vx, vy, vz]`` in ``au`` and
            ``au / day``.
        """
        return jnp.concatenate([self.pos, self.vel], axis=-1)

    @classmethod
    def from_array(
            cls: type[R],
            tdb: "TDBView",
            array: Float[ArrayLike, "... 6"],
    ) -> R:
        """Build a relative state from one stacked Cartesian array.

        Parameters
        ----------
        tdb : TDBView
            Epoch in ``TDB``.
        array : Float[ArrayLike, "... 6"]
            Array ordered as ``[..., x, y, z, vx, vy, vz]`` in ``au`` and
            ``au / day``.

        Returns
        -------
        RelativeState
            Relative state object of ``cls``.
        """
        arr = jnp.asarray(array, dtype=float)
        if arr.shape[-1] != 6:
            raise ValueError(f"RelativeState array last dimension must be 6, got {arr.shape}.")
        return cls(tdb=tdb, pos=arr[..., :3], vel=arr[..., 3:])

    @property
    def dist(self) -> Float[Array, "..."]:
        """Distance of the relative position vector in ``au``."""
        return jnp.linalg.norm(self.pos, axis=-1)

    @property
    def lt(self) -> Float[Array, "..."]:
        """Light time of the relative distance in days."""
        return self.dist / C
