"""Ephemeris bodies based on ``SPK`` kernels.

This module defines :class:`EphemerisBody`, which evaluates its ``BCRS`` state from an ``SPK`` ephemeris.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from typing import Optional
from jax import Array
from jaxtyping import Float

from difforb.core.time.timescale import TDBView
from difforb.core.validate import validate_timeview
from difforb.spk.spk import Ephemeris, MergedSegment
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
        self.segments, self.signs = _eph.load_body(self.naif_name)
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


class EphemerisBodyBatch(eqx.Module):
    """Compact evaluator for a fixed collection of ephemeris bodies."""

    segment_groups: tuple
    group_members: tuple = eqx.field(static=True)
    body_paths: tuple = eqx.field(static=True)

    def __init__(self, bodies: list[EphemerisBody]) -> None:
        """Group shared and structurally identical SPK paths across bodies."""
        if not all(isinstance(body, EphemerisBody) for body in bodies):
            raise TypeError("EphemerisBodyBatch only accepts EphemerisBody instances.")

        unique_segments = []
        segment_indices = {}
        body_paths = []
        for body in bodies:
            path = []
            for segment, sign in zip(body.segments, body.signs):
                identity = id(segment)
                if identity not in segment_indices:
                    segment_indices[identity] = len(unique_segments)
                    unique_segments.append(segment)
                path.append((segment_indices[identity], sign))
            body_paths.append(tuple(path))

        groups = {}
        for index, segment in enumerate(unique_segments):
            signature = tuple(
                (leaf.shape, str(leaf.dtype))
                for leaf in jax.tree_util.tree_leaves(segment)
            )
            groups.setdefault(signature, []).append(index)

        segment_groups = []
        group_members = []
        for indices in groups.values():
            segments = [unique_segments[index] for index in indices]
            segment_groups.append(
                jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *segments)
            )
            group_members.append(tuple(indices))

        self.segment_groups = tuple(segment_groups)
        self.group_members = tuple(group_members)
        self.body_paths = tuple(body_paths)

    @eqx.filter_jit
    def evaluate(self, tdb_jd1: Float[Array, "..."], tdb_jd2: Float[Array, "..."], *,
                 derivatives: bool = False):
        """Evaluate body positions or position, velocity, and acceleration."""
        segment_count = sum(len(members) for members in self.group_members)
        segment_values = [None] * segment_count

        for segments, members in zip(self.segment_groups, self.group_members):
            if derivatives:
                positions, velocities, accelerations = jax.vmap(
                    MergedSegment.pva, in_axes=(0, None, None),
                )(segments, tdb_jd1, tdb_jd2)
                for position, velocity, acceleration, index in zip(
                        positions, velocities, accelerations, members,
                ):
                    segment_values[index] = (position, velocity, acceleration)
            else:
                positions = jax.vmap(
                    MergedSegment.pos, in_axes=(0, None, None),
                )(segments, tdb_jd1, tdb_jd2)
                for position, index in zip(positions, members):
                    segment_values[index] = position

        target_shape = tdb_jd1.shape + (3,)
        if derivatives:
            body_pva = []
            for path in self.body_paths:
                position = jnp.zeros(target_shape)
                velocity = jnp.zeros(target_shape)
                acceleration = jnp.zeros(target_shape)
                for index, sign in path:
                    segment_position, segment_velocity, segment_acceleration = segment_values[index]
                    position = position + sign * segment_position
                    velocity = velocity + sign * segment_velocity
                    acceleration = acceleration + sign * segment_acceleration
                body_pva.append((position / AU_KM, velocity / AU_KM, acceleration / AU_KM))
            return tuple(jnp.stack(values) for values in zip(*body_pva))

        body_positions = []
        for path in self.body_paths:
            position = jnp.zeros(target_shape)
            for index, sign in path:
                position = position + sign * segment_values[index]
            body_positions.append(position / AU_KM)
        return jnp.stack(body_positions)
