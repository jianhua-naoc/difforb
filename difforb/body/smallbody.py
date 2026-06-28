"""Small-body orbit container and propagation interface.

This module defines :class:`SmallBody`, which stores the initial orbit of a small body in canonical ``BCRS`` form with :class:`difforb.core.state.state.State`, an optional photometric model, and an optional propagated trajectory. It bridges orbital elements and frame-aware Cartesian states with the numerical propagation stack in :mod:`difforb.dynamics.force_model` and :mod:`difforb.integrator.integrator`.
"""

from typing import Union, Optional

import jax.numpy as jnp
import equinox as eqx
from jax import Array
from jaxtyping import Float

from difforb.astrometry.reduction.photometry import MagModel
from difforb.body.ephbody import EphemerisBody
from difforb.core.batch import BatchableObject
from difforb.core.element import KepElement
from difforb.core.state.frame import BCRS, Frame
from difforb.core.state.origins import Origin
from difforb.core.state.state import State
from difforb.core.time.timescale import TDBView
from difforb.core.validate import validate_timeview
from difforb.dynamics.force_model import ForceModel
from difforb.integrator.integrator import NumericalIntegrator, BiDirectionalInterpolator
from difforb.report.text import build_repr, format_float_array, format_shape

Orbit = Union[KepElement, State]


class SmallBody(BatchableObject):
    """Small body with an initial orbit and an optional propagated trajectory.

    Parameters
    ----------
    orbit0 : State
        Initial state in canonical ``BCRS`` form. Position is in ``au`` and velocity is in ``au / day``.
    mag_model : MagModel, optional
        Photometric model attached to the body.

    Notes
    -----
    The stored epoch is in ``TDB``. The trajectory field is empty until :meth:`propagate` is called.
    """

    orbit0: State
    mag_model: Optional[MagModel]
    trajectory: Optional[BiDirectionalInterpolator]

    def __init__(self, orbit0: State, mag_model: MagModel = None) -> None:
        """Initialize a small body from an initial canonical ``BCRS`` state.

        Parameters
        ----------
        orbit0 : State
            Initial state in canonical ``BCRS`` form.
        mag_model : MagModel, optional
            Photometric model attached to the body.

        Raises
        ------
        TypeError
            If ``orbit0`` is not an instance of :class:`difforb.core.state.state.State`.
        ValueError
            If ``orbit0.frame`` is not ``BCRS``.
        """
        if not isinstance(orbit0, State):
            raise TypeError(
                f"Invalid initial orbit: `orbit0` must be an instance of `State`, but got `{type(orbit0).__name__}`."
            )
        if orbit0.frame != BCRS:
            raise ValueError("Invalid initial orbit: `orbit0.frame` must be `BCRS`.")
        self.orbit0 = orbit0
        self.mag_model = mag_model
        self.trajectory = None

    @classmethod
    def create(
            cls,
            orbit: Orbit,
            mag_model: MagModel = None,
            *,
            sun: EphemerisBody | None = None,
            earth: EphemerisBody | None = None,
    ) -> 'SmallBody':
        """Build a small body from one supported orbit representation.

        Parameters
        ----------
        orbit : Orbit
            Initial orbit given as a Keplerian element set or a frame-aware Cartesian state.
        mag_model : MagModel, optional
            Photometric model attached to the body.
        sun : EphemerisBody, optional
            Sun ephemeris body used when ``orbit`` must be shifted from a heliocentric origin to ``BCRS``.
        earth : EphemerisBody, optional
            Earth ephemeris body used when ``orbit`` must be shifted from a geocentric origin to ``BCRS``.

        Returns
        -------
        SmallBody
            Small body built from ``orbit``.

        Raises
        ------
        TypeError
            If ``orbit`` is not a supported orbit type.
        ValueError
            If converting the provided Cartesian state to ``BCRS`` requires the Sun or Earth and the corresponding ephemeris body is not available.

        Notes
        -----
        The internal stored orbit is always canonical ``BCRS``. Non-``BCRS`` Cartesian inputs are converted through :meth:`difforb.core.state.state.State.to`.
        """
        if isinstance(orbit, KepElement):
            state = orbit.state()
        elif isinstance(orbit, State):
            state = orbit
        else:
            raise TypeError(
                f"Invalid initial orbit: `orbit` must be an instance of `KepElement` or `State`, but got `{type(orbit).__name__}`."
            )

        if state.frame != BCRS:
            if state.frame.origin is Origin.SUN and sun is None:
                sun = EphemerisBody("sun")
            if state.frame.origin is Origin.EARTH and earth is None:
                earth = EphemerisBody("earth")
            state = state.to(BCRS, sun=sun, earth=earth)

        return cls(state, mag_model)

    @property
    def shape(self):
        """Batch shape of the stored initial state."""
        return self.orbit0.shape

    def propagate(self, t_start: TDBView, t_end: TDBView, force_model: ForceModel,
                  integrator: NumericalIntegrator, grid: bool = False) -> 'SmallBody':
        """Propagate the orbit and store the trajectory interpolator.

        Parameters
        ----------
        t_start, t_end : TDBView
            Start and end epoch in ``TDB``.
        force_model : ForceModel
            Dynamical model used in the equations of motion.
        integrator : NumericalIntegrator
            Numerical integrator used to build the trajectory.
        grid : bool, default=False
            If ``True``, use the Cartesian product of the body batch and the time batch. If ``False``, use point-wise broadcasting.

        Returns
        -------
        SmallBody
            New object with the ``trajectory`` field set.

        Raises
        ------
        TypeError
            If ``t_start`` or ``t_end`` is not a :class:`TDB` epoch.
        """
        validate_timeview(t_start, TDBView, 't_start')
        validate_timeview(t_end, TDBView, 't_end')
        trajectory = integrator(force_model, self.orbit0.array, self.orbit0.tdb.jd1, self.orbit0.tdb.jd2,
                                t_start.jd1, t_start.jd2, t_end.jd1, t_end.jd2, grid)
        return eqx.tree_at(lambda smallbody: smallbody.trajectory, self, trajectory, is_leaf=lambda x: x is None)

    @eqx.filter_jit
    def _bcrs_pv_jd(
            self,
            tdb_jd1: Float[Array, "..."],
            tdb_jd2: Float[Array, "..."],
            grid: bool = False,
    ) -> tuple[Float[Array, "... 3"], Float[Array, "... 3"]]:
        # -------------------------------------------------------------------------
        # Step 1: Check that the propagated trajectory is ready
        # -------------------------------------------------------------------------
        if self.trajectory is None:
            raise RuntimeError("Trajectory is not initialized. Please call 'propagate()' first to integrate the orbit.")

        # -------------------------------------------------------------------------
        # Step 2: Interpolate the trajectory and check the time range
        # -------------------------------------------------------------------------
        is_covered = self.trajectory.is_covered(tdb_jd1, tdb_jd2, grid)
        final_pos, final_vel = self.trajectory(tdb_jd1, tdb_jd2, grid)
        final_pos = eqx.error_if(
            final_pos,
            jnp.logical_not(is_covered),
            "Interpolation Error: The requested time `t` is outside the coverage of the propagated trajectory. Please expand the range in `propagate()`.",
        )
        return final_pos, final_vel

    def state(
            self,
            tdb: TDBView,
            frame: Frame = BCRS,
            *,
            sun: EphemerisBody | None = None,
            earth: EphemerisBody | None = None,
            grid: bool = False,
    ) -> State:
        """Evaluate the propagated orbit in one requested frame.

        Parameters
        ----------
        tdb : TDBView
            Target epoch in ``TDB``.
        frame : Frame, default=``BCRS``
            Target output frame.
        sun : EphemerisBody, optional
            Sun ephemeris body used when the target frame touches the ``SUN`` origin.
        earth : EphemerisBody, optional
            Earth ephemeris body used when the target frame touches the ``EARTH`` origin.
        grid : bool, default=False
            If ``True``, use the Cartesian product of the body batch and the time batch. If ``False``, use point-wise broadcasting.

        Returns
        -------
        State
            Interpolated state in ``frame``.

        Raises
        ------
        TypeError
            If ``tdb`` is not a :class:`TDBView`.
        RuntimeError
            If the trajectory has not been initialized, or if ``tdb`` falls outside the propagated coverage interval.
        ValueError
            If the target frame touches the ``SUN`` origin and ``sun`` is not provided, or touches the ``EARTH`` origin and ``earth`` is not provided.

        Notes
        -----
        This method first evaluates the propagated orbit in canonical ``BCRS`` form with the internal :meth:`_bcrs_pv_jd` evaluator, and then converts the result to ``frame`` through :class:`difforb.core.state.state.State`.
        """
        validate_timeview(tdb, TDBView, 'tdb')

        final_pos, final_vel = self._bcrs_pv_jd(tdb.jd1, tdb.jd2, grid)

        # -------------------------------------------------------------------------
        # Step 3: Broadcast the epoch and build the canonical ``BCRS`` state
        # -------------------------------------------------------------------------
        body_shape = self.orbit0.shape
        time_shape = tdb.shape
        if not grid:
            target_shape = jnp.broadcast_shapes(body_shape, time_shape)

            def bcast(x):
                return jnp.broadcast_to(x, target_shape)
        else:
            target_shape = body_shape + time_shape
            expand_axes = tuple(range(len(body_shape)))

            def bcast(x):
                return jnp.broadcast_to(jnp.expand_dims(x, axis=expand_axes), target_shape)
        tdb_bcast = eqx.tree_at(lambda t: (t.jd1, t.jd2, t.time._tt_jd1, t.time._tt_jd2), tdb, replace_fn=bcast)

        state = State(tdb=tdb_bcast, pos=final_pos, vel=final_vel, frame=BCRS)

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
        mag_model_name = self.mag_model.__class__.__name__ if self.mag_model is not None else "none"
        trajectory_status = "ready" if self.trajectory is not None else "uninitialized"
        return build_repr(
            self.__class__.__name__,
            [
                ("shape", format_shape(self.shape)),
                ("epoch_jd", format_float_array(self.orbit0.tdb.jd, precision=9, scientific=False, signed=False)),
                ("frame", self.orbit0.frame.name),
                ("mag_model", mag_model_name),
                ("trajectory", trajectory_status),
            ],
        )
