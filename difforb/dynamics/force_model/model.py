"""Aggregate force model used by numerical propagators.

The :class:`ForceModel` object sums individual force terms at one ``TDB`` offset and exposes a single estimated-parameter vector for orbit-determination solvers.
"""

from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from difforb.core.batch import BatchableObject
from difforb.core.time.utils import renormalize_split_jd
from difforb.dynamics.force_model.base import Force, ParametrizedForce
from difforb.dynamics.force_model.gravity import NewtonianGravity, PPNGravity
from difforb.report.text import build_repr, format_class_name_array, format_shape, format_string_array


class ForceModel(BatchableObject):
    """Sum a list of force terms for numerical propagation."""
    forces: tuple

    def __init__(self, forces: List['Force']):
        """Initialize the force model.

        Parameters
        ----------
        forces : list[Force]
            Force terms included in the model.

        Raises
        ------
        TypeError
            If any force does not implement the ordinary ``float64`` force interface.
        """
        for force in forces:
            if not isinstance(force, Force):
                raise TypeError("difforb.dynamics.ForceModel only accepts difforb.dynamics.Force terms.")
        self.forces = tuple(forces)

    @jax.jit
    def __call__(self, tdb_offset: float, state: Tuple[Float[Array, "3"], Float[Array, "3"]], args=None) -> Float[
        Array, "3"]:
        """Evaluate the total acceleration.

        Parameters
        ----------
        tdb_offset : float
            Offset in days from the reference ``TDB`` epoch stored in ``args``.
        state : tuple[Float[Array, "3"], Float[Array, "3"]]
            Cartesian state ``(pos, vel)`` in ``BCRS``.
        args : Any, optional
            Extra runtime data. The first two items must be ``(t0_jd1, t0_jd2)``.

        Returns
        -------
        Float[Array, "3"]
            Total acceleration in ``au / day^2``.
        """
        t0_jd1, t0_jd2 = args
        tdb_jd1, tdb_jd2 = renormalize_split_jd(t0_jd1, t0_jd2 + tdb_offset)
        acc = jnp.zeros(3)
        for force in self.forces:
            acc += force(tdb_jd1, tdb_jd2, state, args)
        return acc

    def _prepare(self, tdb_offset, args):
        """Prepare built-in gravity backgrounds for repeated states at one epoch.

        The offset is in days from the split ``TDB`` reference epoch in ``args``. The returned PyTree retains time and force-parameter derivatives and is valid only for this epoch and model. Return ``None`` when no force supports reuse.
        """
        if not any(type(force) in (NewtonianGravity, PPNGravity) for force in self.forces):
            return None
        t0_jd1, t0_jd2 = args
        tdb_jd1, tdb_jd2 = renormalize_split_jd(t0_jd1, t0_jd2 + tdb_offset)
        # Exact types preserve custom subclasses' acceleration implementations.
        backgrounds = tuple(
            force._prepare(tdb_jd1, tdb_jd2)
            if type(force) in (NewtonianGravity, PPNGravity) else None
            for force in self.forces
        )
        return tdb_jd1, tdb_jd2, backgrounds

    def _evaluate_prepared(self, state, args, prepared):
        """Recompute acceleration for the current ``BCRS`` state using prepared backgrounds."""
        tdb_jd1, tdb_jd2, backgrounds = prepared
        acc = jnp.zeros(3)
        for force, background in zip(self.forces, backgrounds):
            if background is None:
                acc += force(tdb_jd1, tdb_jd2, state, args)
            else:
                acc += force._evaluate_prepared(state, background)
        return acc

    def get_all_estimated_params(self) -> Float[Array, "N_all_estimated"]:
        """Collect all estimated parameters into one array."""
        params = [f.get_estimated_params() for f in self.forces if isinstance(f, ParametrizedForce)]
        return jnp.concatenate(params, axis=-1) if params else jnp.array([])

    def update_estimated_params(self, estimated_params: Float[Array, "N_all_estimated"]) -> 'ForceModel':
        """Return a new ``ForceModel`` instance with updated estimated parameters."""
        new_forces = []
        cursor = 0

        for force in self.forces:
            if isinstance(force, ParametrizedForce):
                n = force.n_estimated_params
                new_params = estimated_params[..., cursor:cursor + n]
                new_force = force.update_estimated_params(new_params)
                new_forces.append(new_force)
                cursor += n
            else:
                new_forces.append(force)
        return eqx.tree_at(lambda m: m.forces, self, tuple(new_forces))

    def get_all_estimated_param_names(self) -> List[str]:
        """Return the names of all estimated parameters."""
        names = []
        for force in self.forces:
            if isinstance(force, ParametrizedForce):
                names.extend(force.get_estimated_param_names())
        return names

    @property
    def shape(self):
        """Return the broadcast batch shape of all force terms."""
        shapes = [f.shape for f in self.forces]
        if not shapes: return ()
        return jnp.broadcast_shapes(*shapes)

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("shape", format_shape(self.shape)),
                ("n_forces", str(len(self.forces))),
                ("forces", format_class_name_array(self.forces)),
                ("estimated_params", format_string_array(self.get_all_estimated_param_names(), quote=False)),
            ],
        )
