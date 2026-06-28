"""Heliocentric ``RTN`` non-gravitational force terms.

The models in this module express empirical non-gravitational accelerations in the heliocentric radial-transverse-normal frame and expose selected acceleration components as estimable parameters.
"""

from typing import Any, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, Float

from difforb.body.ephbody import EphemerisBody
from difforb.dynamics.force_model.base import ParametrizedForce


@jax.jit
def compute_rtn_distance_law_non_grav_acceleration(pos: Float[Array, "3"], vel: Float[Array, "3"],
                                                   pos_sun: Float[Array, "3"], vel_sun: Float[Array, "3"],
                                                   A1: Float[Array, ""], A2: Float[Array, ""],
                                                   A3: Float[Array, ""], alpha: Float[Array, ""],
                                                   r0: Float[Array, ""], m: Float[Array, ""],
                                                   n: Float[Array, ""], k: Float[Array, ""]) -> Float[
    Array, "3"]:
    """Compute one ``RTN`` non-gravitational acceleration scaled by ``g(r)``.

    Parameters
    ----------
    pos, vel : Float[Array, "3"]
        State of the integrated body in ``BCRS``.
    pos_sun, vel_sun : Float[Array, "3"]
        State of the Sun in ``BCRS``.
    A1, A2, A3 : Float[Array, ""]
        Radial, transverse, and normal non-gravitational parameters in ``au / day^2``.
    alpha, r0, m, n, k : Float[Array, ""]
        Parameters of the radial distance law ``g(r) = alpha * (r / r0)^(-m) * (1 + (r / r0)^n)^(-k)``.

    Returns
    -------
    Float[Array, "3"]
        Non-gravitational acceleration in ``au / day^2``.
    """
    r = pos - pos_sun
    r_dist = jnp.linalg.norm(r)
    v = vel - vel_sun

    # Distance law ``g(r)``.
    r_r0 = r_dist / r0
    g = alpha * (r_r0 ** (-m)) * ((1. + r_r0 ** n) ** (-k))

    # Build the ``RTN`` unit vectors.
    r_uv = r / r_dist
    h = jnp.cross(r, v)
    h_dist = jnp.linalg.norm(h)
    n_uv = h / h_dist
    t_uv = jnp.cross(n_uv, r_uv)

    acc = g * (A1 * r_uv + A2 * t_uv + A3 * n_uv)
    return acc


class RTNDistanceLawNonGravEffect(ParametrizedForce):
    """Non-gravitational acceleration in the heliocentric ``RTN`` frame."""
    params: Array
    estimated_indices: Array
    alpha: Array
    r0: Array
    m: Array
    n: Array
    k: Array
    estimated_param_names: tuple = eqx.field(static=True)
    param_prefix: str = eqx.field(static=True)
    sun: EphemerisBody

    def __init__(self, sun: EphemerisBody, estimated_params: Tuple[str, ...] = ('A1', 'A2', 'A3'),
                 A1: Float[ArrayLike, "..."] = 0.,
                 A2: Float[ArrayLike, "..."] = 0., A3: Float[ArrayLike, "..."] = 0.,
                 alpha: Float[ArrayLike, "..."] = 0.1112620426,
                 r0: Float[ArrayLike, "..."] = 2.808, m: Float[ArrayLike, ""] = 2.15,
                 n: Float[ArrayLike, "..."] = 5.093,
                 k: Float[ArrayLike, "..."] = 4.6142,
                 param_prefix: str = "NG"):
        """Initialize one heliocentric ``RTN`` non-gravitational effect.

        Parameters
        ----------
        sun : EphemerisBody
            Sun ephemeris body used to build the heliocentric relative state.
        estimated_params : tuple[str, ...], default=('A1', 'A2', 'A3')
            Names of the estimated parameters.
        A1, A2, A3 : Float[ArrayLike, "..."], default=0
            Initial radial, transverse, and normal non-gravitational parameters in ``au / day^2``.
        alpha, r0, m, n, k : Float[ArrayLike, "..."]
            Parameters of the radial distance law ``g(r)``.
        param_prefix : str, default="NG"
            Prefix used when exposing estimated-parameter names through :class:`ForceModel`.
        """
        estimated_params = map(str.upper, estimated_params)
        estimated_params = tuple(sorted(list(set(estimated_params))))
        mapping = {'A1': 0, 'A2': 1, 'A3': 2}
        invalid_names = tuple(name for name in estimated_params if name not in mapping)
        if invalid_names:
            raise ValueError(f"Unsupported non-gravitational parameters: {invalid_names!r}.")
        self.estimated_indices = jnp.array([mapping[t] for t in estimated_params], dtype=jnp.int32)
        self.param_prefix = str(param_prefix).strip() or "NG"
        self.estimated_param_names = tuple(f"{self.param_prefix}_{name}" for name in estimated_params)
        A1, A2, A3, self.alpha, self.r0, self.m, self.n, self.k = jnp.broadcast_arrays(jnp.asarray(A1), jnp.asarray(
            A2), jnp.asarray(A3), jnp.asarray(alpha), jnp.asarray(r0), jnp.asarray(m), jnp.asarray(n), jnp.asarray(k))
        self.params = jnp.stack([A1, A2, A3], axis=-1)
        self.sun = sun

    @property
    def n_estimated_params(self) -> int:
        """Return the number of estimated acceleration parameters."""
        return len(self.estimated_indices)

    def get_estimated_params(self) -> Float[Array, "N_estimated"]:
        """Return the selected ``RTN`` acceleration parameters.

        Returns
        -------
        Float[Array, "N_estimated"]
            Estimated parameters in ``au / day^2``.
        """
        return self.params[..., self.estimated_indices]

    def get_estimated_param_scales(self) -> Float[Array, "N_estimated"]:
        """
        Return characteristic scales for estimated ``RTN`` acceleration parameters.

        The ``A1``, ``A2``, and ``A3`` parameters are accelerations in
        ``au / day^2`` before radial distance-law scaling. The generic
        ``RTN`` model uses one conservative scale for all three components;
        specialized subclasses may override this with narrower model-specific
        values.

        Returns
        -------
        Float[Array, "N_estimated"]
            Characteristic acceleration scales in ``au / day^2``.
        """
        param_scales = jnp.ones_like(self.params) * 1e-12
        return param_scales[..., self.estimated_indices]

    def update_estimated_params(self, new_params: Float[Array, "N_estimated"]) -> 'RTNDistanceLawNonGravEffect':
        """Return a copy with new estimated acceleration parameters.

        Parameters
        ----------
        new_params : Float[Array, "N_estimated"]
            New values for the selected ``RTN`` parameters, in ``au / day^2``.

        Returns
        -------
        RTNDistanceLawNonGravEffect
            Force with updated parameters.
        """
        updated_params = self.params.at[..., self.estimated_indices].set(jnp.asarray(new_params))
        return eqx.tree_at(lambda f: f.params, self, updated_params)

    def get_estimated_param_names(self) -> List[str]:
        """Return the names of the selected ``RTN`` parameters."""
        return list(self.estimated_param_names)

    def __call__(self, tdb_jd1: Float[Array, ""], tdb_jd2: Float[Array, ""], state: Tuple[Float[Array, "3"], Float[Array, "3"]],
                 args: Any = None) -> Float[
        Array, "3"]:
        """Evaluate the non-gravitational acceleration at one ``TDB`` epoch.

        Parameters
        ----------
        tdb_jd1, tdb_jd2 : Float[Array, ""]
            Two parts of the ``TDB`` Julian Date.
        state : tuple[Float[Array, "3"], Float[Array, "3"]]
            Integrated body state ``(pos, vel)`` in ``BCRS``. Position is in ``au`` and velocity is in ``au / day``.
        args : Any, optional
            Extra propagator data. This force does not use it.

        Returns
        -------
        Float[Array, "3"]
            Acceleration in ``au / day^2``.
        """
        A1, A2, A3 = self.params[..., 0], self.params[..., 1], self.params[..., 2]
        sun_pos, sun_vel = self.sun._bcrs_pv_jd(tdb_jd1, tdb_jd2)
        pos, vel = state
        return compute_rtn_distance_law_non_grav_acceleration(pos, vel, sun_pos, sun_vel, A1, A2, A3, self.alpha,
                                                              self.r0, self.m, self.n, self.k)

    @property
    def shape(self):
        """Return the batch shape of the parameter arrays."""
        return self.params.shape[:-1]


class CometOutgassingEffect(RTNDistanceLawNonGravEffect):
    """Symmetric cometary outgassing effect expressed in the heliocentric ``RTN`` frame."""

    def __init__(self, sun: EphemerisBody, estimated_params: Tuple[str, ...] = ('A1', 'A2', 'A3'),
                 A1: Float[ArrayLike, "..."] = 0.,
                 A2: Float[ArrayLike, "..."] = 0., A3: Float[ArrayLike, "..."] = 0.,
                 alpha: Float[ArrayLike, "..."] = 0.1112620426,
                 r0: Float[ArrayLike, "..."] = 2.808, m: Float[ArrayLike, ""] = 2.15,
                 n: Float[ArrayLike, "..."] = 5.093,
                 k: Float[ArrayLike, "..."] = 4.6142,
                 param_prefix: str = "Outgassing"):
        """Initialize the symmetric comet-outgassing effect."""
        super().__init__(
            sun=sun,
            estimated_params=estimated_params,
            A1=A1,
            A2=A2,
            A3=A3,
            alpha=alpha,
            r0=r0,
            m=m,
            n=n,
            k=k,
            param_prefix=param_prefix,
        )

    def get_estimated_param_scales(self) -> Float[Array, "N_estimated"]:
        """
        Return characteristic scales for estimated comet outgassing parameters.

        Returns
        -------
        Float[Array, "N_estimated"]
            Characteristic ``A1``, ``A2``, and ``A3`` scales in ``au / day^2``.
        """
        return jnp.ones_like(self.get_estimated_params()) * 1e-8


class EmpiricalYarkovskyEffect(RTNDistanceLawNonGravEffect):
    """Empirical Yarkovsky-like effect represented by one transverse ``RTN`` term."""

    def __init__(self, sun: EphemerisBody, estimated_params: Tuple[str, ...] = ('A2',),
                 A2: Float[ArrayLike, "..."] = 0.,
                 alpha: Float[ArrayLike, "..."] = 1.0,
                 r0: Float[ArrayLike, "..."] = 1.0, m: Float[ArrayLike, ""] = 2.0,
                 n: Float[ArrayLike, "..."] = 1.0,
                 k: Float[ArrayLike, "..."] = 0.0,
                 param_prefix: str = "Yarkovsky"):
        """Initialize the empirical Yarkovsky effect."""
        estimated_params = tuple(name for name in map(str.upper, estimated_params) if name == "A2")
        super().__init__(
            sun=sun,
            estimated_params=estimated_params,
            A1=0.0,
            A2=A2,
            A3=0.0,
            alpha=alpha,
            r0=r0,
            m=m,
            n=n,
            k=k,
            param_prefix=param_prefix,
        )

    def get_estimated_param_scales(self) -> Float[Array, "N_estimated"]:
        """
        Return the characteristic scale for the estimated transverse acceleration.

        Returns
        -------
        Float[Array, "N_estimated"]
            Characteristic ``A2`` scale in ``au / day^2``.
        """
        return jnp.ones_like(self.get_estimated_params()) * 1e-13


class EmpiricalRadiationPressure(RTNDistanceLawNonGravEffect):
    """Empirical solar-radiation-pressure-like effect represented by one radial ``RTN`` term."""

    def __init__(self, sun: EphemerisBody, estimated_params: Tuple[str, ...] = ('A1',),
                 A1: Float[ArrayLike, "..."] = 0.,
                 alpha: Float[ArrayLike, "..."] = 1.0,
                 r0: Float[ArrayLike, "..."] = 1.0, m: Float[ArrayLike, ""] = 2.0,
                 n: Float[ArrayLike, "..."] = 1.0,
                 k: Float[ArrayLike, "..."] = 0.0,
                 param_prefix: str = "RadiationPressure"):
        """Initialize the empirical radiation-pressure effect."""
        estimated_params = tuple(name for name in map(str.upper, estimated_params) if name == "A1")
        super().__init__(
            sun=sun,
            estimated_params=estimated_params,
            A1=A1,
            A2=0.0,
            A3=0.0,
            alpha=alpha,
            r0=r0,
            m=m,
            n=n,
            k=k,
            param_prefix=param_prefix,
        )

    def get_estimated_param_scales(self) -> Float[Array, "N_estimated"]:
        """
        Return the characteristic scale for the estimated radial acceleration.

        Returns
        -------
        Float[Array, "N_estimated"]
            Characteristic ``A1`` scale in ``au / day^2``.
        """
        return jnp.ones_like(self.get_estimated_params()) * 1e-12
