"""Heliocentric ``RTN`` non-gravitational force terms.

The models in this module express empirical non-gravitational accelerations in the heliocentric radial-transverse-normal frame and expose selected acceleration components and the cometary time lag as estimable parameters.
"""

from typing import Any, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, Float

from difforb.body.ephbody import EphemerisBody
from difforb.dynamics.force_model.base import ParametrizedForce
from difforb.dynamics.two_body import kepler_propagate


@jax.jit
def compute_rtn_distance_law_non_grav_acceleration(pos: Float[Array, "3"], vel: Float[Array, "3"],
                                                   pos_sun: Float[Array, "3"], vel_sun: Float[Array, "3"],
                                                   A1: Float[Array, ""], A2: Float[Array, ""],
                                                   A3: Float[Array, ""], alpha: Float[Array, ""],
                                                   r0: Float[Array, ""], m: Float[Array, ""],
                                                   n: Float[Array, ""], k: Float[Array, ""],
                                                   mu: Float[Array, ""],
                                                   DT: Optional[Float[Array, ""]] = None) -> Float[
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
    mu : Float[Array, ""]
        Solar gravitational parameter in ``au^3 / day^2`` used for the two-body time shift.
    DT : Optional[Float[Array, ""]]
        Lag in TDB days. ``None`` uses the current distance; a numerical value uses the distance propagated by ``-DT`` under solar gravity. The ``RTN`` directions always use the current state.

    Returns
    -------
    Float[Array, "3"]
        Non-gravitational acceleration in ``au / day^2``.
    """
    r = pos - pos_sun
    r_dist = jnp.linalg.norm(r)
    v = vel - vel_sun

    r_r0 = r_dist / r0
    if DT is not None:
        shifted_pos, _ = kepler_propagate(r[None, :], v[None, :], -DT[None], mu)
        r_r0 = jnp.linalg.norm(shifted_pos[0]) / r0
    g = alpha * (r_r0 ** (-m)) * ((1. + r_r0 ** n) ** (-k))

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
    estimated_indices: Array = eqx.field(metadata={"batch_shared": True})
    alpha: Array
    r0: Array
    m: Array
    n: Array
    k: Array
    estimated_param_names: tuple = eqx.field(static=True)
    param_prefix: str = eqx.field(static=True)
    sun: EphemerisBody = eqx.field(metadata={"batch_shared": True})

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
        self._initialize_params(sun, estimated_params, {'A1': A1, 'A2': A2, 'A3': A3},
                                alpha, r0, m, n, k, param_prefix)

    def _initialize_params(self, sun, estimated_params, params, alpha, r0, m, n, k, param_prefix):
        """Initialize the parameter vector, distance law, and solar ephemeris."""
        estimated_params = map(str.upper, estimated_params)
        estimated_params = tuple(sorted(list(set(estimated_params))))
        mapping = {name: i for i, name in enumerate(params)}
        invalid_names = tuple(name for name in estimated_params if name not in mapping)
        if invalid_names:
            raise ValueError(f"Unsupported non-gravitational parameters: {invalid_names!r}.")
        self.estimated_indices = jnp.array([mapping[t] for t in estimated_params], dtype=jnp.int32)
        self.param_prefix = str(param_prefix).strip() or "NG"
        self.estimated_param_names = tuple(f"{self.param_prefix}_{name}" for name in estimated_params)
        values = (*params.values(), alpha, r0, m, n, k)
        arrays = jnp.broadcast_arrays(*[jnp.asarray(value) for value in values])
        self.params = jnp.stack(arrays[:len(params)], axis=-1)
        self.alpha, self.r0, self.m, self.n, self.k = arrays[len(params):]
        self.sun = sun

    @property
    def n_estimated_params(self) -> int:
        """Return the number of estimated force parameters."""
        return len(self.estimated_indices)

    def get_estimated_params(self) -> Float[Array, "N_estimated"]:
        """Return the selected non-gravitational parameters.

        Returns
        -------
        Float[Array, "N_estimated"]
            Acceleration components in ``au / day^2`` and cometary ``DT`` in days, ordered by :meth:`get_estimated_param_names`.
        """
        return self.params[..., self.estimated_indices]

    def update_estimated_params(self, new_params: Float[Array, "N_estimated"]) -> 'RTNDistanceLawNonGravEffect':
        """Return a copy with new estimated force parameters.

        Parameters
        ----------
        new_params : Float[Array, "N_estimated"]
            New acceleration components in ``au / day^2`` and cometary ``DT`` in days, ordered by :meth:`get_estimated_param_names`.

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
                                                              self.r0, self.m, self.n, self.k, self.sun.gm)

    @property
    def shape(self):
        """Return the batch shape of the parameter arrays."""
        return self.params.shape[:-1]


class CometOutgassingEffect(RTNDistanceLawNonGravEffect):
    """Marsden cometary outgassing with a time-shifted distance law.

    The acceleration is ``g(r(t - DT)) * (A1*R + A2*T + A3*N)``. The shifted distance is evaluated by two-body propagation of the current heliocentric state under the Sun's gravity. The ``RTN`` directions use the current state. Positive ``DT`` shifts the activity peak after perihelion; ``DT=0`` gives the symmetric model.
    """

    def __init__(self, sun: EphemerisBody, estimated_params: Tuple[str, ...] = ('A1', 'A2', 'A3'),
                 A1: Float[ArrayLike, "..."] = 0.,
                 A2: Float[ArrayLike, "..."] = 0., A3: Float[ArrayLike, "..."] = 0., DT: Float[ArrayLike, "..."] = 0.,
                 alpha: Float[ArrayLike, "..."] = 0.1112620426,
                 r0: Float[ArrayLike, "..."] = 2.808, m: Float[ArrayLike, ""] = 2.15,
                 n: Float[ArrayLike, "..."] = 5.093,
                 k: Float[ArrayLike, "..."] = 4.6142,
                 param_prefix: str = "Outgassing"):
        """Initialize the comet-outgassing effect.

        Parameters
        ----------
        sun : EphemerisBody
            Sun ephemeris and gravitational parameter used for heliocentric states and the two-body time shift.
        estimated_params : tuple[str, ...], default=('A1', 'A2', 'A3')
            Parameters to estimate. Any subset of ``A1``, ``A2``, ``A3``, and ``DT`` may be selected; unlisted parameters remain fixed.
        A1, A2, A3 : Float[ArrayLike, "..."], default=0
            Radial, transverse, and normal acceleration components in ``au / day^2``.
        alpha, r0, m, n, k : Float[ArrayLike, "..."]
            Fixed parameters of the distance law ``g(r)``; ``r0`` is in ``au``.
        param_prefix : str, default="Outgassing"
            Prefix used in estimated-parameter names.
        DT : Float[ArrayLike, "..."], default=0
            Lag in ``TDB`` days. This is the initial value when ``DT`` is estimated and the fixed value otherwise.
        """
        self._initialize_params(sun, estimated_params, {'A1': A1, 'A2': A2, 'A3': A3, 'DT': DT},
                                alpha, r0, m, n, k, param_prefix)

    def __call__(self, tdb_jd1: Float[Array, ""], tdb_jd2: Float[Array, ""], state: Tuple[Float[Array, "3"], Float[Array, "3"]],
                 args: Any = None) -> Float[Array, "3"]:
        """Evaluate the time-shifted outgassing acceleration in ``au / day^2``."""
        A1, A2, A3, DT = self.params[..., 0], self.params[..., 1], self.params[..., 2], self.params[..., 3]
        sun_pos, sun_vel = self.sun._bcrs_pv_jd(tdb_jd1, tdb_jd2)
        pos, vel = state
        return compute_rtn_distance_law_non_grav_acceleration(pos, vel, sun_pos, sun_vel, A1, A2, A3, self.alpha,
                                                              self.r0, self.m, self.n, self.k, self.sun.gm, DT)


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
