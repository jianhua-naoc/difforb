"""Gravity force terms for point-mass perturbations and relativistic corrections.

This module contains Newtonian point-mass gravity and the built-in point-mass ``PPN`` correction. All terms operate on ``BCRS`` Cartesian states and return accelerations in ``au / day^2``.
"""

from typing import Any, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from difforb.body.ephbody import EphemerisBody, EphemerisBodyBatch
from difforb.core.constants import INV_C2
from difforb.dynamics.force_model.base import Force


@jax.jit
def compute_newtonian_acceleration(r_i: Float[Array, "3"], r_others: Float[Array, "N-1 3"],
                                   mu_others: Float[Array, "N-1"]) -> Float[Array, "3"]:
    """Compute the Newtonian point-mass acceleration.

    Parameters
    ----------
    r_i : Float[Array, "3"]
        Position of the integrated body in ``au``.
    r_others : Float[Array, "N-1 3"]
        Positions of the perturbing bodies in ``au``.
    mu_others : Float[Array, "N-1"]
        Gravitational parameters of the perturbing bodies in ``au^3 / day^2``.

    Returns
    -------
    Float[Array, "3"]
        Total Newtonian acceleration in ``au / day^2``.
    """
    r_ij = r_others - r_i
    r2_ij = jnp.einsum("ij,ij->i", r_ij, r_ij)
    r_ij_inv = jax.lax.rsqrt(r2_ij)
    r_ij3_inv = r_ij_inv * r_ij_inv * r_ij_inv
    return (mu_others * r_ij3_inv) @ r_ij


class NewtonianGravity(Force):
    """Point-mass gravity from a fixed list of ephemeris bodies."""
    bodies: tuple | EphemerisBodyBatch = eqx.field(
        metadata={"batch_shared": True},
    )
    gms: Float[Array, "N"]

    def __init__(self, bodies: List[EphemerisBody]):
        """Initialize the Newtonian gravity model.

        Parameters
        ----------
        bodies : list[EphemerisBody]
            Perturbing bodies.
        """
        self.gms = jnp.asarray([body.gm for body in bodies])
        self.bodies = (
            EphemerisBodyBatch(bodies)
            if all(isinstance(body, EphemerisBody) for body in bodies)
            else tuple(bodies)
        )

    def __call__(self, tdb_jd1: Float[Array, ""], tdb_jd2: Float[Array, ""], state: Tuple[Float[Array, "3"], Float[Array, "3"]],
                 args) -> Float[Array, "3"]:
        """Evaluate the Newtonian acceleration at one ``TDB`` epoch.

        Parameters
        ----------
        tdb_jd1, tdb_jd2 : Float[Array, ""]
            Two parts of the ``TDB`` Julian Date.
        state : tuple[Float[Array, "3"], Float[Array, "3"]]
            Integrated body state ``(pos, vel)`` in ``BCRS``. Position is in ``au``.
        args : Any
            Extra propagator data. This force does not use it.

        Returns
        -------
        Float[Array, "3"]
            Acceleration in ``au / day^2``.
        """
        return self._evaluate_prepared(state, self._prepare(tdb_jd1, tdb_jd2))

    def _prepare(self, tdb_jd1, tdb_jd2):
        """Return perturbing-body positions in ``BCRS`` at one ``TDB`` epoch, in ``au``."""
        if isinstance(self.bodies, EphemerisBodyBatch):
            return self.bodies.evaluate(tdb_jd1, tdb_jd2)
        return jnp.stack([
            body._bcrs_pos_jd(tdb_jd1, tdb_jd2) for body in self.bodies
        ])

    def _evaluate_prepared(self, state, pos_others):
        return compute_newtonian_acceleration(state[0], pos_others, self.gms)

    @property
    def shape(self):
        """Return the batch shape."""
        return self.gms.shape[:-1]


@jax.jit
def compute_planetary_potentials(r_others: Float[Array, "N 3"], mu_others: Float[Array, "N"]) -> Float[Array, "N"]:
    """Compute the potentials between perturbing bodies used by the ``PPN`` model.

    Parameters
    ----------
    r_others : Float[Array, "N 3"]
        Positions of the perturbing bodies in ``au``.
    mu_others : Float[Array, "N"]
        Gravitational parameters of the perturbing bodies in ``au^3 / day^2``.

    Returns
    -------
    Float[Array, "N"]
        Background potentials ``Phi_j = sum_{k != j} (mu_k / r_jk)`` in ``au^2 / day^2``.
    """
    # Step 1: Build the pairwise relative-position matrix.
    r_diff = r_others[:, None, :] - r_others[None, :, :]

    # Step 2: Build the pairwise squared-distance matrix.
    r2_jk = jnp.einsum('ijk,ijk->ij', r_diff, r_diff)

    # Step 3: Compute inverse distances and mask the diagonal.
    inv_r_jk = jnp.where(r2_jk > 0, jax.lax.rsqrt(r2_jk), 0.0)

    # Step 4: Sum the weighted background potentials.
    phi_background = inv_r_jk @ mu_others

    return phi_background


@jax.jit
def compute_ppn_acceleration(
        r_i: Float[Array, "3"],
        v_i: Float[Array, "3"],
        r_others: Float[Array, "N-1 3"],
        v_others: Float[Array, "N-1 3"],
        acc_others: Float[Array, "N-1 3"],
        mu_others: Float[Array, "N-1"],
        phi_others: Float[Array, "N-1"]
) -> Float[Array, "3"]:
    """Compute the ``PPN`` acceleration.

    Parameters
    ----------
    r_i, v_i : Float[Array, "3"]
        Position and velocity of the integrated body in ``BCRS``.
    r_others, v_others, acc_others : Float[Array, "N-1 3"]
        Positions, velocities, and accelerations of the perturbing bodies in ``BCRS``.
    mu_others : Float[Array, "N-1"]
        Gravitational parameters of the perturbing bodies in ``au^3 / day^2``.
    phi_others : Float[Array, "N-1"]
        Background potentials for the perturbing bodies.

    Returns
    -------
    Float[Array, "3"]
        ``PPN`` acceleration in ``au / day^2``.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Sec. 8.3.1.
    """
    # =================================================================
    # 1. Pre-compute the relative state vectors.
    # =================================================================
    r_ij = r_others - r_i
    v_ij = v_others - v_i
    r2_ij = jnp.einsum('ij,ij->i', r_ij, r_ij)
    r_ij_inv = jax.lax.rsqrt(r2_ij)
    r_ij_inv2 = r_ij_inv * r_ij_inv
    r_ij_inv3 = r_ij_inv2 * r_ij_inv

    # =================================================================
    # 2. Build term B from the potentials between perturbing bodies.
    # =================================================================
    term_B_per_j = -INV_C2 * phi_others

    # =================================================================
    # 3. Build the scalar terms (C, D, E, F, G).
    # =================================================================

    term_A_sum = jnp.dot(mu_others, r_ij_inv)
    term_A_const = 1.0 - 4.0 * INV_C2 * term_A_sum
    v_i2 = jnp.dot(v_i, v_i)
    v_j2 = jnp.einsum('ij,ij->i', v_others, v_others)

    vi_dot_vj = v_others @ v_i
    r_ij_dot_vj = jnp.einsum('ij,ij->i', r_ij, v_others)
    r_ij_dot_a_j = jnp.einsum('ij,ij->i', r_ij, acc_others)

    term_C = v_i2 * INV_C2
    term_D = 2.0 * INV_C2 * v_j2
    term_E = -4.0 * INV_C2 * vi_dot_vj

    ratio_rv = r_ij_dot_vj * r_ij_inv
    term_F = -1.5 * INV_C2 * (ratio_rv * ratio_rv)
    term_G = 0.5 * INV_C2 * r_ij_dot_a_j

    bracket = term_A_const + term_B_per_j + term_C + term_D + term_E + term_F + term_G

    # =================================================================
    # 4. Assemble: linear combinations of vectors.
    # =================================================================

    coeff_r = mu_others * bracket * r_ij_inv3
    r_ij_dot_vi = r_ij @ v_i
    dot_product = -4.0 * r_ij_dot_vi + 3.0 * r_ij_dot_vj
    coeff_v = -(INV_C2 * mu_others * r_ij_inv3 * dot_product)
    coeff_a = 3.5 * INV_C2 * mu_others * r_ij_inv
    acc = (coeff_r @ r_ij) + (coeff_v @ v_ij) + (coeff_a @ acc_others)
    return acc


class PPNGravity(Force):
    """Post-Newtonian gravity from a fixed list of ephemeris bodies."""
    bodies: tuple | EphemerisBodyBatch = eqx.field(
        metadata={"batch_shared": True},
    )
    gms: Float[Array, "N"]

    def __init__(self, bodies: List[EphemerisBody]):
        """Initialize the parametrized post-Newtonian gravity model.

        Parameters
        ----------
        bodies : list[EphemerisBody]
            Perturbing bodies.

        References
        ----------
        1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Sec. 8.3.1.
        """
        self.gms = jnp.asarray([body.gm for body in bodies])
        self.bodies = (
            EphemerisBodyBatch(bodies)
            if all(isinstance(body, EphemerisBody) for body in bodies)
            else tuple(bodies)
        )

    def __call__(self, tdb_jd1: Float[Array, ""], tdb_jd2: Float[Array, ""], state: Tuple[Float[Array, "3"], Float[Array, "3"]],
                 args) -> Float[Array, "3"]:
        """Evaluate the ``PPN`` acceleration at one ``TDB`` epoch.

        Parameters
        ----------
        tdb_jd1, tdb_jd2 : Float[Array, ""]
            Two parts of the ``TDB`` Julian Date.
        state : tuple[Float[Array, "3"], Float[Array, "3"]]
            Integrated body state ``(pos, vel)`` in ``BCRS``. Position is in ``au`` and velocity is in ``au / day``.
        args : Any
            Extra propagator data. This force does not use it.

        Returns
        -------
        Float[Array, "3"]
            Acceleration in ``au / day^2``.
        """
        return self._evaluate_prepared(state, self._prepare(tdb_jd1, tdb_jd2))

    def _prepare(self, tdb_jd1, tdb_jd2):
        """Return background ``BCRS`` states and potentials at one ``TDB`` epoch, independently of the integrated body."""
        if isinstance(self.bodies, EphemerisBodyBatch):
            pos_others, vel_others, acc_others = self.bodies.evaluate(
                tdb_jd1, tdb_jd2, derivatives=True,
            )
        else:
            pva = [
                body._bcrs_pva_jd(tdb_jd1, tdb_jd2) for body in self.bodies
            ]
            pos_others = jnp.stack([values[0] for values in pva])
            vel_others = jnp.stack([values[1] for values in pva])
            acc_others = jnp.stack([values[2] for values in pva])
        phi_planetary = compute_planetary_potentials(pos_others, self.gms)
        return pos_others, vel_others, acc_others, phi_planetary

    def _evaluate_prepared(self, state, background):
        pos_others, vel_others, acc_others, phi_planetary = background
        return compute_ppn_acceleration(*state, pos_others, vel_others, acc_others, self.gms, phi_planetary)

    @property
    def shape(self):
        """Return the batch shape."""
        return self.gms.shape[:-1]
