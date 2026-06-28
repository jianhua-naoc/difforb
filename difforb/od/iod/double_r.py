"""Double-r initial orbit determination for angle-only observation triplets.

This module implements the numerical Double-r iteration used by
``difforb.od.iod.solver``. The solver works on batched triplets of observer
positions, line-of-sight vectors, and split Julian dates, and returns the
estimated state at the middle observation epoch together with the final angular
residual norm.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float
from jax.typing import ArrayLike

from difforb.dynamics.two_body import kepler_propagate, lambert_solver

jax.config.update("jax_enable_x64", True)


class DoubleRIODResult(NamedTuple):
    """Batched Double-r solution at the middle observation epoch.

    Parameters
    ----------
    pos_t2 : Float[Array, "N 3"]
        Estimated Cartesian positions at the middle observation epoch, in
        canonical distance units.
    vel_t2 : Float[Array, "N 3"]
        Estimated Cartesian velocities at the middle observation epoch, in
        canonical distance-per-time units.
    epoch_tdb_jd1 : Float[Array, "N"]
        High-order split Julian-date term of the middle observation epoch in
        ``TDB``.
    epoch_tdb_jd2 : Float[Array, "N"]
        Low-order split Julian-date term of the middle observation epoch in
        ``TDB``.
    residual_norm : Float[Array, "N"]
        Final angular residual norm for each candidate, in radians.
    iter_num : int
        Number of performed Newton iterations.
    """

    pos_t2: Float[Array, "N 3"]
    vel_t2: Float[Array, "N 3"]
    epoch_tdb_jd1: Float[Array, "N"]
    epoch_tdb_jd2: Float[Array, "N"]
    residual_norm: Float[Array, "N"]
    iter_num: int


def double_r_iod(site_pos: Float[Array, "N 3 3"],
                 obs_los: Float[Array, "N 3 3"],
                 obs_jd1: Float[Array, "N 3"],
                 obs_jd2: Float[Array, "N 3"],
                 mu: Float[ArrayLike, ""],
                 init_rho: Float[Array, "... 2"] = jnp.array([1.0, 1.0]),
                 min_rho: float = 1e-4,
                 tol: float = 1e-6,
                 max_iter: int = 20) -> DoubleRIODResult:
    """Estimate candidate states from angle-only triplets by Double-r iteration.

    Parameters
    ----------
    site_pos : Float[Array, "N 3 3"]
        Observer positions at ``t1``, ``t2``, and ``t3`` for each candidate.
    obs_los : Float[Array, "N 3 3"]
        Line-of-sight unit vectors aligned with ``site_pos``.
    obs_jd1 : Float[Array, "N 3"]
        High-order split Julian-date term of the observation epochs in ``TDB``.
    obs_jd2 : Float[Array, "N 3"]
        Low-order split Julian-date term of the observation epochs in ``TDB``.
    mu : Float[ArrayLike, ""]
        Central-body gravitational parameter.
    init_rho : Float[Array, "... 2"], default=jnp.array([1.0, 1.0])
        Initial range guesses for the first and third observations, in ``au``.
    min_rho : float, default=1e-4
        Lower bound imposed on the iterated ranges, in ``au``.
    tol : float, default=1e-6
        Convergence threshold for the angular residual norm, in radians.
    max_iter : int, default=20
        Maximum number of Newton iterations.

    Returns
    -------
    DoubleRIODResult
        Estimated state at the middle observation epoch for each candidate.

    Notes
    -----
    The iteration solves for the first and third topocentric ranges. The
    middle-observation residual is represented in an orthonormal plane that is
    perpendicular to the observed line of sight at ``t2``.
    """

    # -------------------------------------------------------------------------
    # Step 1: Split the triplet geometry and epoch arrays
    # -------------------------------------------------------------------------
    site_pos_t1 = site_pos[:, 0]
    site_pos_t2 = site_pos[:, 1]
    site_pos_t3 = site_pos[:, 2]

    los_unit_t1 = obs_los[:, 0]
    los_unit_t2 = obs_los[:, 1]
    los_unit_t3 = obs_los[:, 2]

    t1_tdb_jd1 = obs_jd1[:, 0]
    t2_tdb_jd1 = obs_jd1[:, 1]
    t3_tdb_jd1 = obs_jd1[:, 2]
    t1_tdb_jd2 = obs_jd2[:, 0]
    t2_tdb_jd2 = obs_jd2[:, 1]
    t3_tdb_jd2 = obs_jd2[:, 2]

    dt_t1_to_t3 = (t3_tdb_jd1 - t1_tdb_jd1) + (t3_tdb_jd2 - t1_tdb_jd2)
    dt_t1_to_t2 = (t2_tdb_jd1 - t1_tdb_jd1) + (t2_tdb_jd2 - t1_tdb_jd2)
    rho13_init = jnp.broadcast_to(init_rho, (site_pos_t1.shape[0], 2))

    # -------------------------------------------------------------------------
    # Step 2: Build the residual plane around the observed ``t2`` LOS
    # -------------------------------------------------------------------------
    z_axis = jnp.zeros_like(los_unit_t2).at[:, 2].set(1.0)
    x_axis = jnp.zeros_like(los_unit_t2).at[:, 0].set(1.0)

    use_z_axis = jnp.abs(jnp.sum(los_unit_t2 * z_axis, axis=-1)) < 0.99
    reference_axis = jnp.where(use_z_axis[:, None], z_axis, x_axis)

    los_plane_u_hat = jnp.cross(los_unit_t2, reference_axis)
    los_plane_u_hat = los_plane_u_hat / jnp.linalg.norm(los_plane_u_hat, axis=-1, keepdims=True)
    los_plane_v_hat = jnp.cross(los_unit_t2, los_plane_u_hat)

    def residual_func(rho13: Float[Array, "N 2"]) -> Float[Array, "N 2"]:
        rho_t1 = rho13[:, 0][:, None]
        rho_t3 = rho13[:, 1][:, None]

        obj_pos_t1 = site_pos_t1 + rho_t1 * los_unit_t1
        obj_pos_t3 = site_pos_t3 + rho_t3 * los_unit_t3

        obj_vel_t1, _ = lambert_solver(obj_pos_t1, obj_pos_t3, dt_t1_to_t3, mu=mu)
        obj_pos_t2_est, _ = kepler_propagate(obj_pos_t1, obj_vel_t1, dt_t1_to_t2, mu)

        site2obj_pos_t2_est = obj_pos_t2_est - site_pos_t2
        site2obj_dist_t2_est = jnp.linalg.norm(site2obj_pos_t2_est, axis=-1, keepdims=True)
        los_unit_t2_est = site2obj_pos_t2_est / site2obj_dist_t2_est

        residual_u = jnp.sum(los_unit_t2_est * los_plane_u_hat, axis=-1)
        residual_v = jnp.sum(los_unit_t2_est * los_plane_v_hat, axis=-1)
        return jnp.stack([residual_u, residual_v], axis=-1)

    def residual_jacobian(rho13: Float[Array, "N 2"]) -> Float[Array, "N 2 2"]:
        zeros = jnp.zeros_like(rho13[..., 0])
        ones = jnp.ones_like(rho13[..., 0])
        basis_t1 = jnp.stack([ones, zeros], axis=-1)
        basis_t3 = jnp.stack([zeros, ones], axis=-1)

        _, jacobian_col_t1 = jax.jvp(residual_func, (rho13,), (basis_t1,))
        _, jacobian_col_t3 = jax.jvp(residual_func, (rho13,), (basis_t3,))
        return jnp.stack([jacobian_col_t1, jacobian_col_t3], axis=-1)

    # -------------------------------------------------------------------------
    # Step 3: Iterate on the first and third ranges
    # -------------------------------------------------------------------------
    def iteration_body(carry):
        iteration_idx, rho13, _ = carry
        residual_uv = residual_func(rho13)
        residual_norm = jnp.linalg.norm(residual_uv, axis=-1)
        needs_update = residual_norm > tol

        jacobian = residual_jacobian(rho13)
        delta_rho13 = jnp.linalg.solve(jacobian, residual_uv[..., None]).squeeze(-1)
        delta_rho13 = jnp.where(needs_update[:, None], delta_rho13, 0.0)

        next_rho13 = rho13 - delta_rho13
        next_rho13 = jnp.maximum(next_rho13, min_rho)
        return iteration_idx + 1, next_rho13, residual_norm

    def iteration_cond(carry):
        iteration_idx, _, residual_norm = carry
        return jnp.logical_and(jnp.max(residual_norm) > tol, iteration_idx < max_iter)

    init_residual_norm = jnp.full(site_pos_t1.shape[0], jnp.inf)
    iter_num, rho13_final, residual_norm_final = jax.lax.while_loop(
        iteration_cond,
        iteration_body,
        (0, rho13_init, init_residual_norm),
    )

    # -------------------------------------------------------------------------
    # Step 4: Recover the state at the middle observation epoch
    # -------------------------------------------------------------------------
    obj_pos_t1_final = site_pos_t1 + rho13_final[:, 0][:, None] * los_unit_t1
    obj_pos_t3_final = site_pos_t3 + rho13_final[:, 1][:, None] * los_unit_t3
    obj_vel_t1_final, _ = lambert_solver(obj_pos_t1_final, obj_pos_t3_final, dt_t1_to_t3, mu=mu)
    obj_pos_t2, obj_vel_t2 = kepler_propagate(obj_pos_t1_final, obj_vel_t1_final, dt_t1_to_t2, mu)

    return DoubleRIODResult(
        pos_t2=obj_pos_t2,
        vel_t2=obj_vel_t2,
        epoch_tdb_jd1=t2_tdb_jd1,
        epoch_tdb_jd2=t2_tdb_jd2,
        residual_norm=residual_norm_final,
        iter_num=iter_num,
    )
