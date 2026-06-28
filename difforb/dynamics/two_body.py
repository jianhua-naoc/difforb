"""Two-body propagation and Lambert solving.

This module implements universal-variable solvers for two standard orbital problems: Kepler propagation from an initial state and Lambert transfer between two positions.

Unless noted otherwise, positions are in ``au``, velocities are in ``au / day``, times are in days, and the gravitational parameter is in ``au^3 / day^2``.
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array
from jaxtyping import Float
import equinox as eqx

jax.config.update("jax_enable_x64", True)


def C2C3(psi: Float[Array, "N"]) -> Tuple[Float[Array, "N"], Float[Array, "N"]]:
    """Compute the Stumpff functions ``C2`` and ``C3``.

    Parameters
    ----------
    psi : Float[Array, "N"]
        Universal-variable argument.

    Returns
    -------
    tuple[Float[Array, "N"], Float[Array, "N"]]
        Tuple ``(C2, C3)`` evaluated at ``psi``.

    Notes
    -----
    The implementation switches between elliptic, hyperbolic, and series forms to stay stable near ``psi = 0``.

    References
    ----------
    1. Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 1.
    """
    # psi > 1e-6
    branch1_mask = psi > 1e-6
    safe_psi_branch1 = jnp.where(branch1_mask, psi, 1.)
    sqrt_psi = jnp.sqrt(safe_psi_branch1)
    c2_branch1 = (1. - jnp.cos(sqrt_psi)) / safe_psi_branch1
    c3_branch1 = (sqrt_psi - jnp.sin(sqrt_psi)) / (safe_psi_branch1 * sqrt_psi)
    # psi < -1e-6
    branch2_mask = psi < -1e-6
    safe_psi_branch2 = jnp.where(branch2_mask, psi, -1.)
    sqrt_neg_psi = jnp.sqrt(-safe_psi_branch2)
    c2_branch2 = (1. - jnp.cosh(sqrt_neg_psi)) / safe_psi_branch2
    c3_branch2 = (jnp.sinh(sqrt_neg_psi) - sqrt_neg_psi) / (-safe_psi_branch2 * sqrt_neg_psi)
    # other
    c2_branch3 = 1.0 / 2.0 - psi / 24.0 + psi ** 2 / 720.0 - psi ** 3 / 40320.
    c3_branch3 = 1.0 / 6.0 - psi / 120.0 + psi ** 2 / 5040.0 - psi ** 3 / 362880.

    c2 = jnp.where(branch1_mask, c2_branch1, jnp.where(branch2_mask, c2_branch2, c2_branch3))
    c3 = jnp.where(branch1_mask, c3_branch1, jnp.where(branch2_mask, c3_branch2, c3_branch3))

    return c2, c3


@eqx.filter_jit
def kepler_propagate(init_pos: Float[Array, "N 3"], init_vel: Float[Array, "N 3"], dt: Float[Array, "N"],
                     mu: Float[ArrayLike, ""],
                     tol: float = 1e-15, max_iter=20) -> \
        Tuple[
            Float[Array, "N 3"], Float[Array, "N 3"]]:
    """Propagate Cartesian states with the two-body Kepler solution.

    Parameters
    ----------
    init_pos : Float[Array, "N 3"]
        Initial position vectors in ``au``.
    init_vel : Float[Array, "N 3"]
        Initial velocity vectors in ``au / day``.
    dt : Float[Array, "N"]
        Propagation intervals in days.
    mu : Float[ArrayLike, ""]
        Central-body gravitational parameter in ``au^3 / day^2``.
    tol : float, default=1e-15
        Convergence tolerance for the universal-variable Newton iteration.
    max_iter : int, default=20
        Maximum number of the universal-variable Newton iteration steps.

    Returns
    -------
    tuple[Float[Array, "N 3"], Float[Array, "N 3"]]
        Final position and velocity vectors in ``au`` and ``au / day``.

    Notes
    -----
    The solver uses the universal-variable form, so it supports elliptic, parabolic, and hyperbolic cases in one routine.

    References
    ----------
    1. Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 8.
    """
    init_pos_mag = jnp.linalg.norm(init_pos, axis=-1)
    init_vel_mag2 = jnp.sum(init_vel * init_vel, axis=-1)
    init_pos_vel_dot = jnp.sum(init_pos * init_vel, axis=-1)
    alpha = 2. * jnp.reciprocal(init_pos_mag) - init_vel_mag2 * jnp.reciprocal(mu)
    sqrt_mu = jnp.sqrt(mu)

    # ==============================================================
    # 1. Build the initial guess for the universal variable ``chi``.
    # ==============================================================

    # Circle or ellipse.
    mask_cir_or_ell = alpha > 1e-6
    chi_cir_or_ell = sqrt_mu * dt * alpha

    # Hyperbola.
    mask_hyp = alpha < -1e-6
    safe_alpha_hyp = jnp.where(mask_hyp, alpha, -1.)
    a_hyp = 1. * jnp.reciprocal(safe_alpha_hyp)
    sqrt_neg_a = jnp.sqrt(-a_hyp)
    sign_dt = jnp.sign(dt)
    scal1 = init_pos_vel_dot + sign_dt * sqrt_neg_a * sqrt_mu * (1. - safe_alpha_hyp * init_pos_mag)
    safe_scal1 = jnp.where(jnp.abs(scal1) < 1e-20, 1., scal1)
    scal2 = (-2. * mu * safe_alpha_hyp * dt) * jnp.reciprocal(safe_scal1)
    safe_scal2 = jnp.where(jnp.abs(scal2) <= 0., 1., scal2)
    chi_hyp = sign_dt * sqrt_neg_a * jnp.log(safe_scal2)

    # Parabola.
    h = jnp.cross(init_pos, init_vel, axis=-1)
    h_mag2 = jnp.sum(h * h, axis=-1)
    p = h_mag2 / mu
    sqrt_p = jnp.sqrt(p)
    s = jnp.arctan(p * sqrt_p / (3. * sqrt_mu * dt)) / 2.
    w = jnp.arctan(jnp.power(jnp.tan(s), 1. / 3.))
    chi_para = sqrt_p * 2. * (1. / jnp.tan(2. * w))

    init_chi = jnp.where(mask_cir_or_ell, chi_cir_or_ell, jnp.where(mask_hyp, chi_hyp, chi_para))

    # ==========================================
    # 2. Refine ``chi`` with Newton iteration.
    # ==========================================
    scal = init_pos_vel_dot / sqrt_mu

    def body_func(carry):
        i, cur_chi, *_ = carry
        chi2 = cur_chi * cur_chi
        chi3 = chi2 * cur_chi
        psi = chi2 * alpha
        c2, c3 = C2C3(psi)
        scal2 = chi2 * c2
        scal3 = cur_chi * (1. - psi * c3)
        r = scal2 + scal * scal3 + init_pos_mag * (1. - psi * c2)
        new_chi = cur_chi + (sqrt_mu * dt - chi3 * c3 - scal * scal2 - init_pos_mag * scal3) * jnp.reciprocal(r)
        return i + 1, new_chi, cur_chi, psi, c2, c3, r

    def cond_func(carry):
        i, cur_chi, prev_chi, *_ = carry
        err = jnp.max(jnp.abs(cur_chi - prev_chi), initial=0.)
        return jnp.logical_and(err > tol, i < max_iter)

    n_batch = init_chi.shape[0]
    init_psi, init_c2, init_c3, init_r = jnp.zeros(n_batch), jnp.zeros(n_batch), jnp.zeros(n_batch), jnp.zeros(n_batch)
    init_carry = (0, init_chi, init_chi + 1., init_psi, init_c2, init_c3, init_r)
    _, final_chi, _, psi, c2, c3, r = jax.lax.while_loop(cond_func, body_func, init_carry)

    # ==========================================
    # 3. Build the Lagrange coefficients.
    # ==========================================
    chi2 = final_chi * final_chi
    chi3 = chi2 * final_chi
    f = (1. - chi2 / init_pos_mag * c2).reshape(-1, 1)
    g = (dt - chi3 / sqrt_mu * c3).reshape(-1, 1)
    f_dot = (sqrt_mu / (r * init_pos_mag) * final_chi * (psi * c3 - 1.)).reshape(-1, 1)
    g_dot = (1. - chi2 / r * c2).reshape(-1, 1)

    # ==========================================
    # 4. Build the final Cartesian state.
    # ==========================================
    final_pos = f * init_pos + g * init_vel
    final_vel = f_dot * init_pos + g_dot * init_vel
    return final_pos, final_vel


@eqx.filter_jit
def lambert_solver(init_pos: Float[Array, "N 3"], final_pos: Float[Array, "N 3"], dt: Float[Array, "N"],
                   mu: Float[ArrayLike, ""],
                   tol: float = 1e-8, max_iter=50) -> Tuple[Float[Array, "3"], Float[Array, "3"]]:
    """Solve Lambert's problem with universal variables.

    Parameters
    ----------
    init_pos : Float[Array, "N 3"]
        Initial position vectors in ``au``.
    final_pos : Float[Array, "N 3"]
        Final position vectors in ``au``.
    dt : Float[Array, "N"]
        Transfer times in days.
    mu : Float[ArrayLike, ""]
        Central-body gravitational parameter in ``au^3 / day^2``.
    tol : float, default=1e-8
        Convergence tolerance for the universal-variable iteration.
    max_iter : int, default=50
        Maximum number of the universal-variable iteration steps.

    Returns
    -------
    tuple[Float[Array, "3"], Float[Array, "3"]]
        Initial and final transfer velocities in ``au / day`` with shape ``(N, 3)``.

    Notes
    -----
    This routine solves the short-way transfer with the transfer-direction flag ``tm = 1``.

    References
    ----------
    1. Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 58.
    """
    init_pos_mag = jnp.linalg.norm(init_pos, axis=-1)
    final_pos_mag = jnp.linalg.norm(final_pos, axis=-1)
    sqrt_mu = jnp.sqrt(mu)

    cos_dnu = jnp.sum(init_pos * final_pos, axis=-1) / (init_pos_mag * final_pos_mag)
    # sin_dnu = tm * jnp.sqrt(1. - cos_dnu * cos_dnu)
    tm = 1
    A = tm * jnp.sqrt(init_pos_mag * final_pos_mag * (1. + cos_dnu))

    # ===========================================
    # 1. Build the initial bracket for ``psi``.
    # ===========================================
    n_batch = init_pos.shape[0]
    zero = jnp.zeros(n_batch)
    init_psi = zero
    init_c2, init_c3 = 0.5 + zero, 1. / 6. + zero
    init_psi_up, init_psi_low = 4. * jnp.pi * jnp.pi + zero, -4. * jnp.pi + zero
    init_dt_n = dt + 1.
    init_carray = (0, init_dt_n, init_psi, init_psi_low, init_psi_up, init_c2, init_c3)

    # ===================================================
    # 2. Refine ``psi`` with the transfer-time iteration.
    # ===================================================
    def body_func(carry):
        i, dt_n, psi, psi_low, psi_up, c2, c3 = carry
        y = init_pos_mag + final_pos_mag + (A * (psi * c3 - 1.) / jnp.sqrt(c2))
        y = jnp.maximum(y, 1e-18)
        chi = jnp.sqrt(y / c2)
        new_dt_n = (chi ** 3 * c3 + A * jnp.sqrt(y)) / sqrt_mu
        cond = new_dt_n <= dt
        new_psi_low = jnp.where(cond, psi, psi_low)
        new_psi_up = jnp.where(cond, psi_up, psi)
        new_psi = (new_psi_low + new_psi_up) / 2.
        new_c2, new_c3 = C2C3(new_psi)
        return i + 1, new_dt_n, new_psi, new_psi_low, new_psi_up, new_c2, new_c3

    def cond_func(carry):
        i, dt_n, *_ = carry
        err = jnp.max(jnp.abs(dt_n - dt))
        return jnp.logical_and(i < max_iter, err > tol)

    _, _, psi, _, _, c2, c3 = jax.lax.while_loop(cond_func, body_func, init_carray)

    # ===========================================================
    # 3. Build the Lagrange coefficients and transfer velocities.
    # ===========================================================
    y = init_pos_mag + final_pos_mag + (A * (psi * c3 - 1.) / jnp.sqrt(c2))
    y = jnp.maximum(y, 1e-18)
    f = (1. - y / init_pos_mag)[:, None]
    g_dot = (1. - y / final_pos_mag)[:, None]
    g = (A * jnp.sqrt(y) / sqrt_mu)[:, None]

    # ==================================
    # 4. Build the transfer velocities.
    # ==================================

    init_vel = (final_pos - f * init_pos) / g
    final_vel = (g_dot * final_pos - init_pos) / g

    return init_vel, final_vel
