"""Light-time correction models and path solvers.

This module evaluates one-way and round-trip light-time paths used in astrometric reduction. It includes delay models for relativity, the terrestrial atmosphere, and the solar corona, plus iterative solvers for the down-leg and up-leg signal paths. The main inputs are site states, target trajectories, and epochs in ``UTC`` or ``TDB``. The main outputs are path geometry in ``BCRS`` and solved light time in days.
"""

from functools import partial
from typing import Optional, Tuple

import numpy as np
import jax
from jax import Array, numpy as jnp
from jaxtyping import Float
import equinox as eqx

from difforb.body.ephbody import EphemerisBody
from difforb.body.smallbody import SmallBody
from difforb.core.constants import C, DAY_S, AU_KM, C_KM_SEC
from difforb.body.site import Site
from difforb.core.batch import safe_dispatch, safe_cartesian_dispatch, BatchableObject
from difforb.core.state.frame import BCRS, GCRS
from difforb.core.state.state import State
from difforb.core.time.timescale import TDBView, Time, UTCView
from difforb.report.text import build_repr, format_float_array, format_shape

jax.config.update("jax_enable_x64", True)


# ======================================================
# 1. Correction: relativity, atmosphere and solar corona
# ======================================================

@jax.jit
def relativistic_time_delay(mu: Float[Array, ""], r1: Float[Array, ""], r2: Float[Array, ""],
                            r12: Float[Array, ""]) -> Float[Array, ""]:
    """Return the one-body Shapiro time delay.

    Parameters
    ----------
    mu : Float[Array, ""]
        Gravitational parameter of the perturbing body in ``au^3 / day^2``.
    r1 : Float[Array, ""]
        Distance from the perturbing body to the transmitter in ``au``.
    r2 : Float[Array, ""]
        Distance from the perturbing body to the receiver in ``au``.
    r12 : Float[Array, ""]
        Transmitter-to-receiver distance in ``au``.

    Returns
    -------
    Float[Array, ""]
        Time delay in days.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Sec. 8.7.5.
    """
    r1_plus_r2 = r1 + r2
    safe_denominator = jnp.sqrt(jnp.square(r1 + r2 - r12) + jnp.square(1e-30))
    return 2 * mu / (C ** 3) * jnp.log((r1_plus_r2 + r12) / safe_denominator)


def sum_relativistic_time_delay(
        bodies: tuple[EphemerisBody, ...],
        tx_tdb_jd1: Float[Array, ""],
        tx_tdb_jd2: Float[Array, ""],
        tx_pos: Float[Array, "3"],
        rx_tdb_jd1: Float[Array, ""],
        rx_tdb_jd2: Float[Array, ""],
        rx_pos: Float[Array, "3"],
        path_dist: Float[Array, ""]
) -> Float[Array, ""]:
    """Sum the Shapiro delays from all configured perturbing bodies.

    Parameters
    ----------
    bodies : tuple[EphemerisBody, ...]
        Perturbing bodies.
    tx_tdb_jd1 : Float[Array, ""]
        First part of the split transmit ``TDB`` Julian Date.
    tx_tdb_jd2 : Float[Array, ""]
        Second part of the split transmit ``TDB`` Julian Date.
    tx_pos : Float[Array, "3"]
        Transmitter position in ``BCRS``, in ``au``.
    rx_tdb_jd1 : Float[Array, ""]
        First part of the split receive ``TDB`` Julian Date.
    rx_tdb_jd2 : Float[Array, ""]
        Second part of the split receive ``TDB`` Julian Date.
    rx_pos : Float[Array, "3"]
        Receiver position in ``BCRS``, in ``au``.
    path_dist : Float[Array, ""]
        End-to-end path distance in ``au``.

    Returns
    -------
    Float[Array, ""]
        Total Shapiro delay in days.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Sec. 8.7.5.
    """
    total_delay = jnp.zeros_like(path_dist)
    for body in bodies:
        body_tx_pos = body._bcrs_pos_jd(tx_tdb_jd1, tx_tdb_jd2)
        body_rx_pos = body._bcrs_pos_jd(rx_tdb_jd1, rx_tdb_jd2)
        body2tx_pos = tx_pos - body_tx_pos
        body2rx_pos = rx_pos - body_rx_pos
        total_delay = total_delay + relativistic_time_delay(
            body.gm,
            jnp.linalg.norm(body2tx_pos, axis=-1),
            jnp.linalg.norm(body2rx_pos, axis=-1),
            path_dist
        )
    return total_delay


@jax.jit
def atmosphere_time_delay(cosz: Float[Array, ""]) -> Float[Array, ""]:
    """Return the atmospheric delay from the zenith angle.

    Parameters
    ----------
    cosz : Float[Array, ""]
        Cosine of the antenna zenith angle.

    Returns
    -------
    Float[Array, ""]
        Time delay in days.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Sec. 8.7.7.
    """
    cotz = cosz / jnp.sqrt(1. - cosz * cosz)
    return (7 / (cosz + 0.0014 / (0.045 + cotz))) / (1e9 * DAY_S)


@jax.jit
def solar_corona_electron_density(r: Float[Array, "3"], ) -> Float[Array, ""]:
    """Return the solar-corona electron density along one heliographic vector.

    Parameters
    ----------
    r : Float[Array, "3"]
        Position relative to the Sun center in solar radii. The ``z`` axis must point to the solar north pole.

    Returns
    -------
    Float[Array, ""]
        Electron density in ``cm^-3``.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Sec. 8.7.6.
    """
    A = 1.06e8
    a = 4.89e5
    b = 3.91e5

    # -------------------------------------------------------------------------
    # Step 1: Build the radial and latitude terms
    # -------------------------------------------------------------------------
    dist = jnp.linalg.norm(r, axis=-1)
    dist2 = dist * dist

    z_component = r[..., 2]
    sin_beta = z_component / dist
    sin_beta = jnp.clip(sin_beta, -1.0, 1.0)
    sin_beta_sq = sin_beta ** 2
    cos_beta_sq = 1.0 - sin_beta_sq

    # -------------------------------------------------------------------------
    # Step 2: Evaluate the two-term density model
    # -------------------------------------------------------------------------
    term1 = A / jnp.power(dist, 6)
    num = a * b
    den = jnp.sqrt(a ** 2 * sin_beta_sq + b ** 2 * cos_beta_sq)
    term2 = (num / den) / dist2

    return term1 + term2


@jax.jit
def get_solar_rotation_matrix() -> Float[Array, "3 3"]:
    """Return the rotation matrix from ``BCRS`` to the heliographic working frame.

    Returns
    -------
    Float[Array, "3 3"]
        Rotation matrix that maps ``BCRS`` vectors to a heliographic frame whose ``z`` axis points to the solar north pole.

    Notes
    -----
    The exact longitude origin of the ``x`` and ``y`` axes does not affect the latitude-dependent corona model used in this module.
    """

    # -------------------------------------------------------------------------
    # Step 1: Build the solar-pole direction in ``BCRS``
    # -------------------------------------------------------------------------
    alpha = jnp.deg2rad(286.13)
    delta = jnp.deg2rad(63.87)
    z_axis = jnp.array([
        jnp.cos(delta) * jnp.cos(alpha),
        jnp.cos(delta) * jnp.sin(alpha),
        jnp.sin(delta)
    ])

    # -------------------------------------------------------------------------
    # Step 2: Build an orthonormal basis around the solar pole
    # -------------------------------------------------------------------------
    ref_vec = jnp.array([0., 0., 1.])
    y_axis_raw = jnp.cross(z_axis, ref_vec)
    y_axis = y_axis_raw / jnp.linalg.norm(y_axis_raw)
    x_axis = jnp.cross(y_axis, z_axis)
    x_axis = x_axis / jnp.linalg.norm(x_axis)

    # -------------------------------------------------------------------------
    # Step 3: Assemble the rotation matrix
    # -------------------------------------------------------------------------
    R = jnp.stack([x_axis, y_axis, z_axis], axis=0)

    return R


@jax.jit
def bcrs_to_heliographic_rotation(v_bcrs: Float[Array, "N 3"]) -> Float[Array, "N 3"]:
    """Rotate vectors from ``BCRS`` to the heliographic working frame.

    Parameters
    ----------
    v_bcrs : Float[Array, "N 3"]
        Vectors relative to the Sun center in the ``BCRS``.

    Returns
    -------
    Float[Array, "N 3"]
        Vectors in the heliographic working frame.
    """
    v_in = jnp.atleast_2d(v_bcrs)
    R = get_solar_rotation_matrix()
    v_helio = v_in @ R.T
    return jnp.reshape(v_helio, v_bcrs.shape)


@partial(jax.jit, static_argnames=["num_steps"])
def solar_corona_delay(pos_sun2tx: Float[Array, "3"], pos_sun2rx: Float[Array, "3"],
                       tx_freq: Float[Array, ""],
                       num_steps: int = 64) -> Float[Array, ""]:
    """Return the solar-corona delay along one signal path.

    Parameters
    ----------
    pos_sun2tx : Float[Array, "3"]
        Transmitter position relative to the Sun in ``au``. The ``z`` axis must point to the solar north pole.
    pos_sun2rx : Float[Array, "3"]
        Receiver position relative to the Sun in ``au``. The ``z`` axis must point to the solar north pole.
    tx_freq : Float[Array, ""]
        Signal frequency in ``Hz``.
    num_steps : int, default=64
        Number of integration steps along the ray path.

    Returns
    -------
    Float[Array, ""]
        Time delay in days.

    Notes
    -----
    This function integrates the electron-density model along a straight ray path. The path is split at its closest approach to the Sun and each side is integrated with fixed-order Gauss-Legendre quadrature.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Sec. 8.7.6.
    """
    SOLAR_RADIUS_AU = 695700.0 / AU_KM

    nodes_per_segment = max(num_steps // 2, 1)
    gl_nodes_np, gl_weights_np = np.polynomial.legendre.leggauss(nodes_per_segment)
    gl_nodes = jnp.asarray(gl_nodes_np)
    gl_weights = jnp.asarray(gl_weights_np)

    # -------------------------------------------------------------------------
    # Step 1: Build the straight-line path and split it at closest solar approach
    # -------------------------------------------------------------------------
    diff_vec = pos_sun2rx - pos_sun2tx
    path_len_au = jnp.linalg.norm(diff_vec, axis=-1)
    diff_norm2 = jnp.sum(diff_vec * diff_vec, axis=-1)
    u_closest = jnp.clip(-jnp.sum(pos_sun2tx * diff_vec, axis=-1) / diff_norm2, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # Step 2: Integrate the density on both sides of the closest point
    # -------------------------------------------------------------------------
    def integrate_segment(u0, u1):
        center = 0.5 * (u0 + u1)
        half_width = 0.5 * (u1 - u0)
        u_vals = center + half_width * gl_nodes
        path_points_au = pos_sun2tx + u_vals[:, None] * diff_vec
        path_points_sr = path_points_au / SOLAR_RADIUS_AU
        ne_vals = solar_corona_electron_density(path_points_sr)
        return half_width * jnp.sum(gl_weights * ne_vals)

    path_len_cm = path_len_au * (AU_KM * 1e5)
    integral_u = integrate_segment(0.0, u_closest) + integrate_segment(u_closest, 1.0)
    column_density_cm2 = path_len_cm * integral_u

    # -------------------------------------------------------------------------
    # Step 3: Convert the integrated column density to time delay
    # -------------------------------------------------------------------------
    column_density_m2 = column_density_cm2 * 1e4
    coeff = 40.3 / (C_KM_SEC * 1e3 * tx_freq ** 2)
    delay_seconds = coeff * column_density_m2
    delya_day = delay_seconds / DAY_S

    return delya_day


# ===========================================
# 2. Light Time Solution: down-leg and up-leg
# ===========================================


class LightTimeContext(eqx.Module):
    """Shared configuration for the light-time solver.

    Parameters
    ----------
    sun : EphemerisBody
        Solar ephemeris body used by the relativistic delay and corona delay model.
    earth : EphemerisBody
        Earth ephemeris body used by the site-state model.
    shapiro_bodies : tuple[EphemerisBody, ...], optional
        Bodies included in the Shapiro delay model. If omitted, only the Sun is used.
    atmos_cor_enable : bool, default=False
        If ``True``, include the atmospheric delay model.
    corona_cor_enable : bool, default=False
        If ``True``, include the solar-corona delay model.
    """
    sun: EphemerisBody
    earth: EphemerisBody
    shapiro_bodies: tuple[EphemerisBody, ...]
    atmos_cor_enable: bool = eqx.field(static=True)
    corona_cor_enable: bool = eqx.field(static=True)

    def __init__(self, sun: EphemerisBody, earth: EphemerisBody,
                 shapiro_bodies: Optional[tuple[EphemerisBody, ...]] = None,
                 atmos_cor_enable: bool = False,
                 corona_cor_enable: bool = False):
        """Initialize a light-time solver context.

        Parameters
        ----------
        sun : EphemerisBody
            Solar ephemeris body used by the relativistic delay and corona delay model.
        earth : EphemerisBody
            Earth ephemeris body used by the site-state model.
        shapiro_bodies : tuple[EphemerisBody, ...], optional
            Bodies included in the Shapiro delay model. If omitted, only the Sun is used.
        atmos_cor_enable : bool, default=False
            If ``True``, include the atmospheric delay model.
        corona_cor_enable : bool, default=False
            If ``True``, include the solar-corona delay model.
        """
        self.sun = sun
        self.earth = earth
        self.shapiro_bodies = (sun,) if shapiro_bodies is None else tuple(shapiro_bodies)
        self.atmos_cor_enable = atmos_cor_enable
        self.corona_cor_enable = corona_cor_enable


class LightPath(BatchableObject):
    """Solved one-way light-time path.

    Notes
    -----
    The stored ``pos`` and ``vel`` vectors point from the path end to the target. For a down leg, they point from the receiver to the target. For an up leg, they point from the transmitter to the target.

    ``start`` is the state at the path start point, evaluated at the signal emission time for this leg. ``end`` is the state at the path end point, evaluated at the signal reception time for this leg. For a down leg, ``start`` is the target state and ``end`` is the receiver state. For an up leg, ``start`` is the transmitter state and ``end`` is the target state. The canonical units are ``au`` for position and distance, ``au / day`` for velocity, and days for light time.
    """
    pos: Float[Array, "... 3"]
    vel: Float[Array, "... 3"]
    dist: Float[Array, "..."]
    lt: Float[Array, "..."]  # Light time [day].
    start: State  # State at the path start point.
    end: State  # State at the path end point.

    @property
    def shape(self):
        """Batch shape of the solved path."""
        return self.lt.shape

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("shape", format_shape(self.shape)),
                ("lt_day", format_float_array(self.lt)),
                ("dist_au", format_float_array(self.dist)),
                ("start_type", self.start.__class__.__name__),
                ("end_type", self.end.__class__.__name__),
            ],
        )


def down_leg_light_time_single(t_rec: Time, rx: Site, target: SmallBody,
                               context: LightTimeContext, tx_freq: float = 0.0,
                               tol: float = 1e-14) -> 'LightPath':
    """Solve down-leg light-time path.

    Parameters
    ----------
    t_rec : Time
        Receive epoch.
    rx : Site
        Receiver site.
    target : SmallBody
        Target body with the propagated trajectory.
    context : LightTimeContext
        Delay-model configuration.
    tx_freq : float, default=0.0
        Signal frequency in ``Hz``. This is only used when the corona correction is enabled.
    tol : float, default=1e-14
        Convergence tolerance for the fixed-point light-time iteration, in days.

    Returns
    -------
    LightPath
        Solved down-leg path.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

    Notes
    -----
    The path starts at the target at transmit time and ends at the receiver at receive time.
    """
    sun = context.sun
    earth = context.earth
    max_iters = 6

    # -------------------------------------------------------------------------
    # Step 1: Build the fixed receive-end geometry
    # -------------------------------------------------------------------------
    rx_state_rec = rx.state(t_rec, frame=BCRS, earth=earth)
    rx_pos_rec = rx_state_rec.pos
    rx_vel_rec = rx_state_rec.vel
    t_rec_tdb = rx_state_rec.tdb
    t_rec_tdb_jd1 = t_rec_tdb.jd1
    t_rec_tdb_jd2 = t_rec_tdb.jd2

    if context.atmos_cor_enable:
        earth2rx_state_rec = rx.state(t_rec, frame=GCRS)
        earth2rx_pos_rec = earth2rx_state_rec.pos
        dist_earth2rx_rec = jnp.linalg.norm(earth2rx_pos_rec, axis=-1)
    if context.corona_cor_enable:
        sun_pos_rec = sun._bcrs_pos_jd(t_rec_tdb_jd1, t_rec_tdb_jd2)
        sun2rx_pos_rec = rx_pos_rec - sun_pos_rec
        sun2rx_pos_rec_helio = bcrs_to_heliographic_rotation(sun2rx_pos_rec)

    def body_func(carry):
        i, cur_lt, *_ = carry
        t_trm_tdb_jd1 = t_rec_tdb_jd1
        t_trm_tdb_jd2 = t_rec_tdb_jd2 - cur_lt
        target_pos_trm, target_vel_trm = target._bcrs_pv_jd(t_trm_tdb_jd1, t_trm_tdb_jd2)
        down_pos = target_pos_trm - rx_pos_rec
        down_vel = target_vel_trm - rx_vel_rec
        down_dist = jnp.linalg.norm(down_pos, axis=-1)
        new_lt = down_dist / C
        rel_delay = sum_relativistic_time_delay(
            context.shapiro_bodies,
            t_trm_tdb_jd1,
            t_trm_tdb_jd2,
            target_pos_trm,
            t_rec_tdb_jd1,
            t_rec_tdb_jd2,
            rx_pos_rec,
            down_dist
        )
        new_lt = new_lt + rel_delay

        if context.atmos_cor_enable:
            cosz = jnp.sum(earth2rx_state_rec.pos * down_pos, axis=-1) / (
                    dist_earth2rx_rec * down_dist)
            atm_delay = atmosphere_time_delay(cosz)
            new_lt = new_lt + atm_delay

        if context.corona_cor_enable:
            sun_pos_trm = sun._bcrs_pos_jd(t_trm_tdb_jd1, t_trm_tdb_jd2)
            sun2target_pos_trm = target_pos_trm - sun_pos_trm
            sun2target_pos_trm_helio = bcrs_to_heliographic_rotation(sun2target_pos_trm)
            corona_delay = solar_corona_delay(sun2target_pos_trm_helio, sun2rx_pos_rec_helio, tx_freq)
            new_lt = new_lt + corona_delay
        return i + 1, new_lt, cur_lt, down_pos, down_vel, down_dist, target_pos_trm, target_vel_trm

    def cond_func(carry):
        i, cur_lt, prev_lt, *_ = carry
        err = jnp.max(jnp.abs(cur_lt - prev_lt), initial=0.)
        return (err > tol) & (i < max_iters)

    # -------------------------------------------------------------------------
    # Step 2: Build the initial guess
    # -------------------------------------------------------------------------
    target_pos_rec, target_vel_rec = target._bcrs_pv_jd(t_rec_tdb_jd1, t_rec_tdb_jd2)
    init_down_pos = target_pos_rec - rx_pos_rec
    init_down_vel = target_vel_rec - rx_vel_rec
    init_dist = jnp.linalg.norm(init_down_pos, axis=-1)
    init_lt = init_dist / C
    init_carry = (0, init_lt, init_lt + 1., init_down_pos, init_down_vel, init_dist, target_pos_rec, target_vel_rec)

    # -------------------------------------------------------------------------
    # Step 3: Solve the fixed-point light-time equation
    # -------------------------------------------------------------------------
    _, final_lt, _, down_pos, down_vel, down_dist, target_pos_trm, target_vel_trm = jax.lax.while_loop(cond_func, body_func,
                                                                                                       init_carry)
    t_trm = Time.from_tdb_jd(t_rec_tdb_jd1, t_rec_tdb_jd2 - final_lt, eop=t_rec.eop,
                             gregorian_start=t_rec.gregorian_start)
    t_trm_tdb = t_trm.tdb()
    target_state_trm = State(tdb=t_trm_tdb, pos=target_pos_trm, vel=target_vel_trm, frame=BCRS)
    rx_state_rec = State.from_array(t_rec_tdb, jnp.concatenate([rx_pos_rec, rx_vel_rec]), BCRS)
    return LightPath(pos=down_pos, vel=down_vel, lt=final_lt, dist=down_dist,
                     start=target_state_trm,
                     end=rx_state_rec)


def _down_leg_light_time_single_reorder(target, rx, tx_freq, t_rec, context, tol):
    return down_leg_light_time_single(
        t_rec, rx, target, context, tx_freq, tol
    )


@eqx.filter_jit
def down_leg_light_time(
        t_rec: Time, rx: Site,
        target: SmallBody, context: LightTimeContext, tx_freq: Float[Array, "..."] = None,
        tol: float = 1e-14, grid: bool = False) -> 'LightPath':
    """Solve down-leg light-time path.

    Parameters
    ----------
    t_rec : Time
        Receive epoch.
    rx : Site
        Receiver site.
    target : SmallBody
        Target body with the propagated trajectory.
    context : LightTimeContext
        Delay-model configuration.
    tx_freq : Float[Array, "..."], optional
        Signal frequency in ``Hz``. This is only used when the corona correction is enabled.
    tol : float, default=1e-14
        Convergence tolerance in days.
    grid : bool, default=False
        If ``True``, use the Cartesian product of target, site, and time batches. If ``False``, use point-wise broadcasting.

    Returns
    -------
    LightPath
        Solved down-leg path or batch of paths.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

    Notes
    -----
    Vectorize :func:`down_leg_light_time_single`.
    """
    wrapper = partial(
        _down_leg_light_time_single_reorder,
        context=context,
        tol=tol
    )

    if not grid:
        return safe_dispatch(wrapper, (0, 0, 0, 0), target, rx, tx_freq,
                             t_rec)
    else:
        return safe_cartesian_dispatch(wrapper, ((0,), (target,)), ((0, 0), (rx, tx_freq)), ((0,), (t_rec,)))


def forward_up_leg_light_time_single(t_trm: Time, tx: Site, target: SmallBody,
                                     context: LightTimeContext, tx_freq: float,
                                     tol: float = 1e-14) -> Tuple[LightPath, Time]:
    """Solve a transmit-to-target light-time path from a fixed transmit epoch.

    Parameters
    ----------
    t_trm : Time
        Transmit epoch at the transmitter site.
    tx : Site
        Transmitter site.
    target : SmallBody
        Target body with the propagated trajectory.
    context : LightTimeContext
        Delay-model configuration.
    tx_freq : float
        Signal frequency in ``Hz``. This is only used when the corona correction is enabled.
    tol : float, default=1e-14
        Convergence tolerance for the fixed-point light-time iteration, in days.

    Returns
    -------
    tuple[LightPath, Time]
        Solved up-leg path and target bounce epoch.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

    Notes
    -----
    The path starts at the transmitter at transmit time and ends at the target at bounce time.
    """
    sun = context.sun
    earth = context.earth
    max_iters = 6

    # -------------------------------------------------------------------------
    # Step 1: Build the fixed transmit-end geometry
    # -------------------------------------------------------------------------
    tx_state_trm = tx.state(t_trm, frame=BCRS, earth=earth)
    tx_pos_trm = tx_state_trm.pos
    tx_vel_trm = tx_state_trm.vel
    t_trm_tdb = tx_state_trm.tdb
    t_trm_tdb_jd1 = t_trm_tdb.jd1
    t_trm_tdb_jd2 = t_trm_tdb.jd2
    t_trm_tt_jd1 = t_trm._tt_jd1
    t_trm_tt_jd2 = t_trm._tt_jd2
    eop = t_trm.eop
    gregorian_start = t_trm.gregorian_start
    if context.atmos_cor_enable:
        earth2tx_state_trm = tx.state(t_trm, frame=GCRS)
        earth2tx_pos_trm = earth2tx_state_trm.pos
        dist_earth2tx_trm = jnp.linalg.norm(earth2tx_pos_trm, axis=-1)
    if context.corona_cor_enable:
        sun_pos_trm = sun._bcrs_pos_jd(t_trm_tdb_jd1, t_trm_tdb_jd2)
        sun2tx_pos_trm = tx_pos_trm - sun_pos_trm
        sun2tx_pos_trm_helio = bcrs_to_heliographic_rotation(sun2tx_pos_trm)

    def body_func(carry):
        i, cur_t_bounce_tt_jd2, prev_t_bounce_tt_jd2, *_ = carry
        t_bounce = Time.from_tt_jd(t_trm_tt_jd1, cur_t_bounce_tt_jd2, eop=eop, gregorian_start=gregorian_start)
        t_bounce_tdb = t_bounce.tdb()
        t_bounce_tdb_jd1, t_bounce_tdb_jd2 = t_bounce_tdb.jd1, t_bounce_tdb.jd2
        target_pos_bounce, target_vel_bounce = target._bcrs_pv_jd(t_bounce_tdb_jd1, t_bounce_tdb_jd2)
        up_pos = target_pos_bounce - tx_pos_trm
        up_vel = target_vel_bounce - tx_vel_trm
        up_dist = jnp.linalg.norm(up_pos, axis=-1)
        lt_tdb = up_dist / C
        rel_delay = sum_relativistic_time_delay(
            context.shapiro_bodies,
            t_trm_tdb_jd1, t_trm_tdb_jd2,
            tx_pos_trm,
            t_bounce_tdb_jd1, t_bounce_tdb_jd2,
            target_pos_bounce,
            up_dist
        )
        lt_tdb = lt_tdb + rel_delay

        if context.atmos_cor_enable:
            cosz = jnp.sum(earth2tx_pos_trm * up_pos, axis=-1) / (
                    dist_earth2tx_trm * up_dist)
            atm_delay = atmosphere_time_delay(cosz)
            lt_tdb = lt_tdb + atm_delay

        if context.corona_cor_enable:
            sun_pos_bounce = sun._bcrs_pos_jd(t_bounce_tdb_jd1, t_bounce_tdb_jd2)
            sun2target_pos_bounce = target_pos_bounce - sun_pos_bounce
            sun2target_pos_bounce_helio = bcrs_to_heliographic_rotation(sun2target_pos_bounce)
            corona_delay = solar_corona_delay(sun2tx_pos_trm_helio, sun2target_pos_bounce_helio, tx_freq)
            lt_tdb = lt_tdb + corona_delay

        actual_lt_tdb = (t_bounce_tdb_jd1 - t_trm_tdb_jd1) + (t_bounce_tdb_jd2 - t_trm_tdb_jd2)
        lt_err = lt_tdb - actual_lt_tdb
        new_t_bounce_tt_jd2 = cur_t_bounce_tt_jd2 + lt_err

        return (i + 1, new_t_bounce_tt_jd2, cur_t_bounce_tt_jd2, up_pos, up_vel, up_dist,
                target_pos_bounce, target_vel_bounce, lt_tdb, t_bounce_tdb, t_bounce)

    def cond_func(carry):
        i, cur_t_bounce_tt_jd2, prev_t_bounce_tt_jd2, *_ = carry
        err = jnp.max(jnp.abs(cur_t_bounce_tt_jd2 - prev_t_bounce_tt_jd2), initial=0.)
        return (err > tol) & (i < max_iters)

    # -------------------------------------------------------------------------
    # Step 2: Build the initial guess
    # -------------------------------------------------------------------------
    target_pos_trm, target_vel_trm = target._bcrs_pv_jd(t_trm_tdb_jd1, t_trm_tdb_jd2)
    init_up_pos = target_pos_trm - tx_pos_trm
    init_up_vel = target_vel_trm - tx_vel_trm
    init_up_dist = jnp.linalg.norm(init_up_pos, axis=-1)
    init_lt = init_up_dist / C
    init_t_bounce_tt_jd2 = t_trm_tt_jd2 + init_lt
    init_carry = (0, init_t_bounce_tt_jd2, init_t_bounce_tt_jd2 - 1.0, init_up_pos, init_up_vel, init_up_dist,
                  target_pos_trm, target_vel_trm, init_lt, t_trm_tdb, t_trm)

    # -------------------------------------------------------------------------
    # Step 3: Solve the fixed-point light-time equation
    # -------------------------------------------------------------------------
    _, _, _, up_pos, up_vel, up_dist, target_pos_bounce, target_vel_bounce, final_lt, t_bounce_tdb, t_bounce = jax.lax.while_loop(
        cond_func,
        body_func,
        init_carry)
    target_state_bounce = State(t_bounce_tdb, target_pos_bounce, target_vel_bounce, BCRS)
    return LightPath(pos=up_pos, vel=up_vel, dist=up_dist, lt=final_lt, start=tx_state_trm,
                     end=target_state_bounce), t_bounce


def forward_down_leg_light_time_single(t_bounce_tdb: TDBView, target_state_bounce: State, rx: Site,
                                       context: LightTimeContext, tx_freq: float,
                                       tol: float = 1e-14) -> Tuple[LightPath, Time]:
    """Solve a target-to-receiver light-time path from a fixed bounce epoch.

    Parameters
    ----------
    t_bounce_tdb : TDBView
        Bounce epoch in ``TDB``.
    target_state_bounce : State
        Target state at the bounce epoch, in ``BCRS``.
    rx : Site
        Receiver site.
    context : LightTimeContext
        Delay-model configuration.
    tx_freq : float
        Signal frequency in ``Hz``. This is only used when the corona correction is enabled.
    tol : float, default=1e-14
        Convergence tolerance for the fixed-point light-time iteration, in days.

    Returns
    -------
    tuple[LightPath, Time]
        Solved down-leg path and receive epoch.

    Notes
    -----
    The path starts at the target at bounce time and ends at the receiver at receive time.
    """
    sun = context.sun
    earth = context.earth
    max_iters = 6

    # -------------------------------------------------------------------------
    # Step 1: Build the fixed bounce-start geometry
    # -------------------------------------------------------------------------
    t_bounce = t_bounce_tdb.time
    t_bounce_tdb_jd1 = t_bounce_tdb.jd1
    t_bounce_tdb_jd2 = t_bounce_tdb.jd2
    t_bounce_tt_jd1 = t_bounce._tt_jd1
    t_bounce_tt_jd2 = t_bounce._tt_jd2
    eop = t_bounce.eop
    gregorian_start = t_bounce.gregorian_start
    target_pos_bounce = target_state_bounce.pos
    target_vel_bounce = target_state_bounce.vel
    rx_state_bounce = rx.state(t_bounce, frame=BCRS, earth=earth)
    if context.corona_cor_enable:
        sun_pos_bounce = sun._bcrs_pos_jd(t_bounce_tdb_jd1, t_bounce_tdb_jd2)
        sun2target_pos_bounce = target_pos_bounce - sun_pos_bounce
        sun2target_pos_bounce_helio = bcrs_to_heliographic_rotation(sun2target_pos_bounce)

    def body_func(carry):
        i, cur_t_rec_tt_jd2, prev_t_rec_tt_jd2, *_ = carry
        t_rec = Time.from_tt_jd(t_bounce_tt_jd1, cur_t_rec_tt_jd2, eop=eop, gregorian_start=gregorian_start)
        rx_state_rec = rx.state(t_rec, frame=BCRS, earth=earth)
        rx_pos_rec = rx_state_rec.pos
        rx_vel_rec = rx_state_rec.vel
        t_rec_tdb = rx_state_rec.tdb
        t_rec_tdb_jd1, t_rec_tdb_jd2 = t_rec_tdb.jd1, t_rec_tdb.jd2
        down_pos = target_pos_bounce - rx_pos_rec
        down_vel = target_vel_bounce - rx_vel_rec
        down_dist = jnp.linalg.norm(down_pos, axis=-1)
        lt_tdb = down_dist / C
        rel_delay = sum_relativistic_time_delay(
            context.shapiro_bodies,
            t_bounce_tdb_jd1, t_bounce_tdb_jd2,
            target_pos_bounce,
            t_rec_tdb_jd1, t_rec_tdb_jd2,
            rx_pos_rec,
            down_dist
        )
        lt_tdb = lt_tdb + rel_delay

        if context.atmos_cor_enable:
            earth2rx_pos_rec = rx.state(t_rec, frame=GCRS).pos
            dist_earth2rx_rec = jnp.linalg.norm(earth2rx_pos_rec, axis=-1)
            cosz = jnp.sum(earth2rx_pos_rec * down_pos, axis=-1) / (
                    dist_earth2rx_rec * down_dist)
            atm_delay = atmosphere_time_delay(cosz)
            lt_tdb = lt_tdb + atm_delay

        if context.corona_cor_enable:
            sun_pos_rec = sun._bcrs_pos_jd(t_rec_tdb_jd1, t_rec_tdb_jd2)
            sun2rx_pos_rec = rx_pos_rec - sun_pos_rec
            sun2rx_pos_rec_helio = bcrs_to_heliographic_rotation(sun2rx_pos_rec)
            corona_delay = solar_corona_delay(sun2target_pos_bounce_helio, sun2rx_pos_rec_helio, tx_freq)
            lt_tdb = lt_tdb + corona_delay

        actual_lt_tdb = (t_rec_tdb_jd1 - t_bounce_tdb_jd1) + (t_rec_tdb_jd2 - t_bounce_tdb_jd2)
        lt_err = lt_tdb - actual_lt_tdb
        new_t_rec_tt_jd2 = cur_t_rec_tt_jd2 + lt_err

        return (i + 1, new_t_rec_tt_jd2, cur_t_rec_tt_jd2, down_pos, down_vel, down_dist,
                rx_pos_rec, rx_vel_rec, lt_tdb, t_rec_tdb, t_rec)

    def cond_func(carry):
        i, cur_t_rec_tt_jd2, prev_t_rec_tt_jd2, *_ = carry
        err = jnp.max(jnp.abs(cur_t_rec_tt_jd2 - prev_t_rec_tt_jd2), initial=0.)
        return (err > tol) & (i < max_iters)

    # -------------------------------------------------------------------------
    # Step 2: Build the initial guess
    # -------------------------------------------------------------------------
    init_down_pos = target_pos_bounce - rx_state_bounce.pos
    init_down_vel = target_vel_bounce - rx_state_bounce.vel
    init_down_dist = jnp.linalg.norm(init_down_pos, axis=-1)
    init_lt = init_down_dist / C
    init_t_rec_tt_jd2 = t_bounce_tt_jd2 + init_lt
    init_carry = (0, init_t_rec_tt_jd2, init_t_rec_tt_jd2 - 1.0, init_down_pos, init_down_vel, init_down_dist,
                  rx_state_bounce.pos, rx_state_bounce.vel, init_lt, t_bounce_tdb, t_bounce)

    # -------------------------------------------------------------------------
    # Step 3: Solve the fixed-point light-time equation
    # -------------------------------------------------------------------------
    _, _, _, down_pos, down_vel, down_dist, rx_pos_rec, rx_vel_rec, final_lt, t_rec_tdb, t_rec = jax.lax.while_loop(
        cond_func,
        body_func,
        init_carry)
    rx_state_rec = State(t_rec_tdb, rx_pos_rec, rx_vel_rec, BCRS)
    return LightPath(pos=down_pos, vel=down_vel, dist=down_dist, lt=final_lt, start=target_state_bounce,
                     end=rx_state_rec), t_rec


def up_leg_light_time_single(t_bounce_tdb: TDBView, target_state_bounce: State, tx: Site,
                             context: LightTimeContext, tx_freq: float,
                             tol: float = 1e-14) -> Tuple[LightPath, UTCView]:
    """Solve up-leg light-time path.

    Parameters
    ----------
    t_bounce_tdb : TDBView
        Bounce epoch in ``TDB``.
    target_state_bounce : State
        Target state at the bounce epoch, in ``BCRS``.
    tx : Site
        Transmitter site.
    context : LightTimeContext
        Delay-model configuration.
    tx_freq : float
        Signal frequency in ``Hz``. This is only used when the corona correction is enabled.
    tol : float, default=1e-14
        Convergence tolerance for the fixed-point light-time iteration, in days.

    Returns
    -------
    tuple[LightPath, UTCView]
        Solved up-leg path and the transmit epoch in ``UTC``.

    Notes
    -----
    The path starts at the transmitter at transmit time and ends at the target at bounce time.
    """
    sun = context.sun
    earth = context.earth
    max_iters = 6

    # -------------------------------------------------------------------------
    # Step 1: Build the fixed bounce-end geometry
    # -------------------------------------------------------------------------
    t_bounce = t_bounce_tdb.time
    t_bounce_tdb_jd1 = t_bounce_tdb.jd1
    t_bounce_tdb_jd2 = t_bounce_tdb.jd2
    t_bounce_tt_jd1 = t_bounce._tt_jd1
    t_bounce_tt_jd2 = t_bounce._tt_jd2
    eop = t_bounce.eop
    gregorian_start = t_bounce.gregorian_start
    tx_state_bounce = tx.state(t_bounce, frame=BCRS, earth=earth)
    target_pos_bounce = target_state_bounce.pos
    target_vel_bounce = target_state_bounce.vel
    if context.corona_cor_enable:
        sun_pos_bounce = sun._bcrs_pos_jd(t_bounce_tdb_jd1, t_bounce_tdb_jd2)
        sun2target_pos_bounce = target_pos_bounce - sun_pos_bounce
        sun2target_pos_bounce_helio = bcrs_to_heliographic_rotation(sun2target_pos_bounce)

    def body_func(carry):
        i, cur_t_trm_tt_jd2, prev_t_trm_tt_jd2, *_ = carry
        t_trm = Time.from_tt_jd(t_bounce_tt_jd1, cur_t_trm_tt_jd2, eop=eop, gregorian_start=gregorian_start)
        tx_state_trm = tx.state(t_trm, frame=BCRS, earth=earth)
        tx_pos_trm, tx_vel_trm = tx_state_trm.pos, tx_state_trm.vel
        t_trm_tdb = tx_state_trm.tdb
        t_trm_tdb_jd1, t_trm_tdb_jd2 = t_trm_tdb.jd1, t_trm_tdb.jd2
        up_pos = target_pos_bounce - tx_pos_trm
        up_vel = target_vel_bounce - tx_vel_trm
        up_dist = jnp.linalg.norm(up_pos, axis=-1)
        lt_tdb = up_dist / C
        rel_delay = sum_relativistic_time_delay(
            context.shapiro_bodies,
            t_trm_tdb_jd1, t_trm_tdb_jd2,
            tx_pos_trm,
            t_bounce_tdb_jd1, t_bounce_tdb_jd2,
            target_pos_bounce,
            up_dist
        )
        lt_tdb = lt_tdb + rel_delay

        if context.atmos_cor_enable:
            earth2tx_pos_trm = tx.state(t_trm, frame=GCRS).pos
            dist_earth2tx_trm = jnp.linalg.norm(earth2tx_pos_trm, axis=-1)
            cosz = jnp.sum(earth2tx_pos_trm * up_pos, axis=-1) / (
                    dist_earth2tx_trm * up_dist)
            atm_delay = atmosphere_time_delay(cosz)
            lt_tdb = lt_tdb + atm_delay

        if context.corona_cor_enable:
            sun_pos_trm = sun._bcrs_pos_jd(t_trm_tdb.jd1, t_trm_tdb.jd2)
            sun2tx_pos_trm = tx_pos_trm - sun_pos_trm
            sun2tx_pos_trm_helio = bcrs_to_heliographic_rotation(sun2tx_pos_trm)
            corona_delay = solar_corona_delay(sun2tx_pos_trm_helio, sun2target_pos_bounce_helio, tx_freq)
            lt_tdb = lt_tdb + corona_delay

        actual_lt_tdb = (t_bounce_tdb_jd1 - t_trm_tdb_jd1) + (t_bounce_tdb_jd2 - t_trm_tdb_jd2)
        lt_err = lt_tdb - actual_lt_tdb

        new_t_trm_tt_jd2 = cur_t_trm_tt_jd2 - lt_err

        return (i + 1, new_t_trm_tt_jd2, cur_t_trm_tt_jd2, up_pos, up_vel, up_dist, tx_pos_trm, tx_vel_trm, lt_tdb, t_trm_tdb,
                t_trm)

    def cond_func(carry):
        i, cur_t_trm_tt_jd2, prev_t_trm_tt_jd2, *_ = carry
        err = jnp.max(jnp.abs(cur_t_trm_tt_jd2 - prev_t_trm_tt_jd2), initial=0.)
        return (err > tol) & (i < max_iters)

    # -------------------------------------------------------------------------
    # Step 2: Build the initial guess
    # -------------------------------------------------------------------------
    init_up_pos = target_state_bounce.pos - tx_state_bounce.pos
    init_up_vel = target_state_bounce.vel - tx_state_bounce.vel
    init_up_dist = jnp.linalg.norm(init_up_pos, axis=-1)
    init_lt = init_up_dist / C
    init_t_trm_tt_jd2 = t_bounce_tt_jd2 - init_lt
    init_carry = (0, init_t_trm_tt_jd2, init_t_trm_tt_jd2 + 1.0, init_up_pos, init_up_vel, init_up_dist, tx_state_bounce.pos,
                  tx_state_bounce.vel, init_lt, t_bounce_tdb, t_bounce)

    # -------------------------------------------------------------------------
    # Step 3: Solve the fixed-point light-time equation
    # -------------------------------------------------------------------------
    _, _, _, up_pos, up_vel, up_dist, tx_pos_trm, tx_vel_trm, final_lt, t_trm_tdb, t_trm = jax.lax.while_loop(
        cond_func,
        body_func,
        init_carry)
    tx_state_trm = State(t_trm_tdb, tx_pos_trm, tx_vel_trm, BCRS)
    return LightPath(pos=up_pos, vel=up_vel, dist=up_dist, lt=final_lt, start=tx_state_trm,
                     end=target_state_bounce), t_trm


def _up_leg_light_time_single_reorder(target_state_bounce, tx, tx_freq, t_bounce_tdb, context, tol):
    """Reorder arguments for batch dispatch."""
    return up_leg_light_time_single(
        t_bounce_tdb, target_state_bounce, tx, context, tx_freq, tol
    )


@eqx.filter_jit
def up_leg_light_time(t_bounce_tdb: TDBView, target_state_bounce: State, tx: Site,
                      context: LightTimeContext, tx_freq: Float[Array, "..."],
                      tol: float = 1e-14) -> Tuple[
    LightPath, UTCView]:
    """Solve up-leg light-time path.

    Parameters
    ----------
    t_bounce_tdb : TDBView
        Bounce epoch in ``TDB``.
    target_state_bounce : State
        Target state at the bounce epoch, in ``BCRS``.
    tx : Site
        Transmitter site.
    context : LightTimeContext
        Delay-model configuration.
    tx_freq : Float[Array, "..."]
        Signal frequency in ``Hz``. This is only used when the corona correction is enabled.
    tol : float, default=1e-14
        Convergence tolerance in days.

    Returns
    -------
    tuple[LightPath, UTCView]
        Solved up-leg path and the transmit epoch in ``UTC``.

    Notes
    -----
    Vectorize :func:`up_leg_light_time_single`.

    The transmitter site must expose an ``ITRS`` location.
    """
    wrapper = partial(
        _up_leg_light_time_single_reorder,
        context=context,
        tol=tol
    )

    return safe_dispatch(wrapper, (0, 0, 0, 0), target_state_bounce, tx, tx_freq, t_bounce_tdb)
