"""Radar delay and Doppler measurement reduction.

This module builds two-way radar observables from light-time solutions. The main entry point solves the receive-side down leg and the transmit-side up leg, then combines them into round-trip delay, range, Doppler shift, and range rate.

Receiver times are given in ``UT1``, ``UTC``, or mixed ``UT``. The target state is handled in ``TDB`` through :mod:`difforb.astrometry.reduction.lt`.
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from difforb.body.site import Site
from difforb.astrometry.reduction.lt import (LightPath, LightTimeContext, down_leg_light_time_single,
                                             forward_down_leg_light_time_single, forward_up_leg_light_time_single,
                                             up_leg_light_time_single)
from difforb.body.smallbody import SmallBody
from difforb.core.constants import C, DAY_S
from difforb.core.time.timescale import Time
from difforb.report.text import build_repr, format_float_array, format_shape
from difforb.utils import R3_single

from difforb.core.batch import safe_dispatch, safe_cartesian_dispatch, BatchableObject

jax.config.update("jax_enable_x64", True)


class RadarObservation(BatchableObject):
    """Two-way radar measurements.

    Parameters
    ----------
    up_path : LightPath
        Solved transmit-to-target light path.
    down_path : LightPath
        Solved target-to-receiver light path.
    delay : Float[Array, "..."]
        Two-way light time in microseconds.
    range : Float[Array, "..."]
        Two-way range in ``au``.
    doppler_shift : Float[Array, "..."]
        Two-way Doppler shift in ``Hz``.
    rate : Float[Array, "..."]
        Two-way range rate in ``au / day``.
    tx_azimuth, tx_elevation : Float[Array, "..."]
        Transmitter pointing azimuth and elevation in degrees at the transmit epoch. Space transmitter rows are ``NaN``.
    rx_azimuth, rx_elevation : Float[Array, "..."]
        Receiver pointing azimuth and elevation in degrees at the receive epoch. Space receiver rows are ``NaN``.
    """

    up_path: LightPath
    down_path: LightPath
    delay: Float[Array, "..."]  # in us
    range: Float[Array, "..."]  # in au
    doppler_shift: Float[Array, "..."]  # in Hz
    rate: Float[Array, "..."]  # in au/day
    tx_azimuth: Float[Array, "..."]  # in deg
    tx_elevation: Float[Array, "..."]  # in deg
    rx_azimuth: Float[Array, "..."]  # in deg
    rx_elevation: Float[Array, "..."]  # in deg

    @property
    def shape(self):
        return self.up_path.shape

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("shape", format_shape(self.shape)),
                ("delay_us", format_float_array(self.delay)),
                ("range_au", format_float_array(self.range)),
                ("doppler_hz", format_float_array(self.doppler_shift)),
                ("rate_au_per_d", format_float_array(self.rate)),
                ("tx_azimuth_deg", format_float_array(self.tx_azimuth)),
                ("tx_elevation_deg", format_float_array(self.tx_elevation)),
                ("rx_azimuth_deg", format_float_array(self.rx_azimuth)),
                ("rx_elevation_deg", format_float_array(self.rx_elevation)),
                ("up_path", self.up_path.__class__.__name__),
                ("down_path", self.down_path.__class__.__name__),
            ],
        )


def _radar_pointing_angles_single(t: Time, site: Site, pointing_pos: Float[Array, "3"]) -> tuple[Float[Array, ""],
                                                                                              Float[Array, ""]]:
    """Return ground-site azimuth and elevation for one radar pointing vector.

    Parameters
    ----------
    t : Time
        Epoch at the ground site.
    site : Site
        Ground or space site.
    pointing_pos : Float[Array, "3"]
        Inertial vector from the site to the target path endpoint, in ``au``.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""]]
        Azimuth and elevation in degrees. Space rows are returned as ``NaN``.
    """
    ground_itrs = site.ground_itrs
    cirs_pos = t.gcrs_to_cirs_matrix @ pointing_pos
    tirs_pos = R3_single(t.ERA) @ cirs_pos
    itrs_pos = t.polar_motion_matrix @ tirs_pos

    sin_lat, cos_lat = jnp.sin(ground_itrs.geodetic_lat), jnp.cos(ground_itrs.geodetic_lat)
    sin_lon, cos_lon = jnp.sin(ground_itrs.lon), jnp.cos(ground_itrs.lon)
    enu_mat = jnp.array(
        [[-sin_lon, cos_lon, 0.], [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
         [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]]
    )
    enu_pos = enu_mat @ itrs_pos
    azimuth = jnp.arctan2(enu_pos[0], enu_pos[1]) % (2.0 * jnp.pi)
    elevation = jnp.arctan2(enu_pos[2], jnp.sqrt(enu_pos[0] ** 2 + enu_pos[1] ** 2))
    azimuth = jnp.where(site.is_ground, jnp.rad2deg(azimuth), jnp.full_like(azimuth, jnp.nan))
    elevation = jnp.where(site.is_ground, jnp.rad2deg(elevation), jnp.full_like(elevation, jnp.nan))
    return azimuth, elevation


def _pointing_angles_from_paths(tx: Site, rx: Site, up_path: LightPath,
                                down_path: LightPath) -> tuple[Float[Array, ""], Float[Array, ""],
                                                                Float[Array, ""], Float[Array, ""]]:
    """Return transmitter and receiver pointing angles from solved radar paths."""
    tx_t = up_path.start.tdb.time
    rx_t = down_path.end.tdb.time
    tx_azimuth, tx_elevation = _radar_pointing_angles_single(tx_t, tx, up_path.pos)
    rx_azimuth, rx_elevation = _radar_pointing_angles_single(rx_t, rx, down_path.pos)
    return tx_azimuth, tx_elevation, rx_azimuth, rx_elevation


def compute_radar_obs_single(t_rec: Time, rx: Site, tx: Site, tx_freq: float,
                             target: SmallBody,
                             context: LightTimeContext) -> RadarObservation:
    """Compute two-way radar observation.

    Parameters
    ----------
    t_rec : Time
        Receive epoch.
    rx : Site
        Receiver site.
    tx : Site
        Transmitter site.
    tx_freq : float
        Transmit frequency in ``Hz``.
    target : SmallBody
        Target body with the propagated trajectory.
    context : LightTimeContext
        Shared light-time solver context.

    Returns
    -------
    RadarObservation
        Two-way radar measurements and the solved up-leg and down-leg paths.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

    Notes
    -----
    The delay model is differentiated with respect to the receive epoch to get the Doppler term and the range-rate term.
    """

    def _core_delay_model(_t_rec_tt_jd2):
        # -------------------------------------------------------------------------
        # Step 1: Rebuild the reception epoch as a high-precision ``UT`` object.
        # -------------------------------------------------------------------------
        t_rec_local = Time.from_tt_jd(t_rec_tt_jd1, _t_rec_tt_jd2, eop=t_rec.eop, gregorian_start=t_rec.gregorian_start)

        # -------------------------------------------------------------------------
        # Step 2: Solve the target-to-receiver down leg.
        # -------------------------------------------------------------------------
        down_sol = down_leg_light_time_single(t_rec_local, rx, target, context, tx_freq, 1e-16)
        t_bounce_tdb = down_sol.start.tdb
        target_state_bounce = down_sol.start

        # -------------------------------------------------------------------------
        # Step 3: Solve the transmitter-to-target up leg.
        # -------------------------------------------------------------------------
        up_sol, t_trm = up_leg_light_time_single(t_bounce_tdb, target_state_bounce, tx, context, tx_freq, 1e-16)
        total_light_time = (t_rec_local._tt_jd1 - t_trm._tt_jd1) + (t_rec_local._tt_jd2 - t_trm._tt_jd2)
        aux = jax.lax.stop_gradient((up_sol, down_sol))
        return total_light_time, aux

    t_rec_tt_jd1 = t_rec._tt_jd1
    t_rec_tt_jd2 = t_rec._tt_jd2

    # Differentiate with respect to ``jd2`` only. ``jd1`` stays fixed.

    (delay_day, aux_data), (d_delay, _) = jax.jvp(_core_delay_model, (t_rec_tt_jd2,), (1.0,))
    up_path, down_path = aux_data

    delay_us = delay_day * DAY_S * 1e6
    range = delay_day * C
    doppler_shift = -tx_freq * d_delay
    rate = d_delay * C
    tx_azimuth, tx_elevation, rx_azimuth, rx_elevation = _pointing_angles_from_paths(tx, rx, up_path, down_path)

    return RadarObservation(
        up_path=up_path,
        down_path=down_path,
        delay=delay_us,
        range=range,
        doppler_shift=doppler_shift,
        rate=rate,
        tx_azimuth=tx_azimuth,
        tx_elevation=tx_elevation,
        rx_azimuth=rx_azimuth,
        rx_elevation=rx_elevation,
    )


def compute_radar_obs_transmit_single(t_trm: Time, rx: Site, tx: Site, tx_freq: float,
                                      target: SmallBody,
                                      context: LightTimeContext) -> RadarObservation:
    """Compute a two-way radar observation from a fixed transmit epoch.

    Parameters
    ----------
    t_trm : Time
        Transmit epoch at the transmitter site.
    rx : Site
        Receiver site.
    tx : Site
        Transmitter site.
    tx_freq : float
        Transmit frequency in ``Hz``.
    target : SmallBody
        Target body with the propagated trajectory.
    context : LightTimeContext
        Shared light-time solver context.

    Returns
    -------
    RadarObservation
        Two-way radar measurements and the solved up-leg and down-leg paths.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

    Notes
    -----
    The delay model is differentiated with respect to the transmit epoch to get the Doppler term and the range-rate term.
    """

    def _core_delay_model(_t_trm_tt_jd2):
        # -------------------------------------------------------------------------
        # Step 1: Rebuild the transmit epoch as a high-precision ``Time`` object.
        # -------------------------------------------------------------------------
        t_trm_local = Time.from_tt_jd(t_trm_tt_jd1, _t_trm_tt_jd2, eop=t_trm.eop, gregorian_start=t_trm.gregorian_start)

        # -------------------------------------------------------------------------
        # Step 2: Solve the transmitter-to-target up leg.
        # -------------------------------------------------------------------------
        up_sol, _ = forward_up_leg_light_time_single(t_trm_local, tx, target, context, tx_freq, 1e-16)

        # -------------------------------------------------------------------------
        # Step 3: Solve the target-to-receiver down leg.
        # -------------------------------------------------------------------------
        down_sol, t_rec = forward_down_leg_light_time_single(up_sol.end.tdb, up_sol.end, rx, context, tx_freq, 1e-16)
        total_light_time = (t_rec._tt_jd1 - t_trm_local._tt_jd1) + (t_rec._tt_jd2 - t_trm_local._tt_jd2)
        aux = jax.lax.stop_gradient((up_sol, down_sol))
        return total_light_time, aux

    t_trm_tt_jd1 = t_trm._tt_jd1
    t_trm_tt_jd2 = t_trm._tt_jd2
    (delay_day, aux_data), (d_delay, _) = jax.jvp(_core_delay_model, (t_trm_tt_jd2,), (1.0,))
    up_path, down_path = aux_data

    delay_us = delay_day * DAY_S * 1e6
    range = delay_day * C
    doppler_shift = -tx_freq * d_delay
    rate = d_delay * C
    tx_azimuth, tx_elevation, rx_azimuth, rx_elevation = _pointing_angles_from_paths(tx, rx, up_path, down_path)

    return RadarObservation(
        up_path=up_path,
        down_path=down_path,
        delay=delay_us,
        range=range,
        doppler_shift=doppler_shift,
        rate=rate,
        tx_azimuth=tx_azimuth,
        tx_elevation=tx_elevation,
        rx_azimuth=rx_azimuth,
        rx_elevation=rx_elevation,
    )


def compute_radar_obs_single_reorder(target: SmallBody, rx: Site, tx: Site, tx_freq: float, t_rec: Time,
                                     context: LightTimeContext) -> RadarObservation:
    """Reorder ``compute_radar_obs_single`` arguments for batch dispatch."""
    return compute_radar_obs_single(t_rec, rx, tx, tx_freq, target, context)


@eqx.filter_jit
def compute_radar_obs(t_rec: Time, target: SmallBody, rx: Site, tx: Site, tx_freq: Array,
                      context: LightTimeContext, grid: bool = False) -> RadarObservation:
    """Compute two-way radar observation.

    Parameters
    ----------
    t_rec : Time
        Receive epochs at the receiver site.
    target : SmallBody
        Target body with the propagated trajectory.
    rx : Site
        Receiver sites.
    tx : Site
        Transmitter sites.
    tx_freq : Array
        Transmit frequencies in ``Hz``.
    context : LightTimeContext
        Shared light-time solver context.
    grid : bool, default=False
        If ``False``, use point-wise broadcasting. If ``True``, use the Cartesian product of the input batches.

    Returns
    -------
    RadarObservation
        Two-way radar measurements and the solved light paths.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

    Notes
    -----
    Vectorize :func:`compute_radar_obs_single`.
    """
    wrapper = partial(compute_radar_obs_single_reorder, context=context)
    if not grid:
        return safe_dispatch(wrapper, (0, 0, 0, 0, 0), target, rx, tx, tx_freq, t_rec)
    else:
        return safe_cartesian_dispatch(wrapper, ((0,), (target,)), ((0,), (rx,)), ((0, 0), (tx, tx_freq)),
                                       ((0,), (t_rec,)))
