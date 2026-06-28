"""Radar delay and Doppler measurement reduction.

This module builds two-way radar observables from light-time solutions. The main entry point solves the receive-side down leg and the transmit-side up leg, then combines them into round-trip delay, range, Doppler shift, and range rate.

Receiver times are given in ``UT1``, ``UTC``, or mixed ``UT``. The target state is handled in ``TDB`` through :mod:`difforb.astrometry.reduction.lt`.
"""

from functools import partial

import equinox as eqx
import jax
from jax import Array
from jaxtyping import Float

from difforb.body.site import Site
from difforb.astrometry.reduction.lt import LightPath, LightTimeContext, down_leg_light_time_single, up_leg_light_time_single
from difforb.body.smallbody import SmallBody
from difforb.core.constants import C, DAY_S
from difforb.core.time.timescale import Time
from difforb.report.text import build_repr, format_float_array, format_shape

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
    """

    up_path: LightPath
    down_path: LightPath
    delay: Float[Array, "..."]  # in us
    range: Float[Array, "..."]  # in au
    doppler_shift: Float[Array, "..."]  # in Hz
    rate: Float[Array, "..."]  # in au/day

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
                ("up_path", self.up_path.__class__.__name__),
                ("down_path", self.down_path.__class__.__name__),
            ],
        )


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

    return RadarObservation(
        up_path=up_path,
        down_path=down_path,
        delay=delay_us,
        range=range,
        doppler_shift=doppler_shift,
        rate=rate
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
