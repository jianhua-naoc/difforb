"""Optical measurement reduction.

This module builds geometric and astrometric optical light paths between an observing site and a small body. It also provides helpers for solar light bending and stellar aberration on a solved light path. The main outputs are :class:`LightPath` objects in ``BCRS``.
"""

from functools import partial

import jax
from jax import Array, numpy as jnp
from jaxtyping import Float
import equinox as eqx

from difforb.body.ephbody import EphemerisBody
from difforb.body.site import Site
from difforb.astrometry.reduction.lt import LightPath, LightTimeContext, down_leg_light_time_single
from difforb.body.smallbody import SmallBody

from difforb.core.constants import C
from difforb.core.state.frame import BCRS

from difforb.core.batch import safe_dispatch, safe_cartesian_dispatch
from difforb.core.time.timescale import Time

jax.config.update("jax_enable_x64", True)


# ======================================================
# 1. Light bending & Stellar Aberration
# ======================================================

def correct_light_bending_single(sun: EphemerisBody, light_path: LightPath) -> Float[Array, "3"]:
    """Apply solar light bending.

    Parameters
    ----------
    sun : EphemerisBody
        Solar ephemeris body.
    light_path : LightPath
        Solved optical path in ``BCRS``.

    Returns
    -------
    Float[Array, "3"]
        Corrected path vector in ``au``.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Sec. 7.2.4.
    """
    sun_pos_trm = sun._bcrs_pos_jd(light_path.start.tdb.jd1, light_path.start.tdb.jd2)
    sun_pos_rec = sun._bcrs_pos_jd(light_path.end.tdb.jd1, light_path.end.tdb.jd2)
    path_pos = light_path.pos
    path_dist = light_path.dist
    path_uv = path_pos / path_dist
    sun2target_pos = light_path.start.pos - sun_pos_trm
    sun2target_dist = jnp.linalg.norm(sun2target_pos)
    sun2target_uv = sun2target_pos / sun2target_dist
    sun2site_pos = light_path.end.pos - sun_pos_rec
    sun2site_dist = jnp.linalg.norm(sun2site_pos)
    sun2site_uv = sun2site_pos / sun2site_dist
    g1 = (2. * sun.gm) / (C * C * sun2site_dist)
    g2 = 1. + jnp.dot(sun2target_uv, sun2site_uv)
    cor_path_pos = path_dist * (
            path_uv + g1 / g2 * (
            jnp.dot(path_uv, sun2target_uv) * sun2site_uv - jnp.dot(
        sun2site_uv, path_uv) * sun2target_uv
    ))
    return cor_path_pos


def correct_light_bending(sun: EphemerisBody, light_path: LightPath) -> Float[Array, "... 3"]:
    """Apply solar light bending.

    Parameters
    ----------
    sun : EphemerisBody
        Solar ephemeris body.
    light_path : LightPath
        Solved optical path or batch of paths in ``BCRS``.

    Returns
    -------
    Float[Array, "... 3"]
        Corrected path vector in ``au``.

    Notes
    -----
    Vectorize :func:`correct_light_bending_single`.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Sec. 7.2.4.
    """
    wrapper = partial(correct_light_bending_single, sun)
    return safe_dispatch(wrapper, (0,), light_path)


def correct_stellar_aberration_single(light_path: LightPath) -> Float[Array, "3"]:
    """Apply stellar aberration to one solved light path.

    Parameters
    ----------
    light_path : LightPath
        Solved optical path in ``BCRS``.

    Returns
    -------
    Float[Array, "3"]
        Corrected path vector in ``au``.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.40.
    """
    u = light_path.pos / light_path.dist
    v_over_c = light_path.end.vel / C
    v2 = jnp.dot(v_over_c, v_over_c)
    beta_inv = jnp.sqrt(1. - v2)
    u_dot_v = jnp.dot(u, v_over_c)
    numerator = beta_inv * u + v_over_c + (u_dot_v * v_over_c) / (1. + beta_inv)
    denominator = 1. + u_dot_v
    u1 = numerator / denominator
    return u1 * light_path.dist


# =================================================================
# 2. Optical Observation Model: geometric, astrometric
# =================================================================


def compute_geometric_vector_single(t_obs: Time,
                                    site: Site, target: SmallBody) -> LightPath:
    """Compute geometric optical path at the given observed epoch ``t_obs``.

    Parameters
    ----------
    t_obs : Time
        Observation epoch.
    site : Site
        Observing site.
    target : SmallBody
        Target body with the propagated trajectory.

    Returns
    -------
    LightPath
        Geometric path from the site to the target, without light-time correction.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

    Notes
    -----
    The site state is evaluated at the observed epoch. The target state is evaluated at the same epoch converted to ``TDB``. This is a geometric line of sight, not an astrometric one.
    """
    site_state = site.state(t_obs, frame=BCRS)
    t_obs_tdb = site_state.tdb
    target_state = target.state(t_obs_tdb, frame=BCRS)
    light_path_pos = target_state.pos - site_state.pos
    light_path_dist = jnp.linalg.norm(light_path_pos, axis=-1)
    return LightPath(
        pos=light_path_pos,
        vel=target_state.vel - site_state.vel,
        dist=light_path_dist,
        lt=light_path_dist / C,
        start=target_state,
        end=site_state,
    )


@eqx.filter_jit
def compute_geometric_vector(t_obs: Time, site: Site, target: SmallBody, grid: bool = False) -> LightPath:
    """Compute geometric optical path at the given observed epoch ``t_obs``.

    Parameters
    ----------
    t_obs : Time
        Observation epoch.
    site : Site
        Observing site.
    target : SmallBody
        Target body with the propagated trajectory.
    grid : bool, default=False
        If ``True``, use the Cartesian product of target, site, and time batches. If ``False``, use point-wise broadcasting.

    Returns
    -------
    LightPath
        Geometric path or batch of paths.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

    Notes
    -----
    Vectorize :func:`compute_geometric_vector_single`.
    """
    if not grid:
        return safe_dispatch(compute_geometric_vector_single, (0, 0, 0), t_obs, site, target)
    else:
        wrapper = lambda _target, _site, _t_obs: compute_geometric_vector_single(_t_obs, _site, _target)
        return safe_cartesian_dispatch(wrapper, ((0,), (target,)), ((0,), (site,)), ((0,), (t_obs,)))


def compute_astrometric_vector_single(t_obs: Time, site: Site, target: SmallBody,
                                      context: LightTimeContext) -> LightPath:
    """Compute astrometric optical path at the given observed epoch ``t_obs``.

    Parameters
    ----------
    t_obs : Time
        Observation epoch.
    site : Site
        Observing site.
    target : SmallBody
        Target body with the propagated trajectory.
    context : LightTimeContext
        Light-time correction context.

    Returns
    -------
    LightPath
        Astrometric path with down-leg light-time correction.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

    Notes
    -----
    This function calls :func:`down_leg_light_time_single` with zero transmitter frequency, so the result is the solved down-leg optical path.
    """
    light_path = down_leg_light_time_single(t_obs, site, target, context, 0., 1e-16)
    return light_path


def compute_astrometric_vector_single_reorder(target: SmallBody, site: Site, t_obs: Time,
                                              context: LightTimeContext) -> LightPath:
    """Reorder arguments for batch dispatch."""
    return compute_astrometric_vector_single(t_obs, site, target, context)


@eqx.filter_jit
def compute_astrometric_vector(t_obs: Time, site: Site, target: SmallBody,
                               context: LightTimeContext, grid: bool = False) -> LightPath:
    """Compute astrometric optical path at the given observed epoch ``t_obs``.

    Parameters
    ----------
    t_obs : Time
        Observation epoch.
    site : Site
        Observing site.
    target : SmallBody
        Target body with the propagated trajectory.
    context : LightTimeContext
        Light-time correction context.
    grid : bool, default=False
        If ``True``, use the Cartesian product of target, site, and time batches. If ``False``, use point-wise broadcasting.

    Returns
    -------
    LightPath
        Astrometric path or batch of paths.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

    Notes
    -----
    Vectorize :func:`compute_astrometric_vector_single`.
    """
    wrapper = partial(compute_astrometric_vector_single_reorder, context=context)
    if not grid:
        return safe_dispatch(wrapper, (0, 0, 0), target, site, t_obs)
    else:
        return safe_cartesian_dispatch(wrapper, ((0,), (target,)), ((0,), (site,)), ((0,), (t_obs,)))
