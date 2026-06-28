"""Observation preparation and triplet sampling for initial orbit determination.

This module converts public observation containers into the compact numerical
arrays consumed by the angle-only IOD core. It builds one globally time-sorted
optical arc from the unified optical table, derives heliocentric observer
positions and line-of-sight vectors, resolves the effective sampling window,
and gathers sampled triplets into batched arrays for the numerical solver.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jax import Array
from jaxtyping import Float, Int

from difforb.astrometry.data import ObservationData
from difforb.body.ephbody import EphemerisBody
from difforb.body.site import Site
from difforb.core.state.frame import HELIO_ICRS
from difforb.utils import sph2car

jax.config.update("jax_enable_x64", True)

DEFAULT_T2_SAMPLING_RADIUS = 2


class OpticalIODInputs(NamedTuple):
    """Prepared optical observations for angle-only initial orbit determination.

    Parameters
    ----------
    tdb_jd1 : Float[Array, "N"]
        Observation epochs in split Julian-date form. The time scale is
        ``TDB``.
    tdb_jd2 : Float[Array, "N"]
        Low-order split Julian-date term in ``TDB``.
    tdb_jd : Float[Array, "N"]
        Full ``TDB`` Julian dates.
    site_pos : Float[Array, "N 3"]
        Observer positions in ``HELIO_ICRS``, in ``au``.
    los_unit : Float[Array, "N 3"]
        Line-of-sight unit vectors in ``HELIO_ICRS``.
    input_indices : Int[Array, "N"]
        Observation indices in the original mixed input order. These indices
        are aligned with the globally time-sorted arrays stored in this object.
    """

    tdb_jd1: Float[Array, "N"]
    tdb_jd2: Float[Array, "N"]
    tdb_jd: Float[Array, "N"]
    site_pos: Float[Array, "N 3"]
    los_unit: Float[Array, "N 3"]
    input_indices: Int[Array, "N"]


class IODSamplingWindow(NamedTuple):
    """Sampling window for IOD triplet selection."""

    start_idx: int
    end_idx: int
    center_idx: int


class IODTripletBatch(NamedTuple):
    """Batched triplet arrays passed to the numerical IOD solver.

    Parameters
    ----------
    indices : Int[Array, "M 3"]
        Observation indices of each sampled triplet.
    site_pos : Float[Array, "M 3 3"]
        Observer positions at ``t1``, ``t2``, and ``t3`` in ``HELIO_ICRS``,
        in ``au``.
    los_unit : Float[Array, "M 3 3"]
        Line-of-sight unit vectors aligned with ``site_pos``.
    tdb_jd1 : Float[Array, "M 3"]
        High-order split Julian-date term of the triplet epochs in ``TDB``.
    tdb_jd2 : Float[Array, "M 3"]
        Low-order split Julian-date term of the triplet epochs in ``TDB``.
    input_indices : Int[Array, "M 3"]
        Original mixed-input indices of the sampled observations.
    """

    indices: Int[Array, "M 3"]
    site_pos: Float[Array, "M 3 3"]
    los_unit: Float[Array, "M 3 3"]
    tdb_jd1: Float[Array, "M 3"]
    tdb_jd2: Float[Array, "M 3"]
    input_indices: Int[Array, "M 3"]


def build_optical_iod_inputs(
        data: ObservationData,
        sun: EphemerisBody,
        earth: EphemerisBody | None = None,
) -> OpticalIODInputs:
    """Build one sorted optical arc for angle-only initial orbit determination.

    Parameters
    ----------
    data : ObservationData
        Single-target observation bundle. Radar observations are ignored by
        this preparation step.
    sun : EphemerisBody
        Solar-system body used to build ``HELIO_ICRS`` observer states.
    earth : EphemerisBody or None, optional
        Earth body used when converting observer states from ``GCRS`` to
        ``HELIO_ICRS``.

    Returns
    -------
    OpticalIODInputs
        Prepared and globally time-sorted optical arrays for candidate triplet
        sampling and IOD.
    """
    if earth is None:
        earth = EphemerisBody("earth")
    optical_obs = data.optical
    site = Site.from_code(optical_obs.rx_codes)
    site_state = site.state(optical_obs.t, frame=HELIO_ICRS, sun=sun, earth=earth)
    tdb_jd1 = site_state.tdb.jd1
    tdb_jd2 = site_state.tdb.jd2
    site_pos = site_state.pos
    los_unit = sph2car(optical_obs.values[:, 0], optical_obs.values[:, 1])
    input_indices = jnp.asarray(optical_obs.input_indices, dtype=jnp.int32)
    sort_idx = jnp.argsort(tdb_jd1 + tdb_jd2)
    tdb_jd1 = tdb_jd1[sort_idx]
    tdb_jd2 = tdb_jd2[sort_idx]
    site_pos = site_pos[sort_idx]
    los_unit = los_unit[sort_idx]
    input_indices = input_indices[sort_idx]

    return OpticalIODInputs(
        tdb_jd1=tdb_jd1,
        tdb_jd2=tdb_jd2,
        tdb_jd=tdb_jd1 + tdb_jd2,
        site_pos=site_pos,
        los_unit=los_unit,
        input_indices=input_indices,
    )


def resolve_sampling_window(
        tdb_jd: Float[Array, "N"],
        max_arc_days: float) -> IODSamplingWindow:
    """Resolve the effective observation window used for triplet sampling.

    Parameters
    ----------
    tdb_jd : Float[Array, "N"]
        Sorted observation epochs in ``TDB`` Julian days.
    max_arc_days : float
        Requested sampling-window width in days.

    Returns
    -------
    IODSamplingWindow
        Effective inclusive window bounds and the derived middle index.

    Raises
    ------
    ValueError
        Raised when the filtered observation arc does not contain enough
        observations for angle-only IOD.

    Notes
    -----
    The sampling window is always centered on the middle sorted observation.
    If the nominal window contains fewer than three observations, it is
    expanded to the full filtered arc.
    """
    num_observations = len(tdb_jd)
    if num_observations < 3:
        raise ValueError(f"IOD requires at least 3 observations, but only {num_observations} provided.")

    center_tdb_jd = tdb_jd[(num_observations - 1) // 2]
    half_window_days = max_arc_days / 2.0
    window_start_idx = int(jnp.searchsorted(tdb_jd, center_tdb_jd - half_window_days, side="left"))
    window_end_idx = int(jnp.searchsorted(tdb_jd, center_tdb_jd + half_window_days, side="right")) - 1

    num_window_observations = window_end_idx - window_start_idx + 1
    if num_window_observations < 3:
        window_start_idx = 0
        window_end_idx = num_observations - 1

    center_idx = window_start_idx + (window_end_idx - window_start_idx) // 2
    center_idx = int(jnp.clip(center_idx, window_start_idx + 1, window_end_idx - 1))

    return IODSamplingWindow(
        start_idx=window_start_idx,
        end_idx=window_end_idx,
        center_idx=center_idx,
    )


def sample_triplet_indices(
        sampling_window: IODSamplingWindow,
        num_candidates: int,
        solve_key: Array) -> Int[Array, "N 3"]:
    """Sample random observation triplets inside the effective IOD window.

    Parameters
    ----------
    sampling_window : IODSamplingWindow
        Effective observation window used for sampling.
    num_candidates : int
        Number of candidate triplets to draw.
    solve_key : Array
        JAX pseudo-random key used for the sampling draw.

    Returns
    -------
    Int[Array, "N 3"]
        Sampled triplet indices in ascending time order.

    Notes
    -----
    The middle observation of each triplet is sampled inside a fixed index
    radius around the window center.
    """
    key_t1, key_t2, key_t3 = jrandom.split(solve_key, 3)
    t2_min_idx = jnp.maximum(sampling_window.start_idx + 1, sampling_window.center_idx - DEFAULT_T2_SAMPLING_RADIUS)
    t2_max_idx = jnp.minimum(sampling_window.end_idx - 1, sampling_window.center_idx + DEFAULT_T2_SAMPLING_RADIUS)

    t2_random = jrandom.uniform(key_t2, shape=(num_candidates,))
    t2_idx = jnp.floor(t2_min_idx + t2_random * (t2_max_idx - t2_min_idx + 1)).astype(jnp.int32)

    t1_random = jrandom.uniform(key_t1, shape=(num_candidates,))
    t1_idx = jnp.floor(sampling_window.start_idx + t1_random * (t2_idx - sampling_window.start_idx)).astype(jnp.int32)

    t3_random = jrandom.uniform(key_t3, shape=(num_candidates,))
    t3_idx = jnp.floor((t2_idx + 1) + t3_random * (sampling_window.end_idx - t2_idx)).astype(jnp.int32)

    return jnp.stack([t1_idx, t2_idx, t3_idx], axis=1)


def build_triplet_batch(
        optical_inputs: OpticalIODInputs,
        triplet_indices: Int[Array, "N 3"]) -> IODTripletBatch:
    """Gather batched triplet arrays from prepared optical observations.

    Parameters
    ----------
    optical_inputs : OpticalIODInputs
        Prepared observer, line-of-sight, and epoch arrays.
    triplet_indices : Int[Array, "N 3"]
        Observation indices of the sampled triplets.

    Returns
    -------
    IODTripletBatch
        Batched triplet arrays aligned with the sampled indices.
    """
    return IODTripletBatch(
        indices=triplet_indices,
        site_pos=jnp.take(optical_inputs.site_pos, triplet_indices, axis=0),
        los_unit=jnp.take(optical_inputs.los_unit, triplet_indices, axis=0),
        tdb_jd1=jnp.take(optical_inputs.tdb_jd1, triplet_indices, axis=0),
        tdb_jd2=jnp.take(optical_inputs.tdb_jd2, triplet_indices, axis=0),
        input_indices=jnp.take(optical_inputs.input_indices, triplet_indices, axis=0),
    )
