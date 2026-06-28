"""SPK kernel loading and Chebyshev-segment evaluation.

This module provides the :class:`Ephemeris` container used to load one or more
SPK kernels, merge path segments between NAIF bodies, and evaluate those
segments in ``TDB``. The public ``load_window`` contract is expressed with
scalar :class:`difforb.core.time.timescale.TDBView` objects, while the runtime
storage is normalized to Python floats for kernel filtering and caching.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Dict, Tuple, List, Union, Optional

import jplephem.spk
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, ArrayLike
from jax.tree_util import register_pytree_node_class
import equinox as eqx

from difforb.core.batch import safe_dispatch
from difforb.core.constants import J2000, DAY_S
from difforb.core.validate import validate_timeview

from difforb.spk.naif import naif

if TYPE_CHECKING:
    from difforb.core.time.timescale import TDBView

jax.config.update("jax_enable_x64", True)

J2000_JD = float(J2000)
DAY_S_F64 = float(DAY_S)


def _normalize_load_window(load_window: Optional[Tuple[TDBView, TDBView]]) -> Optional[Tuple[float, float]]:
    """Build a Python-float load window in seconds relative to ``J2000``.

    Parameters
    ----------
    load_window : tuple[TDBView, TDBView] or None
        Inclusive kernel-load window expressed in scalar ``TDB`` epochs.

    Returns
    -------
    tuple[float, float] or None
        Start and end epoch in seconds relative to ``J2000``.

    Raises
    ------
    TypeError
        If either window endpoint is not a ``TDBView``.
    ValueError
        If either endpoint carries a batch shape.
    """
    if load_window is None:
        return None
    from difforb.core.time.timescale import TDBView

    start_tdb, end_tdb = load_window
    validate_timeview(start_tdb, TDBView, "load_window[0]")
    validate_timeview(end_tdb, TDBView, "load_window[1]")
    if start_tdb.shape != ():
        raise ValueError("`load_window[0]` must be a scalar TDBView.")
    if end_tdb.shape != ():
        raise ValueError("`load_window[1]` must be a scalar TDBView.")
    start_jd = float(np.asarray(start_tdb.jd1).reshape(-1)[0] + np.asarray(start_tdb.jd2).reshape(-1)[0])
    end_jd = float(np.asarray(end_tdb.jd1).reshape(-1)[0] + np.asarray(end_tdb.jd2).reshape(-1)[0])
    return (
        (start_jd - J2000_JD) * DAY_S_F64,
        (end_jd - J2000_JD) * DAY_S_F64,
    )


def pad_arrays(array_list: List[np.ndarray]) -> np.ndarray:
    """Pad coefficient arrays to the same last-axis length.

    Parameters
    ----------
    array_list : list[np.ndarray]
        Arrays that differ only on the last axis.

    Returns
    -------
    np.ndarray
        Stacked array. Padded values on the last axis are zero.
    """
    # Only the last axis is padded.
    shapes = [arr.shape for arr in array_list]
    max_len = max(shape[-1] for shape in shapes)

    final_shape = array_list[0].shape
    final_shape = (len(array_list),) + final_shape[:-1] + (max_len,)

    output = np.full(final_shape, 0., dtype=array_list[0].dtype)

    for i, arr in enumerate(array_list):
        output[i, ..., :arr.shape[-1]] = arr

    return output


def compute_chebyshev_polynomial(tdb_scale: float, coefficients: Array) -> Array:
    """
    Compute the Chebyshev polynomial by the Clenshaw recurrence.
    """
    component_num = coefficients.shape[1]
    bk1 = jnp.zeros(component_num)
    bk2 = jnp.zeros(component_num)
    double_tdb_scale = 2. * tdb_scale

    def body_func(carry, coeff_k):
        bk1, bk2 = carry
        bk = coeff_k + double_tdb_scale * bk1 - bk2
        return (bk, bk1), None

    init_carry = (bk1, bk2)

    (bk1, bk2), _ = jax.lax.scan(body_func, init_carry, coefficients[:-1])

    return coefficients[-1] + tdb_scale * bk1 - bk2


def compute_position_single(seg_starts_sec: Array, chunk_nums: Array, chunk_lengths_sec: Array, coefficients: Array, tdb_jd1: float,
                            tdb_jd2: float) -> Array:
    """Evaluate a merged SPK segment at one ``TDB`` epoch.

    Parameters
    ----------
    seg_starts_sec : Array
        Segment start times in seconds from ``J2000``.
    chunk_nums : Array
        Number of valid chunks in each stored segment.
    chunk_lengths_sec : Array
        Chunk lengths in seconds.
    coefficients : Array
        Chebyshev coefficients with shape ``N_segment degree component N_chunk``.
    tdb_jd1, tdb_jd2 : float
        Two parts of the ``TDB`` Julian Date.

    Returns
    -------
    Array
        Position vector in kilometers.
    """
    coefficients = jax.lax.stop_gradient(coefficients)
    seg_starts_sec = jax.lax.stop_gradient(seg_starts_sec)
    chunk_nums = jax.lax.stop_gradient(chunk_nums)
    tdb_sec = (tdb_jd1 - J2000) * DAY_S + tdb_jd2 * DAY_S
    # 1. Find the segment covered the input time
    seg_idx = jnp.searchsorted(seg_starts_sec, tdb_sec, side='right') - 1
    seg_idx = jnp.clip(seg_idx, 0, seg_starts_sec.shape[0] - 1)
    # 2. Extract parameters of the segment
    coefficients = coefficients[seg_idx]
    seg_start_sec = seg_starts_sec[seg_idx]
    chunk_length_sec = chunk_lengths_sec[seg_idx]
    # 3. Compute high-precision relative time
    dt_seg1 = (tdb_jd1 - J2000) * DAY_S - seg_start_sec
    dt_seg2 = tdb_jd2 * DAY_S
    dt_seg = dt_seg1 + dt_seg2
    # 4. Find the chunk in the segment covered the input time
    chunk_idx = jnp.floor(dt_seg / chunk_length_sec).astype(int)
    chunk_idx = jnp.clip(chunk_idx, 0, chunk_nums[seg_idx] - 1)
    coefficient = coefficients[:, :, chunk_idx]
    # 5. Compute normalized time
    t_chunk_start_sec = chunk_idx * chunk_length_sec
    dt_chunk = dt_seg - t_chunk_start_sec
    # 5. Compute position by evaluate Chebyshev polynomial
    tdb_scale = (2. * dt_chunk / chunk_length_sec) - 1.
    tdb_scale = jnp.clip(tdb_scale, -1.0, 1.0)
    return compute_chebyshev_polynomial(tdb_scale, coefficient)


def compute_pv_single(seg_starts_sec: Array, chunk_nums: Array, chunk_lengths_sec: Array, coefficients: Array, tdb_jd1: float,
                      tdb_jd2: float) -> \
        Tuple[
            Array, Array]:
    """Evaluate position and velocity from a merged SPK segment.

    Parameters
    ----------
    seg_starts_sec : Array
        Segment start times in seconds from ``J2000``.
    chunk_nums : Array
        Number of valid chunks in each stored segment.
    chunk_lengths_sec : Array
        Chunk lengths in seconds.
    coefficients : Array
        Chebyshev coefficients.
    tdb_jd1, tdb_jd2 : float
        Two parts of the ``TDB`` Julian Date.

    Returns
    -------
    tuple[Array, Array]
        Position in kilometers and velocity in kilometers per day.
    """
    pos_fn = partial(compute_position_single, seg_starts_sec, chunk_nums, chunk_lengths_sec, coefficients, tdb_jd1)
    pos, vel = jax.jvp(pos_fn, (tdb_jd2,), (1.,))
    return pos, vel


def compute_pva_single(seg_starts_sec: Array, chunk_nums: Array, chunk_lengths_sec: Array, coefficients: Array, tdb_jd1: float,
                       tdb_jd2: float) -> \
        Tuple[
            Array, Array, Array]:
    """Evaluate position, velocity, and acceleration from a merged SPK segment.

    Parameters
    ----------
    seg_starts_sec : Array
        Segment start times in seconds from ``J2000``.
    chunk_nums : Array
        Number of valid chunks in each stored segment.
    chunk_lengths_sec : Array
        Chunk lengths in seconds.
    coefficients : Array
        Chebyshev coefficients.
    tdb_jd1, tdb_jd2 : float
        Two parts of the ``TDB`` Julian Date.

    Returns
    -------
    tuple[Array, Array, Array]
        Position in kilometers, velocity in kilometers per day, and acceleration in kilometers per day squared.
    """
    pos_fn = partial(compute_position_single, seg_starts_sec, chunk_nums, chunk_lengths_sec, coefficients, tdb_jd1)

    def pos_vel_fn(t):
        return jax.jvp(pos_fn, (t,), (1.0,))

    (pos, vel), (_, acc) = jax.jvp(pos_vel_fn, (tdb_jd2,), (1.0,))

    return pos, vel, acc


@jax.jit
def compute_position(seg_starts_sec: Array, chunk_nums: Array, chunk_lengths_sec: Array, coefficients: Array, tdb_jd1: Float[Array, "..."],
                     tdb_jd2: Float[Array, "..."]) -> Float[Array, "... 3"]:
    """Evaluate a merged SPK segment at ``TDB`` epochs.

    Parameters
    ----------
    seg_starts_sec : Array
        Segment start times in seconds from ``J2000``.
    chunk_nums : Array
        Number of valid chunks in each stored segment.
    chunk_lengths_sec : Array
        Chunk lengths in seconds.
    coefficients : Array
        Chebyshev coefficients.
    tdb_jd1, tdb_jd2 : Float[Array, "..."]
        Two parts of each ``TDB`` Julian Date.

    Returns
    -------
    Float[Array, "... 3"]
        Position vectors in kilometers.

    Notes
    -----
    Vectorize :func:`compute_position_single`.
    """
    scalar_fn = partial(compute_position_single, seg_starts_sec, chunk_nums, chunk_lengths_sec, coefficients)
    return safe_dispatch(scalar_fn, (0, 0), tdb_jd1, tdb_jd2)


@jax.jit
def compute_pv(seg_starts_sec: Array, chunk_nums: Array, chunk_lengths_sec: Array, coefficients: Array,
               tdb_jd1: ArrayLike, tdb_jd2: ArrayLike) -> Tuple[Array, Array]:
    """Evaluate position and velocity at ``TDB`` epochs.

    Parameters
    ----------
    seg_starts_sec : Array
        Segment start times in seconds from ``J2000``.
    chunk_nums : Array
        Number of valid chunks in each stored segment.
    chunk_lengths_sec : Array
        Chunk lengths in seconds.
    coefficients : Array
        Chebyshev coefficients.
    tdb_jd1, tdb_jd2 : ArrayLike
        Two parts of each ``TDB`` Julian Date.

    Returns
    -------
    tuple[Array, Array]
        Position in kilometers and velocity in kilometers per day.

    Notes
    -----
    Vectorize :func:`compute_pv_single`.
    """
    scalar_fn = partial(compute_pv_single, seg_starts_sec, chunk_nums, chunk_lengths_sec, coefficients)
    return safe_dispatch(scalar_fn, (0, 0), tdb_jd1, tdb_jd2)


@jax.jit
def compute_pva(seg_starts_sec: Array, chunk_nums: Array, chunk_lengths_sec: Array, coefficients: Array,
                tdb_jd1: ArrayLike, tdb_jd2: ArrayLike) -> Tuple[Array, Array, Array]:
    """Evaluate position, velocity, and acceleration at ``TDB`` epochs.

    Parameters
    ----------
    seg_starts_sec : Array
        Segment start times in seconds from ``J2000``.
    chunk_nums : Array
        Number of valid chunks in each stored segment.
    chunk_lengths_sec : Array
        Chunk lengths in seconds.
    coefficients : Array
        Chebyshev coefficients.
    tdb_jd1, tdb_jd2 : ArrayLike
        Two parts of each ``TDB`` Julian Date.

    Returns
    -------
    tuple[Array, Array, Array]
        Position in kilometers, velocity in kilometers per day, and acceleration in kilometers per day squared.

    Notes
    -----
    Vectorize :func:`compute_pva_single`.
    """
    scalar_fn = partial(compute_pva_single, seg_starts_sec, chunk_nums, chunk_lengths_sec, coefficients)
    return safe_dispatch(scalar_fn, (0, 0), tdb_jd1, tdb_jd2)


@register_pytree_node_class
@dataclass
class MergedSegment:
    """
    A JAX-compatible container for merged SPK ephemeris segments.

    This class holds the Chebyshev coefficients and time metadata required to
    compute positions (and velocities) for a celestial body.
    """
    types: Array  # Type for each segment (2: pos only, 3: pos and vel)
    center_ids: Array  # NAIF code for each segment's center
    target_ids: Array  # NAIF code for each segment's target
    seg_starts_sec: Array  # Start time for each segment
    seg_ends_sec: Array  # End time for each segment
    component_nums: Array  # 3 (Type 2) or 6 (Type 6)
    chunk_nums: jnp.ndarray  # Number of logical chunks in each segment
    chunk_lengths_sec: jnp.ndarray  # Duration of each chunk in seconds
    coefficients: jnp.ndarray  # Chebyshev coefficients [deg_num, component_num, chunk_num]

    def tree_flatten(self):
        """Flatten the segment for JAX PyTree use.

        Returns
        -------
        tuple
            PyTree children and auxiliary data.
        """
        aux_data = None
        children = (self.types, self.center_ids, self.target_ids, self.seg_starts_sec, self.seg_ends_sec,
                    self.component_nums,
                    self.chunk_nums,
                    self.chunk_lengths_sec, self.coefficients)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Build a segment from JAX PyTree data.

        Parameters
        ----------
        aux_data : Any
            Auxiliary PyTree data. This class does not use it.
        children : tuple
            PyTree children returned by :meth:`tree_flatten`.

        Returns
        -------
        MergedSegment
            Rebuilt segment.
        """
        return cls(*children)

    @classmethod
    def from_jplephem_segments(cls, segments: List[jplephem.spk.Segment],
                               load_window: Optional[Tuple[float, float]] = None) -> 'MergedSegment':
        """
        Factory method to create a MergedSegment from a list of raw jplephem segments.
        Parses the list of segments formed by jplephem, extracts coefficients, and formats them for JAX computation.
        """
        seg_data = {
            "types": [], "center_ids": [], "target_ids": [],
            "seg_starts": [], "seg_ends": [],
            "chunk_lengths": [], "chunk_nums": [], "component_nums": [],
            "coefficients": []
        }
        # 1. Sort the segments by its start time
        segments = sorted(segments, key=lambda s: s.start_second)
        # 2. Set load window
        req_start, req_end = (-np.inf, np.inf)
        if load_window is not None:
            req_start, req_end = load_window
        # 3. Extract properties and data of segments
        for seg in segments:
            # Keep metadata filtering on Python/NumPy scalars so first-time segment
            # loading remains valid even when the caller is being traced by JAX.
            if seg.end_second < req_start or seg.start_second > req_end:
                continue

            type = seg.data_type
            # Read metadata from the end of the DAF record
            # intlen: Duration of the chunk in this segment (in seconds)
            # rsize: Record size (number of double precision floats per record) of the chunk in this segment
            # n: Number of chunks in this segment
            intlen, rsize, n = seg.daf.read_array(seg.end_i - 2, seg.end_i)
            rsize, n = int(rsize), int(n)

            # Determine component count based on SPK Type
            # Type 2: Position only (3 components: x, y, z)
            # Type 3: Position + Velocity (6 components: x, y, z, vx, vy, vz)
            if type == 2:
                component_num = 3
            elif type == 3:
                component_num = 6
            else:
                raise NotImplementedError("Only support Type 2 and 3")

            # Calculate the degree of the Chebyshev polynomial
            # Formula: rsize = 2 (mid + radius) + (component_num * degree)
            deg = int((rsize - 2) // component_num)

            # Map the raw binary data to a numpy array (Memory mapping for efficiency)
            # Exclude the last 4 bytes which contain the metadata read above
            records = seg.daf.map_array(seg.start_i, seg.end_i - 4)

            # Reshape raw data into records: (Number of Chunks, Data Size per Chunk)
            records.shape = (n, rsize)

            # Extract time parameters: Midpoint and Radius
            mid = records[:, 0]
            radius = records[:, 1]

            chunk_starts = mid - radius
            chunk_ends = mid + radius

            # Filter at Chunk level
            mask = (chunk_ends >= req_start) & (chunk_starts <= req_end)
            if not np.any(mask):
                continue
            filtered_n = np.sum(mask)

            # Extract coefficients
            # Raw shape: (n, component_num * deg)
            coefficients = records[mask, 2:]
            # Reshape to 3D: (Number of Chunks, Components, Polynomial Degree)
            coefficients.shape = (filtered_n, component_num, deg)
            # Transpose dimensions to optimize for vectorized evaluation
            # From: (n, component_num, deg)
            # To:   (deg, component_num, n)
            # This layout aligns with the memory access pattern of the Clenshaw recurrence.
            coefficients = np.transpose(coefficients, (2, 1, 0))
            # Flip the degree axis (Axis 0)
            # Converts from [Low Degree ... High Degree] to [High Degree ... Low Degree]
            # Essential for the Clenshaw algorithm which iterates from high order down to 0.
            coefficients = coefficients[::-1]

            seg_data["types"].append(type)
            seg_data["center_ids"].append(seg.center)
            seg_data["target_ids"].append(seg.target)
            seg_data["seg_starts"].append(chunk_starts[mask][0])
            seg_data["seg_ends"].append(chunk_ends[mask][-1])
            seg_data["component_nums"].append(component_num)
            seg_data["chunk_nums"].append(filtered_n)
            seg_data["chunk_lengths"].append(intlen)
            seg_data["coefficients"].append(coefficients)

        return cls(jnp.array(seg_data['types']), jnp.array(seg_data['center_ids']), jnp.array(seg_data['target_ids']),
                   jnp.array(seg_data['seg_starts']), jnp.array(seg_data['seg_ends']),
                   jnp.array(seg_data['component_nums']),
                   jnp.array(seg_data['chunk_nums']), jnp.array(seg_data['chunk_lengths']),
                   jnp.array(pad_arrays(seg_data['coefficients'])))

    @property
    def seg_starts_jd(self) -> Array:
        """Return segment start times as ``TDB`` Julian Dates."""
        return self.seg_starts_sec / DAY_S + J2000

    @property
    def seg_ends_jd(self) -> Array:
        """Return segment end times as ``TDB`` Julian Dates."""
        return self.seg_ends_sec / DAY_S + J2000

    @staticmethod
    def _tdb_sec(tdb_jd1: Array, tdb_jd2: Array) -> Array:
        """Return seconds from ``J2000`` for split ``TDB`` Julian Dates."""
        return (jnp.asarray(tdb_jd1) - J2000) * DAY_S + jnp.asarray(tdb_jd2) * DAY_S

    def is_covered_sec(self, tdb_sec: Array) -> Array:
        """Return whether each epoch is inside any stored segment range.

        Parameters
        ----------
        tdb_sec : Array
            Epoch seconds relative to ``J2000``.

        Returns
        -------
        Array
            Boolean scalar or array matching the input epoch shape.
        """
        tdb_sec = jnp.asarray(tdb_sec)
        covered = (self.seg_starts_sec <= tdb_sec[..., None]) & (tdb_sec[..., None] <= self.seg_ends_sec)
        return jnp.any(covered, axis=-1)

    def _check_coverage(self, tdb_jd1: Array, tdb_jd2: Array) -> tuple[Array, Array]:
        """Attach a runtime error to epochs outside the loaded segment coverage."""
        tdb_jd1 = jnp.asarray(tdb_jd1)
        tdb_jd2 = jnp.asarray(tdb_jd2)
        outside = jnp.logical_not(self.is_covered_sec(self._tdb_sec(tdb_jd1, tdb_jd2)))
        message = "SPK evaluation error: requested TDB epoch is outside the loaded SPK coverage."
        return eqx.error_if(tdb_jd1, outside, message), eqx.error_if(tdb_jd2, outside, message)

    def pos(self, tdb_jd1: Array, tdb_jd2: Array) -> Array:
        """Return positions at ``TDB`` epochs.

        Parameters
        ----------
        tdb_jd1, tdb_jd2 : Array
            Two parts of each ``TDB`` Julian Date.

        Returns
        -------
        Array
            Position vectors in kilometers.
        """
        tdb_jd1, tdb_jd2 = self._check_coverage(tdb_jd1, tdb_jd2)
        return compute_position(self.seg_starts_sec, self.chunk_nums, self.chunk_lengths_sec, self.coefficients, tdb_jd1, tdb_jd2)

    def state(self, tdb_jd1: Array, tdb_jd2: Array) -> Tuple[Array, Array]:
        """Return positions and velocities at ``TDB`` epochs.

        Parameters
        ----------
        tdb_jd1, tdb_jd2 : Array
            Two parts of each ``TDB`` Julian Date.

        Returns
        -------
        tuple[Array, Array]
            Position in kilometers and velocity in kilometers per day.
        """
        tdb_jd1, tdb_jd2 = self._check_coverage(tdb_jd1, tdb_jd2)
        return compute_pv(self.seg_starts_sec, self.chunk_nums, self.chunk_lengths_sec, self.coefficients, tdb_jd1, tdb_jd2)

    def pva(self, tdb_jd1: Array, tdb_jd2: Array) -> Tuple[Array, Array, Array]:
        """Return positions, velocities, and accelerations at ``TDB`` epochs.

        Parameters
        ----------
        tdb_jd1, tdb_jd2 : Array
            Two parts of each ``TDB`` Julian Date.

        Returns
        -------
        tuple[Array, Array, Array]
            Position in kilometers, velocity in kilometers per day, and acceleration in kilometers per day squared.
        """
        tdb_jd1, tdb_jd2 = self._check_coverage(tdb_jd1, tdb_jd2)
        return compute_pva(self.seg_starts_sec, self.chunk_nums, self.chunk_lengths_sec, self.coefficients, tdb_jd1, tdb_jd2)

    def is_covered(self, tdb_jd: Array) -> Array:
        """Return whether each date is inside any stored segment range.

        Parameters
        ----------
        tdb_jd : Array
            ``TDB`` Julian Date.

        Returns
        -------
        Array
            True where any stored segment covers the date.
        """
        return self.is_covered_sec((jnp.asarray(tdb_jd) - J2000) * DAY_S)

    def __str__(self):
        return "<Segment center=%s, target=%s, covered=JD %s to %s>" % (self.center_ids[1], self.target_ids[1],
                                                                        self.seg_starts_jd, self.seg_ends_jd)

    def __repr__(self):
        return self.__str__()


class TTMinusTDBKernel(eqx.Module):
    """Kernel wrapper for the ``TT - TDB`` time offset segment."""
    segment: MergedSegment

    def tt_minus_tdb(self, tdb_jd1: Array, tdb_jd2: Array) -> Array:
        """Return ``TT - TDB`` at ``TDB`` epochs.

        Parameters
        ----------
        tdb_jd1, tdb_jd2 : Array
            Two parts of each ``TDB`` Julian Date.

        Returns
        -------
        Array
            Time offset in seconds.
        """
        return self.segment.pos(tdb_jd1, tdb_jd2)[..., 0]

    def dtt_minus_tdb_dtdb(self, tdb_jd1: Array, tdb_jd2: Array) -> Array:
        """Return the derivative of ``TT - TDB`` with respect to ``TDB``.

        Parameters
        ----------
        tdb_jd1, tdb_jd2 : Array
            Two parts of each ``TDB`` Julian Date.

        Returns
        -------
        Array
            Derivative in seconds per second.
        """
        _, rate_per_day = self.segment.state(tdb_jd1, tdb_jd2)
        return rate_per_day[..., 0] / DAY_S


class Ephemeris:
    """SPK ephemeris container with cached merged body paths.

    Parameters
    ----------
    filepath : str or list[str]
        One SPK kernel path or an ordered list of kernel paths.
    load_window : tuple[TDBView, TDBView] or None, optional
        Inclusive scalar ``TDB`` window used when loading SPK segments. If
        omitted, all segments remain eligible for later path loads.
    """

    def __init__(self, filepath: Union[str, List[str]],
                 load_window: Optional[Tuple[TDBView, TDBView]] = None) -> None:
        """Initialize an ephemeris container from one SPK kernel set.

        Parameters
        ----------
        filepath : str or list[str]
            One SPK kernel path or an ordered list of kernel paths.
        load_window : tuple[TDBView, TDBView] or None, optional
            Inclusive scalar ``TDB`` window used to discard SPK segments that
            are entirely outside the requested coverage interval.
        """
        self.spks = self._setup_spks(filepath)
        self.load_window = _normalize_load_window(load_window)
        self.graph = collections.defaultdict(list)
        self._build_graph()
        self.cache_path: Dict[Tuple[str, str], 'MergedSegment'] = {}
        self.cache_tt_minus_tdb_kernel = None

    @staticmethod
    def _setup_spks(filepath: Union[str, List[str]]) -> List[jplephem.spk.SPK]:
        filepath = [filepath] if isinstance(filepath, str) else filepath
        return [jplephem.spk.SPK.open(filepath) for filepath in filepath]

    def _build_graph(self):
        for spk in self.spks:
            for seg in spk.segments:
                center_name = naif[seg.center]
                target_name = naif[seg.target]

                if target_name not in self.graph[center_name]:
                    self.graph[center_name].append(target_name)
                if center_name not in self.graph[target_name]:
                    self.graph[target_name].append(center_name)

    def _find_path_names(self, start_name: str, end_name: str) -> List[str]:
        """Find Segment Link By BFS """
        if start_name == end_name:
            return []
        queue = collections.deque([[start_name]])
        visited = {start_name}
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == end_name:
                return path
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
        raise ValueError(f"No path between {start_name} and {end_name} exists in SPK")

    def _load_path(self, center_name: str, target_name: str) -> 'MergedSegment':
        cache_key = (center_name, target_name)
        if cache_key not in self.cache_path:
            segments = []
            for spk in self.spks:
                for seg in spk.segments:
                    if center_name == naif[seg.center] and target_name == naif[seg.target]:
                        segments.append(seg)
            self.cache_path[cache_key] = MergedSegment.from_jplephem_segments(segments, load_window=self.load_window)
        return self.cache_path[cache_key]

    def load_body(self, target_name: str, center_name="SOLAR SYSTEM BARYCENTER") -> Tuple[
        Tuple['MergedSegment'], Tuple]:
        """Load the SPK path from one center body to one target body.

        Parameters
        ----------
        target_name : str
            NAIF target body name.
        center_name : str, default="SOLAR SYSTEM BARYCENTER"
            NAIF center body name.

        Returns
        -------
        tuple[tuple[MergedSegment, ...], tuple[float, ...]]
            Path segments and signs. A negative sign means that the stored segment is used in reverse.
        """
        center_name = center_name.upper()
        target_name = target_name.upper()
        path_names = self._find_path_names(center_name, target_name)

        segments = []
        signs = []
        for i in range(len(path_names) - 1):
            u_name = path_names[i]
            v_name = path_names[i + 1]
            try:
                seg = self._load_path(u_name, v_name)
                segments.append(seg)
                signs.append(1.)
            except ValueError:
                seg = self._load_path(v_name, u_name)
                segments.append(seg)
                signs.append(-1.)
        return tuple(segments), tuple(signs)

    def load_tt_minus_tdb_kernel(self) -> TTMinusTDBKernel:
        """Load and cache the ``TT - TDB`` kernel segment.

        Returns
        -------
        TTMinusTDBKernel
            Kernel wrapper for the time offset.

        Raises
        ------
        ValueError
            If the loaded kernels do not contain a ``TT - TDB`` segment.
        """
        if self.cache_tt_minus_tdb_kernel is not None:
            return self.cache_tt_minus_tdb_kernel
        segments = []
        for spk in self.spks:
            for seg in spk.segments:
                if seg.center == 1000000000 and seg.target == 1000000001:
                    segments.append(seg)
        if not segments:
            raise ValueError(
                "No TT-TDB segment (1000000001 relative to 1000000000) "
                "was found in the loaded SPK kernels."
            )
        self.cache_tt_minus_tdb_kernel = TTMinusTDBKernel(
            MergedSegment.from_jplephem_segments(segments, load_window=self.load_window)
        )
        return self.cache_tt_minus_tdb_kernel

    @property
    def available_bodies(self) -> List[str]:
        """Return body names that appear in the loaded SPK graph."""
        return list(self.graph.keys())
