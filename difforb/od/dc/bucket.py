"""Shape-bucket helpers for differential-correction observation data."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool

from difforb.astrometry.data import ObservationData, OpticalObservationData, RadarObservationData
from difforb.od.dc.result import DCResult, LSQDiagnostics, OpticalResult, RadarResult


def _normalize_buckets(values: tuple[int, ...], label: str) -> tuple[int, ...]:
    buckets = tuple(sorted({int(value) for value in values}))
    if not buckets or any(value <= 0 for value in buckets):
        raise ValueError(f"`{label}` must contain positive bucket sizes.")
    return buckets


@dataclass(frozen=True)
class DCBucketPolicy:
    """Select fixed observation-count buckets for differential correction.

    Parameters
    ----------
    optical_buckets : tuple of int, optional
        Bucket sizes used for optical observation tables.
    radar_buckets : tuple of int, optional
        Bucket sizes used for radar observation tables.

    Notes
    -----
    Empty modality tables are kept empty. Non-empty tables are padded to the first configured bucket that can hold the real row count. Counts above the largest configured bucket extend by repeated doubling from the largest bucket.
    """

    optical_buckets: tuple[int, ...] = (100, 500, 1000, 2000, 4000, 6000, 8000, 10000)
    radar_buckets: tuple[int, ...] = (5, 10, 15, 20)

    def __post_init__(self) -> None:
        object.__setattr__(self, "optical_buckets", _normalize_buckets(self.optical_buckets, "optical_buckets"))
        object.__setattr__(self, "radar_buckets", _normalize_buckets(self.radar_buckets, "radar_buckets"))

    def optical_bucket_size(self, count: int) -> int:
        """Return the optical bucket size for a real observation count."""
        return self._bucket_size(count, self.optical_buckets)

    def radar_bucket_size(self, count: int) -> int:
        """Return the radar bucket size for a real observation count."""
        return self._bucket_size(count, self.radar_buckets)

    @staticmethod
    def _bucket_size(count: int, buckets: tuple[int, ...]) -> int:
        if count <= 0:
            return 0
        for bucket in buckets:
            if count <= bucket:
                return bucket
        bucket = buckets[-1]
        while bucket < count:
            bucket *= 2
        return bucket


@dataclass(frozen=True)
class PackedDCObservations:
    """Differential-correction observations padded to bucket shapes."""

    data: ObservationData
    n_optical: int
    n_radar: int
    flat_valid_mask: Bool[Array, "N_flat_obs"]
    observation_valid_mask: Bool[Array, "N_obs"]
    flat_valid_indices: np.ndarray

    @property
    def is_padded(self) -> bool:
        """Return whether any modality table was padded."""
        return (
            self.data.num_optical != self.n_optical
            or self.data.num_radar != self.n_radar
        )


def _sentinel_indices(start: int, count: int) -> np.ndarray:
    return -np.arange(start, start + count, dtype=int)


def _pad_optical_table(table: OpticalObservationData, target_size: int, *, sentinel_start: int) -> OpticalObservationData:
    count = len(table)
    if target_size < count:
        raise ValueError("Optical bucket size cannot be smaller than the real observation count.")
    if target_size == count:
        return table
    if count == 0:
        raise ValueError("Empty optical tables are not padded to non-empty buckets.")

    pad_count = target_size - count
    indices = np.concatenate([np.arange(count, dtype=int), np.full(pad_count, count - 1, dtype=int)])
    padded = table[indices]
    input_indices = np.asarray(padded.input_indices, dtype=int).copy()
    input_indices[count:] = _sentinel_indices(sentinel_start, pad_count)
    padded.input_indices = input_indices
    return padded


def _pad_radar_table(table: RadarObservationData, target_size: int, *, sentinel_start: int) -> RadarObservationData:
    count = len(table)
    if target_size < count:
        raise ValueError("Radar bucket size cannot be smaller than the real observation count.")
    if target_size == count:
        return table
    if count == 0:
        raise ValueError("Empty radar tables are not padded to non-empty buckets.")

    pad_count = target_size - count
    indices = np.concatenate([np.arange(count, dtype=int), np.full(pad_count, count - 1, dtype=int)])
    padded = table[indices]
    input_indices = np.asarray(padded.input_indices, dtype=int).copy()
    input_indices[count:] = _sentinel_indices(sentinel_start, pad_count)
    padded.input_indices = input_indices
    return padded


def pack_dc_observations(data: ObservationData, policy: DCBucketPolicy) -> PackedDCObservations:
    """Pad differential-correction observations to stable bucket shapes.

    Parameters
    ----------
    data : ObservationData
        Real observation bundle used by a differential-correction solve.
    policy : DCBucketPolicy
        Bucket policy that selects padded modality table lengths.

    Returns
    -------
    PackedDCObservations
        Padded observation bundle plus flat and observation-level masks that mark the real residuals.

    Notes
    -----
    Padding duplicates the final real row in each non-empty modality table so site lookup, time conversion, and light-time evaluation continue to see valid observation payloads. The structural masks must be applied by the solver so padded residuals cannot affect the least-squares system or outlier recovery.
    """
    n_optical = data.num_optical
    n_radar = data.num_radar

    target_optical = policy.optical_bucket_size(n_optical)
    target_radar = policy.radar_bucket_size(n_radar)

    sentinel_start = 1
    optical = _pad_optical_table(data.optical, target_optical, sentinel_start=sentinel_start)
    sentinel_start += max(0, target_optical - n_optical)
    radar = _pad_radar_table(data.radar, target_radar, sentinel_start=sentinel_start)

    packed = ObservationData(
        name=data.name,
        optical=optical,
        radar=radar,
    )

    optical_valid = np.arange(target_optical, dtype=int) < n_optical
    radar_valid = np.arange(target_radar, dtype=int) < n_radar
    flat_valid_mask_np = np.concatenate([
        np.repeat(optical_valid, 2),
        radar_valid,
    ])
    observation_valid_mask_np = np.concatenate([optical_valid, radar_valid])

    return PackedDCObservations(
        data=packed,
        n_optical=n_optical,
        n_radar=n_radar,
        flat_valid_mask=jnp.asarray(flat_valid_mask_np, dtype=bool),
        observation_valid_mask=jnp.asarray(observation_valid_mask_np, dtype=bool),
        flat_valid_indices=np.nonzero(flat_valid_mask_np)[0],
    )


def _crop_optical_result(result: OpticalResult, count: int) -> OpticalResult:
    return OpticalResult(
        residuals=result.residuals[:count],
        normalized_residuals=result.normalized_residuals[:count],
        weighted_rms=result.weighted_rms,
        unweighted_rms=result.unweighted_rms,
        inlier_masks=result.inlier_masks[:count],
        metrics=result.metrics[:count],
    )


def _crop_radar_result(result: RadarResult, count: int) -> RadarResult:
    return RadarResult(
        residuals=result.residuals[:count],
        normalized_residuals=result.normalized_residuals[:count],
        inlier_masks=result.inlier_masks[:count],
        metrics=result.metrics[:count],
        delay_weighted_rms=result.delay_weighted_rms,
        delay_unweighted_rms=result.delay_unweighted_rms,
        doppler_weighted_rms=result.doppler_weighted_rms,
        doppler_unweighted_rms=result.doppler_unweighted_rms,
    )


def crop_dc_result(result: DCResult, packed: PackedDCObservations) -> DCResult:
    """Remove padded residual rows from a bucketed differential-correction result."""
    if not packed.is_padded:
        return result

    diagnostics = result.lsq_diagnostics
    valid_indices = packed.flat_valid_indices
    cropped_diagnostics = LSQDiagnostics(
        flat_jacobian=diagnostics.flat_jacobian[valid_indices],
        flat_weights=diagnostics.flat_weights[valid_indices],
        optical_weight_matrices=diagnostics.optical_weight_matrices[:packed.n_optical],
        radar_weights=diagnostics.radar_weights[:packed.n_radar],
        cov_mat_prior=diagnostics.cov_mat_prior,
        cov_rank=diagnostics.cov_rank,
        cov_condition=diagnostics.cov_condition,
        cov_valid=diagnostics.cov_valid,
        converged=diagnostics.converged,
        termination_reason=diagnostics.termination_reason,
        lsq_iterations=diagnostics.lsq_iterations,
        outlier_iterations=diagnostics.outlier_iterations,
    )

    return DCResult(
        estimate=result.estimate,
        optical=_crop_optical_result(result.optical, packed.n_optical),
        radar=_crop_radar_result(result.radar, packed.n_radar),
        lsq_diagnostics=cropped_diagnostics,
        normalized_residual_rms=result.normalized_residual_rms,
    )
