"""Canonical tables for orbit-determination result analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from difforb.astrometry.data import ObservationData, ObservationLayout
from difforb.core.constants import RAD_TO_ARCSEC
from difforb.od.dc.result import DCResult
from difforb.od.result import ODResult


def _weighted_normal_trace(
        flat_jacobian: np.ndarray,
        optical_weight_matrices: np.ndarray,
        radar_weights: np.ndarray,
        flat_mask: np.ndarray | None = None,
) -> float:
    """Return ``trace(J.T W J)`` for block optical and scalar radar weights."""
    flat_jacobian = np.asarray(flat_jacobian, dtype=float)
    optical_weight_matrices = np.asarray(optical_weight_matrices, dtype=float)
    radar_weights = np.asarray(radar_weights, dtype=float)
    if flat_mask is None:
        flat_mask = np.ones(flat_jacobian.shape[0], dtype=bool)
    flat_mask = np.asarray(flat_mask, dtype=bool)

    n_optical = optical_weight_matrices.shape[0]
    n_flat_optical = 2 * n_optical
    total = 0.0

    if n_optical:
        optical_jacobian = flat_jacobian[:n_flat_optical].reshape((n_optical, 2, flat_jacobian.shape[1]))
        optical_mask = flat_mask[:n_flat_optical].reshape((n_optical, 2)).all(axis=1)
        optical_weighted_jacobian = np.einsum("nij,njk->nik", optical_weight_matrices, optical_jacobian)
        optical_trace = np.sum(optical_jacobian * optical_weighted_jacobian, axis=(1, 2))
        total += float(np.sum(optical_trace[optical_mask]))

    radar_jacobian = flat_jacobian[n_flat_optical:]
    radar_mask = flat_mask[n_flat_optical:]
    if len(radar_weights):
        radar_trace = np.sum(radar_jacobian * radar_jacobian, axis=1) * radar_weights
        total += float(np.sum(radar_trace[radar_mask]))

    return total


OBSERVATION_COLUMNS = [
    "input_index",
    "time_ut_jd",
    "time_ut_iso",
    "station_key",
    "station",
    "transmitter_station",
    "tracklet_id",
    "modality",
    "observer_type",
    "observation_type",
    "obs_mode",
    "inlier",
    "inlier_state",
    "chi2",
    "ra_deg",
    "dec_deg",
    "radar_measurement",
    "radar_unit",
    "tx_freq_hz",
    "ra_residual_arcsec",
    "dec_residual_arcsec",
    "delay_residual_us",
    "doppler_residual_hz",
    "normalized_ra_residual",
    "normalized_dec_residual",
    "normalized_delay_residual",
    "normalized_doppler_residual",
    "combined_normalized_residual",
    "adopted_ra_sigma_arcsec",
    "adopted_dec_sigma_arcsec",
    "adopted_ra_dec_correlation",
    "adopted_radar_sigma",
    "adopted_radar_sigma_unit",
    "reported_ra_sigma_arcsec",
    "reported_dec_sigma_arcsec",
    "reported_ra_dec_correlation",
    "reported_radar_sigma",
]

RESIDUAL_COLUMNS = [
    "input_index",
    "time_ut_jd",
    "time_ut_iso",
    "station_key",
    "station",
    "tracklet_id",
    "modality",
    "observer_type",
    "observation_type",
    "obs_mode",
    "inlier",
    "inlier_state",
    "chi2",
    "component",
    "residual",
    "residual_unit",
    "normalized_residual",
    "adopted_sigma",
    "adopted_sigma_unit",
]


@dataclass(slots=True)
class ODAnalysis:
    """Analysis tables derived from one differential-correction result."""

    observations_data: ObservationData = field(repr=False)
    result: DCResult = field(repr=False)
    observations: pd.DataFrame
    residuals: pd.DataFrame
    _layout: ObservationLayout = field(repr=False)

    @classmethod
    def from_result(cls, observations: ObservationData, result: DCResult | ODResult) -> "ODAnalysis":
        """Build standardized analysis tables from observations and one OD or DC result.

        Parameters
        ----------
        observations : ObservationData
            Observation bundle used by the fit.
        result : DCResult or ODResult
            Differential-correction result, or an orbit-determination result that contains one.

        Returns
        -------
        ODAnalysis
            Analysis object with observation-level and residual-component tables.

        Raises
        ------
        ValueError
            Raised when an ``ODResult`` has no differential-correction result.
        """
        dc_result = _dc_result(result)
        layout = ObservationLayout(observations)
        observation_table = build_observation_table(observations, dc_result, layout=layout)
        residual_table = build_residual_table(observation_table)
        return cls(
            observations_data=observations,
            result=dc_result,
            observations=observation_table,
            residuals=residual_table,
            _layout=layout,
        )

    def station_summary(self) -> pd.DataFrame:
        """Return station-level residual and outlier statistics."""
        return build_station_summary(self.observations, self.residuals)

    def tracklet_summary(self, *, include_contribution: bool = False) -> pd.DataFrame:
        """Return tracklet-level residual and outlier statistics.

        Parameters
        ----------
        include_contribution : bool, default=False
            If ``True``, add weighted and geometric normal-matrix contribution percentages computed from the final Jacobian.
        """
        return build_tracklet_summary(
            self.observations,
            self.residuals,
            result=self.result if include_contribution else None,
            layout=self._layout if include_contribution else None,
        )


def build_observation_table(
    observations: ObservationData,
    result: DCResult | ODResult,
    *,
    layout: ObservationLayout | None = None,
) -> pd.DataFrame:
    """Build one observation-level analysis table.

    The returned table has one row per observation. Optical residuals are in arcseconds. Radar delay residuals are in microseconds, and radar Doppler residuals are in hertz. Normalized residuals and Chi2 values are dimensionless.
    """
    dc_result = _dc_result(result)
    layout = ObservationLayout(observations) if layout is None else layout
    optical_weight_matrices = np.asarray(dc_result.lsq_diagnostics.optical_weight_matrices, dtype=float)
    radar_weights = np.asarray(dc_result.lsq_diagnostics.radar_weights, dtype=float)
    optical_sigmas = np.empty((0, 2), dtype=float)
    optical_correlations = np.empty((0,), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if optical_weight_matrices.shape[-2:] == (2, 2) and len(optical_weight_matrices) > 0:
            optical_covariance = np.linalg.inv(optical_weight_matrices)
            optical_sigmas = np.sqrt(np.stack([optical_covariance[:, 0, 0], optical_covariance[:, 1, 1]], axis=1))
            optical_correlations = optical_covariance[:, 0, 1] / (optical_sigmas[:, 0] * optical_sigmas[:, 1])
        radar_sigmas = np.asarray(np.where(radar_weights > 0.0, 1.0 / np.sqrt(radar_weights), np.nan), dtype=float)

    frames: list[pd.DataFrame] = []
    if observations.has_optical:
        frame = observations.optical.to_dataframe().rename(
            columns={
                "t_ut_jd": "time_ut_jd",
                "t_iso": "time_ut_iso",
                "trk_id": "tracklet_id",
                "ra_uncertainty_arcsec": "reported_ra_sigma_arcsec",
                "dec_uncertainty_arcsec": "reported_dec_sigma_arcsec",
                "ra_dec_correlation": "reported_ra_dec_correlation",
            }
        )
        if "obs_type" in frame:
            frame = frame.drop(columns=["obs_type"])
        if "transmitter_station" not in frame:
            frame["transmitter_station"] = np.nan
        frame["modality"] = "optical"
        frame["observation_type"] = "optical"
        frame["inlier"] = np.asarray(dc_result.optical.inlier_masks, dtype=bool)
        frame["inlier_state"] = np.where(frame["inlier"], "inlier", "outlier")
        frame["chi2"] = np.asarray(dc_result.optical.metrics, dtype=float)
        frame["ra_residual_arcsec"] = np.asarray(dc_result.optical.residuals[:, 0], dtype=float) * RAD_TO_ARCSEC
        frame["dec_residual_arcsec"] = np.asarray(dc_result.optical.residuals[:, 1], dtype=float) * RAD_TO_ARCSEC
        frame["normalized_ra_residual"] = np.asarray(dc_result.optical.normalized_residuals[:, 0], dtype=float)
        frame["normalized_dec_residual"] = np.asarray(dc_result.optical.normalized_residuals[:, 1], dtype=float)
        frame["combined_normalized_residual"] = np.sqrt(
            np.square(frame["normalized_ra_residual"]) + np.square(frame["normalized_dec_residual"])
        )
        if optical_sigmas.shape == (len(frame), 2):
            frame["adopted_ra_sigma_arcsec"] = optical_sigmas[:, 0] * RAD_TO_ARCSEC
            frame["adopted_dec_sigma_arcsec"] = optical_sigmas[:, 1] * RAD_TO_ARCSEC
            frame["adopted_ra_dec_correlation"] = optical_correlations
        else:
            frame["adopted_ra_sigma_arcsec"] = np.nan
            frame["adopted_dec_sigma_arcsec"] = np.nan
            frame["adopted_ra_dec_correlation"] = np.nan
        frames.append(frame)

    if observations.has_radar:
        frame = observations.radar.to_dataframe().rename(
            columns={
                "t_ut_jd": "time_ut_jd",
                "t_iso": "time_ut_iso",
                "tx_code": "transmitter_station",
                "radar_uncertainty": "reported_radar_sigma",
            }
        )
        if "obs_type" in frame:
            frame = frame.drop(columns=["obs_type"])
        for column in ("tracklet_id", "observer_type"):
            if column not in frame:
                frame[column] = np.nan
        is_delay = np.asarray(observations.radar.is_delay, dtype=bool)
        is_doppler = np.asarray(observations.radar.is_doppler, dtype=bool)
        residuals = np.asarray(dc_result.radar.residuals, dtype=float)
        normalized_residuals = np.asarray(dc_result.radar.normalized_residuals, dtype=float)
        frame["modality"] = "radar"
        frame["observation_type"] = np.where(is_delay, "radar_delay", "radar_doppler")
        frame["inlier"] = np.asarray(dc_result.radar.inlier_masks, dtype=bool)
        frame["inlier_state"] = np.where(frame["inlier"], "inlier", "outlier")
        frame["chi2"] = np.asarray(dc_result.radar.metrics, dtype=float)
        frame["delay_residual_us"] = np.where(is_delay, residuals, np.nan)
        frame["doppler_residual_hz"] = np.where(is_doppler, residuals, np.nan)
        frame["normalized_delay_residual"] = np.where(is_delay, normalized_residuals, np.nan)
        frame["normalized_doppler_residual"] = np.where(is_doppler, normalized_residuals, np.nan)
        frame["combined_normalized_residual"] = np.abs(normalized_residuals)
        frame["adopted_radar_sigma"] = radar_sigmas if radar_sigmas.shape == (len(frame),) else np.nan
        frame["adopted_radar_sigma_unit"] = frame["radar_unit"]
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
    table = pd.concat(frames, ignore_index=True, sort=False)
    table = table.sort_values("input_index", kind="stable").reset_index(drop=True)
    return _ordered_columns(table, OBSERVATION_COLUMNS)


def build_residual_table(observations: pd.DataFrame) -> pd.DataFrame:
    """Build one residual-component table from an observation-level table.

    The returned table has one row per scalar residual component. Optical observations produce one right-ascension row and one declination row. Radar observations produce one delay or Doppler row.
    """
    if observations.empty:
        return pd.DataFrame(columns=RESIDUAL_COLUMNS)

    common = [
        "input_index",
        "time_ut_jd",
        "time_ut_iso",
        "station_key",
        "station",
        "tracklet_id",
        "modality",
        "observer_type",
        "observation_type",
        "obs_mode",
        "inlier",
        "inlier_state",
        "chi2",
    ]
    frames: list[pd.DataFrame] = []
    component_specs = [
        ("optical", "ra", "ra_residual_arcsec", "normalized_ra_residual", "adopted_ra_sigma_arcsec", "arcsec"),
        ("optical", "dec", "dec_residual_arcsec", "normalized_dec_residual", "adopted_dec_sigma_arcsec", "arcsec"),
        ("radar_delay", "delay", "delay_residual_us", "normalized_delay_residual", "adopted_radar_sigma", "us"),
        ("radar_doppler", "doppler", "doppler_residual_hz", "normalized_doppler_residual", "adopted_radar_sigma", "Hz"),
    ]
    for observation_type, component, residual_column, normalized_column, sigma_column, unit in component_specs:
        if residual_column not in observations or normalized_column not in observations or sigma_column not in observations:
            continue
        source = observations[observations["observation_type"] == observation_type].copy()
        if source.empty:
            continue
        for column in common:
            if column not in source:
                source[column] = np.nan
        frame = source[common].copy()
        frame["component"] = component
        frame["residual"] = pd.to_numeric(source[residual_column], errors="coerce")
        frame["residual_unit"] = unit
        frame["normalized_residual"] = pd.to_numeric(source[normalized_column], errors="coerce")
        frame["adopted_sigma"] = pd.to_numeric(source[sigma_column], errors="coerce")
        frame["adopted_sigma_unit"] = unit
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=RESIDUAL_COLUMNS)
    table = pd.concat(frames, ignore_index=True, sort=False)
    table = table.sort_values(["input_index", "component"], kind="stable").reset_index(drop=True)
    return _ordered_columns(table, RESIDUAL_COLUMNS)


def build_group_summary(observations: pd.DataFrame, residuals: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Build grouped residual and outlier statistics for observation groups."""
    if observations.empty:
        return pd.DataFrame(columns=group_columns)

    observations = observations.copy()
    residuals = residuals.copy() if residuals is not None else pd.DataFrame()
    if "station_key" in group_columns:
        if "station_key" not in observations and "station" in observations:
            observations["station_key"] = observations["station"]
        if "station_key" not in residuals and "station" in residuals:
            residuals["station_key"] = residuals["station"]
    for column in group_columns:
        if column not in observations:
            observations[column] = np.nan
        if column not in residuals:
            residuals[column] = np.nan
        if column in observations:
            observations[column] = observations[column].map(_canonical_group_key)
        if column in residuals:
            residuals[column] = residuals[column].map(_canonical_group_key)

    grouped = observations.groupby(group_columns, dropna=False)
    summary = grouped.size().rename("obs").reset_index()
    summary = _coerce_group_key_columns(summary, group_columns)
    summary["inliers"] = grouped["inlier"].sum().astype(int).values
    summary["outliers"] = summary["obs"] - summary["inliers"]
    summary["outlier_percent"] = np.where(summary["obs"] > 0, summary["outliers"] / summary["obs"] * 100.0, np.nan)
    summary["chi2_max"] = grouped["chi2"].max().values
    summary["chi2_p95"] = grouped["chi2"].quantile(0.95).values

    inlier_residuals = residuals[residuals["inlier"]].copy() if not residuals.empty else pd.DataFrame()
    if inlier_residuals.empty:
        return summary

    residual_grouped = inlier_residuals.groupby(group_columns, dropna=False)
    residual_summary = residual_grouped["normalized_residual"].apply(_rms).rename("normalized_residual_rms").reset_index()
    residual_summary = _coerce_group_key_columns(residual_summary, group_columns)
    summary = summary.merge(residual_summary, on=group_columns, how="left")

    for component, residual_name, normalized_std_name in [
        ("ra", "ra_residual_arcsec", "std_normalized_ra_residual"),
        ("dec", "dec_residual_arcsec", "std_normalized_dec_residual"),
        ("delay", "delay_residual_us", "std_normalized_delay_residual"),
        ("doppler", "doppler_residual_hz", "std_normalized_doppler_residual"),
    ]:
        component_frame = inlier_residuals[inlier_residuals["component"] == component]
        if component_frame.empty:
            continue
        component_grouped = component_frame.groupby(group_columns, dropna=False)
        component_summary = pd.DataFrame(index=component_grouped.size().index)
        component_summary[f"mean_{residual_name}"] = component_grouped["residual"].mean()
        component_summary[f"std_{residual_name}"] = component_grouped["residual"].std()
        component_summary[normalized_std_name] = component_grouped["normalized_residual"].std()
        component_summary = _coerce_group_key_columns(component_summary.reset_index(), group_columns)
        summary = summary.merge(component_summary, on=group_columns, how="left")

    if {"std_normalized_ra_residual", "std_normalized_dec_residual"} <= set(summary.columns):
        ra_std = pd.to_numeric(summary["std_normalized_ra_residual"], errors="coerce")
        dec_std = pd.to_numeric(summary["std_normalized_dec_residual"], errors="coerce")
        summary["normalized_residual_spread"] = np.sqrt((np.square(ra_std) + np.square(dec_std)) / 2.0)
        summary["max_normalized_residual_std"] = np.maximum(ra_std, dec_std)
    return summary


def build_station_summary(observations: pd.DataFrame, residuals: pd.DataFrame) -> pd.DataFrame:
    """Build station-level residual and outlier statistics."""
    group_columns = ["station_key", "station", "observer_type", "modality", "observation_type"]
    summary = build_group_summary(observations, residuals, group_columns)
    return summary.sort_values("obs", ascending=False, kind="stable").reset_index(drop=True) if "obs" in summary else summary


def build_tracklet_summary(
    observations: pd.DataFrame,
    residuals: pd.DataFrame,
    *,
    result: DCResult | None = None,
    layout: ObservationLayout | None = None,
) -> pd.DataFrame:
    """Build tracklet-level residual and outlier statistics."""
    if observations.empty or "tracklet_id" not in observations:
        return pd.DataFrame(columns=["tracklet_id"])
    tracklet_mask = observations["tracklet_id"].notna() & (observations["tracklet_id"].astype(str).str.strip() != "")
    tracklet_observations = observations[tracklet_mask].copy()
    if tracklet_observations.empty:
        return pd.DataFrame(columns=["tracklet_id"])
    tracklet_observations["tracklet_id"] = tracklet_observations["tracklet_id"].map(_canonical_group_key)

    summary = build_group_summary(tracklet_observations, residuals, ["tracklet_id"])
    grouped = tracklet_observations.groupby("tracklet_id", dropna=False)
    station_by_tracklet = grouped["station"].apply(lambda values: next((str(value) for value in values if value is not None and not pd.isna(value) and str(value)), None))
    summary["station"] = summary["tracklet_id"].map(station_by_tracklet)
    summary["start_time_ut_iso"] = summary["tracklet_id"].map(grouped["time_ut_iso"].min())
    summary["end_time_ut_iso"] = summary["tracklet_id"].map(grouped["time_ut_iso"].max())
    summary["duration_days"] = summary["tracklet_id"].map(grouped["time_ut_jd"].max() - grouped["time_ut_jd"].min())

    if result is not None and layout is not None:
        flat_jacobian = np.asarray(result.lsq_diagnostics.flat_jacobian, dtype=float)
        optical_weight_matrices = np.asarray(result.lsq_diagnostics.optical_weight_matrices, dtype=float)
        radar_weights = np.asarray(result.lsq_diagnostics.radar_weights, dtype=float)
        global_weighted_trace = _weighted_normal_trace(flat_jacobian, optical_weight_matrices, radar_weights)
        global_geometric_trace = float(np.sum(flat_jacobian * flat_jacobian))
        weighted: dict[Any, float] = {}
        geometric: dict[Any, float] = {}
        for tracklet_id, group in tracklet_observations.groupby("tracklet_id", dropna=False):
            input_indices = group["input_index"].to_numpy(dtype=int)
            flat_mask = np.asarray(layout.input_indices_to_flat_mask(input_indices), dtype=bool)
            local_jacobian = flat_jacobian[flat_mask]
            local_weighted_trace = _weighted_normal_trace(flat_jacobian, optical_weight_matrices, radar_weights, flat_mask)
            local_geometric_trace = float(np.sum(local_jacobian * local_jacobian))
            weighted[tracklet_id] = local_weighted_trace / global_weighted_trace * 100.0 if global_weighted_trace > 0 else 0.0
            geometric[tracklet_id] = local_geometric_trace / global_geometric_trace * 100.0 if global_geometric_trace > 0 else 0.0
        summary["weighted_contribution_percent"] = summary["tracklet_id"].map(weighted)
        summary["geometric_contribution_percent"] = summary["tracklet_id"].map(geometric)

    summary = _ordered_columns(
        summary,
        ["tracklet_id", "station", "start_time_ut_iso", "end_time_ut_iso", "duration_days"],
    )
    return summary.sort_values("obs", ascending=False, kind="stable").reset_index(drop=True)


def _dc_result(result: DCResult | ODResult) -> DCResult:
    if isinstance(result, DCResult):
        return result
    if isinstance(result, ODResult):
        if result.dc_result is None:
            raise ValueError("ODResult does not contain a differential-correction result.")
        return result.dc_result
    raise TypeError("`result` must be a DCResult or ODResult.")


def _rms(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(numeric))))


def _ordered_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    ordered = [column for column in columns if column in frame.columns]
    rest = [column for column in frame.columns if column not in ordered]
    return frame[ordered + rest]


def _coerce_group_key_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = frame[column].map(_canonical_group_key).astype(object)
    return frame


def _canonical_group_key(value: Any) -> str | float:
    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return np.nan
        return str(int(number)) if number.is_integer() else str(number)
    text = str(value).strip()
    return text if text else np.nan
