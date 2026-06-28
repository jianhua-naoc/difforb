"""Observation-weight policies for optical and radar astrometry.

This module defines the rule and policy layer used to convert reported observation uncertainties into the inverse-variance weights consumed by orbit-determination code. It works on :class:`difforb.astrometry.data.ObservationData` bundles and returns matched uncertainty tables for optical and radar observations.

Optical uncertainties are carried as two-component tangent-plane sigmas in radians, with an optional per-row correlation coefficient. Radar uncertainties are carried as one scalar sigma per row in the native measurement unit already stored by the radar observation table. Interactive overrides operate on the original mixed-input indices exposed by the observation containers.
"""

import os
from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd
import numpy as np
from jaxtyping import Float, Bool, Shaped, ArrayLike

from difforb.astrometry.data import (
    OpticalObservationData,
    ObservationData, ObsType, ObsMode, ObserverType,
)
from difforb.body.site import parse_site_keys
from difforb.core.config import get_data_path, missing_data_message
from difforb.core.constants import ARCSEC_TO_DEG, DAY_S
from difforb.core.time.timescale import Time

DEFAULT_WEIGHT_SCHEME_FILENAME = str(get_data_path("weights.csv", dataset="weights", must_exist=False))


def _tracklet_group_counts(trk_ids: Shaped[np.ndarray, "N"]) -> np.ndarray:
    """Return per-row tracklet counts without grouping missing identifiers."""
    trk_series = pd.Series(np.asarray(trk_ids, dtype=object)).map(
        lambda value: "" if pd.isna(value) else str(value).strip()
    )
    valid_mask = (trk_series != "") & (trk_series.str.lower() != "nan")
    counts = np.ones(len(trk_series), dtype=int)
    if valid_mask.any():
        valid_counts = trk_series[valid_mask].groupby(trk_series[valid_mask]).transform("count")
        counts[valid_mask.to_numpy()] = valid_counts.to_numpy(dtype=int)
    return counts


class WeightRule:
    """Single rule row in a tabular astrometric weighting scheme.

    The rule stores categorical selectors and optional time bounds for one uncertainty assignment. String fields use ``"*"`` as a wildcard and ``"|"`` as an OR separator for multi-valued selectors.
    """
    WILDCARD = "*"
    OR = "|"

    def __init__(self, obs_type: str, obs_mode: str, observer_type: str, obs_note_code: str, site_code: str,
                 program_code: str, catalog_code: str, sub_frame: str, date_min: Time | None,
                 date_max: Time | None,
                 sigma: float, comment: str):
        """Initialize a weighting-rule row.

        Parameters
        ----------
        obs_type : str
            Observation-type selector keyed by :class:`difforb.astrometry.data.ObsType`, or ``"*"`` for all types.
        obs_mode : str
            Observation-mode selector keyed by :class:`difforb.astrometry.data.ObsMode`. Multiple values may be joined with ``"|"``.
        observer_type : str
            Observer-type selector keyed by :class:`difforb.astrometry.data.ObserverType`. Multiple values may be joined with ``"|"``.
        obs_note_code : str
            Required substring in the optical note-code field, or ``"*"`` for no filter.
        site_code : str
            Required receiver or site code, or ``"*"`` for no filter.
        program_code : str
            Allowed ADES ``prog`` codes. Multiple two-character values may be joined with ``"|"``.
        catalog_code : str
            Allowed catalog codes. Multiple values may be joined with ``"|"``.
        sub_frame : str
            ADES ``subFrm`` selector for the originally submitted angular frame.
            Multiple values may be joined with ``"|"``.
        date_min : Time or None
            Inclusive lower time bound in ``UTC`` for rule validity.
        date_max : Time or None
            Exclusive upper time bound in ``UTC`` for rule validity.
        sigma : float
            Scalar one-axis optical uncertainty in arcseconds before conversion to radians by the policy.
        comment : str
            Human-readable rule annotation carried from the source table.
        """
        self.obs_type = obs_type
        self.obs_mode = obs_mode
        self.observer_type = observer_type
        self.obs_note_code = obs_note_code
        self.site_code = site_code
        self.program_code = program_code
        self.catalog_code = catalog_code
        self.sub_frame = sub_frame
        self.date_min = date_min
        self.date_max = date_max
        self.sigma = sigma
        self.comment = comment

    def match_mask(self, obs: OpticalObservationData) -> Bool[np.ndarray, "N"]:
        """Return the rows matched by this rule.

        Parameters
        ----------
        obs : OpticalObservationData
            Optical observation table to test. Matching uses observation type, observation mode, observer type, note code, receiver code, program code, catalog code, submitted frame, and optional time bounds.

        Returns
        -------
        Bool[np.ndarray, "N"]
            Boolean mask with one entry per input row. ``True`` marks rows that satisfy every active selector in the rule.

        Notes
        -----
        Time comparisons are performed against the observation epochs stored in ``obs.t``. ``date_min`` is inclusive and ``date_max`` is exclusive.
        """
        mask = np.ones(len(obs), dtype=bool)
        if self.obs_type != self.WILDCARD:
            obs_type_id = ObsType[self.obs_type].id
            mask &= obs.obs_type_ids == obs_type_id

        if self.obs_mode != self.WILDCARD:
            obs_mode_ids = [ObsMode[m].id for m in self.obs_mode.split(self.OR)]
            mask &= np.isin(obs.obs_mode_ids, obs_mode_ids)

        if self.observer_type != self.WILDCARD:
            observer_type_ids = [ObserverType[t].id for t in self.observer_type.split(self.OR)]
            mask &= np.isin(obs.observer_type_ids, observer_type_ids)

        if self.obs_note_code != self.WILDCARD:
            mask &= np.char.find(obs.note_codes, self.obs_note_code) >= 0

        if self.site_code != self.WILDCARD:
            mask &= np.asarray(parse_site_keys(obs.rx_codes).codes, dtype=str) == self.site_code

        if self.program_code != self.WILDCARD:
            program_codes = [c for c in self.program_code.split(self.OR)]
            mask &= np.isin(obs.program_codes, program_codes)

        if self.catalog_code != self.WILDCARD:
            catalog_codes = [c for c in self.catalog_code.split(self.OR)]
            mask &= np.isin(obs.catalog_codes, catalog_codes)

        if self.sub_frame != self.WILDCARD:
            sub_frames = [s for s in self.sub_frame.split(self.OR)]
            mask &= np.isin(obs.sub_frames, sub_frames)

        if self.date_min is not None:
            mask &= np.asarray(obs.t >= self.date_min)
        if self.date_max is not None:
            mask &= np.asarray(obs.t < self.date_max)

        return mask


class WeightResult:
    """Resolved uncertainties, correlations, and derived weight matrices."""

    def __init__(
            self,
            optical_uncertainties: Float[np.ndarray, "N_optical 2"],
            radar_uncertainties: Float[np.ndarray, "N_radar"],
            optical_sources: Shaped[np.ndarray, "N_optical"],
            radar_sources: Shaped[np.ndarray, "N_radar"],
            optical_correlations: Float[np.ndarray, "N_optical"],
            optical_time_uncertainties: Float[np.ndarray, "N_optical"],
    ):
        """Initialize a weight-result bundle."""
        optical_uncertainties = np.asarray(optical_uncertainties, dtype=float)
        radar_uncertainties = np.asarray(radar_uncertainties, dtype=float)
        optical_sources = np.asarray(optical_sources)
        radar_sources = np.asarray(radar_sources)
        optical_correlations = np.asarray(optical_correlations, dtype=float)
        optical_time_uncertainties = np.asarray(optical_time_uncertainties, dtype=float)

        if optical_uncertainties.ndim != 2 or optical_uncertainties.shape[1] != 2:
            raise ValueError(
                f"`optical_uncertainties` must have shape (N_optical, 2), got {optical_uncertainties.shape}."
            )
        if radar_uncertainties.ndim != 1:
            raise ValueError(
                f"`radar_uncertainties` must have shape (N_radar,), got {radar_uncertainties.shape}."
            )
        if optical_sources.shape != (optical_uncertainties.shape[0],):
            raise ValueError(
                f"`optical_sources` must have shape ({optical_uncertainties.shape[0]},), got {optical_sources.shape}."
            )
        if radar_sources.shape != (radar_uncertainties.shape[0],):
            raise ValueError(
                f"`radar_sources` must have shape ({radar_uncertainties.shape[0]},), got {radar_sources.shape}."
            )
        if optical_correlations.shape != (optical_uncertainties.shape[0],):
            raise ValueError(
                f"`optical_correlations` must have shape ({optical_uncertainties.shape[0]},), got {optical_correlations.shape}."
            )
        if not np.all(np.isfinite(optical_correlations)):
            raise ValueError("`optical_correlations` must contain finite values.")
        if np.any(np.abs(optical_correlations) >= 1.0):
            raise ValueError("`optical_correlations` must be strictly between -1 and 1.")
        if optical_time_uncertainties.shape != (optical_uncertainties.shape[0],):
            raise ValueError(
                f"`optical_time_uncertainties` must have shape ({optical_uncertainties.shape[0]},), got {optical_time_uncertainties.shape}."
            )
        if np.any(optical_time_uncertainties[np.isfinite(optical_time_uncertainties)] < 0.0):
            raise ValueError("`optical_time_uncertainties` must be non-negative where finite.")

        self.optical_uncertainties = optical_uncertainties
        self.optical_correlations = optical_correlations
        self.optical_time_uncertainties = optical_time_uncertainties
        self.radar_uncertainties = radar_uncertainties
        self.optical_sources = optical_sources
        self.radar_sources = radar_sources

    @staticmethod
    def _covariance_to_uncertainty_components(
            optical_covariances: Float[np.ndarray, "N_optical 2 2"],
    ) -> tuple[Float[np.ndarray, "N_optical 2"], Float[np.ndarray, "N_optical"]]:
        """Return marginal sigmas and correlations from covariance blocks."""
        covariances = np.asarray(optical_covariances, dtype=float)
        if covariances.ndim != 3 or covariances.shape[-2:] != (2, 2):
            raise ValueError(f"`optical_covariances` must have shape (N_optical, 2, 2), got {covariances.shape}.")
        if not np.all(np.isfinite(covariances)):
            raise ValueError("`optical_covariances` must contain finite values.")
        if not np.allclose(covariances, np.swapaxes(covariances, -1, -2)):
            raise ValueError("`optical_covariances` must be symmetric.")
        sigma_x = np.sqrt(covariances[:, 0, 0])
        sigma_y = np.sqrt(covariances[:, 1, 1])
        if np.any(~np.isfinite(sigma_x)) or np.any(~np.isfinite(sigma_y)) or np.any(sigma_x <= 0.0) or np.any(sigma_y <= 0.0):
            raise ValueError("`optical_covariances` must have positive finite marginal variances.")
        correlations = covariances[:, 0, 1] / (sigma_x * sigma_y)
        eps = np.finfo(float).eps
        correlations = np.clip(correlations, -1.0 + eps, 1.0 - eps)
        return np.stack([sigma_x, sigma_y], axis=1), correlations

    @property
    def optical_weights(self) -> Float[np.ndarray, "N_optical 2"]:
        """Return the marginal optical inverse-variance weights."""
        return 1. / (self.optical_uncertainties * self.optical_uncertainties)

    @property
    def optical_covariances(self) -> Float[np.ndarray, "N_optical 2 2"]:
        """Return per-row optical covariance blocks."""
        covariances = np.zeros((len(self.optical_uncertainties), 2, 2), dtype=float)
        sigma_x = self.optical_uncertainties[:, 0]
        sigma_y = self.optical_uncertainties[:, 1]
        covariance_xy = self.optical_correlations * sigma_x * sigma_y
        covariances[:, 0, 0] = sigma_x * sigma_x
        covariances[:, 1, 1] = sigma_y * sigma_y
        covariances[:, 0, 1] = covariance_xy
        covariances[:, 1, 0] = covariance_xy
        return covariances

    def optical_covariances_with_time(self, optical_rates: Float[np.ndarray, "N_optical 2"]) -> Float[np.ndarray, "N_optical 2 2"]:
        """Return optical covariance blocks inflated by time uncertainty.

        Parameters
        ----------
        optical_rates : ndarray, shape (N_optical, 2)
            Tangent-plane angular rates in radians per day, ordered as
            ``(ra_dot_cos_dec, dec_dot)``. The rates must use the same angular
            units as ``optical_uncertainties``.

        Returns
        -------
        ndarray, shape (N_optical, 2, 2)
            Optical covariance matrices after adding
            ``rate rate^T * (sigma_time / 86400)^2`` for finite, non-zero time
            uncertainties.
        """
        rates = np.asarray(optical_rates, dtype=float)
        if rates.shape != self.optical_uncertainties.shape:
            raise ValueError(f"`optical_rates` must have shape {self.optical_uncertainties.shape}, got {rates.shape}.")
        covariances = self.optical_covariances.copy()
        finite_mask = np.isfinite(self.optical_time_uncertainties) & (self.optical_time_uncertainties != 0.0)
        if finite_mask.any():
            time_days = self.optical_time_uncertainties[finite_mask] / float(DAY_S)
            covariances[finite_mask] += np.einsum("ni,nj,n->nij", rates[finite_mask], rates[finite_mask], time_days * time_days)
        return covariances

    def with_optical_time_rates(self, optical_rates: Float[np.ndarray, "N_optical 2"]) -> "WeightResult":
        """Return a weight result whose optical covariances include time uncertainty.

        Parameters
        ----------
        optical_rates : ndarray, shape (N_optical, 2)
            Tangent-plane angular rates in radians per day, ordered as
            ``(ra_dot_cos_dec, dec_dot)``.

        Returns
        -------
        WeightResult
            New bundle with effective optical sigmas and correlations derived
            from the time-inflated covariance matrices. The returned
            ``optical_time_uncertainties`` are zeroed because their contribution
            has already been folded into the covariance.
        """
        optical_covariances = self.optical_covariances_with_time(optical_rates)
        optical_uncertainties, optical_correlations = self._covariance_to_uncertainty_components(optical_covariances)
        return WeightResult(
            optical_uncertainties,
            self.radar_uncertainties.copy(),
            self.optical_sources.copy(),
            self.radar_sources.copy(),
            optical_correlations,
            np.zeros_like(self.optical_time_uncertainties),
        )

    @property
    def optical_weight_matrices(self) -> Float[np.ndarray, "N_optical 2 2"]:
        """Return per-row optical weight matrices."""
        weight_matrices = np.zeros((len(self.optical_uncertainties), 2, 2), dtype=float)
        sigma_x = self.optical_uncertainties[:, 0]
        sigma_y = self.optical_uncertainties[:, 1]
        rho = self.optical_correlations
        scale = 1.0 / (1.0 - rho * rho)
        weight_matrices[:, 0, 0] = scale / (sigma_x * sigma_x)
        weight_matrices[:, 1, 1] = scale / (sigma_y * sigma_y)
        off_diag = -rho * scale / (sigma_x * sigma_y)
        weight_matrices[:, 0, 1] = off_diag
        weight_matrices[:, 1, 0] = off_diag
        return weight_matrices

    def optical_weight_matrices_with_time(self, optical_rates: Float[np.ndarray, "N_optical 2"]) -> Float[np.ndarray, "N_optical 2 2"]:
        """Return optical inverse-covariance blocks inflated by time uncertainty."""
        return np.linalg.inv(self.optical_covariances_with_time(optical_rates))

    @property
    def radar_weights(self) -> Float[np.ndarray, "N_radar"]:
        """Return radar inverse-variance weights."""
        return 1. / (self.radar_uncertainties * self.radar_uncertainties)

    @property
    def flat_weights(self) -> Float[np.ndarray, "N_flat"]:
        """Return the marginal inverse-variance view in optical-then-radar order."""
        return np.concatenate([self.optical_weights.reshape(-1), self.radar_weights.reshape(-1)], axis=0)


class WeightPolicy(ABC):
    """Abstract interface for mapping observations to weighting results."""

    @abstractmethod
    def weights(self, obs: ObservationData) -> WeightResult:
        """Build modality-specific uncertainties for one observation bundle.

        Parameters
        ----------
        obs : ObservationData
            Single-target observation bundle containing optical and radar modality tables.

        Returns
        -------
        WeightResult
            Resolved uncertainties and derived source labels for the available observation rows.
        """
        pass


class UnitWeightPolicy(WeightPolicy):
    """Base class for deterministic weighting policies with one named source."""

    @property
    def name(self) -> str:
        """Return a human-readable policy name."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the source label attached to uncertainties from this policy."""
        pass


class InteractiveWeightPolicy(WeightPolicy):
    """Compose a default unit policy with row-level manual and policy overrides.

    The interactive policy starts from a default :class:`UnitWeightPolicy` result and then replaces selected rows by either manual sigmas or another registered policy. All override bookkeeping is keyed by the original mixed-input indices stored in the observation tables.
    """
    MANUAL_SOURCE = "MANUAL"

    def __init__(self, default_policy: UnitWeightPolicy, additional_policies: list[UnitWeightPolicy] = None):
        """Initialize an interactive weighting controller.

        Parameters
        ----------
        default_policy : UnitWeightPolicy
            Base policy applied to every row before interactive overrides are merged in.
        additional_policies : list[UnitWeightPolicy], optional
            Extra selectable policies available for row-level replacement.
        """
        self.default_policy = default_policy
        self.available_policies = [default_policy]
        if additional_policies:
            self.available_policies += additional_policies
        # Per-row policy overrides keyed by original mixed-input indices.
        self._selections: dict[int, UnitWeightPolicy] = {}

        # Manual uncertainty overrides, separated by modality.
        self._manual_index: set[int] = set()
        self._optical_manual_dict: dict[int, tuple[float, float, float]] = {}
        self._radar_manual_dict: dict[int, float] = {}

    @property
    def name(self) -> str:
        """Return a human-readable policy name."""
        return "Interactive Weight"

    def get_all_policy_results(self, obs: ObservationData) -> dict[WeightPolicy, WeightResult]:
        """Evaluate every registered unit policy on one observation bundle.

        Parameters
        ----------
        obs : ObservationData
            Observation bundle to evaluate.

        Returns
        -------
        dict[WeightPolicy, WeightResult]
            Mapping from registered policy object to its resolved uncertainty bundle.
        """
        return {policy: policy.weights(obs) for policy in self.available_policies}

    @staticmethod
    def _normalize_indices(input_indices: int | Iterable[int] | Float[ArrayLike, "..."]) -> list[int]:
        """Convert scalar or array-like index input to a Python integer list."""
        if isinstance(input_indices, int):
            return [int(input_indices)]

        return [int(i.item()) if hasattr(i, 'item') else int(i) for i in input_indices]

    def select_scheme(self, input_indices: int | Iterable[int] | Float[ArrayLike, "..."], policy: UnitWeightPolicy):
        """Assign a registered policy to specific input rows.

        Parameters
        ----------
        input_indices : int or iterable of int or array-like
            Original mixed-input indices of the rows to override.
        policy : UnitWeightPolicy
            Registered policy whose uncertainties should replace the default result on those rows.

        Raises
        ------
        ValueError
            Raised when ``policy`` has not been registered in ``available_policies``.
        """
        if policy not in self.available_policies:
            raise ValueError(f"Policy {policy.name} is not registered in available_policies.")
        for idx in self._normalize_indices(input_indices):
            self._selections[idx] = policy
            self._manual_index.discard(idx)

    def set_manual_optical(
            self,
            input_indices: int | Iterable[int] | Float[ArrayLike, "..."],
            ra_unc: float,
            dec_unc: float,
            correlation: float = 0.0,
    ):
        """Set manual optical sigmas for specific input rows.

        Parameters
        ----------
        input_indices : int or iterable of int or array-like
            Original mixed-input indices of the rows to override.
        ra_unc : float
            Right-ascension sigma in radians.
        dec_unc : float
            Declination sigma in radians.
        correlation : float, default=0.0
            Correlation coefficient for the two tangent-plane components.
        """
        for idx in self._normalize_indices(input_indices):
            self._manual_index.add(idx)
            self._optical_manual_dict[idx] = (ra_unc, dec_unc, correlation)

    def set_manual_radar(self, input_indices: int | Iterable[int] | Float[ArrayLike, "..."], unc: float):
        """Set manual radar sigmas for specific input rows.

        Parameters
        ----------
        input_indices : int or iterable of int or array-like
            Original mixed-input indices of the rows to override.
        unc : float
            Radar sigma in the native unit of the stored radar measurement.
        """
        for idx in self._normalize_indices(input_indices):
            self._manual_index.add(idx)
            self._radar_manual_dict[idx] = unc

    def restore_default_policy(self, input_indices: int | Iterable[int] | Float[ArrayLike, "..."] = None):
        """Remove interactive overrides and fall back to the default policy.

        Parameters
        ----------
        input_indices : int or iterable of int or array-like, optional
            Original mixed-input indices to reset. When omitted, every stored override is cleared.
        """
        if input_indices is not None:
            for idx in self._normalize_indices(input_indices):
                self._selections.pop(idx, None)
                self._manual_index.discard(idx)
                self._optical_manual_dict.pop(idx, None)
                self._radar_manual_dict.pop(idx, None)
        else:
            self._selections.clear()
            self._manual_index.clear()
            self._optical_manual_dict.clear()
            self._radar_manual_dict.clear()

    def weights(self, obs: ObservationData) -> WeightResult:
        """Merge the default policy with interactive row-level overrides.

        Parameters
        ----------
        obs : ObservationData
            Observation bundle to weight.

        Returns
        -------
        WeightResult
            Weight bundle obtained from the default policy after manual overrides and selected policy substitutions have been applied row by row.

        Notes
        -----
        Manual overrides take precedence over selected alternate policies. If an alternate policy returns ``NaN`` for a row, the default-policy uncertainty is kept for that row.
        """
        used_policies = set(self._selections.values())
        used_policies.add(self.default_policy)

        all_results = {p: p.weights(obs) for p in used_policies}

        default_result = all_results[self.default_policy]
        optical_unc = default_result.optical_uncertainties.copy()
        optical_corr = default_result.optical_correlations.copy()
        optical_time_unc = default_result.optical_time_uncertainties.copy()
        optical_source = default_result.optical_sources.copy()
        radar_unc, radar_source = default_result.radar_uncertainties.copy(), default_result.radar_sources.copy()

        def apply_optical_overrides(optical_obs, unc, corr, time_unc, source):
            if len(optical_obs) == 0:
                return
            for i, idx in enumerate(optical_obs.input_indices):
                idx = int(idx.item()) if hasattr(idx, "item") else int(idx)
                if idx in self._manual_index and idx in self._optical_manual_dict:
                    ra_unc, dec_unc, correlation = self._optical_manual_dict[idx]
                    unc[i] = (ra_unc, dec_unc)
                    corr[i] = correlation
                    source[i] = self.MANUAL_SOURCE
                    continue
                selected_policy = self._selections.get(idx, None)
                if selected_policy:
                    selected_result = all_results[selected_policy]
                    selected_unc = selected_result.optical_uncertainties[i]
                    if not np.isnan(selected_unc).any():
                        unc[i] = selected_unc
                        corr[i] = selected_result.optical_correlations[i]
                        time_unc[i] = selected_result.optical_time_uncertainties[i]
                        source[i] = selected_result.optical_sources[i]

        def apply_radar_overrides(radar_obs, unc, source):
            if len(radar_obs) == 0:
                return
            for i, idx in enumerate(radar_obs.input_indices):
                idx = int(idx.item()) if hasattr(idx, "item") else int(idx)
                if idx in self._manual_index and idx in self._radar_manual_dict:
                    unc[i] = self._radar_manual_dict[idx]
                    source[i] = self.MANUAL_SOURCE
                    continue
                selected_policy = self._selections.get(idx, None)
                if selected_policy:
                    selected_result = all_results[selected_policy]
                    selected_unc = selected_result.radar_uncertainties[i]
                    if not np.isnan(selected_unc).any():
                        unc[i] = selected_unc
                        source[i] = selected_result.radar_sources[i]

        if obs.has_optical:
            apply_optical_overrides(obs.optical, optical_unc, optical_corr, optical_time_unc, optical_source)
        if obs.has_radar:
            apply_radar_overrides(obs.radar, radar_unc, radar_source)

        return WeightResult(optical_unc, radar_unc, optical_source, radar_source, optical_corr, optical_time_unc)


class ADESWeightPolicy(UnitWeightPolicy):
    """Use the uncertainties already reported in the observation records."""

    @property
    def name(self) -> str:
        """Return a human-readable policy name."""
        return "ADES Reported"

    @property
    def source_name(self) -> str:
        """Return the source label attached to this policy."""
        return "ADES"

    def weights(self, obs: ObservationData) -> WeightResult:
        """Copy the reported observation uncertainties into a weight bundle.

        Parameters
        ----------
        obs : ObservationData
            Observation bundle whose stored uncertainties should be used directly.

        Returns
        -------
        WeightResult
            Weight bundle built from the modality-specific uncertainties already stored in ``obs``.
        """
        optical_uncertainties = obs.optical.uncertainties.copy()
        optical_correlations = obs.optical.correlations.copy()
        optical_time_uncertainties = obs.optical.time_uncertainties.copy()
        radar_uncertainties = obs.radar.uncertainties.copy()
        optical_sources = np.array([self.source_name] * len(optical_uncertainties))
        radar_sources = np.array([self.source_name] * len(radar_uncertainties))
        return WeightResult(
            optical_uncertainties,
            radar_uncertainties,
            optical_sources,
            radar_sources,
            optical_correlations,
            optical_time_uncertainties,
        )


class VFCC17WeightPolicy(UnitWeightPolicy):
    """Apply the Vereš-Farnocchia-Chesley-Chatelain 2017 optical weighting table.

    The policy reads a CSV rule table and assigns one scalar optical sigma per matched observation row. The scalar sigma is then duplicated onto right ascension and declination. Radar rows are passed through unchanged from the input observation bundle.
    """
    WILDCARD = "*"
    COMMENT = "!"

    def __init__(self, filepath: str = DEFAULT_WEIGHT_SCHEME_FILENAME):
        """Initialize the ``VFCC17`` weighting policy from a CSV rule table.

        Parameters
        ----------
        filepath : str, default=DEFAULT_WEIGHT_SCHEME_FILENAME
            Path to the weighting-rule CSV file. The file is read with ``"!"`` as the comment prefix.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(missing_data_message("weights", filepath))
        df = pd.read_csv(filepath, comment=self.COMMENT)
        df = df.where(pd.notnull(df), None)  # convert empty string to None
        rules = []
        for _, row in df.iterrows():
            if row.date_min != self.WILDCARD:
                year, month, day = row.date_min.split('-')
                date_min = Time.from_ut_date(float(year.strip()), float(month.strip()),
                                             float(day.strip()))
            else:
                date_min = None
            if row.date_max != self.WILDCARD:
                year, month, day = row.date_max.split('-')
                date_max = Time.from_ut_date(float(year.strip()), float(month.strip()),
                                             float(day.strip()))
            else:
                date_max = None
            rules.append(WeightRule(obs_type=row.obs_type, obs_mode=row.obs_mode,
                                    observer_type=row.observer_type, obs_note_code=row.obs_note_code, site_code=row.site_code,
                                    program_code=row.program_code, catalog_code=row.catalog_code,
                                    sub_frame=row.sub_frame,
                                    date_min=date_min, date_max=date_max,
                                    sigma=row.sigma,
                                    comment=row.comment))
        self.rules = rules

    @property
    def name(self) -> str:
        """Return a human-readable policy name."""
        return "VFCC17"

    @property
    def source_name(self) -> str:
        """Return the source label attached to this policy."""
        return "VFCC17"

    def uncertainties(self, obs: ObservationData) -> Float[np.ndarray, "N_optical 2"]:
        """Resolve ``VFCC17`` optical sigmas for one observation bundle.

        Parameters
        ----------
        obs : ObservationData
            Observation bundle whose optical rows should be matched against the
            loaded weighting rules.

        Returns
        -------
        Float[np.ndarray, "N_optical 2"]
            Optical sigmas in radians, stored as ``(sigma_ra, sigma_dec)`` with
            one row per optical observation.

        Notes
        -----
        The rule table stores one optical sigma value in arcseconds per matched row. This method converts that scalar to radians and applies it identically to right ascension and declination.

        The over-observation scaling follows the ``N/4`` rule from
        Vereš et al. (2017). Rows are counted per tracklet identifier within
        the unified optical table.

        References
        ----------
        1. Vereš, P., Farnocchia, D., Chesley, S. R., & Chamberlin, A. B. (2017). Statistical analysis of astrometric errors for the most productive asteroid surveys. Icarus, 296, 139-144.
        """
        optical_sigmas = np.deg2rad(np.zeros(len(obs.optical)) * ARCSEC_TO_DEG)
        matched_mask = np.zeros(len(obs.optical), dtype=bool)
        for rule in self.rules:
            rule_mask = rule.match_mask(obs.optical)
            matched_mask |= rule_mask
            optical_sigmas[rule_mask] = np.deg2rad(rule.sigma * ARCSEC_TO_DEG)
        n_max = 4
        tracklet_counts = _tracklet_group_counts(obs.optical.trk_ids)
        over_mask = (tracklet_counts > n_max) & matched_mask
        optical_sigmas[over_mask] *= np.sqrt(tracklet_counts[over_mask] / n_max)
        return np.repeat(optical_sigmas, 2).reshape(-1, 2)

    def time_uncertainties(self, obs: ObservationData) -> Float[np.ndarray, "N_optical"]:
        """Resolve VFCC17 optical time uncertainties in seconds.

        Missing ordinary optical times receive the GRSS/VFCC17 default of
        ``1.0`` second. Gaia, occultation, and space-observer rows are left as
        ``NaN`` so downstream covariance inflation skips them.
        """
        time_uncertainties = obs.optical.time_uncertainties.copy()
        if len(time_uncertainties) == 0:
            return time_uncertainties
        site_keys = parse_site_keys(obs.optical.rx_codes)
        site_codes = np.asarray(site_keys.codes, dtype=str)
        gaia_mask = site_codes == "258"
        occultation_mask = obs.optical.obs_mode_ids == ObsMode.OCCULTATION.id
        space_mask = obs.optical.observer_type_ids == ObserverType.SPACE_BASED.id
        no_default_mask = gaia_mask | occultation_mask | space_mask
        time_uncertainties[no_default_mask] = np.nan
        default_mask = ~np.isfinite(time_uncertainties) & ~no_default_mask
        time_uncertainties[default_mask] = 1.0
        return time_uncertainties

    def weights(self, obs: ObservationData) -> WeightResult:
        """Build a full weight bundle from the ``VFCC17`` optical sigmas.

        Parameters
        ----------
        obs : ObservationData
            Observation bundle to weight.

        Returns
        -------
        WeightResult
            Weight bundle with ``VFCC17`` optical uncertainties and reported radar uncertainties.
        """
        optical_uncertainties = self.uncertainties(obs)
        optical_correlations = np.zeros(len(optical_uncertainties), dtype=float)
        optical_time_uncertainties = self.time_uncertainties(obs)
        radar_uncertainties = obs.radar.uncertainties
        optical_sources = np.array([self.source_name] * len(optical_uncertainties))
        radar_sources = np.array(["ADES"] * len(radar_uncertainties))
        return WeightResult(
            optical_uncertainties,
            radar_uncertainties,
            optical_sources,
            radar_sources,
            optical_correlations,
            optical_time_uncertainties,
        )
