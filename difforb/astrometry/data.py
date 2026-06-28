"""Observation containers for single-target astrometry.

This module stores optical observations in one table regardless of observer
location. Observer coordinates live in the receiver key string and are resolved
by :mod:`difforb.body.site` when a site state is needed.
"""

from __future__ import annotations

import math
from enum import Enum, unique
from typing import Tuple, Iterable

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int, Shaped

from difforb.core.time.timescale import Time
from difforb.body.site import parse_site_keys, site_display_labels, site_identity_keys
from difforb.report.text import build_repr, format_float_array, format_optional_shape

jax.config.update("jax_enable_x64", True)


def _format_time_bound(jd: np.ndarray, bound: str) -> str | None:
    values = np.asarray(jd)
    if values.size == 0:
        return None
    scalar = values.min() if bound == "min" else values.max()
    return format_float_array(scalar, precision=9, scientific=False, signed=False)


def add_station_identity_columns(
        frame,
        *,
        station_column: str = "rx_code",
):
    """Add station identity and display columns to an observation dataframe.

    Parameters
    ----------
    frame : pandas.DataFrame
        Observation dataframe with one row per observation.
    station_column : str, default="rx_code"
        Column that stores observer keys.

    Returns
    -------
    pandas.DataFrame
        Copy of ``frame`` with ``station_key`` and ``station`` columns.
    """
    keyed = frame.copy()
    keys = keyed[station_column].to_numpy(dtype=str)
    keyed["station_key"] = site_identity_keys(keys)
    keyed["station"] = site_display_labels(keys)
    return keyed


class ObsType(Enum):
    OPTICAL = (1, 'optical')
    OCCULATION = (2, 'occulation')
    RADAR = (3, 'radar')

    __id_lookup = {}

    def __init__(self, id: int, label: str):
        self.id = id
        self.label = label

    @classmethod
    def from_ids(cls, ids) -> list['ObsType']:
        if not cls.__id_lookup:
            cls.__id_lookup = {t.id: t for t in cls}
        try:
            return [cls.__id_lookup[id.item()] for id in ids]
        except KeyError as e:
            raise ValueError(f"Unknown observation type: {e}")


@unique
class ObsMode(Enum):
    """
    Observation Mode Enumeration
    Value format: (Internal ID, ADES MODE, Description)
    """
    CCD = (101, 'CCD', "CCD")
    CMOS = (102, 'CMO', "CMOS")
    VIDEO = (103, 'VID', "Mini-normal place from Video frames")
    PHOTO = (104, 'PHO', "Photographic")
    ENCODER = (105, 'ENC', "Encoder")
    PMT = (106, 'PMT', "Photo-Multiplier Tube")
    MICRO = (107, 'MIC', "Micrometer")
    MERIDIAN = (108, 'MER', "Meridian/Transit circle")
    TDI = (109, 'TDI', "Time-delay Integration CCD")
    OCCULTATION = (210, 'OCC', "Occultation")
    DELAY_CENTER = (301, None, "Radar Delay (Center)")
    DELAY_BOUNCE = (302, None, "Radar Delay")
    DOPPLER_CENTER = (303, None, "Radar Doppler (Center)")
    DOPPLER_BOUNCE = (304, None, "Radar Doppler")
    UNKNOWN = (401, 'UNK', "Unknown mode")

    __id_lookup = {}

    def __init__(self, id: int, ades_mode: str | list[str], desc: str):
        self.id = id
        self.ades_mode = ades_mode
        self.desc = desc

    @classmethod
    def from_ids(cls, ids) -> list['ObsMode']:
        if not cls.__id_lookup:
            cls.__id_lookup = {mode.id: mode for mode in cls}
        try:
            return [cls.__id_lookup[id.item()] for id in ids]
        except KeyError as e:
            raise ValueError(f"Unknown observation mode: {e}")

    @classmethod
    def from_ades_mode(cls, ades_mode: str, is_delay: bool = None, com: int = None):
        if isinstance(ades_mode, float) and math.isnan(ades_mode):
            if com is None:
                raise ValueError(
                    f"``com`` must be specified if ``ades_mode`` is ``{cls.DELAY_CENTER.ades_mode}`` or ``{cls.DOPPLER_CENTER.ades_mode}``")
            elif not com in [1, 2]:
                raise ValueError(f"``com`` must be 1 or 2, not {com}``")

            if is_delay is None:
                raise ValueError(
                    f"``is_delay`` must be specified if ``ades_mode`` is ``{cls.DELAY_CENTER.ades_mode}`` or ``{cls.DOPPLER_CENTER.ades_mode}``")

            if is_delay:
                mode = cls.DELAY_CENTER if com == 1 else cls.DELAY_BOUNCE
            else:
                mode = cls.DOPPLER_CENTER if com == 1 else cls.DOPPLER_BOUNCE
        else:
            for m in cls:
                if ades_mode == m.ades_mode:
                    mode = m
                    break
            else:
                raise ValueError(f"Unknown ADES observation mode: {ades_mode}")
        return mode


class ObserverType(Enum):
    GROUND_FIXED = (1, 'fixed-ground')
    GROUND_ROVING = (2, 'roving-ground')
    SPACE_BASED = (3, 'space')

    __id_lookup = {}

    def __init__(self, id: int, label: str):
        self.id = id
        self.label = label

    @classmethod
    def from_ades_sys(cls, sys: str | float) -> 'ObserverType':
        if (isinstance(sys, float) and math.isnan(sys)) or (sys is None):
            return cls.GROUND_FIXED
        elif sys == 'WGS84':
            return cls.GROUND_ROVING
        elif sys in ['ICRFAU', 'ICRFKM', 'ICRF_AU', 'ICRF_KM']:
            return cls.SPACE_BASED
        else:
            raise ValueError(f"Unknown observer type: {sys}")

    @classmethod
    def from_ids(cls, ids) -> list['ObserverType']:
        if not cls.__id_lookup:
            cls.__id_lookup = {t.id: t for t in cls}
        try:
            return [cls.__id_lookup[id.item()] for id in ids]
        except KeyError as e:
            raise ValueError(f"Unknown observer type: {e}")


ADES_OBS_NOTE_INFO = {
    "A": "Earlier approximate position inferior",
    "a": "Sense of motion ambiguous",
    "B": "Bright sky/black or dark plate",
    "b": "Bad seeing",
    "c": "Crowded star field",
    "D": "Declination uncertain",
    "d": "Diffuse image",
    "E": "At or near edge of plate/frame",
    "e": "Extrapolation to zero aperture technique used",
    "F": "Faint image",
    "f": "Involved with emulsion, plate or CCD flaw",
    "G": "Poor guiding",
    "g": "No guiding",
    "H": "Hand measurement of CCD image",
    "h": "Observed through cloud/haze",
    "I": "Involved with star",
    "i": "Inkdot measured",
    "L": "Color corrected observations",
    "K": "Stacked image",
    "k": "Stare-mode observation by scanning system",
    "M": "Measurement difficult",
    "m": "Image tracked on object motion",
    "N": "Near edge of plate, measurement uncertain",
    "n": "Normal place",
    "O": "Image out of focus",
    "o": "Plate measured in one direction only",
    "P": "Position uncertain",
    "p": "Poor image",
    "R": "Right ascension uncertain",
    "r": "Poor distribution of reference stars",
    "S": "Poor sky",
    "s": "Streaked image",
    "T": "Time uncertain",
    "t": "Trailed image",
    "U": "Uncertain image",
    "u": "Unconfirmed image",
    "V": "Very faint image",
    "W": "Weak image",
    "w": "Weak solution",
    "Z": "astrometry from F51, F52, G96, I41, 703 (large surveys without program codes) reported by a non-survey measurerer/pipeline"
}


class OpticalObservationData:
    """Single-target optical observation table.

    Parameters
    ----------
    t : Time
        Observation epochs. The table is one-dimensional and ``t`` must have
        shape ``(N,)``.
    mode_ids : Int[np.ndarray, "N"]
        Internal observation-mode identifiers. These should map to optical
        ground-observation modes in :class:`difforb.type.ObsMode`.
    values : Float[np.ndarray, "N 2"]
        Angular observations in radians, ordered as right ascension and
        declination.
    uncertainties : Float[np.ndarray, "N 2"]
        Adopted angular uncertainties aligned with ``values``.
    correlations : Float[np.ndarray, "N"]
        Correlation coefficients for the two optical tangent-plane components.
    time_uncertainties : Float[np.ndarray, "N"]
        Observation-time uncertainties in seconds. Missing values are stored
        as ``NaN``.
    rx_codes : array-like of str
        Receiver observer keys. Fixed ground keys are plain observatory codes.
        Roving ground keys use ``"code @ lon_deg, lat_deg, alt_m"``. Space
        keys use ``"code # x_au, y_au, z_au"``.
    note1_codes : array-like of str
        Raw ``Note 1`` or equivalent program-code field from the source
        observation record.
    note2_codes : array-like of str
        Raw ``Note 2`` or equivalent technology-code field from the source
        observation record.
    catalog_codes : array-like of str
        Astrometric catalog codes. Missing values are represented by empty
        strings.
    magnitudes : Float[np.ndarray, "N"]
        Reported apparent magnitudes. Missing values are represented by
        ``NaN``.
    band_codes : array-like of str
        Reported magnitude-band codes. Missing values are represented by empty
        strings.
    sub_frames : array-like of str
        ADES ``subFrm`` values for the originally submitted angular frame.
        Missing values are represented by empty strings.
    input_indices : Int[np.ndarray, "N"], optional
        Positions of the rows in the original mixed input order. If omitted,
        the rows are assumed to already be in their original order.
    """

    def __init__(
            self,
            t: Time,
            trk_ids: Shaped[np.ndarray, "N"],
            obs_type_ids: Int[np.ndarray, "N"],
            obs_mode_ids: Int[np.ndarray, "N"],
            values: Float[np.ndarray, "N 2"],
            uncertainties: Float[np.ndarray, "N 2"],
            correlations: Float[np.ndarray, "N"],
            time_uncertainties: Float[np.ndarray, "N"],
            rx_codes: Shaped[np.ndarray, "N"],
            program_codes: Shaped[np.ndarray, "N"],
            catalog_codes: Shaped[np.ndarray, "N"],
            note_codes: Shaped[np.ndarray, "N"],
            magnitudes: Float[np.ndarray, "N"],
            band_codes: Shaped[np.ndarray, "N"],
            sub_frames: Shaped[np.ndarray, "N"],
            input_indices: Int[np.ndarray, "N"],
    ) -> None:
        values_arr = np.asarray(values, dtype=float)
        if values_arr.ndim != 2 or values_arr.shape[1] != 2:
            raise ValueError(f"`values` must have shape (N, 2), got {values_arr.shape}.")
        uncertainties_arr = np.asarray(uncertainties, dtype=float)
        if uncertainties_arr.ndim != 2 or uncertainties_arr.shape != values_arr.shape:
            raise ValueError(f"`uncertainties` must have shape {values_arr.shape}, got {uncertainties_arr.shape}.")
        correlations_arr = np.asarray(correlations, dtype=float)
        if correlations_arr.ndim == 0 and values_arr.shape[0] == 1:
            correlations_arr = correlations_arr.reshape(1)
        if correlations_arr.shape != (values_arr.shape[0],):
            raise ValueError(f"`correlations` must have shape ({values_arr.shape[0]},), got {correlations_arr.shape}.")
        if not np.all(np.isfinite(correlations_arr)):
            raise ValueError("`correlations` must contain finite values.")
        if np.any(np.abs(correlations_arr) >= 1.0):
            raise ValueError("`correlations` must be strictly between -1 and 1.")
        time_uncertainties_arr = np.asarray(time_uncertainties, dtype=float)
        if time_uncertainties_arr.ndim == 0 and values_arr.shape[0] == 1:
            time_uncertainties_arr = time_uncertainties_arr.reshape(1)
        if time_uncertainties_arr.shape != (values_arr.shape[0],):
            raise ValueError(
                f"`time_uncertainties` must have shape ({values_arr.shape[0]},), got {time_uncertainties_arr.shape}."
            )
        if np.any(time_uncertainties_arr[np.isfinite(time_uncertainties_arr)] < 0.0):
            raise ValueError("`time_uncertainties` must be non-negative where finite.")

        self.t = t
        self.trk_ids = trk_ids
        self.obs_type_ids = obs_type_ids
        self.obs_mode_ids = obs_mode_ids
        self.values = values_arr
        self.uncertainties = uncertainties_arr
        self.correlations = correlations_arr
        self.time_uncertainties = time_uncertainties_arr
        self.rx_codes = rx_codes
        self.program_codes = program_codes
        self.catalog_codes = catalog_codes
        self.note_codes = note_codes
        self.magnitudes = magnitudes
        self.band_codes = band_codes
        self.sub_frames = sub_frames
        self.input_indices = input_indices

    @property
    def shape(self) -> tuple[int]:
        """Return the table shape ``(N,)``."""
        return (self.values.shape[0],)

    @property
    def obs_types(self) -> list[ObsType]:
        return ObsType.from_ids(self.obs_type_ids)

    @property
    def obs_modes(self) -> list[ObsMode]:
        """Return the resolved optical observation modes."""
        return ObsMode.from_ids(self.obs_mode_ids)

    @property
    def observer_types(self) -> list[ObserverType]:
        return ObserverType.from_ids(self.observer_type_ids)

    @property
    def observer_type_ids(self) -> Int[np.ndarray, "N"]:
        parsed = parse_site_keys(self.rx_codes)
        ids = np.empty(len(parsed.kind_ids), dtype=int)
        ids[parsed.kind_ids == 0] = ObserverType.GROUND_FIXED.id
        ids[parsed.kind_ids == 1] = ObserverType.GROUND_ROVING.id
        ids[parsed.kind_ids == 2] = ObserverType.SPACE_BASED.id
        return ids

    @property
    def notes(self) -> list[str]:
        notes: list[str] = []
        for code_str in self.note_codes:
            if not code_str:
                notes.append('')
                continue
            row_notes = []
            for ch in code_str:
                row_notes.append(ADES_OBS_NOTE_INFO.get(ch, f"Unknown note code: {ch}"))
            notes.append('; '.join(row_notes))
        return notes

    @property
    def is_roving(self) -> Bool[np.ndarray, "N"]:
        """Mask roving-observer optical observations."""
        return self.observer_type_ids == ObserverType.GROUND_ROVING.id

    @property
    def is_space(self) -> Bool[np.ndarray, "N"]:
        return self.observer_type_ids == ObserverType.SPACE_BASED.id

    def sort_by_date(self, ascending: bool = True) -> Tuple["OpticalObservationData", Int[np.ndarray, "N"]]:
        """Sort the optical table by epoch."""
        jd = np.asarray(self.t.tt.jd)
        indices = np.argsort(jd if ascending else -jd)
        return self[indices], indices

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, idx) -> "OpticalObservationData":
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return self.__class__(
            t=self.t[idx],
            trk_ids=self.trk_ids[idx],
            obs_type_ids=self.obs_type_ids[idx],
            obs_mode_ids=self.obs_mode_ids[idx],
            values=self.values[idx],
            uncertainties=self.uncertainties[idx],
            time_uncertainties=self.time_uncertainties[idx],
            correlations=self.correlations[idx],
            rx_codes=self.rx_codes[idx],
            program_codes=self.program_codes[idx],
            catalog_codes=self.catalog_codes[idx],
            note_codes=self.note_codes[idx],
            magnitudes=self.magnitudes[idx],
            band_codes=self.band_codes[idx],
            sub_frames=self.sub_frames[idx],
            input_indices=self.input_indices[idx],
        )

    def to_dataframe(self):
        """Convert the optical table to a pandas dataframe.

        Returns
        -------
        pandas.DataFrame
            Row-wise tabular view of the optical observations. The
            dataframe keeps one row per observation and exposes the angular
            values, adopted uncertainties, observer keys, catalog codes, and
            original input-order indices.

        Notes
        -----
        This method is intended for inspection, export, and debugging. It does
        not participate in the numerical ``JAX`` pipelines.
        """
        import pandas as pd

        obs_type_names = [t.label for t in self.obs_types]
        obs_mode_names = [mode.ades_mode for mode in self.obs_modes]
        observer_type_names = [t.label for t in self.observer_types]
        frame = pd.DataFrame({
            "input_index": self.input_indices,
            "t_ut_jd": np.asarray(self.t.ut.jd),
            "t_iso": np.asarray(self.t.ut.iso_string),
            "trk_id": self.trk_ids,
            "obs_type": obs_type_names,
            "obs_mode": obs_mode_names,
            "ra_deg": np.rad2deg(self.values[:, 0]),
            "dec_deg": np.rad2deg(self.values[:, 1]),
            "ra_uncertainty_arcsec": np.rad2deg(self.uncertainties[:, 0]) * 3600.0,
            "dec_uncertainty_arcsec": np.rad2deg(self.uncertainties[:, 1]) * 3600.0,
            "ra_dec_correlation": self.correlations,
            "time_uncertainty_s": self.time_uncertainties,
            "rx_code": self.rx_codes,
            "program_code": self.program_codes,
            "catalog_code": self.catalog_codes,
            "note_code": self.note_codes,
            "notes": self.notes,
            "magnitude": self.magnitudes,
            "mag_band_code": self.band_codes,
            "sub_frame": self.sub_frames,
            "observer_type": observer_type_names,
        })
        return add_station_identity_columns(frame)

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("n_obs", str(len(self))),
                ("ut_start_jd", _format_time_bound(self.t.ut.jd, "min")),
                ("ut_end_jd", _format_time_bound(self.t.ut.jd, "max")),
            ],
        )


class RadarObservationData:
    """Single-target radar observation table.

    Parameters
    ----------
    t : Time
        Observation epochs with shape ``(N,)``.
    mode_ids : Int[np.ndarray, "N"]
        Internal observation-mode identifiers. These should map to radar modes
        in :class:`difforb.type.ObsMode`.
    values : Float[np.ndarray, "N"]
        Radar measurements stored as a one-dimensional series. The physical
        meaning is determined by ``mode_ids``, for example delay/range or
        Doppler/range-rate.
    uncertainties : Float[np.ndarray, "N"]
        Adopted measurement uncertainties aligned with ``values``.
    rx_codes : array-like of str
        Receiver station codes.
    tx_codes : array-like of str
        Transmitter station codes.
    tx_freq : Float[np.ndarray, "N"], optional
        Full-length transmitter frequencies in ``Hz``. Missing values are
        initialized to ``NaN``.
    input_indices : Int[np.ndarray, "N"], optional
        Positions of the rows in the original mixed input order. If omitted,
        the rows are assumed to already be in their original order.
    """

    def __init__(
            self,
            t: Time,
            obs_type_ids: Int[np.ndarray, "N"],
            obs_mode_ids: Int[np.ndarray, "N"],
            values: Float[np.ndarray, "N"],
            uncertainties: Float[np.ndarray, "N"],
            rx_codes: Shaped[np.ndarray, "N"],
            tx_codes: Shaped[np.ndarray, "N"],
            tx_freq: Float[np.ndarray, "N"],
            input_indices: Int[np.ndarray, "N"],
    ) -> None:
        values_arr = np.asarray(values, dtype=float)
        if values_arr.ndim == 0:
            values_arr = values_arr.reshape(1)
        if values_arr.ndim != 1:
            raise ValueError(f"`values` must have shape (N,), got {values_arr.shape}.")

        self.t = t
        self.obs_type_ids = obs_type_ids
        self.obs_mode_ids = obs_mode_ids
        self.values = values_arr
        self.uncertainties = uncertainties
        self.rx_codes = rx_codes
        self.tx_codes = tx_codes
        self.tx_freq = tx_freq
        self.input_indices = input_indices

    @property
    def shape(self) -> tuple[int]:
        """Return the table shape ``(N,)``."""
        return (self.values.shape[0],)

    @property
    def obs_types(self) -> list[ObsType]:
        return ObsType.from_ids(self.obs_type_ids)

    @property
    def obs_modes(self) -> list[ObsMode]:
        """Return the resolved radar observation modes."""
        return ObsMode.from_ids(self.obs_mode_ids)

    @property
    def is_delay(self) -> Bool[np.ndarray, "N"]:
        """Mask radar range or delay observations."""
        return (self.obs_mode_ids == ObsMode.DELAY_CENTER.id) | (self.obs_mode_ids == ObsMode.DELAY_BOUNCE.id)

    @property
    def is_doppler(self) -> Bool[np.ndarray, "N"]:
        """Mask radar Doppler or range-rate observations."""
        return (self.obs_mode_ids == ObsMode.DOPPLER_CENTER.id) | (self.obs_mode_ids == ObsMode.DOPPLER_BOUNCE.id)

    def sort_by_date(self, ascending: bool = True) -> Tuple["RadarObservationData", Int[np.ndarray, "N"]]:
        """Sort the radar table by epoch."""
        jd = np.asarray(self.t.tt.jd)
        indices = np.argsort(jd if ascending else -jd)
        return self[indices], indices

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, idx) -> "RadarObservationData":
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return self.__class__(
            t=self.t[idx],
            obs_type_ids=self.obs_type_ids[idx],
            obs_mode_ids=self.obs_mode_ids[idx],
            values=self.values[idx],
            uncertainties=self.uncertainties[idx],
            rx_codes=self.rx_codes[idx],
            tx_codes=self.tx_codes[idx],
            tx_freq=self.tx_freq[idx],
            input_indices=self.input_indices[idx],
        )

    def to_dataframe(self):
        """Convert the radar table to a pandas dataframe.

        Returns
        -------
        pandas.DataFrame
            Row-wise tabular view of the radar observations. The dataframe
            keeps one row per observation and exposes the scalar measurement,
            adopted uncertainty, receiver and transmitter codes, transmitter
            frequency, and original input-order index.

        Notes
        -----
        This method is intended for inspection, export, and debugging. It does
        not participate in the numerical ``JAX`` pipelines.
        """
        import pandas as pd

        obs_type_names = [t.label for t in self.obs_types]
        obs_mode_names = [mode.desc for mode in self.obs_modes]
        value_units = np.where(self.is_delay, "us", "Hz")
        frame = pd.DataFrame({
            "input_index": self.input_indices,
            "t_ut_jd": np.asarray(self.t.ut.jd),
            "t_iso": np.asarray(self.t.ut.iso_string),
            "obs_type": obs_type_names,
            "obs_mode": obs_mode_names,
            "radar_measurement": self.values,
            "radar_uncertainty": self.uncertainties,
            "radar_unit": value_units,
            "rx_code": self.rx_codes,
            "tx_code": self.tx_codes,
            "tx_freq_hz": self.tx_freq,
        })
        return add_station_identity_columns(frame)

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("n_obs", str(len(self))),
                ("ut_start_jd", _format_time_bound(self.t.ut.jd, "min")),
                ("ut_end_jd", _format_time_bound(self.t.ut.jd, "max"))
            ],
        )


class ObservationData:
    """Single-target observation bundle with optical and radar tables."""

    def __init__(
            self,
            name: str,
            optical: OpticalObservationData,
            radar: RadarObservationData,
    ) -> None:
        self.name = name
        self.optical = optical
        self.radar = radar

    @property
    def has_optical(self) -> bool:
        """Return ``True`` when the bundle contains optical observations."""
        return len(self.optical) > 0

    @property
    def has_radar(self) -> bool:
        """Return ``True`` when the bundle contains radar observations."""
        return len(self.radar) > 0

    @property
    def is_mixed(self) -> bool:
        """Return ``True`` when more than one modality table is present."""
        return int(self.has_optical) + int(self.has_radar) > 1

    @property
    def num_optical(self) -> int:
        """Return the number of bundled optical observations."""
        return len(self.optical)

    @property
    def num_radar(self) -> int:
        """Return the number of bundled radar observations."""
        return len(self.radar)

    @property
    def num_observations(self) -> int:
        """Return the total number of bundled observations."""
        return self.num_optical + self.num_radar

    @property
    def t_start(self) -> Time:
        """Return the earliest bundled observation epoch."""
        return self.select_time_bound("min")

    @property
    def t_end(self) -> Time:
        """Return the latest bundled observation epoch."""
        return self.select_time_bound("max")

    def select_time_bound(self, bound: str) -> Time:
        """Select the earliest or latest epoch in the bundle."""
        candidates: list[Time] = []
        if len(self.optical) > 0:
            candidates.append(self.optical.t)
        if len(self.radar) > 0:
            candidates.append(self.radar.t)
        if not candidates:
            raise ValueError("ObservationData does not contain any epochs.")

        best_time = candidates[0]
        best_values = np.asarray(best_time.tt.jd)
        best_index = int(np.argmin(best_values) if bound == "min" else np.argmax(best_values))
        best_scalar = best_time[best_index]
        best_jd = float(np.asarray(best_scalar.tt.jd).item())

        for candidate in candidates[1:]:
            candidate_values = np.asarray(candidate.tt.jd)
            candidate_index = int(np.argmin(candidate_values) if bound == "min" else np.argmax(candidate_values))
            candidate_scalar = candidate[candidate_index]
            candidate_jd = float(np.asarray(candidate_scalar.tt.jd).item())
            is_better = candidate_jd < best_jd if bound == "min" else candidate_jd > best_jd
            if is_better:
                best_scalar = candidate_scalar
                best_jd = candidate_jd

        return best_scalar

    def to_dataframe(self, sort_by: str = "input"):
        """Convert the mixed observation bundle to a pandas dataframe."""
        import pandas as pd

        frames = []
        optical_df = self.optical.to_dataframe()
        if len(optical_df) > 0:
            frames.append(optical_df)
        radar_df = self.radar.to_dataframe()
        if len(radar_df) > 0:
            frames.append(radar_df)

        if len(frames) == 0:
            df = pd.DataFrame(columns=["input_index", "t_ut_jd", "t_iso", "obs_mode"])
        else:
            df = pd.concat(frames, axis=0, ignore_index=True, sort=False)

        if sort_by == "input":
            return df.sort_values("input_index", kind="stable").reset_index(drop=True)
        if sort_by == "time":
            return df.sort_values("t_ut_jd", kind="stable").reset_index(drop=True)
        if sort_by == "group":
            return df.reset_index(drop=True)
        raise ValueError("`sort_by` must be one of {'input', 'time', 'group'}.")

    def __len__(self) -> int:
        return self.num_observations

    def __repr__(self) -> str:
        ut_start_jd = None
        ut_end_jd = None
        if self.num_observations > 0:
            ut_start_jd = _format_time_bound(np.asarray([self.t_start.ut.jd]), "min")
            ut_end_jd = _format_time_bound(np.asarray([self.t_end.ut.jd]), "max")
        return build_repr(
            self.__class__.__name__,
            [
                ("name", repr(self.name) if self.name is not None else None),
                ("n_obs", str(self.num_observations)),
                ("n_optical", str(self.num_optical) if self.has_optical else None),
                ("n_radar", str(self.num_radar) if self.has_radar else None),
                ("ut_start_jd", ut_start_jd),
                ("ut_end_jd", ut_end_jd),
                ("optical_shape", format_optional_shape(self.optical.shape)),
                ("radar_shape", format_optional_shape(self.radar.shape)),
            ],
        )


class ObservationLayout:
    """Flat residual layout for optical and radar observations."""

    def __init__(self, data: ObservationData):
        self.data = data
        self.n_optical = data.num_optical
        self.n_radar = data.num_radar
        self.n_2d = self.n_optical
        self.n_1d = self.n_radar
        self.n_obs = self.data.num_observations

        optical_input_indices = np.asarray(self.data.optical.input_indices, dtype=int)
        radar_input_indices = np.asarray(self.data.radar.input_indices, dtype=int)
        self.arranged_input_indices = np.concatenate((optical_input_indices, radar_input_indices))
        self.revert_arrange_indices = np.argsort(self.arranged_input_indices)
        self.flat_input_indices = np.concatenate([
            np.repeat(optical_input_indices, 2),
            radar_input_indices,
        ])

    def input_indices_to_flat_mask(self, input_indices: Iterable[int]) -> Bool[Array, "N_flat_obs"]:
        mask = np.isin(self.flat_input_indices, list(input_indices))
        return jnp.asarray(mask, dtype=bool)

    def flat_mask_to_mask(self, flat_mask: Bool[Array, "N_flat_obs"]) -> Bool[Array, "N_obs"]:
        cut_optical = 2 * self.n_optical
        mask_2d = flat_mask[:cut_optical][::2]
        mask_1d = flat_mask[cut_optical:]
        return jnp.concatenate([mask_2d, mask_1d])

    def concat_to_flat_array(
            self,
            optical_array: Float[Array, "N_optical_obs 2"],
            radar_array: Float[Array, "N_radar_obs"],
    ) -> Float[Array, "N_flat_obs"]:
        """Concatenate optical and radar residual arrays into one flat array."""
        return jnp.concatenate([optical_array.reshape(-1), radar_array.reshape(-1)], axis=0)

    def split_flat_array(
            self,
            flat_array: Float[Array, "N_flat_obs"],
    ) -> tuple[Float[Array, "N_flat_optical_obs"], Float[Array, "N_flat_radar_obs"]]:
        """Split one flat residual-like array into optical and radar blocks."""
        cut_optical = 2 * self.n_optical
        return flat_array[:cut_optical], flat_array[cut_optical:]

    def split_array(
            self,
            array: Float[Array, "N_obs"],
    ) -> tuple[Float[Array, "N_optical_obs"], Float[Array, "N_radar_obs"]]:
        """Split one observation-aligned array into optical and radar blocks."""
        return array[:self.n_optical], array[self.n_optical:]

    def flat_array_to_2d_array(self, flat_array: Float[Array, "N_flat_obs"]) -> Float[Array, "N_obs 2"]:
        """Reshape a flat optical block into ``(N, 2)``."""
        return flat_array.reshape(-1, 2)

    def split_flat_array_to_array(
            self,
            flat_array: Float[Array, "N_flat_obs"],
    ) -> tuple[Float[Array, "N_optical_obs 2"], Float[Array, "N_radar_obs"]]:
        """Split a flat residual-like array into shaped optical and radar blocks."""
        flat_optical_array, flat_radar_array = self.split_flat_array(flat_array)
        return self.flat_array_to_2d_array(flat_optical_array), flat_radar_array

    def restore_flat_array(self, flat_array: Float[Array, "N_flat_obs"]) -> Float[Array, "N_obs 2"]:
        """Restore a flat residual-like array to original input order."""
        optical_array, radar_array = self.split_flat_array(flat_array)
        radar_pad_array = jnp.stack([radar_array, jnp.zeros_like(radar_array)], axis=1)
        combined_array = jnp.concatenate([self.flat_array_to_2d_array(optical_array), radar_pad_array], axis=0)
        return combined_array[self.revert_arrange_indices]

    def restore_array(self, array: Float[Array, "N_obs"]) -> Float[Array, "N_obs"]:
        """Restore an observation-aligned array to original input order."""
        return array[self.revert_arrange_indices]
