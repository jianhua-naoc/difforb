"""Load ADES PSV observations into ``ObservationData``.

This module keeps two related row contracts:

- an internal normalized row schema used by the online MPC or JPL adapters,
- the standard ADES PSV presentation used for local files and online saves.

The public local file path implemented here accepts only ADES PSV files with a
``.psv`` suffix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from difforb.astrometry.data import ObsMode
from difforb.astrometry.data import ObservationData
from difforb.astrometry.data import ObsType
from difforb.astrometry.data import ObserverType
from difforb.astrometry.data import OpticalObservationData
from difforb.astrometry.data import RadarObservationData
from difforb.body.site import Site
from difforb.body.site import format_site_key
from difforb.core.constants import ARCSEC_TO_DEG
from difforb.core.constants import AU_KM
from difforb.core.time.timescale import Time

REQUIRED_ADES_COLUMNS = (
    "obstime",
    "Obstype",
    "mode",
    "sys",
    "ra",
    "dec",
    "rmsra",
    "rmsdec",
    "rmscorr",
    "rmstime",
    "stn",
    "prog",
    "astcat",
    "notes",
    "mag",
    "band",
    "subfrm",
    "pos1",
    "pos2",
    "pos3",
    "trkid",
    "provid",
    "permid",
    "com",
    "delay",
    "rmsdelay",
    "doppler",
    "rmsdoppler",
    "rcv",
    "trx",
    "frq",
)

_ADES_PSV_VERSION = "2022"
_PSV_SUFFIX = ".psv"
_STANDARD_SPACE_SYS_TO_INTERNAL = {
    "ICRF_AU": "ICRFAU",
    "ICRF_KM": "ICRFKM",
}
_STANDARD_ADES_FIELD_ORDER = (
    "permID",
    "provID",
    "trkSub",
    "trkID",
    "trkMPC",
    "obsSubID",
    "obsID",
    "mode",
    "stn",
    "prog",
    "sys",
    "ctr",
    "obsTime",
    "ra",
    "dec",
    "rmsRA",
    "rmsDec",
    "rmsCorr",
    "rmsTime",
    "astCat",
    "mag",
    "rmsMag",
    "band",
    "photCat",
    "photAp",
    "logSNR",
    "seeing",
    "exp",
    "notes",
    "remarks",
    "ref",
    "disc",
    "subFmt",
    "subFrm",
    "precTime",
    "precRA",
    "precDec",
    "uncTime",
    "raStar",
    "decStar",
    "obsCenter",
    "deltaRA",
    "deltaDec",
    "dist",
    "rmsDist",
    "pa",
    "rmsPA",
    "pos1",
    "pos2",
    "pos3",
    "posCov11",
    "posCov12",
    "posCov13",
    "posCov22",
    "posCov23",
    "posCov33",
    "vel1",
    "vel2",
    "vel3",
    "artSat",
    "deprecated",
    "localUse",
    "nStars",
    "nucMag",
    "fltr",
    "shapeOcc",
    "delay",
    "rmsDelay",
    "doppler",
    "rmsDoppler",
    "trx",
    "rcv",
    "frq",
    "com",
)
_STANDARD_ADES_FIELD_BY_KEY = {field.lower(): field for field in _STANDARD_ADES_FIELD_ORDER}
_FORBIDDEN_RADAR_STANDARD_FIELDS = (
    "obsSubID",
    "trkID",
    "trkMPC",
    "mode",
    "stn",
    "sys",
    "ctr",
    "pos1",
    "pos2",
    "pos3",
    "vel1",
    "vel2",
    "vel3",
    "ra",
    "dec",
    "rmsRA",
    "rmsDec",
    "rmsCorr",
    "astCat",
    "mag",
    "rmsMag",
    "band",
    "fltr",
    "photCat",
    "photAp",
    "nucMag",
    "precTime",
    "precRA",
    "precDec",
)


def _is_missing(value: Any) -> bool:
    """Return ``True`` when one scalar should be treated as missing."""
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        return text == "" or text.lower() == "nan"
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _normalize_obstime(value: Any) -> str | None:
    """Normalize one ADES timestamp to the internal parser convention."""
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text.rstrip("Z").replace(" ", "T")


def normalize_ades_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build one normalized internal ADES row dictionary.

    Parameters
    ----------
    record : mapping
        One flattened ADES-like row in the internal normalized schema.

    Returns
    -------
    dict
        Copy of ``record`` with parser-required columns present and with a
        normalized ``obstime`` field.
    """
    normalized = {column: np.nan for column in REQUIRED_ADES_COLUMNS}
    normalized.update(dict(record))
    if "subFrm" in normalized and _is_missing(normalized["subfrm"]):
        normalized["subfrm"] = normalized["subFrm"]
    if "rmsCorr" in normalized and _is_missing(normalized["rmscorr"]):
        normalized["rmscorr"] = normalized["rmsCorr"]
    if "rmsTime" in normalized and _is_missing(normalized["rmstime"]):
        normalized["rmstime"] = normalized["rmsTime"]
    for key, value in list(normalized.items()):
        if isinstance(value, str) and value.strip() == "":
            normalized[key] = np.nan
    normalized["obstime"] = _normalize_obstime(normalized.get("obstime"))
    return normalized


def _fill_missing_ades_columns(ades_df: pd.DataFrame) -> pd.DataFrame:
    """Fill parser-required internal ADES columns on a dataframe copy."""
    filled = ades_df.copy()
    for column in REQUIRED_ADES_COLUMNS:
        if column not in filled.columns:
            filled[column] = np.nan
    filled["sys"] = filled["sys"].replace("", np.nan)
    return filled


def _convert_to_str_numpy_array(arr: np.ndarray) -> np.ndarray:
    """Convert a possibly mixed array to a string array with empty missing values."""
    safe_arr = np.where(pd.isna(arr), "", arr)
    return safe_arr.astype(str)


def build_ades_site_keys(stations: np.ndarray, systems: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Build self-contained observer keys from ADES station fields.

    Parameters
    ----------
    stations : numpy.ndarray
        ADES ``stn`` field values.
    systems : numpy.ndarray
        ADES ``sys`` field values.
    positions : numpy.ndarray
        ADES ``pos1``, ``pos2``, and ``pos3`` fields. Roving ground rows are
        interpreted as WGS84 longitude, latitude, and height. Space rows are
        interpreted as ICRF/GCRS position, in ``au`` unless the system is
        ``ICRFKM`` or ``ICRF_KM``.

    Returns
    -------
    numpy.ndarray
        Canonical observer keys for ``OpticalObservationData.rx_codes``.
    """
    station_values = _convert_to_str_numpy_array(stations)
    system_values = np.asarray(systems, dtype=object)
    position_values = np.asarray(positions, dtype=float)
    keys = []
    for station, system, position in zip(station_values, system_values, position_values):
        observer_type = ObserverType.from_ades_sys(system)
        if observer_type is ObserverType.GROUND_FIXED:
            keys.append(format_site_key(station, Site.TYPE_COMMON_GROUND))
        elif observer_type is ObserverType.GROUND_ROVING:
            keys.append(format_site_key(station, Site.TYPE_ROVING_GROUND, position))
        elif observer_type is ObserverType.SPACE_BASED:
            space_position = position / AU_KM if system in ("ICRFKM", "ICRF_KM") else position
            keys.append(format_site_key(station, Site.TYPE_SATELLITE, space_position))
        else:
            raise ValueError(f"Unsupported ADES observer type: {observer_type}.")
    return np.asarray(keys, dtype=str)


def _parse_iso_time_strings(iso_time_strings: Sequence[str | None]) -> tuple[list[int], list[int], list[int], list[int], list[int], list[float]]:
    """Parse normalized ADES ISO timestamps into calendar fields."""
    years: list[int] = []
    months: list[int] = []
    days: list[int] = []
    hours: list[int] = []
    minutes: list[int] = []
    seconds: list[float] = []
    for value in iso_time_strings:
        normalized = _normalize_obstime(value)
        if normalized is None:
            raise ValueError("Every ADES row must define a non-empty `obstime`.")
        date_part, time_part = normalized.split("T")
        year_text, month_text, day_text = date_part.split("-")
        hour_text, minute_text, second_text = time_part.split(":")
        years.append(int(year_text))
        months.append(int(month_text))
        days.append(int(day_text))
        hours.append(int(hour_text))
        minutes.append(int(minute_text))
        seconds.append(float(second_text))
    return years, months, days, hours, minutes, seconds


def _first_non_missing(series: pd.Series) -> str | None:
    """Return the first non-missing scalar from one dataframe column."""
    for value in series.tolist():
        if not _is_missing(value):
            return str(value)
    return None


def _resolve_observation_name(ades_df: pd.DataFrame) -> str | None:
    """Resolve one bundle name from the available identification columns."""
    for column in ("provid", "permid", "trkid"):
        if column in ades_df.columns:
            value = _first_non_missing(ades_df[column])
            if value is not None:
                return value
    return None


def records_to_observations(records: Sequence[Mapping[str, Any]]) -> ObservationData:
    """Build ``ObservationData`` from normalized ADES-like rows.

    Parameters
    ----------
    records : sequence of mapping
        Flattened ADES-like row dictionaries in the internal normalized schema.

    Returns
    -------
    ObservationData
        Single-target observation bundle with unified optical and radar
        modality tables.

    Raises
    ------
    ValueError
        Raised when ``records`` is empty or when one row is missing a usable
        ``obstime`` field.

    Notes
    -----
    Space-observer positions are stored internally in ``AU``. Input rows tagged
    as ``ICRFKM`` are converted from ``km`` to ``AU`` on ingestion. Radar delay
    values in the internal row schema are stored in microseconds to match the
    rest of the ``difforb`` radar pipeline.
    """
    internal_records = [normalize_ades_record(record) for record in records]
    if len(internal_records) == 0:
        raise ValueError("No ADES observation rows were provided.")

    ades_df = _fill_missing_ades_columns(pd.DataFrame(internal_records))
    t_year, t_month, t_day, t_hour, t_minute, t_second = _parse_iso_time_strings(ades_df["obstime"].tolist())
    t = Time.from_ut_date(t_year, t_month, t_day, t_hour, t_minute, t_second)

    optical_mask = ades_df["Obstype"] == ObsType.OPTICAL.label
    optical_obs = ades_df[optical_mask]
    optical_obs_t = t[optical_mask.to_numpy()]
    optical_obs_type_ids = np.full(len(optical_obs), ObsType.OPTICAL.id, dtype=int)
    optical_obs_mode_ids = np.array([ObsMode.from_ades_mode(mode).id for mode in optical_obs["mode"].tolist()], dtype=int)
    optical_obs_values = np.stack(
        [
            optical_obs["ra"].to_numpy(dtype=float),
            optical_obs["dec"].to_numpy(dtype=float),
        ],
        axis=-1,
    )
    optical_obs_values = np.deg2rad(optical_obs_values)
    optical_obs_uncertainties = np.stack(
        [
            optical_obs["rmsra"].to_numpy(dtype=float),
            optical_obs["rmsdec"].to_numpy(dtype=float),
        ],
        axis=-1,
    )
    optical_obs_uncertainties = np.deg2rad(optical_obs_uncertainties * ARCSEC_TO_DEG)
    optical_obs_correlations = optical_obs["rmscorr"].to_numpy(dtype=float)
    optical_obs_correlations = np.where(np.isfinite(optical_obs_correlations), optical_obs_correlations, 0.0)
    if np.any(np.abs(optical_obs_correlations) >= 1.0):
        raise ValueError("Optical ADES `rmscorr` values must be strictly between -1 and 1.")
    optical_time_uncertainties = optical_obs["rmstime"].to_numpy(dtype=float)
    if np.any(optical_time_uncertainties[np.isfinite(optical_time_uncertainties)] < 0.0):
        raise ValueError("Optical ADES `rmsTime` values must be non-negative where finite.")
    optical_site_payloads = np.stack(
        [
            optical_obs["pos1"].to_numpy(dtype=float),
            optical_obs["pos2"].to_numpy(dtype=float),
            optical_obs["pos3"].to_numpy(dtype=float),
        ],
        axis=-1,
    )
    optical_obs_data = OpticalObservationData(
        t=optical_obs_t,
        trk_ids=_convert_to_str_numpy_array(optical_obs["trkid"].to_numpy()),
        obs_type_ids=optical_obs_type_ids,
        obs_mode_ids=optical_obs_mode_ids,
        values=optical_obs_values,
        uncertainties=optical_obs_uncertainties,
        time_uncertainties=optical_time_uncertainties,
        correlations=optical_obs_correlations,
        rx_codes=build_ades_site_keys(
            optical_obs["stn"].to_numpy(),
            optical_obs["sys"].to_numpy(),
            optical_site_payloads,
        ),
        program_codes=_convert_to_str_numpy_array(optical_obs["prog"].to_numpy()),
        catalog_codes=_convert_to_str_numpy_array(optical_obs["astcat"].to_numpy()),
        note_codes=_convert_to_str_numpy_array(optical_obs["notes"].to_numpy()),
        magnitudes=optical_obs["mag"].to_numpy(dtype=float),
        band_codes=_convert_to_str_numpy_array(optical_obs["band"].to_numpy()),
        sub_frames=_convert_to_str_numpy_array(optical_obs["subfrm"].to_numpy()),
        input_indices=optical_obs.index.to_numpy(dtype=int),
    )

    radar_mask = ades_df["Obstype"] == ObsType.RADAR.label
    radar_obs = ades_df[radar_mask]
    radar_obs_t = t[radar_mask.to_numpy()]
    radar_obs_type_ids = np.full(len(radar_obs), ObsType.RADAR.id, dtype=int)
    delay_value = radar_obs["delay"].to_numpy(dtype=float)
    doppler_value = radar_obs["doppler"].to_numpy(dtype=float)
    delay_mask = np.isfinite(delay_value)
    doppler_mask = np.isfinite(doppler_value)
    if np.any(delay_mask == doppler_mask):
        raise ValueError("Internal radar rows must define exactly one of `delay` or `doppler`.")
    delay_rms = radar_obs["rmsdelay"].to_numpy(dtype=float)
    doppler_rms = radar_obs["rmsdoppler"].to_numpy(dtype=float)
    if np.any(delay_mask & ~np.isfinite(delay_rms)):
        raise ValueError("Internal radar delay rows must define `rmsdelay`.")
    if np.any(doppler_mask & ~np.isfinite(doppler_rms)):
        raise ValueError("Internal radar Doppler rows must define `rmsdoppler`.")
    for column in ("trx", "rcv", "frq"):
        if radar_obs[column].isna().any():
            raise ValueError(f"Internal radar rows must define `{column}`.")
    radar_com = radar_obs["com"].fillna(1).to_numpy(dtype=int)
    radar_obs_mode_ids = np.array(
        [
            ObsMode.from_ades_mode(mode, is_delay=is_delay, com=com).id
            for mode, is_delay, com in zip(
                radar_obs["mode"].tolist(),
                delay_mask,
                radar_com,
            )
        ],
        dtype=int,
    )
    radar_values = np.zeros(len(radar_obs), dtype=float)
    radar_uncertainties = np.zeros(len(radar_obs), dtype=float)
    radar_values[delay_mask] = delay_value[delay_mask]
    radar_uncertainties[delay_mask] = delay_rms[delay_mask]
    radar_values[doppler_mask] = doppler_value[doppler_mask]
    radar_uncertainties[doppler_mask] = doppler_rms[doppler_mask]
    radar_obs_data = RadarObservationData(
        t=radar_obs_t,
        obs_type_ids=radar_obs_type_ids,
        obs_mode_ids=radar_obs_mode_ids,
        values=radar_values,
        uncertainties=radar_uncertainties,
        rx_codes=_convert_to_str_numpy_array(radar_obs["rcv"].to_numpy()),
        tx_codes=_convert_to_str_numpy_array(radar_obs["trx"].to_numpy()),
        tx_freq=radar_obs["frq"].to_numpy(dtype=float),
        input_indices=radar_obs.index.to_numpy(dtype=int),
    )

    return ObservationData(
        name=_resolve_observation_name(ades_df),
        optical=optical_obs_data,
        radar=radar_obs_data,
    )


def _require_psv_suffix(filepath: str) -> Path:
    """Return one path after enforcing the ADES PSV suffix."""
    path = Path(filepath)
    if path.suffix.lower() != _PSV_SUFFIX:
        raise ValueError("ADES PSV files must use the `.psv` suffix.")
    return path


def _is_internal_radar_record(record: Mapping[str, Any]) -> bool:
    """Return ``True`` when one internal row represents a radar observation."""
    obs_type = str(record.get("Obstype") or record.get("obstype") or "").strip().lower()
    if obs_type == ObsType.RADAR.label:
        return True
    return any(not _is_missing(record.get(field)) for field in ("delay", "doppler", "rmsdelay", "rmsdoppler", "trx", "rcv"))


def _standard_ades_field_name(key: str) -> str | None:
    """Return the standard ADES PSV field name for one flattened record key."""
    return _STANDARD_ADES_FIELD_BY_KEY.get(str(key).strip().lower())


def _standard_psv_number(value: Any, scale: float = 1.0) -> str:
    """Format one numeric ADES PSV field after applying a unit scale."""
    if _is_missing(value):
        return ""
    scalar = float(value) * scale
    if not np.isfinite(scalar):
        return ""
    return format(scalar, ".15g")


def _standard_psv_text(value: Any) -> str:
    """Format one scalar as a safe ADES PSV field."""
    if _is_missing(value):
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return _standard_psv_number(value)
    text = str(value).strip()
    return text.replace("|", " ").replace("\n", " ").replace("\r", " ")


def _standard_obstime_from_internal(value: Any) -> str:
    """Return one ADES UTC timestamp string."""
    text = _standard_psv_text(value)
    if not text:
        return ""
    if len(text) >= 19 and text[4] == "-" and text[7] == "-" and text[10] == " ":
        text = f"{text[:10]}T{text[11:]}"
    if "T" in text and not text.endswith("Z"):
        text = f"{text}Z"
    return text


def _standard_psv_row_from_internal_record(record: Mapping[str, Any]) -> dict[str, str]:
    """Map one internal ADES-like row to one standard ADES PSV row."""
    row: dict[str, str] = {}
    for key, value in record.items():
        field = _standard_ades_field_name(str(key))
        if field is None or field in {"delay", "rmsDelay", "doppler", "rmsDoppler", "frq"}:
            continue
        text = _standard_obstime_from_internal(value) if field == "obsTime" else _standard_psv_text(value)
        if text:
            row[field] = text

    if _is_internal_radar_record(record):
        row.pop("mode", None)
        if not _is_missing(record.get("delay")):
            row["delay"] = _standard_psv_number(record.get("delay"), 1.0e-6)
        if not _is_missing(record.get("rmsdelay")):
            row["rmsDelay"] = _standard_psv_number(record.get("rmsdelay"))
        if not _is_missing(record.get("doppler")):
            row["doppler"] = _standard_psv_number(record.get("doppler"))
        if not _is_missing(record.get("rmsdoppler")):
            row["rmsDoppler"] = _standard_psv_number(record.get("rmsdoppler"))
        if not _is_missing(record.get("frq")):
            row["frq"] = _standard_psv_number(record.get("frq"), 1.0e-6)
    else:
        for field in ("delay", "rmsDelay", "doppler", "rmsDoppler", "trx", "rcv", "frq", "com"):
            row.pop(field, None)
    return row


def records_to_ades_psv_text(records: Sequence[Mapping[str, Any]]) -> str:
    """Return canonical ADES PSV text for internal ADES-like rows.

    Parameters
    ----------
    records : sequence of mapping
        Flattened ADES-like rows in the internal online-loader schema.

    Returns
    -------
    str
        ADES PSV text using the standard ADES 2022 field names and deterministic field ordering.

    Raises
    ------
    ValueError
        Raised when no writable ADES fields are present.

    Notes
    -----
    Radar delay values are stored internally in microseconds and are written to
    ADES PSV in seconds. Radar delay uncertainties are stored and written in
    microseconds. Radar transmitter frequencies are stored internally in ``Hz``
    and are written to ADES PSV in ``MHz``.
    """
    rows = [_standard_psv_row_from_internal_record(record) for record in records]
    if not rows:
        raise ValueError("No ADES rows were provided.")

    row_columns = [
        tuple(field for field in _STANDARD_ADES_FIELD_ORDER if row.get(field))
        for row in rows
    ]
    if not any(row_columns):
        raise ValueError("No ADES PSV fields were provided.")
    empty_row_index = next((index for index, columns in enumerate(row_columns) if not columns), None)
    if empty_row_index is not None:
        raise ValueError(f"ADES row {empty_row_index} does not contain any writable PSV fields.")

    lines = [f"# version={_ADES_PSV_VERSION}"]
    current_columns: tuple[str, ...] | None = None
    for row, columns in zip(rows, row_columns):
        if columns != current_columns:
            lines.append("|".join(columns))
            current_columns = columns
        lines.append("|".join(row.get(column, "") for column in columns))
    return "\n".join(lines) + "\n"


def write_ades_psv_records(filepath: str, records: Sequence[Mapping[str, Any]]) -> None:
    """Write internal ADES-like rows to one ADES PSV file.

    Parameters
    ----------
    filepath : str
        Destination path. The filename must use the ``.psv`` suffix.
    records : sequence of mapping
        Flattened ADES-like rows in the internal online-loader schema.

    Raises
    ------
    ValueError
        Raised when the destination does not use ``.psv`` or when no writable ADES fields are present.

    Notes
    -----
    Radar delay values are stored internally in microseconds and are written to ADES PSV in seconds. Radar delay uncertainties are stored and written in microseconds. Radar transmitter frequencies are stored internally in ``Hz`` and are written to ADES PSV in ``MHz``.
    """
    path = _require_psv_suffix(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(records_to_ades_psv_text(records), encoding="utf-8", newline="")


_write_ades_psv_records = write_ades_psv_records


def _split_psv_fields(line: str) -> list[str]:
    """Split one PSV record into stripped fields."""
    return [field.strip() for field in line.rstrip("\n").split("|")]


def _is_keyword_record(line: str) -> bool:
    """Return ``True`` when one PSV line is a keyword record."""
    if line.startswith("#") or line.startswith("!"):
        return False
    fields = [field for field in _split_psv_fields(line) if field]
    return len(fields) > 0 and all("a" <= field[0] <= "z" for field in fields)


def _row_is_radar(row: Mapping[str, Any]) -> bool:
    """Return ``True`` when one standard ADES row is a radar row."""
    return any(not _is_missing(row.get(field)) for field in ("delay", "doppler", "rmsDelay", "rmsDoppler", "trx", "rcv"))


def _row_has_offset_or_occultation_fields(row: Mapping[str, Any]) -> bool:
    """Return ``True`` when one standard ADES row uses unsupported observation fields."""
    return any(not _is_missing(row.get(field)) for field in ("raStar", "decStar", "obsCenter", "deltaRA", "deltaDec", "dist", "pa"))


def _validate_standard_radar_row(row: Mapping[str, Any]) -> None:
    """Validate one standard ADES radar row against the supported radar subset."""
    forbidden_field = next(
        (field for field in _FORBIDDEN_RADAR_STANDARD_FIELDS if not _is_missing(row.get(field))),
        None,
    )
    if forbidden_field is not None:
        raise ValueError(f"Field `{forbidden_field}` is not permitted on ADES radar observations.")

    has_delay = not _is_missing(row.get("delay"))
    has_doppler = not _is_missing(row.get("doppler"))
    if has_delay == has_doppler:
        raise ValueError("ADES radar observations must define exactly one of `delay` or `doppler`.")
    if has_delay and _is_missing(row.get("rmsDelay")):
        raise ValueError("ADES radar delay observations must define `rmsDelay`.")
    if has_doppler and _is_missing(row.get("rmsDoppler")):
        raise ValueError("ADES radar Doppler observations must define `rmsDoppler`.")
    if not has_delay and not _is_missing(row.get("rmsDelay")):
        raise ValueError("`rmsDelay` is only permitted when `delay` is reported.")
    if not has_doppler and not _is_missing(row.get("rmsDoppler")):
        raise ValueError("`rmsDoppler` is only permitted when `doppler` is reported.")
    for required_field in ("trx", "rcv", "frq"):
        if _is_missing(row.get(required_field)):
            raise ValueError(f"ADES radar observations must define `{required_field}`.")


def _standard_sys_to_internal(sys: Any) -> Any:
    """Map one standard ADES ``sys`` token to the internal row convention."""
    if _is_missing(sys):
        return np.nan
    text = str(sys).strip()
    return _STANDARD_SPACE_SYS_TO_INTERNAL.get(text, text)


def _standard_delay_to_internal(value: Any) -> float:
    """Convert standard ADES radar delay seconds to internal microseconds."""
    if _is_missing(value):
        return np.nan
    return float(value) * 1.0e6


def _standard_microseconds_to_internal(value: Any) -> float:
    """Read one standard ADES microsecond field into the internal convention."""
    if _is_missing(value):
        return np.nan
    return float(value)


def _standard_radar_com_to_internal(value: Any) -> int:
    """Convert standard ADES ``com`` values to the internal center or bounce code."""
    if _is_missing(value):
        return 1
    return 1 if int(value) == 1 else 2


def _standard_frq_to_internal(value: Any) -> float:
    """Convert standard ADES carrier frequency MHz to internal Hz."""
    if _is_missing(value):
        return np.nan
    return float(value) * 1.0e6


def _standard_psv_row_to_internal_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Map one standard ADES PSV row to the internal normalized row schema."""
    if _row_has_offset_or_occultation_fields(row):
        raise ValueError("Current local ADES loading supports optical and radar rows, not offset or occultation rows.")

    trk_id = row.get("trkID")
    if _is_missing(trk_id):
        trk_id = row.get("trkMPC")

    is_radar = _row_is_radar(row)
    if is_radar:
        _validate_standard_radar_row(row)
    record = normalize_ades_record(
        {
            "obstime": row.get("obsTime"),
            "Obstype": ObsType.RADAR.label if is_radar else ObsType.OPTICAL.label,
            "mode": np.nan if is_radar else row.get("mode"),
            "sys": _standard_sys_to_internal(row.get("sys")),
            "ra": row.get("ra"),
            "dec": row.get("dec"),
            "rmsra": row.get("rmsRA"),
            "rmsdec": row.get("rmsDec"),
            "rmscorr": row.get("rmsCorr"),
            "rmstime": row.get("rmsTime"),
            "stn": row.get("stn"),
            "prog": row.get("prog"),
            "astcat": row.get("astCat"),
            "notes": row.get("notes"),
            "mag": row.get("mag"),
            "band": row.get("band"),
            "subfrm": row.get("subFrm"),
            "pos1": row.get("pos1"),
            "pos2": row.get("pos2"),
            "pos3": row.get("pos3"),
            "trkid": trk_id,
            "provid": row.get("provID"),
            "permid": row.get("permID"),
            "com": _standard_radar_com_to_internal(row.get("com")) if is_radar else np.nan,
            "delay": _standard_delay_to_internal(row.get("delay")),
            "rmsdelay": _standard_microseconds_to_internal(row.get("rmsDelay")),
            "doppler": row.get("doppler"),
            "rmsdoppler": row.get("rmsDoppler"),
            "rcv": row.get("rcv"),
            "trx": row.get("trx"),
            "frq": _standard_frq_to_internal(row.get("frq")),
        }
    )
    return record


def _load_psv_records(filepath: str) -> list[dict[str, Any]]:
    """Load normalized ADES rows from one standard PSV file."""
    path = _require_psv_suffix(filepath)
    records: list[dict[str, Any]] = []
    current_keywords: list[str] | None = None
    with path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#") or stripped.startswith("!"):
                continue
            if _is_keyword_record(stripped):
                current_keywords = _split_psv_fields(line)
                continue
            if current_keywords is None:
                raise ValueError("Encountered an ADES PSV data record before any keyword record.")
            values = _split_psv_fields(line)
            if len(values) < len(current_keywords):
                values.extend([""] * (len(current_keywords) - len(values)))
            row = {keyword: value for keyword, value in zip(current_keywords, values) if keyword}
            records.append(_standard_psv_row_to_internal_record(row))
    return records


def load_local_observations(filepath: str) -> ObservationData:
    """Load local ADES observations into one ``ObservationData`` bundle.

    Parameters
    ----------
    filepath : str
        Path to a local ADES PSV file. The filename must use the ``.psv``
        suffix.

    Returns
    -------
    ObservationData
        Parsed single-target observation bundle.

    Raises
    ------
    ValueError
        Raised when ``filepath`` does not use ``.psv`` or when the file does
        not contain usable ADES PSV rows.

    Notes
    -----
    The local carrier is ADES pipe-separated values (PSV). The returned
    ``ObservationData`` uses DiffOrb's internal units: optical angles in
    radians, radar delays in microseconds, and radar frequencies in ``Hz``.
    """
    records = _load_psv_records(filepath)
    return records_to_observations(records)
