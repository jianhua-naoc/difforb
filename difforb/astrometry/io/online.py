"""Fetch online MPC and JPL observations as ``ObservationData``.

Online loading fetches flattened optical rows from the MPC Observations API and
radar rows from the JPL small-body radar API. Both feeds are translated to the
internal row schema used by :mod:`difforb.astrometry.io.ades`. The same rows
can be written to a local ADES PSV file before they are converted to
``ObservationData``.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

import requests

from difforb.astrometry.data import ObservationData
from difforb.astrometry.io.ades import records_to_observations
from difforb.astrometry.io.ades import write_ades_psv_records

IAU_MPC_API = "https://data.minorplanetcenter.net/api/get-obs"
JPL_SB_RADAR_API = "https://ssd-api.jpl.nasa.gov/sb_radar.api"
DEFAULT_HTTP_TIMEOUT = 30.0
_PROVISIONAL_DESIGNATION_PATTERNS = (
    re.compile(r"^\d{4}\s"),
    re.compile(r"^[A-Z]/\d{4}\s"),
    re.compile(r"^[A-Z]\d{3}\s"),
)
_NUMBERED_COMET_NAME_PATTERN = re.compile(r"^(\d+)\s*([PDCXAI])\s*/", re.IGNORECASE)
_COMET_PROVISIONAL_PATTERN = re.compile(r"^([PDCXAI])\s*/\s*(\d{4})\s*([A-Z]{1,2})\s*(\d+)$", re.IGNORECASE)
_MPC_DISAMBIGUATION_PATTERN = re.compile(r"disambiguation_list=(\[[\s\S]*?\])\)")


def _decode_json_like_payload(payload: Any) -> Any:
    """Decode one response body that may itself contain one JSON string.

    Notes
    -----
    Some upstream services occasionally return a JSON document as one quoted
    string instead of returning the decoded mapping directly. This helper
    unwraps that extra layer so downstream code always sees the semantic
    payload rather than the transport quirk.
    """
    decoded = payload
    while isinstance(decoded, str):
        text = decoded.strip()
        if len(text) == 0:
            break
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            break
    return decoded


def _extract_error_payload(exc: requests.HTTPError) -> Any | None:
    """Decode one JSON error payload attached to ``requests.HTTPError``."""
    response = exc.response
    if response is None:
        return None
    try:
        return _decode_json_like_payload(response.json())
    except ValueError:
        return None


def _parse_mpc_disambiguation_options(message: str) -> list[dict[str, Any]]:
    """Extract one MPC disambiguation list from the ``get-obs`` error text."""
    match = _MPC_DISAMBIGUATION_PATTERN.search(message)
    if match is None:
        return []
    try:
        options = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return []
    return [dict(option) for option in options if isinstance(option, dict)]


def _format_mpc_disambiguation_options(options: list[dict[str, Any]]) -> str:
    """Format one short human-facing list of MPC designation alternatives."""
    formatted: list[str] = []
    for option in options:
        label = option.get("permid") or option.get("unpacked_primary_provisional_designation") or option.get("name")
        if label is None:
            continue
        group = option.get("group")
        name = option.get("name")
        parts = [str(label)]
        details: list[str] = []
        if group:
            details.append(str(group))
        if name and str(name) != str(label):
            details.append(str(name))
        if details:
            parts.append(f"({' / '.join(details)})")
        formatted.append(" ".join(parts))
    return ", ".join(formatted)


def _raise_mpc_get_obs_error(desig: str, exc: requests.HTTPError) -> None:
    """Raise one stable local error for MPC designation-identifier failures.

    Notes
    -----
    The MPC ``get-obs`` endpoint currently reports some designation-identifier
    failures as HTTP 500 with one serialized internal exception message. This
    helper converts the upstream text to a clearer local ``ValueError`` so the
    caller can distinguish ambiguous designations from ordinary transport
    failures.
    """
    payload = _extract_error_payload(exc)
    if isinstance(payload, dict):
        message = str(payload.get("message", "")).strip()
        options = _parse_mpc_disambiguation_options(message)
        if options:
            formatted = _format_mpc_disambiguation_options(options)
            raise ValueError(
                f"Ambiguous MPC designation '{desig}'. Use one unique designation such as: {formatted}."
            ) from exc
        if "Bad Label from designation identifier" in message:
            numbered_alias_match = _NUMBERED_COMET_NAME_PATTERN.match(desig)
            if numbered_alias_match is not None:
                numbered_identifier = f"{numbered_alias_match.group(1)}{numbered_alias_match.group(2).upper()}"
                raise ValueError(
                    f"MPC could not resolve designation '{desig}'. Use the numbered periodic-comet identifier '{numbered_identifier}' instead."
                ) from exc
            raise ValueError(
                f"MPC could not resolve designation '{desig}'. Try one permanent or provisional designation accepted by MPC."
            ) from exc
    raise exc


def _candidate_designations(desig: str) -> list[str]:
    """Build one small fallback list of MPC identifiers for one user input.

    Notes
    -----
    The MPC ``get-obs`` service accepts one broad set of designation formats,
    but the local loader should not silently replace one explicit identifier
    with another semantically different alias. This helper therefore limits
    normalization to spacing and case cleanup for one already specific cometary
    provisional designation while leaving numbered comet aliases untouched.
    """
    text = str(desig).strip()
    candidates: list[str] = []

    def _append(candidate: str) -> None:
        normalized = candidate.strip()
        if len(normalized) == 0 or normalized in candidates:
            return
        candidates.append(normalized)

    _append(text)

    match = _COMET_PROVISIONAL_PATTERN.match(text)
    if match is not None:
        prefix, year, half_month, sequence = match.groups()
        _append(f"{prefix.upper()}/{year} {half_month.upper()}{sequence}")

    return candidates


def _request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_payload: dict[str, Any] | None = None,
) -> Any:
    """Request JSON content from one remote observation service."""
    response = session.get(url, params=params, json=json_payload, timeout=DEFAULT_HTTP_TIMEOUT)
    response.raise_for_status()
    return _decode_json_like_payload(response.json())


def _fetch_mpc_ades_records(session: requests.Session, desig: str, ades_version: str) -> list[dict[str, Any]]:
    """Fetch flattened optical ADES rows from the MPC Observations API."""
    last_error: Exception | None = None
    for candidate in _candidate_designations(desig):
        try:
            response_json = _request_json(
                session,
                IAU_MPC_API,
                json_payload={
                    "desigs": [candidate],
                    "output_format": ["ADES_DF"],
                    "ades_version": ades_version,
                },
            )
        except requests.HTTPError as exc:
            try:
                _raise_mpc_get_obs_error(candidate, exc)
            except ValueError as local_error:
                last_error = local_error
                continue
            raise
        if not response_json:
            return []
        records = response_json[0].get("ADES_DF", [])
        if records is None:
            return []
        optical_records = [dict(record) for record in records]
        for record in optical_records:
            if not record.get("permid") and not record.get("provid"):
                record.update(_designation_to_internal_id_fields(candidate))
        return optical_records
    if last_error is not None:
        raise last_error
    return []


def _table_records_from_jpl_response(response_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the JPL table-style response into row dictionaries."""
    response_json = _decode_json_like_payload(response_json)
    if not isinstance(response_json, dict):
        return []
    if str(response_json.get("count", "0")).strip() == "0":
        return []
    fields = response_json.get("fields", [])
    rows = response_json.get("data", [])
    return [dict(zip(fields, row)) for row in rows]


def _designation_to_internal_id_fields(desig: str) -> dict[str, Any]:
    """Map one target designation to the internal ``permid`` or ``provid`` field."""
    text = str(desig).strip()
    if any(pattern.match(text) for pattern in _PROVISIONAL_DESIGNATION_PATTERNS):
        return {"provid": text}
    return {"permid": text}


def _jpl_bp_to_internal_com(value: Any) -> int:
    """Map JPL bounce-point codes to the internal radar ``com`` convention."""
    return 1 if str(value).strip().upper() == "C" else 2


def _convert_jpl_radar_record(record: dict[str, Any], desig: str) -> dict[str, Any]:
    """Convert one JPL radar row to the internal normalized radar-row schema.

    Notes
    -----
    The JPL small-body radar API reports transmitter frequencies in ``MHz``.
    ``difforb`` stores internal radar transmitter frequencies in ``Hz``, so
    the value is converted here before the internal row is built.

    Radar observations do not use the optical-only ``mode`` or ``trkID`` fields
    defined elsewhere in ADES. The identification group is therefore carried with
    ``permid`` or ``provid`` according to the current designation instead of
    forcing the target identifier into ``trkid``.
    """
    units = str(record.get("units", "")).strip().lower()
    if units == "us":
        delay = record.get("value")
        rms_delay = record.get("sigma")
        doppler = None
        rms_doppler = None
    elif units == "hz":
        delay = None
        rms_delay = None
        doppler = record.get("value")
        rms_doppler = record.get("sigma")
    else:
        raise ValueError(f"Unsupported JPL radar units: {record.get('units')}")

    freq_mhz = record.get("freq")
    freq_hz = None if freq_mhz is None else float(freq_mhz) * 1.0e6
    identifier = str(record.get("des") or desig).strip() or desig
    return {
        "obstime": record.get("epoch"),
        "Obstype": "radar",
        **_designation_to_internal_id_fields(identifier),
        "com": _jpl_bp_to_internal_com(record.get("bp")),
        "delay": delay,
        "rmsdelay": rms_delay,
        "doppler": doppler,
        "rmsdoppler": rms_doppler,
        "rcv": record.get("rcvr"),
        "trx": record.get("xmit"),
        "frq": freq_hz,
        "notes": record.get("notes"),
    }


def _fetch_jpl_radar_records(session: requests.Session, desig: str) -> list[dict[str, Any]]:
    """Fetch radar observations from the JPL small-body radar API."""
    response_json = _request_json(session, JPL_SB_RADAR_API, params={"des": desig})
    return [_convert_jpl_radar_record(record, desig) for record in _table_records_from_jpl_response(response_json)]


def load_online_observations(desig: str, save_path: str | None = None) -> ObservationData:
    """Load current MPC optical observations and JPL radar observations.

    Parameters
    ----------
    desig : str
        Permanent designation, provisional designation, or object name accepted
        by the MPC Observations API. The same identifier is forwarded to the
        JPL radar API.
    save_path : str or None, optional
        Destination path for one ADES PSV file. The filename must use the
        ``.psv`` suffix. If ``None``, no local file is written.

    Returns
    -------
    ObservationData
        Parsed single-target observation bundle assembled from the current MPC
        optical feed and the JPL radar feed.

    Raises
    ------
    ValueError
        Raised when neither remote service returns any observation rows.
    requests.HTTPError
        Raised when one of the remote services returns an unsuccessful status.

    Notes
    -----
    Optical observations are requested as ``ADES_DF`` rows from the MPC
    Observations API. Radar observations are requested from the JPL small-body
    radar API and converted into the shared ADES-compatible row schema before
    the final :class:`ObservationData` bundle is built.

    If ``save_path`` is provided, the function writes the fetched observations
    to ADES PSV before converting the same rows to ``ObservationData``. Radar
    delay rows fetched from JPL are converted from the JPL table units to ADES
    PSV units during the write step.

    References
    ----------
    1. https://docs.minorplanetcenter.net/mpc-ops-docs/apis/get-obs/
    2. https://ssd-api.jpl.nasa.gov/doc/sb_radar.html
    """
    with requests.Session() as session:
        optical_records = _fetch_mpc_ades_records(session, desig, ades_version="2022")
        try:
            radar_records = _fetch_jpl_radar_records(session, desig)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if optical_records and status_code == 400:
                radar_records = []
            else:
                raise

    records = optical_records + radar_records
    if len(records) == 0:
        raise ValueError(f"No online observations were found for designation '{desig}'.")
    if save_path is not None:
        write_ades_psv_records(save_path, records)
    return records_to_observations(records)
