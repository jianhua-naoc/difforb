"""Observation-loading helpers built around ADES observations.

This package keeps the public observation-ingestion surface small:

- :func:`load_local_observations` reads local ADES PSV files,
- :func:`load_online_observations` fetches current MPC optical and JPL radar data.

The internal numerical model remains the modality-separated
``difforb.astrometry.data.ObservationData`` container. The ``io`` package only
handles conversion from external ADES-related records into that container.
"""

from difforb.astrometry.io.ades import load_local_observations
from difforb.astrometry.io.ades import records_to_ades_psv_text
from difforb.astrometry.io.ades import write_ades_psv_records
from difforb.astrometry.io.online import load_online_observations

__all__ = [
    "load_local_observations",
    "load_online_observations",
    "records_to_ades_psv_text",
    "write_ades_psv_records",
]
