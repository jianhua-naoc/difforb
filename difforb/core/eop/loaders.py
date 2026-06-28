"""Load Earth orientation parameter tables distributed by IERS.

This module reads the IERS EOP C04 series used by DiffOrb and returns :class:`difforb.core.eop.container.EarthOrientationData`.

Table samples are stored on daily Modified Julian Dates (``MJD``). Date ranges follow the values stored in the source file.
"""

import datetime
import os
import re

import jax.numpy as jnp

from difforb.core.config import get_writable_data_path
from difforb.core.constants import MJD_START
from difforb.core.eop.container import EarthOrientationData
from difforb.utils import download

DEFAULT_IERS_EOP_C04_FILENAME = str(get_writable_data_path("iers/eopc04.dPsi_dEps.1962-now.txt"))
DEFAULT_IERS_EOP_C04_URL = "https://datacenter.iers.org/data/latestVersion/EOP_20u24_C04_0h_dPsi_dEps_1962-now.txt"
DEFAULT_IERS_EOP_C04_FALLBACK_URL = "https://hpiers.obspm.fr/iers/eop/eopc04/eopc04.dPsi_dEps.1962-now"
DEFAULT_IERS_EOP_C04_URLS = (
    DEFAULT_IERS_EOP_C04_URL,
    DEFAULT_IERS_EOP_C04_FALLBACK_URL,
)


def parse_iers_eopc04(filepath: str) -> EarthOrientationData:
    """Parse an IERS EOP C04 text file into an interpolation container.

    Parameters
    ----------
    filepath : str
        Path to the downloaded C04 text file.

    Returns
    -------
    EarthOrientationData
        Daily ``MJD`` samples with polar-motion coordinates in arcseconds (``xpole`` and ``ypole``), ``UT1-UTC`` in seconds, and additive ``dPsi`` and ``dEps`` correction terms in arcseconds. ``final_date_range`` stores the file span in ``UTC`` Julian dates.
    """
    with open(filepath, "r") as f:
        rows = f.readlines()
        iers_oep_data = []
        for row in rows:
            if row.startswith("#"):
                continue
            else:
                part = re.split(r"\s+", row.strip())
                mjd = float(part[4])
                # in arcsecond
                xpole = float(part[5])
                ypole = float(part[6])
                # in second
                ut1dutc = float(part[7])
                # in arcsecond
                dpsi = float(part[8])
                deps = float(part[9])
                iers_oep_data.append([mjd, xpole, ypole, ut1dutc, dpsi, deps])
        iers_oep_data = jnp.array(iers_oep_data)
        date_range = jnp.array([iers_oep_data[0][0] + MJD_START, iers_oep_data[-1][0] + MJD_START])
        return EarthOrientationData(
            mjds=iers_oep_data[:, 0],
            xpoles=iers_oep_data[:, 1],
            ypoles=iers_oep_data[:, 2],
            ut1dutcs=iers_oep_data[:, 3],
            dpsis=iers_oep_data[:, 4],
            depss=iers_oep_data[:, 5],
            final_date_range=date_range,
        )


def load_iers_eopc04(auto_update: bool = True, *, force_update: bool = False) -> EarthOrientationData:
    """Load the local IERS EOP C04 table, downloading updates when needed.

    Parameters
    ----------
    auto_update : bool, optional
        If ``True``, refresh the local file when the current ``UTC`` Julian Date is more than 35 days beyond the last final record
        in the file.
    force_update : bool, default=False
        If ``True``, download the latest file before returning, regardless of the local file age.

    Returns
    -------
    EarthOrientationData
        Parsed C04 Earth Orientation Parameter (EOP) data.
    """
    if force_update:
        _download_iers_eopc04(DEFAULT_IERS_EOP_C04_FILENAME)
        return parse_iers_eopc04(DEFAULT_IERS_EOP_C04_FILENAME)

    from difforb.core.time.utils import julian_date
    if os.path.exists(DEFAULT_IERS_EOP_C04_FILENAME):
        eop = parse_iers_eopc04(DEFAULT_IERS_EOP_C04_FILENAME)
        cur_time = datetime.datetime.now()
        cur_jd1, cur_jd2 = julian_date(jnp.array(cur_time.year),
                                       jnp.array(cur_time.month),
                                       jnp.array(cur_time.day), 0, 0, 0)
        cur_jd = cur_jd1 + cur_jd2
        # need update?
        if auto_update and cur_jd > eop.final_date_range[1] + 35.:
            _download_iers_eopc04(DEFAULT_IERS_EOP_C04_FILENAME)
            eop = parse_iers_eopc04(DEFAULT_IERS_EOP_C04_FILENAME)
    else:
        _download_iers_eopc04(DEFAULT_IERS_EOP_C04_FILENAME)
        eop = parse_iers_eopc04(DEFAULT_IERS_EOP_C04_FILENAME)
    return eop


def _download_iers_eopc04(filepath: str) -> None:
    errors = []
    for url in DEFAULT_IERS_EOP_C04_URLS:
        try:
            download(filepath, url)
            return
        except Exception as exc:
            errors.append((url, exc))

    detail = "\n".join(
        f"    - {url}: {type(exc).__name__}: {exc}"
        for url, exc in errors
    )
    raise OSError(
        "Could not download the IERS EOP C04 table from any configured source.\n"
        f"Tried:\n{detail}"
    ) from errors[-1][1]
