"""Astrometric debias maps and debias policies.

This module loads the Eggl et al. 2018 bias table. It applies catalog-based
right-ascension and declination corrections to optical observations. It returns
one optical bias array in radians.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, NamedTuple

import numpy as np
import pandas as pd
from astropy import units as u
from astropy_healpix import lonlat_to_healpix
from jaxtyping import Float

from difforb.astrometry.data import ObservationData
from difforb.core.config import get_data_path, missing_data_message
from difforb.core.constants import ARCSEC_TO_DEG, J2000
from difforb.utils import as_str_array

DEFAULT_BIAS_FILENAME = str(get_data_path("debias_2018/bias.dat", dataset="debias2018", must_exist=False))


class AstrometricDebiasMap:
    """Map catalog code and sky position to optical bias values."""

    def __init__(self, bias_data: Float[np.ndarray, "N"], supported_codes: Union[List, np.ndarray],
                 used_codes: Union[List, np.ndarray], n_side: int):
        """Initialize an astrometric debias map.

        Parameters
        ----------
        bias_data : Float[np.ndarray, "N"]
            Bias table with shape ``N_pixel N_catalog 4``. The last axis stores ``dRA``, ``dDEC``, ``dPMRA``, and ``dPMDEC``.
        supported_codes : list or np.ndarray
            Catalog codes in the same order as the table.
        used_codes : list or np.ndarray
            Catalog codes that can be corrected.
        n_side : int
            HEALPix ``NSIDE`` value.
        """
        self.bias_data = bias_data
        self.catalog_codes = as_str_array(supported_codes)
        self.used_catalog_codes = as_str_array(used_codes)
        self.n_side = n_side

    def _get_bias_from_arrays(self, tt_jd: Float[np.ndarray, "N"], ra: Float[np.ndarray, "N"],
                              dec: Float[np.ndarray, "N"], query_codes: Union[List, np.ndarray]) -> \
            Tuple[Float[np.ndarray, "N"], Float[np.ndarray, "N"]]:
        # 1. Mask of observations needed to be debiased
        query_codes = as_str_array(query_codes)
        mask = np.isin(query_codes,
                       self.used_catalog_codes)
        masked_t_jd = tt_jd[mask]
        masked_ra = ra[mask]
        masked_dec = dec[mask]
        masked_query_codes = query_codes[mask]
        # 2. Map ``RA`` and ``DEC`` to the same RING-ordered HEALPix pixels as the debias table.
        pixel_ids = lonlat_to_healpix(
            masked_ra * u.rad,
            masked_dec * u.rad,
            self.n_side,
            order="ring",
        )
        # 3. Get catalog id
        sorter = np.argsort(self.catalog_codes)
        sorted_catalog_ids = np.searchsorted(self.catalog_codes, masked_query_codes, sorter=sorter)
        catalog_ids = sorter[sorted_catalog_ids]
        # 4. Get bias parameter: dRA, dDEC, dPMRA, dPMDEC
        bias_ra = np.zeros_like(ra)
        bias_dec = np.zeros_like(dec)
        dRA, dDEC, dPMRA, dPMDEC = self.bias_data[pixel_ids, catalog_ids].T
        bias_ra[mask] = (dRA + (masked_t_jd - J2000) / 365.25 * dPMRA / 1000.) / np.cos(masked_dec)
        bias_ra = np.deg2rad(bias_ra * ARCSEC_TO_DEG)
        bias_dec[mask] = dDEC + (masked_t_jd - J2000) / 365.25 * dPMDEC / 1000.
        bias_dec = np.deg2rad(bias_dec * ARCSEC_TO_DEG)
        return bias_ra, bias_dec

    def get_bias(self, obs: ObservationData) -> Float[np.ndarray, "N 2"]:
        """Return optical bias arrays for one observation set.

        Parameters
        ----------
        obs : ObservationData
            Observation data for one target.

        Returns
        -------
        Float[np.ndarray, "N 2"]
            Optical bias arrays in radians. Columns are right ascension and
            declination.
        """
        ra_bias, dec_bias = self._get_bias_from_arrays(
            np.asarray(obs.optical.t.tt.jd),
            obs.optical.values[:, 0],
            obs.optical.values[:, 1],
            obs.optical.catalog_codes,
        )
        return np.stack([ra_bias, dec_bias], axis=-1)

    @classmethod
    def from_bias2018(cls, filepath: str = DEFAULT_BIAS_FILENAME) -> 'AstrometricDebiasMap':
        """Build a debias map from the Eggl et al. 2018 table file.

        Parameters
        ----------
        filepath : str, default=DEFAULT_BIAS_FILENAME
            Path to the bias table.

        Returns
        -------
        AstrometricDebiasMap
            Loaded debias map.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(missing_data_message("debias2018", filepath))
        annotation_label = "!"
        nside_pattern = r'NSIDE=\s*(\d+)'
        npix_pattern = r'NPIX=\s*(\d+)'
        supported_catalog_codes = ['USNOA1', 'USNOSA1', 'USNOA2', 'USNOSA2', 'UCAC1', 'Tyc2', 'GSC1.1', 'GSC1.2', 'ACT', 'GSCACT',
                                   'SDSS8',
                                   'USNOB1',
                                   'PPM',
                                   'UCAC4',
                                   'UCAC2', 'PPMXL',
                                   'UCAC3',
                                   'NOMAD',
                                   'CMC14', '2MASS',
                                   'SDSS7', 'CMC15', 'SSTRC4', 'URAT1', 'Gaia1', 'UCAC5']
        used_catalog_codes = ['USNOA1', 'USNOSA1', 'USNOA2', 'USNOSA2', 'UCAC1', 'GSC1.1', 'GSC1.2', 'GSCACT', 'SDSS8', 'USNOB1',
                              'PPM',
                              'UCAC4', 'UCAC2',
                              'PPMXL',
                              'UCAC3',
                              'NOMAD', 'CMC14',
                              '2MASS',
                              'SDSS7',
                              'CMC15', 'SSTRC4', 'URAT1']
        # Read header information
        with open(filepath, 'r') as f:
            for row in f:
                row = row.strip()
                # Empty row
                if not row:
                    continue
                # Annotation row
                if row.startswith(annotation_label):
                    # Extract NSIDE
                    nside_match = re.search(nside_pattern, row)
                    if nside_match:
                        side_num = int(nside_match.group(1))
                    # Extract NPIX
                    npix_match = re.search(npix_pattern, row)
                    if npix_match:
                        pixel_num = int(npix_match.group(1))
                    continue
                break

        # Read bias data
        df = pd.read_csv(filepath, sep=r'\s+', comment=annotation_label, header=None, engine='c')
        bias_data = df.to_numpy()
        bias_data = bias_data.reshape((pixel_num, len(supported_catalog_codes), 4))
        return AstrometricDebiasMap(bias_data, supported_catalog_codes, used_catalog_codes, side_num)


class DebiasResult(NamedTuple):
    """Bias arrays returned by a debias policy."""
    optical_bias: Float[np.ndarray, "N_optical 2"]


class DebiasPolicy(ABC):
    """Base class for optical debias policies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the policy name."""
        pass

    @abstractmethod
    def bias(self, obs: ObservationData) -> DebiasResult:
        """Return optical bias arrays.

        Parameters
        ----------
        obs : ObservationData
            Observation data for one target.

        Returns
        -------
        DebiasResult
            Bias arrays in radians.
        """
        pass


class NoDebiasPolicy(DebiasPolicy):
    """Policy that returns zero optical bias."""

    @property
    def name(self) -> str:
        """Return the policy name."""
        return "No Policy"

    def bias(self, obs: ObservationData) -> DebiasResult:
        """Return zero bias arrays for each optical observation.

        Parameters
        ----------
        obs : ObservationData
            Observation data for one target.

        Returns
        -------
        DebiasResult
            Zero bias arrays in radians.
        """
        return DebiasResult(np.zeros((obs.num_optical, 2)))


class EgglDebiasPolicy(DebiasPolicy):
    """Policy for the Eggl et al. 2018 astrometric debias map."""

    def __init__(self):
        """Initialize the policy with the default debias map."""
        self.debias_map = AstrometricDebiasMap.from_bias2018()

    @property
    def name(self) -> str:
        """Return the policy name."""
        return "Eggl"

    def bias(self, obs: ObservationData) -> DebiasResult:
        """Return Eggl et al. 2018 bias arrays for optical observations.

        Parameters
        ----------
        obs : ObservationData
            Observation data for one target.

        Returns
        -------
        DebiasResult
            Bias arrays in radians.
        """
        return DebiasResult(self.debias_map.get_bias(obs))
