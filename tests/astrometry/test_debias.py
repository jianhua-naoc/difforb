import numpy as np

from difforb.astrometry.debias import AstrometricDebiasMap


def test_debias_map_uses_ring_ordered_healpix_pixels():
    n_side = 4
    bias_data = np.zeros((12 * n_side * n_side, 1, 4), dtype=float)
    expected_pixels = np.array([72, 58, 160, 11])
    for i, pixel_id in enumerate(expected_pixels, start=1):
        bias_data[pixel_id, 0] = (float(i), -float(i), 0.0, 0.0)

    debias_map = AstrometricDebiasMap(
        bias_data=bias_data,
        supported_codes=["UCAC4"],
        used_codes=["UCAC4"],
        n_side=n_side,
    )

    tt_jd = np.full(4, 2451545.0)
    ra = np.array([0.0, 1.0, 3.2, 6.0])
    dec = np.array([0.0, 0.3, -0.7, 1.1])
    query_codes = np.array(["UCAC4"] * 4)

    ra_bias, dec_bias = debias_map._get_bias_from_arrays(tt_jd, ra, dec, query_codes)

    expected_ra_arcsec = np.arange(1.0, 5.0) / np.cos(dec)
    expected_dec_arcsec = -np.arange(1.0, 5.0)
    np.testing.assert_allclose(ra_bias, np.deg2rad(expected_ra_arcsec / 3600.0))
    np.testing.assert_allclose(dec_bias, np.deg2rad(expected_dec_arcsec / 3600.0))
