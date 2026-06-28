import numpy as np

from difforb.astrometry.io.ades import records_to_observations
from difforb.astrometry.weight import ADESWeightPolicy, VFCC17WeightPolicy


def optical_record(
        obstime: str, *, mode: str = "CCD", station: str = "568", prog: str = "",
        astcat: str = "Gaia3", subfrm: str = "", rmstime: float = np.nan,
        system: str = "", position: tuple[float, float, float] | None = None,
        trkid: str = "") -> dict[str, object]:
    pos1, pos2, pos3 = (np.nan, np.nan, np.nan) if position is None else position
    return {
        "obstime": obstime,
        "Obstype": "optical",
        "mode": mode,
        "sys": system,
        "ra": 10.0,
        "dec": 20.0,
        "rmsra": 0.5,
        "rmsdec": 0.5,
        "rmscorr": 0.3,
        "rmsTime": rmstime,
        "stn": station,
        "prog": prog,
        "astcat": astcat,
        "notes": "",
        "mag": np.nan,
        "band": "",
        "subFrm": subfrm,
        "pos1": pos1,
        "pos2": pos2,
        "pos3": pos3,
        "trkid": trkid,
        "provid": "synthetic",
    }


def test_vfcc17_sub_frame_rules_assign_b1950_table5_sigmas():
    observations = records_to_observations(
        [
            optical_record("2025-01-01T00:00:00"),
            optical_record("2025-01-01T00:00:00", subfrm="B1950.0"),
            optical_record("1900-01-01T00:00:00", subfrm="B1950.0"),
            optical_record("1880-01-01T00:00:00", subfrm="B1950.0"),
        ]
    )

    weights = VFCC17WeightPolicy().weights(observations)
    ra_sigmas_arcsec = np.rad2deg(weights.optical_uncertainties[:, 0]) * 3600.0
    dec_sigmas_arcsec = np.rad2deg(weights.optical_uncertainties[:, 1]) * 3600.0

    np.testing.assert_allclose(ra_sigmas_arcsec, np.asarray([0.5, 2.5, 5.0, 10.0]), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(dec_sigmas_arcsec, np.asarray([0.5, 2.5, 5.0, 10.0]), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(weights.optical_correlations, np.zeros(4), rtol=0.0, atol=0.0)


def test_vfcc17_grss_station_defaults_survive_orbfit_cleanup():
    observations = records_to_observations(
        [
            optical_record("2025-01-01T00:00:00", station="F52"),
            optical_record("2025-01-01T00:00:00", station="F52", prog="15"),
            optical_record("2025-01-01T00:00:00", station="F52", prog="Z"),
            optical_record("2025-01-01T00:00:00", station="V39"),
            optical_record("2025-01-01T00:00:00", station="Z24"),
            optical_record("2025-01-01T00:00:00", station="Z31"),
        ]
    )

    weights = VFCC17WeightPolicy().weights(observations)
    ra_sigmas_arcsec = np.rad2deg(weights.optical_uncertainties[:, 0]) * 3600.0

    np.testing.assert_allclose(ra_sigmas_arcsec, np.asarray([0.2, 1.0, 0.2, 0.4, 0.4, 0.4]), rtol=0.0, atol=1.0e-12)


def test_vfcc17_program_rules_use_ades_two_character_codes():
    observations = records_to_observations(
        [
            optical_record("2025-01-01T00:00:00", station="T09", prog="01"),
            optical_record("2025-01-01T00:00:00", station="T09", prog="04"),
            optical_record("2025-01-01T00:00:00", station="T09", prog="14"),
        ]
    )

    weights = VFCC17WeightPolicy().weights(observations)
    ra_sigmas_arcsec = np.rad2deg(weights.optical_uncertainties[:, 0]) * 3600.0

    np.testing.assert_allclose(ra_sigmas_arcsec, np.asarray([1.0, 1.0, 0.5]), rtol=0.0, atol=1.0e-12)


def test_vfcc17_over_observation_uses_tracklet_ids_not_roving_station_code():
    observations = records_to_observations(
        [
            optical_record(
                f"2025-01-01T00:{index:02d}:00",
                station="270",
                system="WGS84",
                position=(-70.0 + index, 35.0 + index * 0.1, 100.0 + index),
            )
            for index in range(5)
        ]
    )

    weights = VFCC17WeightPolicy().weights(observations)
    ra_sigmas_arcsec = np.rad2deg(weights.optical_uncertainties[:, 0]) * 3600.0

    np.testing.assert_allclose(
        ra_sigmas_arcsec,
        np.full(5, 2.0),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_vfcc17_over_observation_scales_nonempty_tracklet():
    observations = records_to_observations(
        [
            optical_record(
                f"2025-01-01T00:{index:02d}:00",
                station="T08",
                trkid="tracklet-a",
            )
            for index in range(5)
        ]
    )

    weights = VFCC17WeightPolicy().weights(observations)
    ra_sigmas_arcsec = np.rad2deg(weights.optical_uncertainties[:, 0]) * 3600.0

    np.testing.assert_allclose(
        ra_sigmas_arcsec,
        np.full(5, 0.8 * np.sqrt(5.0 / 4.0)),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_vfcc17_over_observation_does_not_count_entire_station_globally():
    observations = records_to_observations(
        [
            optical_record(
                f"2025-01-{day:02d}T00:{minute:02d}:00",
                station="T08",
            )
            for day in range(1, 9)
            for minute in range(4)
        ]
    )

    weights = VFCC17WeightPolicy().weights(observations)
    ra_sigmas_arcsec = np.rad2deg(weights.optical_uncertainties[:, 0]) * 3600.0

    np.testing.assert_allclose(
        ra_sigmas_arcsec,
        np.full(32, 0.8),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_vfcc17_grss_default_mode_rules_match_current_csv():
    observations = records_to_observations(
        [
            optical_record("2025-01-01T00:00:00", station="000", astcat="UNK"),
            optical_record("2025-01-01T00:00:00", mode="VID", station="000", astcat="UNK"),
            optical_record("2025-01-01T00:00:00", mode="OCC"),
            optical_record("2025-01-01T00:00:00", mode="PMT"),
            optical_record("2025-01-01T00:00:00", mode="TDI"),
        ]
    )

    weights = VFCC17WeightPolicy().weights(observations)
    ra_sigmas_arcsec = np.rad2deg(weights.optical_uncertainties[:, 0]) * 3600.0

    np.testing.assert_allclose(ra_sigmas_arcsec, np.asarray([1.5, 1.5, 0.05, 0.2, 1.5]), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(weights.optical_time_uncertainties, np.asarray([1.0, 1.0, np.nan, 1.0, 1.0]), rtol=0.0, atol=0.0)


def test_vfcc17_grss_time_uncertainty_defaults_and_reported_values():
    observations = records_to_observations(
        [
            optical_record("2025-01-01T00:00:00", station="000"),
            optical_record("2025-01-01T00:00:00", station="000", rmstime=2.5),
            optical_record("2025-01-01T00:00:00", mode="OCC", station="000"),
            optical_record("2025-01-01T00:00:00", station="258"),
            optical_record("2025-01-01T00:00:00", mode="OCC", station="000", rmstime=2.5),
            optical_record("2025-01-01T00:00:00", station="258", rmstime=2.5),
        ]
    )

    weights = VFCC17WeightPolicy().weights(observations)

    np.testing.assert_allclose(weights.optical_time_uncertainties[:2], np.asarray([1.0, 2.5]), rtol=0.0, atol=0.0)
    assert np.isnan(weights.optical_time_uncertainties[2])
    assert np.isnan(weights.optical_time_uncertainties[3])
    assert np.isnan(weights.optical_time_uncertainties[4])
    assert np.isnan(weights.optical_time_uncertainties[5])


def test_weight_result_inflates_optical_covariance_with_time_uncertainty():
    observations = records_to_observations([optical_record("2025-01-01T00:00:00", station="000", rmstime=2.0)])
    weights = ADESWeightPolicy().weights(observations)
    rates = np.asarray([[3.0e-4, -4.0e-4]], dtype=float)

    covariance = weights.optical_covariances_with_time(rates)[0]
    base_covariance = weights.optical_covariances[0]
    time_days = 2.0 / 86400.0
    expected = base_covariance + np.outer(rates[0], rates[0]) * time_days * time_days

    np.testing.assert_allclose(covariance, expected, rtol=1.0e-12, atol=0.0)

    effective_weights = weights.with_optical_time_rates(rates)
    np.testing.assert_allclose(effective_weights.optical_covariances[0], expected, rtol=1.0e-12, atol=0.0)
    np.testing.assert_allclose(effective_weights.optical_time_uncertainties, np.asarray([0.0]), rtol=0.0, atol=0.0)


def test_ades_weight_policy_uses_reported_correlation():
    observations = records_to_observations([optical_record("2025-01-01T00:00:00", rmstime=3.0)])

    weights = ADESWeightPolicy().weights(observations)

    assert weights.optical_correlations[0] == 0.3
    assert weights.optical_time_uncertainties[0] == 3.0
    covariance = weights.optical_covariances[0]
    weight_matrix = weights.optical_weight_matrices[0]

    np.testing.assert_allclose(covariance @ weight_matrix, np.eye(2), rtol=1.0e-12, atol=1.0e-12)
