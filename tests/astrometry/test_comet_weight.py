import numpy as np
import pytest

from difforb.astrometry import CometWeightPolicy, InteractiveWeightPolicy, ADESWeightPolicy
from difforb.astrometry.io.ades import records_to_observations
from tests.astrometry.test_weight import optical_record


def arcsec(result):
    return np.rad2deg(result.optical_uncertainties) * 3600


def test_density_counts_each_station_and_tracklet_without_grouping_missing_ids():
    rows = [optical_record(f"2025-01-01T01:{i:02d}:00", trkid="shared") for i in range(8)]
    rows += [optical_record(f"2025-01-01T02:{i:02d}:00", station="F51", trkid="shared") for i in range(3)]
    rows += [optical_record(f"2025-01-01T03:{i:02d}:00", trkid="other") for i in range(4)]
    rows += [optical_record(f"2025-01-01T04:{i:02d}:00") for i in range(8)]
    result = CometWeightPolicy().weights(records_to_observations(rows))
    expected = np.repeat(np.array([np.sqrt(2)] * 8 + [1.] * 15)[:, None], 2, axis=1)
    np.testing.assert_allclose(arcsec(result), expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.optical_weights[:8].sum(axis=0), np.full(2, 4 / np.deg2rad(1 / 3600)**2))
    assert set(result.optical_sources) == {"COMET_1ARCSEC"}
    np.testing.assert_array_equal(result.optical_correlations, 0)
    np.testing.assert_array_equal(result.optical_time_uncertainties, 0)


def test_density_uses_station_code_without_roving_position_payload():
    obs = records_to_observations([
        optical_record(f"2025-01-01T01:{i:02d}:00", station="270", system="WGS84",
                       position=(-70. + i, 35., 100.), trkid="shared") for i in range(5)
    ])
    np.testing.assert_allclose(arcsec(CometWeightPolicy().weights(obs)), np.sqrt(5 / 4), rtol=0, atol=1e-12)


@pytest.mark.parametrize(("date", "next_date"), [("2025-01-01", "2025-01-02"), ("1900-01-01", "1900-01-02")])
def test_density_keeps_same_tracklet_across_midnight(date, next_date):
    rows = [optical_record(f"{date}T23:59:{i:02d}", trkid="same") for i in range(4)]
    rows += [optical_record(f"{next_date}T00:00:{i:02d}", trkid="same") for i in range(4)]
    np.testing.assert_allclose(arcsec(CometWeightPolicy().weights(records_to_observations(rows))), np.sqrt(2), rtol=0, atol=1e-12)


def test_interactive_ades_override_preserves_default_density_for_other_rows():
    obs = records_to_observations([optical_record(f"2025-01-01T01:{i:02d}:00", trkid="shared") for i in range(8)])
    baseline = CometWeightPolicy(); ades = ADESWeightPolicy()
    policy = InteractiveWeightPolicy(baseline, [ades]); policy.select_scheme([2], ades)
    result = policy.weights(obs)
    np.testing.assert_allclose(arcsec(result)[[0, 1, 3, 4, 5, 6, 7]], np.sqrt(2), rtol=0, atol=1e-12)
    np.testing.assert_allclose(arcsec(result)[2], .5, rtol=0, atol=1e-12)
    assert result.optical_sources[2] == "ADES"
    policy.restore_default_policy([2])
    np.testing.assert_allclose(arcsec(policy.weights(obs)), np.sqrt(2), rtol=0, atol=1e-12)


def test_empty_optical_bundle_and_radar_passthrough():
    from types import SimpleNamespace
    optical = records_to_observations([optical_record("2025-01-01T00:00:00")]).optical[:0]
    obs = SimpleNamespace(optical=optical, radar=SimpleNamespace(uncertainties=np.array([.3, .7])))
    result = CometWeightPolicy().weights(obs)
    assert result.optical_uncertainties.shape == (0, 2)
    np.testing.assert_array_equal(result.radar_uncertainties, [.3, .7])
    np.testing.assert_array_equal(result.radar_sources, ["ADES", "ADES"])


@pytest.mark.parametrize("fields", [
    {"mode": "PHO"},
    {"mode": "UNK"},
    {"notes": "n"},
    {"subFrm": "B1950.0"},
])
@pytest.mark.parametrize(("date", "sigma"), [
    ("1889-12-31", 10.0),
    ("1890-01-01", 5.0),
    ("1949-12-31", 5.0),
    ("1950-01-01", 2.5),
])
def test_general_photographic_rules_at_date_boundaries(fields, date, sigma):
    record = optical_record(f"{date}T00:00:00", station="F51")
    record.update(fields)
    result = CometWeightPolicy().weights(records_to_observations([record]))
    np.testing.assert_allclose(arcsec(result), sigma, rtol=0, atol=1e-12)


@pytest.mark.parametrize(("mode", "sigma"), [
    ("OCC", .05),
    ("PMT", .2),
    ("MER", .5),
    ("ENC", .75),
    ("MIC", 2.),
    ("TDI", 1.5),
    ("VID", 1.),
])
def test_general_observation_mode_rules(mode, sigma):
    obs = records_to_observations([optical_record("2025-01-01T00:00:00", mode=mode)])
    np.testing.assert_allclose(arcsec(CometWeightPolicy().weights(obs)), sigma, rtol=0, atol=1e-12)


@pytest.mark.parametrize(("station", "sigma"), [("C51", 1.5), ("248", .2)])
def test_general_satellite_and_hipparcos_rules(station, sigma):
    obs = records_to_observations([
        optical_record("2025-01-01T00:00:00", station=station,
                       system="ICRFAU", position=(1., 0., 0.)),
    ])
    np.testing.assert_allclose(arcsec(CometWeightPolicy().weights(obs)), sigma, rtol=0, atol=1e-12)


@pytest.mark.parametrize("mode", ["CCD", "CMO"])
@pytest.mark.parametrize("catalog", ["Gaia3", "UNK"])
def test_ccd_station_program_and_catalog_rules_do_not_override_comet_default(mode, catalog):
    obs = records_to_observations([
        optical_record("2025-01-01T00:00:00", mode=mode, station=station, astcat=catalog, prog=program)
        for station, program in [("F51", ""), ("G96", ""), ("568", "0T"), ("M28", "")]
    ])
    np.testing.assert_allclose(arcsec(CometWeightPolicy().weights(obs)), 1., rtol=0, atol=1e-12)


def test_general_rules_are_applied_before_station_tracklet_density_scaling():
    rows = [optical_record(f"1900-01-01T01:{i:02d}:00", mode="PHO", trkid="shared") for i in range(8)]
    result = CometWeightPolicy().weights(records_to_observations(rows))
    np.testing.assert_allclose(arcsec(result), 5 * np.sqrt(2), rtol=0, atol=1e-12)


def test_tracklet_counts_include_rows_with_different_base_uncertainties():
    rows = [optical_record(f"1900-01-01T01:{i:02d}:00", mode="PHO", trkid="shared") for i in range(4)]
    rows += [optical_record(f"1900-01-01T02:{i:02d}:00", trkid="shared") for i in range(4)]
    result = CometWeightPolicy().weights(records_to_observations(rows))
    expected = np.repeat(np.array([5.] * 4 + [1.] * 4)[:, None], 2, axis=1) * np.sqrt(2)
    np.testing.assert_allclose(arcsec(result), expected, rtol=0, atol=1e-12)
