import numpy as np

from difforb.astrometry.data import ObserverType
from difforb.astrometry.io.ades import load_local_observations, records_to_observations, write_ades_psv_records
from difforb.body.site import Site, format_site_key


def optical_record(
        *,
        station: str,
        system: str = "",
        pos1: float | None = None,
        pos2: float | None = None,
        pos3: float | None = None,
        subfrm: str = "",
        rmscorr: float | None = None,
        rmstime: float | None = None,
) -> dict[str, object]:
    return {
        "obstime": "2025-01-01T00:00:00",
        "Obstype": "optical",
        "mode": "CCD",
        "sys": system,
        "ra": 10.0,
        "dec": 20.0,
        "rmsra": 0.5,
        "rmsdec": 0.6,
        "rmscorr": np.nan if rmscorr is None else rmscorr,
        "rmstime": np.nan if rmstime is None else rmstime,
        "stn": station,
        "prog": "",
        "astcat": "",
        "notes": "",
        "mag": np.nan,
        "band": "",
        "subFrm": subfrm,
        "pos1": pos1,
        "pos2": pos2,
        "pos3": pos3,
        "trkid": "",
        "provid": "synthetic",
    }


def test_records_to_observations_builds_unified_optical_site_keys():
    records = [
        optical_record(station="568"),
        optical_record(station="247", system="WGS84", pos1=1.0, pos2=2.0, pos3=3.0),
        optical_record(station="C51", system="ICRFAU", pos1=1.0e-4, pos2=2.0e-4, pos3=3.0e-4),
    ]

    observations = records_to_observations(records)

    expected_rx_codes = np.asarray(
        [
            "568",
            format_site_key("247", Site.TYPE_ROVING_GROUND, [1.0, 2.0, 3.0]),
            format_site_key("C51", Site.TYPE_SATELLITE, [1.0e-4, 2.0e-4, 3.0e-4]),
        ],
        dtype=str,
    )
    np.testing.assert_array_equal(observations.optical.rx_codes, expected_rx_codes)
    np.testing.assert_array_equal(
        observations.optical.observer_type_ids,
        np.asarray(
            [
                ObserverType.GROUND_FIXED.id,
                ObserverType.GROUND_ROVING.id,
                ObserverType.SPACE_BASED.id,
            ],
            dtype=int,
        ),
    )
    assert observations.num_optical == 3
    assert observations.num_radar == 0


def test_records_to_observations_preserves_ades_sub_frame():
    observations = records_to_observations(
        [
            optical_record(station="568", subfrm="B1950.0"),
            optical_record(station="G96"),
        ]
    )

    np.testing.assert_array_equal(observations.optical.sub_frames, np.asarray(["B1950.0", ""], dtype=str))
    assert observations.optical.to_dataframe().loc[0, "sub_frame"] == "B1950.0"


def test_records_to_observations_preserves_rms_correlation():
    observations = records_to_observations(
        [
            optical_record(station="568", rmscorr=0.25),
            optical_record(station="G96"),
        ]
    )

    np.testing.assert_allclose(observations.optical.correlations, np.asarray([0.25, 0.0]), rtol=0.0, atol=0.0)
    assert observations.optical.to_dataframe().loc[0, "ra_dec_correlation"] == 0.25


def test_ades_psv_write_and_load_round_trips_sub_frame(tmp_path):
    path = tmp_path / "sub-frame.psv"
    write_ades_psv_records(str(path), [optical_record(station="568", subfrm="B1950.0")])

    text = path.read_text(encoding="utf-8")
    assert "subFrm" in text

    observations = load_local_observations(str(path))

    np.testing.assert_array_equal(observations.optical.sub_frames, np.asarray(["B1950.0"], dtype=str))


def test_ades_psv_write_and_load_round_trips_rms_correlation(tmp_path):
    path = tmp_path / "rms-corr.psv"
    write_ades_psv_records(str(path), [optical_record(station="568", rmscorr=-0.4)])

    text = path.read_text(encoding="utf-8")
    assert "rmsCorr" in text

    observations = load_local_observations(str(path))

    np.testing.assert_allclose(observations.optical.correlations, np.asarray([-0.4]), rtol=0.0, atol=0.0)


def test_ades_psv_write_and_load_round_trips_rms_time(tmp_path):
    path = tmp_path / "rms-time.psv"
    write_ades_psv_records(str(path), [optical_record(station="568", rmstime=2.5)])

    text = path.read_text(encoding="utf-8")
    assert "rmsTime" in text

    observations = load_local_observations(str(path))

    np.testing.assert_allclose(observations.optical.time_uncertainties, np.asarray([2.5]), rtol=0.0, atol=0.0)
