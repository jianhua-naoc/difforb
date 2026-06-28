import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from difforb.astrometry.data import ObsMode, ObsType, ObservationData, OpticalObservationData, RadarObservationData
from difforb.core.constants import RAD_TO_ARCSEC
from difforb.core.state.frame import BCRS
from difforb.core.state.state import State
from difforb.core.time.timescale import Time
from difforb.od.analysis import ODAnalysis, build_observation_table, build_residual_table
from difforb.od.dc.result import DCResult, DCEstimate, LSQDiagnostics, OpticalResult, RadarResult
from difforb.od.result import ODResult
from tests.assertions import assert_allclose, assert_array_equal


EPOCH_TDB_JD = 2460000.5
ARCSEC_TO_RAD = 1.0 / float(RAD_TO_ARCSEC)


def analysis_case():
    optical_t = Time.from_tdb_jd(
        EPOCH_TDB_JD + jnp.asarray([0.1, 0.3], dtype=jnp.float64),
        jnp.zeros(2, dtype=jnp.float64),
    )
    radar_t = Time.from_tdb_jd(
        EPOCH_TDB_JD + jnp.asarray([0.0, 0.2], dtype=jnp.float64),
        jnp.zeros(2, dtype=jnp.float64),
    )
    optical = OpticalObservationData(
        t=optical_t,
        trk_ids=np.asarray(["A", "B"], dtype=object),
        obs_type_ids=np.full(2, ObsType.OPTICAL.id, dtype=int),
        obs_mode_ids=np.full(2, ObsMode.CCD.id, dtype=int),
        values=np.deg2rad(np.asarray([[10.0, 20.0], [30.0, -5.0]], dtype=float)),
        uncertainties=np.asarray([[0.4, 0.5], [0.6, 0.7]], dtype=float) * ARCSEC_TO_RAD,
        correlations=np.asarray([0.0, 0.1], dtype=float),
        time_uncertainties=np.full(2, np.nan, dtype=float),
        rx_codes=np.asarray(["568", "G96"], dtype=str),
        program_codes=np.asarray(["A", "B"], dtype=object),
        catalog_codes=np.asarray(["Gaia2", "Gaia2"], dtype=object),
        note_codes=np.asarray(["", ""], dtype=object),
        magnitudes=np.asarray([18.1, 19.2], dtype=float),
        band_codes=np.asarray(["G", "G"], dtype=object),
        sub_frames=np.asarray(["ICRF", "ICRF"], dtype=object),
        input_indices=np.asarray([1, 3], dtype=int),
    )
    radar = RadarObservationData(
        t=radar_t,
        obs_type_ids=np.full(2, ObsType.RADAR.id, dtype=int),
        obs_mode_ids=np.asarray([ObsMode.DELAY_CENTER.id, ObsMode.DOPPLER_CENTER.id], dtype=int),
        values=np.asarray([100.0, -2.5], dtype=float),
        uncertainties=np.asarray([8.0, 0.5], dtype=float),
        rx_codes=np.asarray(["568", "G96"], dtype=str),
        tx_codes=np.asarray(["856", "856"], dtype=str),
        tx_freq=np.asarray([2.38e9, 8.56e9], dtype=float),
        input_indices=np.asarray([0, 2], dtype=int),
    )
    observations = ObservationData(name="analysis-synthetic", optical=optical, radar=radar)

    optical_adopted_sigmas = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float) * ARCSEC_TO_RAD
    optical_weight_matrices = np.asarray(
        [
            np.diag(1.0 / (optical_adopted_sigmas[0] * optical_adopted_sigmas[0])),
            np.diag(1.0 / (optical_adopted_sigmas[1] * optical_adopted_sigmas[1])),
        ],
        dtype=float,
    )
    radar_weights = np.asarray([1.0 / 25.0, 1.0 / 36.0], dtype=float)
    flat_jacobian = jnp.diag(jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.float64))
    orbit = State(
        Time.from_tdb_jd(EPOCH_TDB_JD, 0.0).tdb(),
        jnp.asarray([1.0, 0.2, -0.1], dtype=jnp.float64),
        jnp.asarray([-0.002, 0.011, 0.001], dtype=jnp.float64),
        BCRS,
    )
    result = DCResult(
        estimate=DCEstimate(
            orbit=orbit,
            model_params=jnp.asarray([], dtype=jnp.float64),
            model_param_names=[],
            cov_mat_post=jnp.eye(6, dtype=jnp.float64),
        ),
        optical=OpticalResult(
            residuals=jnp.asarray([[2.0, -3.0], [-4.0, 6.0]], dtype=jnp.float64) * ARCSEC_TO_RAD,
            normalized_residuals=jnp.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=jnp.float64),
            weighted_rms=1.0,
            unweighted_rms=2.0,
            inlier_masks=jnp.asarray([True, False]),
            metrics=jnp.asarray([5.0, 25.0], dtype=jnp.float64),
        ),
        radar=RadarResult(
            residuals=jnp.asarray([3.0, -4.0], dtype=jnp.float64),
            normalized_residuals=jnp.asarray([0.5, -2.0], dtype=jnp.float64),
            inlier_masks=jnp.asarray([True, False]),
            metrics=jnp.asarray([0.25, 4.0], dtype=jnp.float64),
            delay_weighted_rms=3.0,
            delay_unweighted_rms=3.0,
            doppler_weighted_rms=4.0,
            doppler_unweighted_rms=4.0,
        ),
        lsq_diagnostics=LSQDiagnostics(
            flat_jacobian=flat_jacobian,
            flat_weights=jnp.asarray([1.0, 0.25, 1.0 / 9.0, 0.0625, 0.04, 1.0 / 36.0], dtype=jnp.float64),
            optical_weight_matrices=jnp.asarray(optical_weight_matrices, dtype=jnp.float64),
            radar_weights=jnp.asarray(radar_weights, dtype=jnp.float64),
            cov_mat_prior=jnp.eye(6, dtype=jnp.float64),
            cov_rank=jnp.asarray(6, dtype=jnp.int32),
            cov_condition=jnp.asarray(1.0, dtype=jnp.float64),
            cov_valid=jnp.asarray(True),
            converged=True,
            termination_reason="gradient_converged",
            lsq_iterations=2,
            outlier_iterations=1,
        ),
        normalized_residual_rms=1.5,
    )
    return observations, result


def test_observation_table_mixed():
    observations, result = analysis_case()

    table = build_observation_table(observations, result)

    print(
        "[od.analysis.observations] "
        f"rows={len(table)} "
        f"modalities={table['modality'].tolist()} "
        f"inlier_states={table['inlier_state'].tolist()}"
    )

    assert_array_equal(table["input_index"].to_numpy(dtype=int), np.asarray([0, 1, 2, 3]))
    assert table["modality"].tolist() == ["radar", "optical", "radar", "optical"]
    assert table["observation_type"].tolist() == ["radar_delay", "optical", "radar_doppler", "optical"]
    assert table["inlier_state"].tolist() == ["inlier", "inlier", "outlier", "outlier"]

    optical_row = table[table["input_index"] == 1].iloc[0]
    assert_allclose(optical_row["ra_residual_arcsec"], 2.0, atol=1.0e-12, rtol=0.0)
    assert_allclose(optical_row["dec_residual_arcsec"], -3.0, atol=1.0e-12, rtol=0.0)
    assert_allclose(optical_row["normalized_ra_residual"], 1.0, atol=0.0, rtol=0.0)
    assert_allclose(optical_row["normalized_dec_residual"], -2.0, atol=0.0, rtol=0.0)
    assert_allclose(optical_row["combined_normalized_residual"], np.sqrt(5.0), atol=1.0e-15, rtol=0.0)
    assert_allclose(optical_row["adopted_ra_sigma_arcsec"], 1.0, atol=1.0e-12, rtol=0.0)
    assert_allclose(optical_row["adopted_dec_sigma_arcsec"], 2.0, atol=1.0e-12, rtol=0.0)
    assert_allclose(optical_row["reported_ra_sigma_arcsec"], 0.4, atol=1.0e-12, rtol=0.0)

    delay_row = table[table["input_index"] == 0].iloc[0]
    assert_allclose(delay_row["delay_residual_us"], 3.0, atol=0.0, rtol=0.0)
    assert_allclose(delay_row["normalized_delay_residual"], 0.5, atol=0.0, rtol=0.0)
    assert_allclose(delay_row["adopted_radar_sigma"], 5.0, atol=1.0e-15, rtol=0.0)
    assert delay_row["adopted_radar_sigma_unit"] == "us"

    doppler_row = table[table["input_index"] == 2].iloc[0]
    assert_allclose(doppler_row["doppler_residual_hz"], -4.0, atol=0.0, rtol=0.0)
    assert_allclose(doppler_row["normalized_doppler_residual"], -2.0, atol=0.0, rtol=0.0)
    assert_allclose(doppler_row["adopted_radar_sigma"], 6.0, atol=1.0e-15, rtol=0.0)
    assert doppler_row["adopted_radar_sigma_unit"] == "Hz"


def test_residual_table_components():
    observations, result = analysis_case()
    observation_table = build_observation_table(observations, result)

    residuals = build_residual_table(observation_table)

    print(
        "[od.analysis.residuals] "
        f"components={residuals['component'].tolist()} "
        f"units={residuals['residual_unit'].tolist()}"
    )

    assert_array_equal(residuals["input_index"].to_numpy(dtype=int), np.asarray([0, 1, 1, 2, 3, 3]))
    assert residuals["component"].tolist() == ["delay", "dec", "ra", "doppler", "dec", "ra"]
    assert residuals["residual_unit"].tolist() == ["us", "arcsec", "arcsec", "Hz", "arcsec", "arcsec"]
    assert_allclose(residuals["residual"].to_numpy(dtype=float), np.asarray([3.0, -3.0, 2.0, -4.0, 6.0, -4.0]))
    assert_allclose(residuals["normalized_residual"].to_numpy(dtype=float), np.asarray([0.5, -2.0, 1.0, -2.0, 4.0, 3.0]))
    assert_allclose(residuals["adopted_sigma"].to_numpy(dtype=float), np.asarray([5.0, 2.0, 1.0, 6.0, 4.0, 3.0]))


def test_station_summary():
    observations, result = analysis_case()
    analysis = ODAnalysis.from_result(observations, result)

    summary = analysis.station_summary()

    print(
        "[od.analysis.station_summary] "
        f"groups={len(summary)} "
        f"obs={summary['obs'].tolist()} "
        f"outliers={summary['outliers'].tolist()}"
    )

    optical_568 = summary[
        (summary["station_key"] == "568")
        & (summary["modality"] == "optical")
        & (summary["observation_type"] == "optical")
    ].iloc[0]
    assert int(optical_568["obs"]) == 1
    assert int(optical_568["inliers"]) == 1
    assert int(optical_568["outliers"]) == 0
    assert_allclose(optical_568["chi2_max"], 5.0, atol=0.0, rtol=0.0)
    assert_allclose(optical_568["normalized_residual_rms"], np.sqrt(2.5), atol=1.0e-15, rtol=0.0)

    doppler_g96 = summary[
        (summary["station_key"] == "G96")
        & (summary["modality"] == "radar")
        & (summary["observation_type"] == "radar_doppler")
    ].iloc[0]
    assert int(doppler_g96["obs"]) == 1
    assert int(doppler_g96["inliers"]) == 0
    assert int(doppler_g96["outliers"]) == 1
    assert_allclose(doppler_g96["outlier_percent"], 100.0, atol=0.0, rtol=0.0)
    assert np.isnan(doppler_g96["normalized_residual_rms"])


def test_tracklet_summary_contribution():
    observations, result = analysis_case()
    analysis = ODAnalysis.from_result(observations, result)

    summary = analysis.tracklet_summary(include_contribution=True).sort_values("tracklet_id").reset_index(drop=True)

    print(
        "[od.analysis.tracklet_summary] "
        f"tracklets={summary['tracklet_id'].tolist()} "
        f"weighted={summary['weighted_contribution_percent'].tolist()} "
        f"geometric={summary['geometric_contribution_percent'].tolist()}"
    )

    sigma = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float) * ARCSEC_TO_RAD
    weighted_a = 1.0 / sigma[0, 0] ** 2 + 4.0 / sigma[0, 1] ** 2
    weighted_b = 9.0 / sigma[1, 0] ** 2 + 16.0 / sigma[1, 1] ** 2
    weighted_global = weighted_a + weighted_b + 2.0
    geometric_global = 1.0 + 4.0 + 9.0 + 16.0 + 25.0 + 36.0

    assert summary["tracklet_id"].tolist() == ["A", "B"]
    assert_allclose(summary["weighted_contribution_percent"].to_numpy(dtype=float),
                    np.asarray([weighted_a, weighted_b]) / weighted_global * 100.0,
                    atol=1.0e-12, rtol=1.0e-12)
    assert_allclose(summary["geometric_contribution_percent"].to_numpy(dtype=float),
                    np.asarray([5.0, 25.0]) / geometric_global * 100.0,
                    atol=1.0e-15, rtol=0.0)


def test_analysis_from_result_contract():
    observations, result = analysis_case()
    od_result = ODResult(iod_result=None, dc_result=result)

    from_dc = ODAnalysis.from_result(observations, result)
    from_od = ODAnalysis.from_result(observations, od_result)

    assert len(from_dc.observations) == 4
    assert len(from_dc.residuals) == 6
    assert len(from_od.observations) == 4
    assert len(from_od.residuals) == 6
    with pytest.raises(ValueError, match="does not contain a differential-correction result"):
        ODAnalysis.from_result(observations, ODResult(iod_result=None, dc_result=None))
