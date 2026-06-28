# Analyze Residuals By Station And Tracklet

This guide shows how to join a `DCResult` or `ODResult` back to the input observations. It also shows how to run station-level and tracklet-level statistical analysis.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- Load observations from one local observation file.
- Run orbit determination or differential correction and keep both `obs` and `result`. The `obs` object must be the same observation set used to produce `result`.
- Keep the original `input_index` values from the observation file. See [Load Local ADES Observations](load-local-ades-observations.md).
- Use this guide after you have chosen weights and outlier settings.

For the residual and outlier rules, read [Differential Correction](../concepts/differential-correction.md) and [Outlier Rejection](../concepts/outlier-rejection.md).

## 1. Build the analysis tables

`ODAnalysis.from_result(...)` builds standardized pandas tables from the observations and the solved result. The object stores:

- `analysis.observations`: one row per observation, with residuals, normalized residuals, chi-square metrics, inlier flags, and adopted uncertainty columns.
- `analysis.residuals`: one row per scalar residual component.
- `analysis.station_summary()`: grouped station-level residual and outlier statistics.
- `analysis.tracklet_summary(include_contribution=True)`: grouped tracklet statistics with optional normal-matrix contribution metrics.

```python
from difforb.od import ODAnalysis

analysis = ODAnalysis.from_result(obs, result)

cols = [
    "input_index",
    "station_key",
    "inlier",
    "ra_residual_arcsec",
    "dec_residual_arcsec",
    "chi2",
]

print(analysis.observations[cols].head(3).to_string(index=False))
```

```text title="Output"
 input_index station_key  inlier  ra_residual_arcsec  dec_residual_arcsec      chi2
         350         203    True           -0.262155             0.026162  0.074983
         351         203    True           -0.000118            -0.230026  0.056159
         352         938    True           -0.886054            -0.152916  0.253836
```

## 2. Analyze station residuals and weights

Use station summaries to find stations with possible systematic residual bias, wrong weight scale, or unusual rejection rate. The mean residual columns show station bias. The normalized residual scatter columns show whether the adopted optical uncertainties match the fitted residuals.

```python
station_stats = analysis.station_summary()

cols = [
    "station_key",
    "obs",
    "inliers",
    "outlier_percent",
    "mean_ra_residual_arcsec",
    "std_normalized_ra_residual",
    "std_normalized_dec_residual",
    "normalized_residual_spread",
]

print(station_stats[cols].head(5).to_string(index=False))
```

```text title="Output"
station_key  obs  inliers  outlier_percent  mean_ra_residual_arcsec  std_normalized_ra_residual  std_normalized_dec_residual  normalized_residual_spread
        168   13       13              0.0                 0.229473                    0.171732                    0.241504                    0.209517
        938   13       13              0.0                -0.045656                    0.187568                    0.262132                    0.228001
        M31    8        8              0.0                -0.374215                    0.258697                    0.301722                    0.281038
        C23    6        6              0.0                -0.000909                    0.613301                    0.573920                    0.594937
        K74    4        4              0.0                -0.098672                    0.048700                    0.081203                    0.066997
```

The returned station table groups by station, observer type, modality, and observation type. Its common columns are:

- `station_key`: canonical station key used for grouping.
- `station`: display station label.
- `observer_type`: observer category from the observation table.
- `modality`: `optical` or `radar`.
- `observation_type`: `optical`, `radar_delay`, or `radar_doppler`.
- `obs`: number of observations in the group.
- `inliers`: number of observations from the group used in the final fit.
- `outliers`: number of rejected observations.
- `outlier_percent`: rejected observations as a percentage of `obs`.
- `chi2_max`: maximum chi-square rejection metric in the group.
- `chi2_p95`: 95th percentile chi-square rejection metric in the group.
- `normalized_residual_rms`: root-mean-square normalized residual for inlier residual components.
- `mean_ra_residual_arcsec`: mean right ascension residual for inliers, in arcseconds.
- `std_ra_residual_arcsec`: standard deviation of right ascension residuals for inliers, in arcseconds.
- `std_normalized_ra_residual`: standard deviation of normalized right ascension residuals for inliers.
- `mean_dec_residual_arcsec`: mean declination residual for inliers, in arcseconds.
- `std_dec_residual_arcsec`: standard deviation of declination residuals for inliers, in arcseconds.
- `std_normalized_dec_residual`: standard deviation of normalized declination residuals for inliers.
- `normalized_residual_spread`: `sqrt((std_normalized_ra_residual^2 + std_normalized_dec_residual^2) / 2)`.
- `max_normalized_residual_std`: larger value of `std_normalized_ra_residual` and `std_normalized_dec_residual`.
- `mean_delay_residual_us`: mean radar delay residual for inliers, in microseconds.
- `std_delay_residual_us`: standard deviation of radar delay residuals for inliers, in microseconds.
- `std_normalized_delay_residual`: standard deviation of normalized radar delay residuals for inliers.
- `mean_doppler_residual_hz`: mean radar Doppler residual for inliers, in hertz.
- `std_doppler_residual_hz`: standard deviation of radar Doppler residuals for inliers, in hertz.
- `std_normalized_doppler_residual`: standard deviation of normalized radar Doppler residuals for inliers.

Optical columns are present when the group has optical observations. Radar columns are present when the group has radar observations.

Use `normalized_residual_spread` as a weight-scale check. A value near `1` means the adopted optical uncertainty matches the residual scatter. A value far below `1` means the adopted uncertainty is too large, so the weight policy is too conservative. A value far above `1` means the adopted uncertainty is too small, so the weight policy is too optimistic.

## 3. Analyze tracklets

Use tracklet summaries to find tracklets with possible local residual bias, wrong weight scale, high rejection rate, or high contribution to the fit. The contribution columns show how much one tracklet affects the final normal matrix.

```python
tracklet_stats = analysis.tracklet_summary(include_contribution=True)

cols = [
    "tracklet_id",
    "obs",
    "inliers",
    "station",
    "duration_days",
    "normalized_residual_spread",
    "weighted_contribution_percent",
]

print(tracklet_stats[cols].head(5).to_string(index=False))
```

```text title="Output"
tracklet_id  obs  inliers station  duration_days  normalized_residual_spread  weighted_contribution_percent
00000Il85h    8        8     M31       0.002750                    0.281038                     20.738288
00000IjyBI   13       13     938       0.099734                    0.228001                      5.159399
00000IkVWa    4        4     W68       0.044334                    0.417620                     11.158416
00000IkzNj    2        2     168       0.038990                    0.209517                     10.470362
00000IkCA2   11       11     168       0.193280                    0.318442                      3.200758
```

The returned tracklet table groups by `tracklet_id`. Its columns are:

- `tracklet_id`: tracklet identifier from the observation table.
- `station`: display station label for the tracklet.
- `start_time_ut_iso`: earliest observation time in the tracklet.
- `end_time_ut_iso`: latest observation time in the tracklet.
- `duration_days`: time span of the tracklet in days.
- `obs`: number of observations in the tracklet.
- `inliers`: number of observations from the tracklet used in the final fit.
- `outliers`: number of rejected observations.
- `outlier_percent`: rejected observations as a percentage of `obs`.
- `mean_ra_residual_arcsec`: mean right ascension residual for inliers, in arcseconds.
- `std_ra_residual_arcsec`: standard deviation of right ascension residuals for inliers, in arcseconds.
- `std_normalized_ra_residual`: standard deviation of normalized right ascension residuals for inliers.
- `mean_dec_residual_arcsec`: mean declination residual for inliers, in arcseconds.
- `std_dec_residual_arcsec`: standard deviation of declination residuals for inliers, in arcseconds.
- `std_normalized_dec_residual`: standard deviation of normalized declination residuals for inliers.
- `normalized_residual_spread`: `sqrt((std_normalized_ra_residual^2 + std_normalized_dec_residual^2) / 2)`.
- `max_normalized_residual_std`: larger value of `std_normalized_ra_residual` and `std_normalized_dec_residual`.
- `weighted_contribution_percent`: percentage of the final weighted normal-matrix trace contributed by the tracklet.
- `geometric_contribution_percent`: percentage of the unweighted normal-matrix trace contributed by the tracklet.

Let `J` be the final flattened Jacobian. Let `W` be the diagonal matrix made from the final flattened weights. For one tracklet `k`, let `J_k` and `W_k` contain only the residual rows from that tracklet. DiffOrb computes:

```text
weighted_contribution_percent = 100 * trace(J_k.T @ W_k @ J_k) / trace(J.T @ W @ J)
```

`weighted_contribution_percent` shows how much the tracklet affects the fit after the current weights are applied. A high value means that the tracklet has a large weighted influence on the final solution. `geometric_contribution_percent` uses the same formula with all weights set to one, so it shows the tracklet contribution from geometry alone. Use `normalized_residual_spread` beside these contribution metrics to separate a high-contribution tracklet from one whose normalized residuals are unusually concentrated or broad.

## Verification

The output above came from the short `2025 BC10` differential-correction example. That run used a local `2025_BC10-online.psv` file saved from the online loader and a local `de441.bsp` kernel.

## Common Mistakes

- A station with few observations can give noisy standard deviations.
- A high geometric contribution is not automatically bad. It means the tracklet has a large effect on the final solution.

## Next Steps

- Return to [Choose And Override Observation Weights](choose-and-override-observation-weights.md) when residual scatter suggests a weight change.
- Return to [Configure Outlier Rejection For Orbit Determination](configure-outlier-rejection-for-orbit-determination.md) when specific `input_index` rows need review.
- Use the [OD API](../api/od.md) for details on `ODAnalysis` and result fields.
