# Inspect Differential Correction Results

This guide shows how to read the main fields of a `DCResult` after differential correction.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- Load observations from one local observation file before the solve.
- Run a `DCSolver` call as shown in [Run Differential Correction From An Initial Orbit](run-differential-correction-from-an-initial-orbit.md), or run `ODSolver` as shown in [Run Integrated Orbit Determination With ODSolver](run-integrated-orbit-determination-with-odsolver.md).
- Keep the returned `DCResult` object in memory. If you used `ODSolver`, use `result = od_result.dc_result`.

For what the fields mean, read [Differential Correction](../concepts/differential-correction.md).

## 1. Inspect least-squares diagnostics

`result.lsq_diagnostics` stores the final least-squares diagnostics. It includes convergence fields, covariance fields, and the flattened residual-system fields.

```python
diag = result.lsq_diagnostics

print("NORMALIZED_RESIDUAL_RMS", f"{result.normalized_residual_rms:.6f}")
print("CONVERGED", diag.converged)
print("TERMINATION", diag.termination_reason)
print("LSQ_ITERS", diag.lsq_iterations)
print("OUTLIER_ITERS", diag.outlier_iterations)
print("COV_VALID", bool(diag.cov_valid))
print("COV_RANK", int(diag.cov_rank))
print("COV_CONDITION", f"{float(diag.cov_condition):.3e}")
print("COV_PRIOR_SHAPE", tuple(diag.cov_mat_prior.shape))
print("FLAT_JAC_SHAPE", tuple(diag.flat_jacobian.shape))
print("FLAT_WEIGHTS_SHAPE", tuple(diag.flat_weights.shape))
```

Output for the short `2025 BC10` example:

```text title="Output"
NORMALIZED_RESIDUAL_RMS 0.426434
CONVERGED True
TERMINATION gradient_converged
LSQ_ITERS 4
OUTLIER_ITERS 1
COV_VALID True
COV_RANK 6
COV_CONDITION 7.092e+03
COV_PRIOR_SHAPE (6, 6)
FLAT_JAC_SHAPE (160, 6)
FLAT_WEIGHTS_SHAPE (160,)
```

The fields mean:

- `normalized_residual_rms`: RMS of normalized residuals. It is dimensionless.
- `converged`: whether the final least-squares solve met a convergence condition.
- `termination_reason`: why the final least-squares solve stopped.
- `lsq_iterations`: total number of inner least-squares iterations.
- `outlier_iterations`: number of outer outlier-rejection iterations.
- `cov_valid`: whether the covariance matrix passed the solver checks.
- `cov_rank`: numerical rank of the covariance matrix.
- `cov_condition`: condition number of the covariance matrix. A very large value means the covariance is poorly conditioned.
- `cov_mat_prior`: covariance matrix before applying the post-fit residual scale.
- `flat_jacobian`: final Jacobian matrix for the flattened residual vector.
- `flat_weights`: final weights for the flattened residual vector.

Possible `termination_reason` values are:

- `gradient_converged`: the scaled gradient was small enough.
- `step_converged`: the accepted parameter step was small enough.
- `max_iter_reached`: the solver reached the maximum accepted iteration count.
- `damping_failed`: no damped trial step could be accepted.

## 2. Inspect orbit and covariance

`result.estimate` is the final accepted estimate. `result.estimate.orbit` is its orbit state. In the default `DCSolver` path, it is a `State` object with `frame=BCRS`.

```python
orbit = result.estimate.orbit
unc = result.estimate.uncertainties

print("EPOCH_TDB_JD", f"{float(orbit.tdb.jd):.9f}")
print("FRAME", orbit.frame.name)
print("POS_AU", [round(float(x), 9) for x in orbit.pos.tolist()])
print("VEL_AU_PER_D", [round(float(x), 9) for x in orbit.vel.tolist()])
print("N_UNCERTAINTIES", len(unc))
print("COV_POST_SHAPE", tuple(result.estimate.cov_mat_post.shape))
print("MODEL_PARAM_NAMES", result.estimate.model_param_names)
print("MODEL_PARAMS", result.estimate.model_params.tolist())
```

```text title="Output"
EPOCH_TDB_JD 2460762.500000000
FRAME BCRS
POS_AU [-1.106644219, -0.13528989, -0.039702689]
VEL_AU_PER_D [0.014502814, -0.011577568, -0.00660712]
N_UNCERTAINTIES 6
COV_POST_SHAPE (6, 6)
MODEL_PARAM_NAMES []
MODEL_PARAMS []
```

The estimate fields mean:

- `orbit`: fitted orbit.
- `model_params`: fitted force-model parameters other than the six orbit parameters. It is empty for a Cartesian-only fit.
- `model_param_names`: names for `model_params`.
- `cov_mat_post`: posterior covariance matrix after applying the post-fit residual scale.
- `uncertainties`: square root of the diagonal of `cov_mat_post`.

The uncertainty vector has one entry for each estimated parameter. For a Cartesian-only fit, its length is `6`.

For a fit with fitted force-model parameters, read [Estimate Nongravitational Parameters In Differential Correction](estimate-nongravitational-parameters-in-differential-correction.md).

## 3. Inspect the MPC quality code

`result.quality_code` is the IAU MPC Uncertainty Parameter `U`. Its range is `0` to `9`. A smaller value means a better constrained orbit. A value of `9` means a poorly constrained orbit. See the MPC reference for the `U` value: [MPC U value](https://www.minorplanetcenter.net/iau/info/UValue.html).

```python
print("QUALITY_CODE", int(result.quality_code))
```

```text title="Output"
QUALITY_CODE 9
```

## 4. Inspect residual blocks

`DCResult` stores optical and radar results in separate blocks.

```python
print("OPTICAL", result.optical.n_inliers, result.optical.n_obs)
print("RADAR", result.radar.n_inliers, result.radar.n_obs)
print("OPTICAL_WEIGHTED_RMS", f"{result.optical.weighted_rms:.6f}")
print("OPTICAL_UNWEIGHTED_RMS_RAD", f"{result.optical.unweighted_rms:.6e}")
```

```text title="Output"
OPTICAL 80 80
RADAR 0 0
OPTICAL_WEIGHTED_RMS 0.000002
OPTICAL_UNWEIGHTED_RMS_RAD 2.235370e-06
```

Each residual block also stores one rejection metric per observation and the `inlier_masks` field for the final fit.

```python
print("OPTICAL_INLIERS", result.optical.n_inliers, result.optical.n_obs)
print("OPTICAL_OUTLIERS", result.optical.n_outliers)
print("OPTICAL_METRIC_HEAD", [round(float(x), 6) for x in result.optical.metrics[:5].tolist()])
print("OPTICAL_MASK_HEAD", result.optical.inlier_masks[:10].tolist())
```

```text title="Output"
OPTICAL_INLIERS 80 80
OPTICAL_OUTLIERS 0
OPTICAL_METRIC_HEAD [0.074983, 0.056159, 0.253836, 0.008104, 0.004224]
OPTICAL_MASK_HEAD [True, True, True, True, True, True, True, True, True, True]
```

A `True` value in `inlier_masks` means the observation was used in the final fit. `OPTICAL_INLIERS 80 80` and
`OPTICAL_OUTLIERS 0` mean all 80 optical observations were used.

The optical result block, `optical`, has these fields:

- `residuals`: optical residuals in radians. It has two columns: right ascension and declination.
- `normalized_residuals`: dimensionless optical residuals made from the residuals and the adopted weights.
- `weighted_rms`: weighted RMS of the optical residuals in the block.
- `unweighted_rms`: unweighted RMS of the optical residuals in radians.
- `inlier_masks`: boolean field. `True` means the observation was used by the final fit.
- `metrics`: rejection metric for each observation.
- `n_obs`: number of observations in the block.
- `n_inliers`: number of observations used in the final fit.
- `n_outliers`: number of observations rejected from the final fit.

The radar result block has these fields:

- `residuals`: radar residuals. Delay and Doppler rows use their own units.
- `normalized_residuals`: dimensionless radar residuals made from the residuals and the adopted weights.
- `inlier_masks`: boolean field. `True` means the observation was used by the final fit.
- `metrics`: rejection metric for each radar observation.
- `delay_weighted_rms`: weighted RMS for delay observations.
- `delay_unweighted_rms`: unweighted RMS for delay observations.
- `doppler_weighted_rms`: weighted RMS for Doppler observations.
- `doppler_unweighted_rms`: unweighted RMS for Doppler observations.
- `n_obs`: number of radar observations.
- `n_inliers`: number of radar observations used in the final fit.
- `n_outliers`: number of radar observations rejected from the final fit.

## 5. Convert orbit representation

Use `transform(...)` to move the result to another frame or to Keplerian elements. The covariance matrix is propagated with the Jacobian of the same transform. DiffOrb computes this Jacobian with JAX automatic differentiation, as it does for differential-correction Jacobians.

`DCResult.transform(...)` accepts one argument:

- `target`: a `Frame` object or the `KepElement` class. Use `KepElement` when you want Keplerian elements. Use a frame object when you want a `State` object in another frame.

```python
from difforb.core import KepElement

kep_result = result.transform(KepElement)
kep = kep_result.estimate.orbit

print("A_AU", f"{float(kep.a):.9f}")
print("E", f"{float(kep.e):.9f}")
print("INC_DEG", f"{float(kep.inc * 180.0 / 3.141592653589793):.9f}")
```

```text title="Output"
A_AU 2.033934810
E 0.736058905
INC_DEG 4.904109856
```

`transform(...)` returns a new `DCResult`. It does not change `result` in place.

In the returned object, only these values differ from the original result:

- `estimate.orbit`
- `estimate.cov_mat_post`

These values do not change:

- `estimate.model_params`
- `estimate.model_param_names`
- `optical`
- `radar`
- `lsq_diagnostics`
- `normalized_residual_rms`

## Verification

The snippets above were checked with the `DCResult` from [Run Differential Correction From An Initial Orbit](run-differential-correction-from-an-initial-orbit.md). That run used a local `2025_BC10-online.psv` file saved from the online loader and a local `de441.bsp` kernel.

## Common Mistakes

- Do not use `normalized_residual_rms` alone as a convergence flag.
- If `cov_valid` is false, do not report values made from uncertainties as physical results.
- `lsq_iterations` counts inner least-squares iterations. It does not count rejected damping trials.
- `outlier_iterations` counts outer outlier-rejection iterations. It does not count least-squares steps.

## Next Steps

- Return to [Configure Outlier Rejection For Orbit Determination](configure-outlier-rejection-for-orbit-determination.md) when you need to change the policy.
- Continue to [Analyze Residuals By Station And Tracklet](analyze-residuals-by-station-and-tracklet.md) when you need row-level checks.
- Use the [OD API](../api/od.md) for details on `DCResult`, estimates, diagnostics, and
  analyzer objects.
