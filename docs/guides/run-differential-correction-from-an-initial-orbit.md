# Run Differential Correction From An Initial Orbit

This guide shows how to run one `DCSolver.solve(...)` call from an initial guess. The result is a fitted orbit with residuals, inlier counts, and covariance checks.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- Prepare one local observation file. See [Load Online Observations From MPC And JPL](load-online-observations-from-mpc-and-jpl.md) and [Load Local ADES Observations](load-local-ades-observations.md).
- Configure a planetary SPK kernel that covers the observation arc.
- Start from an initial guess, usually from [Solve Initial Orbit From Optical Observations](solve-initial-orbit-from-optical-observations.md).
- Install the local data files required by the selected weight and debias policies.
- Choose a force model, integrator, weight policy, debias policy, and outlier policy before calling the solver.

For the role of differential correction in orbit determination, read [Differential Correction](../concepts/differential-correction.md).

## 1. Prepare the example inputs

The code below uses a short optical slice so it runs quickly. For a real fit, choose the observations and arc for your object.

The snippet also makes simple policy choices so the `DCSolver.solve(...)` call is complete:

- `VFCC17WeightPolicy()` assigns the default statistical observation weights. See
  [Choose And Override Observation Weights](choose-and-override-observation-weights.md) when you need to compare
  weight sources or add row-level overrides.
- `EgglDebiasPolicy()` applies the local optical catalog debias model. See
  [Inspect Optical Debias Corrections](inspect-optical-debias-corrections.md) when you need to inspect the corrections.
- `InteractiveOutlierPolicy(Chi2OutlierRejecter(), ...)` enables automatic chi-square rejection. See
  [Configure Outlier Rejection For Orbit Determination](configure-outlier-rejection-for-orbit-determination.md) when
  you need manual inlier/outlier settings or a different rejection rule.

```python
from difforb.od import DCSolver, IODSolver
from difforb.dynamics import DynamicSystem
from difforb.astrometry import (
    EgglDebiasPolicy,
    ObservationData,
    VFCC17WeightPolicy,
    load_local_observations,
)
from difforb.spk import set_default_ephemeris
from difforb.body import EphemerisBody
from difforb.integrator import NumericalIntegrator
from difforb.od import Chi2OutlierRejecter, InteractiveOutlierPolicy

observation_file = "/path/to/2025_BC10-online.psv"
planetary_kernel = "/path/to/de441.bsp"

set_default_ephemeris(planetary_kernel)
obs_all = load_local_observations(observation_file)

obs = ObservationData(
    name=obs_all.name,
    optical=obs_all.optical[350:430],
    radar=obs_all.radar[:0],
)

sun = EphemerisBody("sun")
earth = EphemerisBody("earth")

initial_orbit = IODSolver(max_iter=20, tol=1e-8).solve(
    obs,
    max_arc_days=3.0,
    candidates_num=5,
).initial_orbit

system = DynamicSystem()
system.add_body(sun)
force_model = system.build_force_model()
integrator = NumericalIntegrator(method="DOPRI8", tol=1e-8, max_steps=512)
weight_policy = VFCC17WeightPolicy()
debias_policy = EgglDebiasPolicy()
outlier_policy = InteractiveOutlierPolicy(
    Chi2OutlierRejecter(),
    enable_auto_rejecter=True,
    max_iters=3,
)
```

This example uses a `DynamicSystem` with only Sun Newtonian gravity, so the code stays focused on the `DCSolver` call.

## 2. Create `DCSolver`

`DCSolver` runs differential correction with Levenberg-Marquardt least squares. It starts from `initial_orbit` and fits the six `BCRS` Cartesian state components. If the force model has estimated parameters, it fits them with the state.

The `DCSolver(...)` constructor accepts these arguments:

- `lsq_tol`: convergence threshold for the least-squares solve. A smaller value is stricter.
- `lsq_max_iters`: maximum number of least-squares iterations.
- `sun`: an `EphemerisBody` object for the Sun.
- `earth`: an `EphemerisBody` object for the Earth.
- `bucket_policy`: optional `DCBucketPolicy` that controls observation-count buckets. Simple calls leave it unset.

If `sun` or `earth` is omitted, `DCSolver` creates `EphemerisBody("sun")` or `EphemerisBody("earth")` during construction.

```python
dc = DCSolver(lsq_tol=1e-5, lsq_max_iters=8, sun=sun, earth=earth)
```

## 3. Run `DCSolver.solve`

`DCSolver.solve(...)` runs one differential-correction solve. Its main arguments are:

- `data`: the `ObservationData` object used in the fit.
- `initial_orbit`: the starting orbit. It can be a `KepElement` or `State` object. The solver converts it to a `State` object with `frame=BCRS` before the fit.
- `force_model`: the dynamical model used during propagation.
- `integrator`: the numerical integrator used with the force model.
- `weight_policy`: the rule that assigns observation weights.
- `debias_policy`: the rule that applies astrometric debias corrections.
- `outlier_policy`: the rule that controls automatic rejection and manual inlier/outlier settings.
- `photocenter_correction`: optional `PhotocenterCorrection` object for comet optical photocenter correction.
- `event_handler`: optional callback that receives solver events.
- `log_detail`: minimum event detail passed to `event_handler`. The choices are `"quiet"`, `"summary"`, `"iter"`, and `"trial"`. Use `"quiet"` when you only need the final result.
- `event_logger`: optional structured event logger. Simple calls leave it unset.

It returns a `DCResult`. The result stores the fitted orbit, residual blocks, outlier counts, and least-squares
diagnostics.

```python
result = dc.solve(
    obs,
    initial_orbit,
    force_model,
    integrator,
    weight_policy,
    debias_policy,
    outlier_policy,
    log_detail="quiet",
)

orbit = result.estimate.orbit

print("N_OBS", len(obs))
print("NORMALIZED_RESIDUAL_RMS", f"{result.normalized_residual_rms:.6f}")
print("CONVERGED", result.lsq_diagnostics.converged)
print("REASON", result.lsq_diagnostics.termination_reason)
print("ITERS", result.lsq_diagnostics.lsq_iterations, result.lsq_diagnostics.outlier_iterations)
print("OPTICAL_INLIERS", result.optical.n_inliers, result.optical.n_obs)
print("OPTICAL_OUTLIERS", result.optical.n_outliers)
print("COV_VALID", bool(result.lsq_diagnostics.cov_valid))
print("COV_RANK", int(result.lsq_diagnostics.cov_rank))
print("EPOCH_TDB_JD", f"{float(orbit.tdb.jd):.9f}")
print("FRAME", orbit.frame.name)
print("POS_AU", [round(float(x), 9) for x in orbit.pos.tolist()])
print("VEL_AU_PER_D", [round(float(x), 9) for x in orbit.vel.tolist()])
```

```text title="Output"
N_OBS 80
NORMALIZED_RESIDUAL_RMS 0.426434
CONVERGED True
REASON gradient_converged
ITERS 4 1
OPTICAL_INLIERS 80 80
OPTICAL_OUTLIERS 0
COV_VALID True
COV_RANK 6
EPOCH_TDB_JD 2460762.500000000
FRAME BCRS
POS_AU [-1.106644219, -0.13528989, -0.039702689]
VEL_AU_PER_D [0.014502814, -0.011577568, -0.00660712]
```

For uncertainty fields and orbit conversion, see [Inspect Differential Correction Results](inspect-differential-correction-results.md).

## Verification

The output above used a local `2025_BC10-online.psv` file saved from the online loader and a local `de441.bsp` kernel. The example uses only 80 optical observations and a simple force model. Treat the numbers as reference output for the API path, not as a final orbit for `2025 BC10`.

## Common Mistakes

- Do not pass observations outside the SPK time range.

## Next Steps

- Continue to [Run Integrated Orbit Determination With ODSolver](run-integrated-orbit-determination-with-odsolver.md) when you want one call that runs IOD and DC together.
- Continue to [Inspect Differential Correction Results](inspect-differential-correction-results.md).
- Return to [Configure Outlier Rejection For Orbit Determination](configure-outlier-rejection-for-orbit-determination.md) when you are ready to change the outlier settings.
- Read [Weighting And Debiasing Models](../concepts/weighting-and-debiasing-models.md) before changing weight or debias policies.
- Read [Dynamical Models](../concepts/dynamical-models.md) before changing the force model.
- Use the [OD API](../api/od.md) for details on `DCSolver`, `DCResult`, and diagnostics.
