# Run Integrated Orbit Determination With ODSolver

This guide shows how to run `ODSolver` on one observation file. `ODSolver` runs IOD first, then runs differential correction on one or more staged arcs.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- Prepare one local observation file. See [Load Online Observations From MPC And JPL](load-online-observations-from-mpc-and-jpl.md) and [Load Local ADES Observations](load-local-ades-observations.md).
- Configure a planetary SPK kernel that covers the observation arc.
- Install the local data files required by the selected weight and debias policies.
- Choose a force model, integrator, weight policy, debias policy, and outlier policy.

For the solver roles, read [Orbit Determination Overview](../concepts/orbit-determination-overview.md).

## 1. Prepare the example inputs

The code below uses a short optical slice so it runs quickly. For a real fit, choose the observations and arc for your object.

The snippet also prepares the force model, integrator, and policies used by `ODSolver.solve(...)`. This guide uses
these simple policy defaults:

- `VFCC17WeightPolicy()` assigns the default statistical observation weights. See
  [Choose And Override Observation Weights](choose-and-override-observation-weights.md) when you need to compare
  weight sources or add row-level overrides.
- `EgglDebiasPolicy()` applies the local optical catalog debias model. See
  [Inspect Optical Debias Corrections](inspect-optical-debias-corrections.md) when you need to inspect the corrections.
- `InteractiveOutlierPolicy(Chi2OutlierRejecter(), ...)` enables automatic chi-square rejection. See
  [Configure Outlier Rejection For Orbit Determination](configure-outlier-rejection-for-orbit-determination.md) when
  you need manual inlier/outlier settings or a different rejection rule.

```python
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

This example uses a `DynamicSystem` with only Sun Newtonian gravity, so the code stays focused on the `ODSolver` call.

## 2. Build the integrated solver

`ODSolver` combines one `IODSolver` and one `DCSolver`. It runs `IODSolver` first, then passes the initial orbit to
`DCSolver`.

The `ODSolver(...)` constructor accepts these arguments:

- `iod_solver`: an `IODSolver` object.
- `dc_solver`: a `DCSolver` object.

```python
from difforb.od import DCSolver, IODSolver, ODSolver

solver = ODSolver(
    IODSolver(max_iter=20, tol=1e-8),
    DCSolver(lsq_tol=1e-5, lsq_max_iters=8, sun=sun, earth=earth),
)
```

## 3. Configure the IOD and DC strategies

`ODSolver` is useful because it can run IOD on a chosen short arc, then run DC on one or more selected arc widths.
`IODStrategy` controls the IOD arc, candidate count, iteration limit, and range guesses. `DCStrategy` controls the DC
arc list, minimum observation count, and solve epoch for each stage.

`IODStrategy(...)` accepts these arguments:

- `arc_days`: width of the optical arc used by `IODSolver`, in days.
- `max_candidates`: number of candidate triplets to try.
- `max_iterations`: maximum number of Double-r iterations for each candidate triplet.
- `init_rho`: initial line-of-sight range guesses for Double-r, in `au`.

`DCStrategy(...)` accepts these arguments:

- `incremental_arc_days`: list of DC arc widths, in days.
- `min_observations`: minimum number of observations required to run one DC stage.
- `epoch_strategy`: how each DC stage chooses the solve epoch. Use `"keep_initial"`, `"arc_midpoint"`, or
  `"weighted_mean"`.

`incremental_arc_days` is used when a direct full-arc fit may be hard to converge. This often happens for objects with
a long observation arc, long orbital period, or a weak initial orbit. Instead of using all observations at once,
`ODSolver` can run several DC stages. Each stage uses a wider arc than the previous stage. The result from one stage
becomes the initial orbit for the next stage. For each arc width, `ODSolver` selects observations around the current
orbit epoch. If the selected arc has fewer than `min_observations`, the stage is skipped. If a wider arc selects the
same observation rows as the previous stage, it is also skipped because it would repeat the same solve.

```python
from difforb.od import DCStrategy, IODStrategy

iod_strategy = IODStrategy(
    arc_days=3.0,
    max_candidates=5,
    max_iterations=20,
    init_rho=(1.0, 1.0),
)

dc_strategy = DCStrategy(
    incremental_arc_days=[3.0],
    min_observations=3,
    epoch_strategy="keep_initial",
)

print("IOD_ARC_DAYS", iod_strategy.arc_days)
print("IOD_MAX_CANDIDATES", iod_strategy.max_candidates)
print("IOD_MAX_ITERATIONS", iod_strategy.max_iterations)
print("DC_ARCS", dc_strategy.incremental_arc_days)
print("DC_MIN_OBSERVATIONS", dc_strategy.min_observations)
print("DC_EPOCH_STRATEGY", dc_strategy.epoch_strategy)
```

```text title="Output"
IOD_ARC_DAYS 3.0
IOD_MAX_CANDIDATES 5
IOD_MAX_ITERATIONS 20
DC_ARCS [3.0]
DC_MIN_OBSERVATIONS 3
DC_EPOCH_STRATEGY keep_initial
```

## 4. Run `ODSolver`

This example uses one short DC stage. For a real fit, set `incremental_arc_days` to the staged arc widths you want to try.

`ODSolver.solve(...)` runs one integrated orbit-determination solve. Its main arguments are:

- `obs`: the `ObservationData` object used in the fit.
- `force_model`: the dynamical model used during propagation.
- `integrator`: the numerical integrator used with the force model.
- `weight_policy`: the rule that assigns observation weights.
- `debias_policy`: the rule that applies astrometric debias corrections.
- `outlier_policy`: the rule that controls automatic rejection and manual inlier/outlier settings.
- `iod_strategy`: the settings for the initial orbit stage.
- `dc_strategy`: the staged differential-correction arc settings.
- `photocenter_correction`: optional `PhotocenterCorrection` object for comet optical photocenter correction.
- `event_handler`: optional callback that receives solver events.
- `log_detail`: minimum event detail passed to `event_handler`. The choices are `"quiet"`, `"summary"`, `"iter"`, and `"trial"`.
- `event_logger`: optional structured event logger. Simple calls leave it unset.

```python
od_result = solver.solve(
    obs,
    force_model=force_model,
    integrator=integrator,
    weight_policy=weight_policy,
    debias_policy=debias_policy,
    outlier_policy=outlier_policy,
    iod_strategy=iod_strategy,
    dc_strategy=dc_strategy,
    log_detail="quiet",
)

dc_result = od_result.dc_result
orbit = dc_result.estimate.orbit

print("HAS_IOD", od_result.iod_result is not None)
print("HAS_DC", dc_result is not None)
print("IOD_ERR_RAD", f"{od_result.iod_result.err:.6e}")
print("FINAL_NORMALIZED_RESIDUAL_RMS", f"{dc_result.normalized_residual_rms:.6f}")
print("FINAL_OPTICAL_INLIERS", dc_result.optical.n_inliers, dc_result.optical.n_obs)
print("FINAL_OPTICAL_OUTLIERS", dc_result.optical.n_outliers)
print("EPOCH_TDB_JD", f"{float(orbit.tdb.jd):.9f}")
print("FRAME", orbit.frame.name)
print("POS_AU", [round(float(x), 9) for x in orbit.pos.tolist()])
print("VEL_AU_PER_D", [round(float(x), 9) for x in orbit.vel.tolist()])
```

```text title="Output"
HAS_IOD True
HAS_DC True
IOD_ERR_RAD 3.267089e-06
FINAL_NORMALIZED_RESIDUAL_RMS 0.465159
FINAL_OPTICAL_INLIERS 70 70
FINAL_OPTICAL_OUTLIERS 0
EPOCH_TDB_JD 2460762.500000000
FRAME BCRS
POS_AU [-1.1059695, -0.135195846, -0.039775405]
VEL_AU_PER_D [0.014412874, -0.011604554, -0.006607118]
```

`od_result.iod_result` stores the initial guess result. `od_result.dc_result` stores the final differential-correction result.

For uncertainty fields and orbit conversion, see [Inspect Differential Correction Results](inspect-differential-correction-results.md).

## Verification

The output above used a local `2025_BC10-online.psv` file saved from the online loader and a local `de441.bsp` kernel. The example uses a short optical slice and a simple dynamic system. Treat the numbers as reference output for the API path, not as a final orbit for `2025 BC10`.

## Common Mistakes

- Do not use `ODSolver` without choosing the force model, integrator, weight policy, debias policy, and outlier policy.
- If all DC stages are skipped, `od_result.dc_result` is `None`. Check it before reading `od_result.dc_result.estimate`.
- After `ODSolver` returns a fit, still inspect residuals, rejected observations, covariance, and station or tracklet summaries before using the orbit.

## Next Steps

- Continue to [Inspect Differential Correction Results](inspect-differential-correction-results.md) to read `od_result.dc_result`.
- Return to [Configure Outlier Rejection For Orbit Determination](configure-outlier-rejection-for-orbit-determination.md) when you are ready to change the outlier settings.
- Read [Weighting And Debiasing Models](../concepts/weighting-and-debiasing-models.md) before changing weight or debias policies.
- Use the [OD API](../api/od.md) for details on `ODSolver`, strategies, and result objects.
