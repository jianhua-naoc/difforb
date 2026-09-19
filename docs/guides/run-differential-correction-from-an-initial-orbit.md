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

- `lsq_max_iters`: maximum number of least-squares iterations.
- `solver_jit`: whether to compile the complete fitting and rejection loops; default `False`. The default favors lower first-call compilation cost, while residual, Jacobian, and propagation calculations still use JAX compilation.
- `sun`: an `EphemerisBody` object for the Sun.
- `earth`: an `EphemerisBody` object for the Earth.

If `sun` or `earth` is omitted, `DCSolver` creates `EphemerisBody("sun")` or `EphemerisBody("earth")` during construction.

```python
dc = DCSolver(lsq_max_iters=8, sun=sun, earth=earth)
```

For repeated fits, enable whole-solver compilation with `DCSolver(solver_jit=True, sun=sun, earth=earth)`. The default avoids this large compilation step, but individual numerical steps and model kernels still compile. Compare first-call and repeated-call times for your workload; neither mode is always faster.

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
- `verbose`: set `True` to print solver progress, leave `False` for no progress output, or pass a callback that accepts an event name and keyword data.
- `device`: optional JAX device that receives the numerical arrays.
- `grid`: whether sequence-valued strategy arguments form a Cartesian product instead of using point-wise broadcasting.
- `batch_size`: optional maximum number of compatible strategies in one mapped solve.

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
    verbose=False,
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

```text title="Output excerpt"
N_OBS 80
NORMALIZED_RESIDUAL_RMS 0.426434
OPTICAL_INLIERS 80 80
OPTICAL_OUTLIERS 0
COV_VALID True
COV_RANK 6
EPOCH_TDB_JD 2460762.500000000
FRAME BCRS
POS_AU [-1.106644219, -0.13528989, -0.039702689]
VEL_AU_PER_D [0.014502814, -0.011577568, -0.00660712]
```

Convergence settings are fixed: `delnor < 1e-3` or ten consecutive accepted steps with less than 0.01 percent RMS decrease. The same settings apply before and after outlier rejection.

A converged solve reports `correction_converged` when the correction norm met its threshold, or `rms_stagnated` when RMS stopped improving under the fixed stopping policy. Iteration counts can differ with solver versions and numerical precision. For uncertainty fields and orbit conversion, see [Inspect Differential Correction Results](inspect-differential-correction-results.md).

## 4. Run a strategy batch on a selected device

Sequence-valued force models, weight policies, or outlier policies request a strategy batch. Scalar arguments are
broadcast across the batch. With the default `grid=False`, sequence arguments are paired point by point and must have
length one or a common batch length.

The following call compares two weight policies on one orbit-estimation problem and places the numerical work on the
first JAX GPU:

```python
import jax

from difforb.astrometry import UnitWeightPolicy

gpu = jax.devices("gpu")[0]
batch_dc = DCSolver(
    lsq_max_iters=8,
    solver_jit=True,
    sun=sun,
    earth=earth,
)

batch_results = batch_dc.solve(
    obs,
    initial_orbit,
    force_model,
    integrator,
    [VFCC17WeightPolicy(), UnitWeightPolicy()],
    debias_policy,
    outlier_policy,
    verbose=False,
    device=gpu,
    batch_size=2,
)

print(batch_results.shape)
print([float(item.normalized_residual_rms) for item in batch_results])
```

The result is a NumPy object array with shape `(2,)`, and each element is a `DCResult`. With `solver_jit=True`,
compatible strategies are mapped together. `batch_size` bounds the number of strategies in each mapped solve and can
reduce peak device memory. Keep `verbose=False` when mapped execution is required, because live progress callbacks use
the host-driven path.

Set `grid=True` to evaluate the Cartesian product of multiple force-model, weight-policy, or outlier-policy sequences.
The returned object array preserves that strategy-grid shape. Passing `device=jax.devices("cpu")[0]` uses the same API
for an explicitly selected CPU device.

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
