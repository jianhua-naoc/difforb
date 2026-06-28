# Estimate Nongravitational Parameters In Differential Correction

This guide shows how to add non-gravitational parameters to a `DynamicSystem`, choose which parameters are fitted, set their initial values, and read the fitted values from `DCResult`.

For comet optical photocenter offsets, use [Estimate A Comet Photocenter Offset In Differential Correction](estimate-comet-photocenter-offset.md). `S0` belongs to the observation model, not the force model.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- Load observations from one local observation file.
- Run a differential-correction setup like [Run Differential Correction From An Initial Orbit](run-differential-correction-from-an-initial-orbit.md).
- Use an observation arc long enough to solve for the parameter. The short example here only shows the API path.
- Configure a planetary SPK kernel that covers the observation arc.

## Supported built-in models

DiffOrb currently supports these built-in non-gravitational models:

| Model | Fitted parameters | Main use |
| --- | --- | --- |
| `CometOutgassingEffect` | Any subset of `A1`, `A2`, and `A3` | Symmetric comet outgassing |
| `EmpiricalYarkovskyEffect` | `A2` | Empirical transverse drift |
| `EmpiricalRadiationPressure` | `A1` | Empirical radial radiation pressure |

For the meaning of these non-gravitational models, read [Dynamical Models](../concepts/dynamical-models.md#non-gravitational-terms).

## 1. Add a parametrized effect

The example below estimates one transverse `A2` term with `EmpiricalYarkovskyEffect`.

`EmpiricalYarkovskyEffect(...)` accepts these common arguments:

- `sun`: an `EphemerisBody` object for the Sun.
- `A2`: transverse acceleration parameter, in `au / day^2`. If `A2` is estimated, this is the initial value. If it is not estimated, this value is fixed.
- `estimated_params`: names to estimate. For this term, use `("A2",)`.
- `param_prefix`: prefix used in the exposed parameter name. The default is `"Yarkovsky"`.

`estimated_params` is the boundary between fitted and fixed force parameters: the solver changes listed parameters and keeps the others at the constructor values. See [Fit only selected model parameters](#3-fit-only-selected-model-parameters) for an example with several parameters.

```python
from difforb.dynamics import DynamicSystem
from difforb.dynamics import EmpiricalYarkovskyEffect

system = DynamicSystem()
system.add_body(sun)
system.add_non_grav_force(
    EmpiricalYarkovskyEffect(
        sun,
        A2=0.0,
        estimated_params=("A2",),
    )
)

force_model = system.build_force_model()

print("PARAM_NAMES", force_model.get_all_estimated_param_names())
print("PARAM_INIT", force_model.get_all_estimated_params().tolist())
```

```text title="Output"
PARAM_NAMES ['Yarkovsky_A2']
PARAM_INIT [0.0]
```

## 2. Run DC with the effect

Pass the `force_model` with non-gravitational parameters to `DCSolver.solve(...)`. The solver estimates the force-model parameters together with the six Cartesian state parameters.

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

print("NORMALIZED_RESIDUAL_RMS", f"{result.normalized_residual_rms:.6f}")
print("PARAM_RESULT", [float(x) for x in result.estimate.model_params.tolist()])
print("PARAM_RESULT_NAMES", result.estimate.model_param_names)
print("N_UNCERTAINTIES", len(result.estimate.uncertainties))
print("COV_RANK", int(result.lsq_diagnostics.cov_rank))
print("COV_VALID", bool(result.lsq_diagnostics.cov_valid))
```

```text title="Output"
NORMALIZED_RESIDUAL_RMS 0.284288
PARAM_RESULT [1.0999534171435365e-06]
PARAM_RESULT_NAMES ['Yarkovsky_A2']
N_UNCERTAINTIES 7
COV_RANK 7
COV_VALID True
```

The uncertainty vector has length `7` because the solve estimated six Cartesian state values plus one model parameter.

## 3. Fit only selected model parameters

You can give all acceleration parameters initial values and fit only a subset of them. Parameters not listed in `estimated_params` remain fixed during the differential-correction solve.

```python
from difforb.dynamics import DynamicSystem
from difforb.dynamics import CometOutgassingEffect

system = DynamicSystem()
system.add_body(sun)
system.add_non_grav_force(
    CometOutgassingEffect(
        sun,
        A1=1.0e-8,
        A2=2.0e-8,
        A3=0.0,
        estimated_params=("A1", "A2"),
    )
)

force_model = system.build_force_model()

print("PARAM_NAMES", force_model.get_all_estimated_param_names())
print("PARAM_INIT", force_model.get_all_estimated_params().tolist())
```

```text title="Output"
PARAM_NAMES ['Outgassing_A1', 'Outgassing_A2']
PARAM_INIT [1e-08, 2e-08]
```

This model uses `A1=1.0e-8` and `A2=2.0e-8` as initial fitted values. It keeps `A3=0.0` fixed because `A3` is not listed in `estimated_params`.

## 4. Set outgassing law parameters

`CometOutgassingEffect` also lets you set distance-law parameters.

```python
outgassing = CometOutgassingEffect(
    sun,
    A1=1.0e-8,
    A2=2.0e-8,
    A3=0.0,
    estimated_params=("A1", "A2"),
    r0=2.8,
    alpha=0.1112620426,
    m=2.15,
    n=5.093,
    k=4.6142,
)
```

The `CometOutgassingEffect(...)` arguments are:

- `sun`: an `EphemerisBody` object for the Sun.
- `estimated_params`: names of the `A` parameters to fit. Use any subset of `("A1", "A2", "A3")`. Use `()` when all `A` parameters should stay fixed.
- `A1`, `A2`, `A3`: radial, transverse, and normal acceleration parameters in `au / day^2`. Listed parameters are initial fitted values. Unlisted parameters are fixed values.
- `r0`: distance scale in `au`.
- `alpha`, `m`, `n`, `k`: fixed shape parameters for the radial distance law `g(r) = alpha * (r / r0)^(-m) * (1 + (r / r0)^n)^(-k)`.
- `param_prefix`: prefix used in exposed parameter names.

Choose the term that matches the effect you want to test. Do not treat a parameter as physical unless the observation arc, force model, and covariance checks support it.

## Verification

The force-model snippets above were checked with a local `de441.bsp` kernel. The differential-correction output in section 2 was produced with the full setup from [Run Differential Correction From An Initial Orbit](run-differential-correction-from-an-initial-orbit.md), but with the force model from section 1. That setup provides the local `2025_BC10-online.psv` observations, initial orbit, integrator, weight policy, debias policy, and outlier policy. The fitted `A2` value is not a physical estimate for `2025 BC10`. It only checks the API path and result fields.

## Common Mistakes

- Do not estimate non-gravitational parameters on arcs too short to solve for them.
- Check `cov_valid`, `cov_rank`, and the parameter uncertainty before you use a fitted value.
- Use `get_all_estimated_param_names()` to make parameter order clear.

## Next Steps

- Continue to [Configure Force Models And Dynamic Systems](configure-force-models-and-dynamic-systems.md) for more force-model setup.
- Continue to [Inspect Differential Correction Results](inspect-differential-correction-results.md) to check covariance and transformed elements.
- Use the [Dynamics API](../api/dynamics.md) and [OD API](../api/od.md) for details on estimated
  force parameters and result fields.
