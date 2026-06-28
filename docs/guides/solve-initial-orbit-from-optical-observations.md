# Solve Initial Orbit From Optical Observations

This guide shows how to run `IODSolver` on one observation file. The result is an initial guess for differential correction.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- Prepare one local observation file. You can create it with [Load Online Observations From MPC And JPL](load-online-observations-from-mpc-and-jpl.md). Then reopen it with [Load Local ADES Observations](load-local-ades-observations.md).
- Configure a planetary SPK kernel that covers the observation arc.
- The file must contain at least three optical observations. Radar observations can be present. `IODSolver` does not use them.

For how IOD fits into orbit determination, read [Initial Orbit Determination](../concepts/initial-orbit-determination.md).

## 1. Load observations and set the ephemeris

Use one local observation file and one planetary SPK kernel. Replace both paths with files that exist on your machine.

```python
from difforb.astrometry import load_local_observations
from difforb.spk import set_default_ephemeris

observation_file = "/path/to/2025_BC10-online.psv"
planetary_kernel = "/path/to/de441.bsp"

set_default_ephemeris(planetary_kernel)
obs = load_local_observations(observation_file)

print("NAME", obs.name)
print("N_OBS", len(obs))
print("N_OPTICAL", obs.num_optical)
print("N_RADAR", obs.num_radar)
```

## 2. Run `IODSolver`

`IODSolver` uses the Double-r method for each candidate observation triplet.

The `IODSolver(...)` constructor controls the Double-r iteration:

- `max_iter`: the maximum number of Newton updates for each candidate triplet.
- `tol`: the convergence threshold for the final angular residual norm, in radians.

`IODSolver.solve(...)` then samples candidate observation triplets inside a time window. Its main arguments are:

- `data`: the `ObservationData` object. `IODSolver` uses optical observations from it. Radar observations are ignored.
- `max_arc_days`: the width of the optical sampling window in days. The window is centered on the middle sorted optical observation. If the window has fewer than three observations, the solver uses the full filtered optical arc.
- `candidates_num`: the number of candidate triplets to sample and solve.
- `init_rho`: the initial line-of-sight range guesses for the Double-r iteration. The two values are for the first and third observations in each triplet, in `au`. The default is `(1.0, 1.0)`.

`IOD` uses a two-body model, so `max_arc_days` should stay short. The default is `1.0` day. Increase it only when the short window does not provide enough useful triplets.

Most runs only need to set `data`, `max_arc_days`, and `candidates_num`.

```python
from difforb.od import IODSolver

iod = IODSolver(max_iter=20, tol=1e-8)
iod_result = iod.solve(
    obs,
    max_arc_days=1.0,
    candidates_num=3,
)

initial_orbit = iod_result.initial_orbit

print("IOD_ERR_RAD", f"{iod_result.err:.6e}")
print("IOD_ITER", int(iod_result.iter_num))
print("USED_INDICES", iod_result.used_indices.tolist())
print("EPOCH_TDB_JD", f"{float(initial_orbit.tdb.jd):.9f}")
print("FRAME", initial_orbit.frame.name)
print("POS_AU", [round(float(x), 9) for x in initial_orbit.pos.tolist()])
print("VEL_AU_PER_D", [round(float(x), 9) for x in initial_orbit.vel.tolist()])
```

```text title="Output"
IOD_ERR_RAD 3.302645e-06
IOD_ITER 20
USED_INDICES [380, 385, 390]
EPOCH_TDB_JD 2460762.500000000
FRAME BCRS
POS_AU [-1.003974355, -0.121002875, -0.05077215]
VEL_AU_PER_D [0.002948315, -0.015542008, -0.006814519]
```

`used_indices` contains `input_index` values from the original observation file. Keep these values if you need to inspect the rows used for the initial guess.

## Verification

The output above used a local `2025_BC10-online.psv` file saved from the online loader and a local `de441.bsp` kernel. Small numerical differences can appear with different JAX, XLA, or dependency versions.

## Common Mistakes

- Do not use the IOD result as the final fitted orbit. It is only an initial guess.
- `IODSolver` uses only optical observations. Radar rows are used later in differential correction.
- A shorter `max_arc_days` can fail if the time window has fewer than three optical observations.

## Next Steps

- Continue to [Run Differential Correction From An Initial Orbit](run-differential-correction-from-an-initial-orbit.md).
- Read [Light-Time Model](../concepts/light-time-model.md) for the one-way optical light-time model used by optical
  predictions.
- Read the [OD API](../api/od.md) for parameter-level details.
