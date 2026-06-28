# Estimate A Comet Photocenter Offset In Differential Correction

This guide shows how to estimate a single comet photocenter offset, `S0`, during differential correction. The fitted orbit is still the comet center-of-mass orbit. `S0` only changes the optical measurement model for right ascension and declination.

For the model behind `S0`, read [Photocenter Correction](../concepts/photocenter-correction.md).

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- Prepare a differential-correction setup like [Run Differential Correction From An Initial Orbit](run-differential-correction-from-an-initial-orbit.md).
- Use comet optical observations that may be biased by coma or tail structure.
- Configure a planetary SPK kernel that covers the observation arc.

For non-gravitational comet accelerations such as `A1`, `A2`, and `A3`, use [Estimate Nongravitational Parameters In Differential Correction](estimate-nongravitational-parameters-in-differential-correction.md). `S0` belongs to the optical observation model, not the force model.

## 1. Create the correction

`PhotocenterCorrection(...)` uses this scalar `S0` center-of-light model:[^yeomans-1994][^farnocchia-2021]

```text
offset_distance = S0 / r_h**2
```

`S0` is in `km`. `r_h` is the heliocentric distance in `au`. A positive `S0` moves the optical point away from the Sun along the Sun-comet direction. JPL `SBDB` also exposes `S0` as an orbit model parameter field.[^jpl-sbdb-s0]

```python
from difforb.astrometry import PhotocenterCorrection

photocenter = PhotocenterCorrection(
    s0=0.0,
    estimate=True,
)

print("PARAM_NAMES", photocenter.get_estimated_param_names())
print("PARAM_INIT", photocenter.get_estimated_params().tolist())
print("PARAM_SCALE", photocenter.get_estimated_param_scales().tolist())
```

```text title="Output"
PARAM_NAMES ['S0']
PARAM_INIT [0.0]
PARAM_SCALE [1000.0]
```

The `s0` argument is the initial `S0` value in `km`. To use a fixed value, set `estimate=False`.

## 2. Run DC with `S0`

Pass the correction to `DCSolver.solve(...)`. The solver stores estimated photocenter parameters after the force-model parameters in `result.estimate.model_params`. `S0` values are in `km`.

```python
result = dc.solve(
    obs,
    initial_orbit,
    force_model,
    integrator,
    weight_policy,
    debias_policy,
    outlier_policy,
    photocenter_correction=photocenter,
    log_detail="quiet",
)

print("PARAM_NAMES", result.estimate.model_param_names)
print("PARAM_VALUES", [float(x) for x in result.estimate.model_params.tolist()])
print("N_UNCERTAINTIES", len(result.estimate.uncertainties))
print("COV_VALID", bool(result.lsq_diagnostics.cov_valid))
```

If the force model has no estimated parameters, `PARAM_NAMES` contains only `["S0"]`. The uncertainty vector has length `7`: six Cartesian state values plus one photocenter parameter.

The measurement model applies the correction to optical observation rows. It skips rows with the MPC/ADES note code `e`, which marks zero-aperture extrapolated astrometry.

## Common Mistakes

- Do not treat `S0` as a non-gravitational acceleration. It changes only right ascension and declination predictions.
- Do not use a fitted `S0` before checking `cov_valid`, covariance rank, residuals, and the parameter uncertainty.
- Be careful with mixed-aperture or mixed-reduction datasets. One global `S0` is an empirical average, not a detailed coma model.
- Keep radar observations in the same fit when available. `S0` does not change radar delay or Doppler predictions.

## Next Steps

- Continue to [Inspect Differential Correction Results](inspect-differential-correction-results.md) to check covariance and parameter uncertainties.
- Continue to [Analyze Residuals By Station And Tracklet](analyze-residuals-by-station-and-tracklet.md) to check station or tracklet structure in optical residuals.
- Use [Estimate Nongravitational Parameters In Differential Correction](estimate-nongravitational-parameters-in-differential-correction.md) when you also need to estimate comet outgassing accelerations.

## References

[^yeomans-1994]: Yeomans, D. K. (1994). *A review of comets and nongravitational forces*. In A. Milani, M. di Martino, & A. Cellino (eds.), *Asteroids, Comets, Meteors 1993*, IAU Symposium, Vol. 160, 241-254. <https://doi.org/10.1017/S007418090004657X>
[^farnocchia-2021]: Farnocchia, D., Bellerose, J., Bhaskaran, S., Micheli, M., & Weryk, R. (2021). *High-fidelity comet 67P ephemeris and predictions based on Rosetta data*. Icarus, 358, 114276. <https://doi.org/10.1016/j.icarus.2020.114276>
[^jpl-sbdb-s0]: NASA/JPL Solar System Dynamics. *SBDB API* documents orbit `model_pars`; *SBDB Query API* lists `S0` and `S0_sigma` as query fields. <https://ssd-api.jpl.nasa.gov/doc/sbdb.html> and <https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html>
