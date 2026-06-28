# Photocenter Correction

Optical comet astrometry can measure a center of light that is not the comet center of mass.[^farnocchia-2021] In
DiffOrb, the propagated state still represents the center of mass. A photocenter correction changes only the modeled
optical direction before right ascension and declination are formed.

DiffOrb uses a scalar global form of the `S0` offset model used in comet orbit fitting:[^yeomans-1994][^farnocchia-2021]

```text
offset_distance = S0 / r_h**2
```

`S0` is in `km`. `r_h` is the heliocentric distance in `au`. A positive `S0` moves the modeled optical point away from
the Sun along the Sun-comet direction. JPL `SBDB` also exposes `S0` as an orbit model parameter field for small-body
solutions.[^jpl-sbdb-s0]

The correction applies to optical right ascension and declination. It does not change the propagated orbit, force
model, radar delay, or radar Doppler prediction. `S0` is therefore an optical observation-model parameter, not a
dynamical acceleration parameter.

## Read Next

- Read [Light-Time Model](light-time-model.md) for the optical one-way light-time model used before the angular
  prediction is formed.
- Read [Differential Correction](differential-correction.md) for how observation-model parameters can be estimated
  together with the orbit.
- Use [Estimate A Comet Photocenter Offset In Differential Correction](../guides/estimate-comet-photocenter-offset.md)
  for the concrete `S0` fitting path.

## References

[^farnocchia-2021]: Farnocchia, D., Bellerose, J., Bhaskaran, S., Micheli, M., & Weryk, R. (2021). *High-fidelity
comet 67P ephemeris and predictions based on Rosetta data*. Icarus, 358, 114276.
<https://doi.org/10.1016/j.icarus.2020.114276>
[^yeomans-1994]: Yeomans, D. K. (1994). *A review of comets and nongravitational forces*. In A. Milani, M. di
Martino, & A. Cellino (eds.), *Asteroids, Comets, Meteors 1993*, IAU Symposium, Vol. 160, 241-254.
<https://doi.org/10.1017/S007418090004657X>
[^jpl-sbdb-s0]: NASA/JPL Solar System Dynamics. *SBDB API* documents orbit `model_pars`; *SBDB Query API* lists `S0`
and `S0_sigma` as query fields. <https://ssd-api.jpl.nasa.gov/doc/sbdb.html> and
<https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html>
