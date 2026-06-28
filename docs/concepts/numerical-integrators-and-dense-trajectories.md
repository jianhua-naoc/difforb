# Numerical Integrators And Dense Trajectories

DiffOrb propagates a small-body state by solving an ordinary differential equation. The state is a Cartesian position
and velocity in `BCRS`. The independent variable is `TDB`.

## Core Model

A force model gives acceleration. A numerical integrator advances the target position and velocity through time.

Position is in `au`. Velocity is in `au / day`. Acceleration is in `au / day^2`.

## Supported Integrators

DiffOrb uses `diffrax` as the integration layer.[^diffrax]

The public integrator interface supports:

- `IAS15`, the default high-order Gauss-Radau integrator.[^rein-spiegel]
- `DOPRI8`, an eighth-order Dormand-Prince method from `diffrax`.
- `DOPRI5`, a fifth-order Dormand-Prince method from `diffrax`.

The method changes numerical error, step count, and run time. It does not change the meaning of the propagated state.

## Dense Trajectories

DiffOrb returns a dense trajectory over the solved time span. A dense trajectory is an interpolated solution. It can be
queried at times between accepted integration steps.

This is part of the model, not only a convenience. Light-time solvers need target states at trial emission or bounce
times. These trial times are usually not accepted integration step endpoints.

Dense trajectories let those queries use the same propagated path and the same force model. They avoid a new
integration for every light-time iteration.

## Where Dense Trajectories Are Used

Dense trajectories connect propagation to later model layers.

- Optical reduction queries the target at the one-way emission time.
- Radar reduction queries the target at down-leg and up-leg signal times.
- Ephemeris products query many output epochs from one propagated path.
- Differential correction repeats propagation as the fitted parameters change.

## Model Boundary

The integrator only solves the state equation. It does not define the force law. It does not define optical or radar
observables. It supplies the trajectory used by those layers.

## Read Next

- Read [Dynamical Models](dynamical-models.md) for the acceleration model used by the integrator.
- Read [Ephemeris Products](ephemeris-products.md) for products built from dense trajectories.
- Use [Propagate A SmallBody And Evaluate Dense Trajectories](../guides/propagate-a-smallbody-and-evaluate-dense-trajectories.md)
  when you need a concrete propagation call.
- Use the [Integrator API](../api/integrator.md) for constructor arguments and return fields.

## References

[^rein-spiegel]: Rein, H., & Spiegel, D. S. (2015). *IAS15: A fast, adaptive, high-order integrator for gravitational
dynamics, accurate to machine precision over a billion orbits*. Monthly Notices of the Royal Astronomical Society,
446(2), 1424-1437. <https://doi.org/10.1093/mnras/stu2164>
[^diffrax]: Diffrax documentation, *ODE solvers*. <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/>
