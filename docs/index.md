# DiffOrb

![DiffOrb banner](assets/brand/difforb-banner.png){ loading=lazy }

DiffOrb is a differentiable, batchable Python framework for small-body orbit propagation, orbit determination, and
Horizons-like ephemeris products.

## Features

Key features:

- **Native JAX with an object-oriented interface.** DiffOrb exposes object-oriented, PyTree-compatible public objects
  for
  common astrodynamical workflows, while keeping the numerical core compatible with JIT compilation, vectorized
  execution, and automatic differentiation.
- **Automatic differentiation support and usage.** DiffOrb keeps its core computations differentiable and uses this
  capability internally, including residual Jacobians for differential correction, radar Doppler from the derivative of
  the converged delay model, and velocity or acceleration derived from SPK position interpolation.
- **Batch computation and automatic broadcasting.** DiffOrb supports scalar inputs and batch inputs. For batch inputs,
  public APIs support automatic broadcasting, including aligned evaluation and Cartesian-product grid evaluation. This
  allows one call to evaluate many targets, epochs, observers, states, or observations without user-written loops or
  multiprocessing code.
- **Dense orbit propagation.** DiffOrb propagates small-body states in `BCRS` as functions of `TDB` and returns dense
  trajectories over the solved time interval. The same trajectory can be reused by light-time iteration, optical
  reduction, radar reduction, and ephemeris-product generation.
- **Composable orbit-determination and ephemeris-generation workflow.** DiffOrb exposes the major stages from
  observations or supplied orbits to fitted solutions and prediction products as composable components, including
  observation loading, initial orbit determination, differential correction, outlier rejection, weighting policies,
  propagation, and observation modelling.
- **Horizons-like ephemeris products.** From a fitted or supplied orbit, DiffOrb can generate optical tables, radar
  tables, vector states, osculating elements, apsides, and close-approach products using shared propagation and
  reduction models.
- **Reusable astrometric and astrodynamical infrastructure.** DiffOrb provides reusable interfaces for epoch creation,
  time-scale and reference-frame conversion, Earth-orientation modelling, observatory position calculation, and SPK/DE
  ephemeris access. These are the same components used internally by the orbit-determination and ephemeris-generation
  workflow.

## Documentation Coverage

- Time scales, Earth rotation, frames, states, sites, and orbital elements.
- Observation loading, optical debiasing, weighting, and measurement reduction.
- Force models, numerical propagation, and prediction products.
- `SPK` ephemeris access and ephemeris-backed body queries.
- Initial orbit determination, differential correction, outlier handling, and high-level orbit-determination workflows.

## Suggested Reading Order

1. Start with [Installation](installation.md) to install DiffOrb.
2. Read [Concepts](concepts/index.md) to understand the models and algorithms used by DiffOrb.
3. Use [Guides](guides/index.md) to learn how to use the common DiffOrb APIs.
4. Use [Workflows](workflows/index.md) to learn how to use DiffOrb for real orbit-determination and ephemeris-generation tasks.
5. Use the [API](api/index.md) pages when you need exact arguments and return fields.

## License

DiffOrb source code, documentation, and DiffOrb-maintained bundled data files
are licensed under the Apache License, Version 2.0. Data files downloaded from
external sources are still governed by the original providers' terms.
