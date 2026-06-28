# Concepts

Concepts explain the models, algorithms, and data conventions used by DiffOrb. Read these pages when you want to understand what DiffOrb computes and what assumptions its APIs rely on.

## Recommended Reading Order

1. Read [Time Scales And Epoch Storage](time-scales-and-epoch-storage.md) first. It defines the epoch rules used across the library.
2. Continue with [Frames And State Representation](frames-and-state-representation.md). It explains how `State` combines a Cartesian vector, an epoch, and a reference frame.
3. Read [Keplerian Elements](keplerian-elements.md) if your inputs or outputs use osculating element sets rather than Cartesian states.
4. Read [Batch Inputs And Shapes](batch-inputs-and-shapes.md) to understand how scalar and batch inputs share the same interface style.
5. Read [Earth Rotation And Terrestrial Geometry](earth-rotation-and-terrestrial-geometry.md) next if you work with `ITRS` sites or `ITRS -> GCRS` conversion.
6. Read [Earth Orientation Parameters](earth-orientation-parameters.md) next if you work with modern `UT1`, polar motion, or Earth-based observations.
7. Continue with [Observer Site Keys And Observer Types](observer-site-classes-and-observer-types.md) to see how fixed ground, roving ground, and space observers map into one `Site` model.
8. Read [Light-Time Model](light-time-model.md) if you work with one-way optical light time, two-way radar light time, or radar Doppler.
9. Read [Photocenter Correction](photocenter-correction.md) if you work with comet center-of-light corrections.
10. Continue with [Dynamical Models](dynamical-models.md) to see how accelerations are defined.
11. Continue with [Numerical Integrators And Dense Trajectories](numerical-integrators-and-dense-trajectories.md) to see how states are propagated and queried.
12. Read [Ephemeris Products](ephemeris-products.md) to see how optical, radar, vector, element, apsides, and close-approach products share one propagated path.
13. Read [Orbit Determination Overview](orbit-determination-overview.md) before the orbit-fitting pages.
14. Read [Initial Orbit Determination](initial-orbit-determination.md), then [Differential Correction](differential-correction.md).
15. Finish with [Weighting And Debiasing Models](weighting-and-debiasing-models.md) and [Outlier Rejection](outlier-rejection.md) when you need the error model and rejection rules.

## Current Concepts

- [Time Scales And Epoch Storage](time-scales-and-epoch-storage.md): canonical `TT`, `UT1`, `UTC`, `TAI`, and `TDB` handling.
- [Frames And State Representation](frames-and-state-representation.md): Cartesian state-vector storage and reference-frame labels.
- [Keplerian Elements](keplerian-elements.md): osculating element storage, units, epoch rules, and Cartesian conversion boundaries.
- [Batch Inputs And Shapes](batch-inputs-and-shapes.md): scalar and batch inputs, point-wise broadcasting, Cartesian-product grids, and PyTree dispatch.
- [Earth Rotation And Terrestrial Geometry](earth-rotation-and-terrestrial-geometry.md): the `ITRS -> GCRS` transformation model used for ground sites.
- [Earth Orientation Parameters](earth-orientation-parameters.md): measured Earth-rotation data used by modern time conversion and ground geometry.
- [Observer Site Keys And Observer Types](observer-site-classes-and-observer-types.md): fixed-ground, roving-ground, and space-based observers, and how their observer keys map into `Site`.
- [Light-Time Model](light-time-model.md): one-way optical light time, two-way radar light time, and radar Doppler.
- [Photocenter Correction](photocenter-correction.md): comet center-of-light corrections for optical astrometry.
- [Dynamical Models](dynamical-models.md): ephemeris-backed perturbing bodies and built-in acceleration laws.
- [Numerical Integrators And Dense Trajectories](numerical-integrators-and-dense-trajectories.md): integration methods and dense trajectories.
- [Ephemeris Products](ephemeris-products.md): optical, radar, vector, element, apsides, and close-approach products.
- [Orbit Determination Overview](orbit-determination-overview.md): the whole path from an initial orbit to a fitted solution.
- [Initial Orbit Determination](initial-orbit-determination.md): the short-arc optical seed used before fitting.
- [Differential Correction](differential-correction.md): residuals, weights, automatic differentiation, and damped least squares.
- [Weighting And Debiasing Models](weighting-and-debiasing-models.md): inverse-variance weighting and catalog-based correction of optical astrometry.
- [Outlier Rejection](outlier-rejection.md): chi-square rejection and recovery around the differential-correction solve.

## Where To Go Next

- Use [Guides](../guides/index.md) when you know the task and want the shortest reliable path.
- Use [Workflows](../workflows/index.md) for end-to-end workflows such as ephemeris generation and orbit determination.
- Use the [API](../api/index.md) when you need exact arguments and return fields.

## Scope

These pages describe the model layer only. They cover time, state vectors, reference frames, Keplerian elements, batch shapes, Earth orientation data, observer sites, terrestrial geometry, light-time models, photocenter correction, dynamical models, propagation, ephemeris products, observation weighting, debiasing, differential correction, and outlier rejection. They do not replace the API pages or full workflows.
