# Guides

Guides show how to use the common DiffOrb APIs for one task at a time. Use them when you want concrete examples for creating objects, loading data, propagating orbits, generating ephemerides, or running solvers.

[Concepts](../concepts/index.md) explain the models behind those tasks. [Workflows](../workflows/index.md) show full runs.

## Time And Epochs

- [Configure Earth Orientation Data](configure-earth-orientation-data.md): check the local `EOP` file, refresh it when network access is available, and verify one modern epoch.
- [Create And Convert Time Objects](create-and-convert-time-objects.md): build `Time` objects from calendar dates or split Julian dates, inspect their fields, batch them, and shift them by uniform intervals.
- [Convert Between UTC, TT, TDB, UT1](convert-between-utc-tt-tdb-ut1.md): derive the main time-scale views from one epoch, measure offsets, and apply the topocentric `TDB - TT` correction.

## Ephemerides And Bodies

- [Load SPK Kernels And Query Major Bodies](load-spk-kernels-and-query-major-bodies.md): configure the default ephemeris, inspect available body names, and query major-body states.
- [Create And Convert Keplerian Elements](create-and-convert-keplerian-elements.md): build `KepElement` objects, convert them to Cartesian states, and round-trip them back to elements.
- [Create A SmallBody From State Or Elements](create-a-smallbody-from-state-or-elements.md): initialize `SmallBody` objects from Cartesian states or Keplerian elements and verify the stored canonical orbit.

## States And Frames

- [Build State From Cartesian Data](build-state-from-cartesian-data.md): create a `State` from canonical Cartesian data and inspect its stacked array form.
- [Transform State Between Frames](transform-state-between-frames.md): convert a `State` between reference frames and roundtrip the result.

## Observer Geometry

- [Get Earth Rotation Quantities And Matrices](get-earth-rotation-quantities-and-matrices.md): read polar-motion coordinates, the Earth Rotation Angle, the `CIO`-based matrices, and the equinox-based matrices from one `Time` object.
- [Create A Ground Site And Get Its GCRS State](create-a-groundsite-and-get-its-gcrs-state.md): create a ground observer from codes, geodetic coordinates, or geocentric constants and evaluate it in canonical `GCRS`.
- [Create A Space Site And Reuse Its GCRS State](create-a-spacesite-and-reuse-its-gcrs-state.md): create a space observer from a canonical `GCRS` state and reuse it through the site API.

## Dynamics And Propagation

- [Configure Force Models And Dynamic Systems](configure-force-models-and-dynamic-systems.md): build the standard major-body system, the extended system with asteroid perturbers, or one custom `ForceModel` with explicit gravity and non-gravitational terms.
- [Propagate A SmallBody And Evaluate Dense Trajectories](propagate-a-smallbody-and-evaluate-dense-trajectories.md): propagate one `SmallBody`, check trajectory coverage, and query interpolated `BCRS` states inside the solved interval.

## Observation Products

- [Get Optical Outputs In Observer And Vector Modes](get-optical-outputs-in-observer-and-vector-modes.md): sample observer-mode optical outputs and vector-mode relative states from one propagated target, one observer, and one observation-time grid.
- [Get Radar Outputs In Monostatic And Bistatic Geometry](get-radar-outputs-in-monostatic-and-bistatic-geometry.md): sample 2025 BC10 radar delay and Doppler predictions at DSS-14 and compare them with JPL radar astrometry records.

## Observation Preparation

- [Load Online Observations From MPC And JPL](load-online-observations-from-mpc-and-jpl.md): fetch online observations and save them to a local ADES PSV file.
- [Load Local ADES Observations](load-local-ades-observations.md): reopen that PSV file or another local ADES PSV file and inspect the mixed rows.
- [Choose And Override Observation Weights](choose-and-override-observation-weights.md): compare `VFCC17` with reported ADES uncertainties and add row-level overrides.
- [Inspect Optical Debias Corrections](inspect-optical-debias-corrections.md): run the Eggl model and inspect row-level optical corrections.

## Orbit Determination

- [Solve Initial Orbit From Optical Observations](solve-initial-orbit-from-optical-observations.md): run `IODSolver` on optical observations and inspect the initial guess.
- [Configure Outlier Rejection For Orbit Determination](configure-outlier-rejection-for-orbit-determination.md): configure automatic chi-square rejection and manual inlier/outlier settings.
- [Run Differential Correction From An Initial Orbit](run-differential-correction-from-an-initial-orbit.md): run one `DCSolver` solve from an initial guess and inspect convergence.
- [Run Integrated Orbit Determination With ODSolver](run-integrated-orbit-determination-with-odsolver.md): run IOD and staged differential correction through one solver, including `IODStrategy` and `DCStrategy`.
- [Inspect Differential Correction Results](inspect-differential-correction-results.md): read the fitted orbit, RMS, covariance status, and inlier counts.
- [Analyze Residuals By Station And Tracklet](analyze-residuals-by-station-and-tracklet.md): join residuals back to observations and analyze station or tracklet diagnostics.
- [Estimate Nongravitational Parameters In Differential Correction](estimate-nongravitational-parameters-in-differential-correction.md): add estimated force-model parameters and read them from `DCResult`.
- [Estimate A Comet Photocenter Offset In Differential Correction](estimate-comet-photocenter-offset.md): estimate a global optical `S0` parameter and read it from `DCResult`.

## Shared Patterns

- [Use Batch Inputs Across DiffOrb APIs](use-batch-inputs-across-difforb-apis.md): build batched objects, inspect batch shapes, slice rows, and choose point-wise or grid calls.

## When To Leave The Guides

- Go back to [Concepts](../concepts/index.md) when you need the rules behind a task.
- Move forward to [Workflows](../workflows/index.md) when you need a multi-stage workflow such as ephemeris generation or orbit determination.
- Use the [API](../api/index.md) pages when you need exact arguments and return fields.
