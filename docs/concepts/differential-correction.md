# Differential Correction

Differential correction (`DC`) refines an initial orbit by fitting predicted observations to measured optical and
radar observations. It requires an initial orbit that is already close enough for local nonlinear optimization.

## Core Model

For each candidate parameter vector, DiffOrb propagates the target across the observation arc and evaluates the
corresponding observation models. The differences between measured and predicted values form the observed minus
computed (`O-C`) residuals.

The default fitted parameters are the six Cartesian components of the target state in the Barycentric Celestial
Reference System (`BCRS`) at the reference epoch. Selected force-model or photocenter parameters can be estimated with
the state.

Optical angles, radar delay, and radar Doppler use different units and uncertainty models. DiffOrb therefore solves a
weighted problem rather than treating every residual component equally. Optical catalog debias corrections are
applied before the optical residuals are formed. See
[Weighting And Debiasing Models](weighting-and-debiasing-models.md) for the weighting and debiasing contracts.

## Algorithms Used

DiffOrb uses Levenberg-Marquardt (`LM`) nonlinear least squares. LM behaves like a Gauss-Newton method near a good
solution and limits corrections when the local linear model is less reliable.

The residual Jacobian is computed with forward-mode automatic differentiation through the same propagation and
observation models used to calculate the residuals. This keeps derivatives aligned with the selected physical model
without requiring a separate set of hand-maintained variational equations.

Optional outlier rejection surrounds the least-squares solve. The orbit is refitted when the selected observation set
changes, so the returned parameters and diagnostics describe the final inlier set. See
[Outlier Rejection](outlier-rejection.md) for the rejection model.

## Convergence And Results

The solver can report convergence because the orbit correction is sufficiently small or because the weighted
residual root-mean-square (`RMS`) has stopped improving. These are numerical stopping conditions; they do not by
themselves establish that every fitted parameter is well constrained.

A differential-correction result includes the fitted orbit, residual summaries, the final observation selection,
covariance diagnostics, iteration counts, and the termination reason. Covariance rank and conditioning should be
inspected before interpreting parameter uncertainties.

## Execution Model

DiffOrb provides host-driven and fully JAX-compiled solver control flow. Both modes use the same numerical model. The
host-driven mode favors lower first-call compilation cost and live progress reporting, while the fully compiled mode
can improve throughput for repeated compatible solves. Residual evaluation, automatic differentiation, and orbit
propagation remain JAX-based in either mode.

Operational details for scalar solves, strategy batches, and device selection belong in
[Run Differential Correction From An Initial Orbit](../guides/run-differential-correction-from-an-initial-orbit.md).

## Read Next

- Read [Weighting And Debiasing Models](weighting-and-debiasing-models.md) for the weights in the objective.
- Read [Outlier Rejection](outlier-rejection.md) for the outer rejection loop.
- Read [Dynamical Models](dynamical-models.md) for the force model used during propagation.
- Use [Run Differential Correction From An Initial Orbit](../guides/run-differential-correction-from-an-initial-orbit.md)
  for one concrete solve.
- Use [Inspect Differential Correction Results](../guides/inspect-differential-correction-results.md) for result fields.
