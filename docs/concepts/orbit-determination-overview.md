# Orbit Determination Overview

DiffOrb is designed for precise orbit determination, not quick short-arc orbit finding. In this use case, the number
of observations is usually much larger than the number of fitted parameters. DiffOrb treats the fit as an
overdetermined nonlinear problem and solves it with weighted nonlinear least squares.

The least-squares solve is local. It needs a starting orbit that is already close to the true orbit. DiffOrb therefore
separates orbit determination into two stages.

Initial orbit determination builds a starting orbit from a short optical arc. Differential correction then refines that
orbit with the selected force model, observation model, adopted weights, and outlier rejection.

## Read Next

- Read [Initial Orbit Determination](initial-orbit-determination.md) for the starting-orbit stage.
- Read [Differential Correction](differential-correction.md) for the weighted nonlinear solve.
- Read [Weighting And Debiasing Models](weighting-and-debiasing-models.md) for the error model used by the fit.
- Read [Outlier Rejection](outlier-rejection.md) for the outer rejection loop.
- Use [Orbit Determination Workflow](../workflows/orbit-determination-workflow.md) for an end-to-end example.
