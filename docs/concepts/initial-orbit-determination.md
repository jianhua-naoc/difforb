# Initial Orbit Determination

Initial orbit determination (`IOD`) gives differential correction a starting orbit. It is not the final fit.

## Core Model

DiffOrb currently uses optical observations for `IOD`. It selects short optical arcs and samples observation triplets.
Each triplet is used to build one candidate orbit.

The best candidate becomes the initial orbit for differential correction.

## Double-r Method

The current `IOD` solver uses the Double-r method.[^escobal][^vallado]

A triplet has three ordered epochs, usually called \(t_1\), \(t_2\), and \(t_3\). The measured right ascension and
declination values give three line-of-sight directions. They do not give the target distance along each direction.

Double-r iterates on two unknown topocentric ranges: the range at \(t_1\) and the range at \(t_3\). These ranges define
two endpoint positions. The endpoint positions and time span define a Lambert problem. The Lambert solution predicts
the middle direction at \(t_2\). The difference from the observed middle direction drives the iteration.

## Candidate Selection

One triplet can be sensitive to noise and geometry. DiffOrb samples several triplets. It solves each candidate and
compares the angular residuals. The candidate with the smallest residuals is selected.

## Model Boundary

The initial orbit is only a seed. It does not use the full final model. It does not provide the final covariance, final
weights, radar fit, or final outlier set.

## Read Next

- Read [Differential Correction](differential-correction.md) for the stage that refines the initial orbit.
- Read [Light-Time Model](light-time-model.md) for the one-way optical light-time model used by optical predictions.
- Use [Solve Initial Orbit From Optical Observations](../guides/solve-initial-orbit-from-optical-observations.md) for
  the concrete solver call.

## References

[^escobal]: Escobal, P. R. (1965). *Methods of Orbit Determination*. New York: John Wiley & Sons.
[^vallado]: Vallado, D. A. (2022). *Fundamentals of Astrodynamics and Applications* (5th ed.). Microcosm Press.
