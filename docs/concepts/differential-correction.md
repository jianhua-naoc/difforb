# Differential Correction

Differential correction (`DC`) refines an initial orbit by solving a weighted nonlinear least-squares problem. It starts
from a nearby orbit and improves it by comparing predicted observations with measured observations.

For each trial parameter vector, DiffOrb propagates the target over the observation arc and evaluates the optical or
radar observation model. The difference between the observed values and the modeled values forms the observed minus
computed (`O-C`) residual vector. The solver uses these residuals, the adopted weights, and the residual Jacobian to
compute an orbit correction.

## Fitted Parameters

Let \(\boldsymbol{\theta}\) be the fitted parameter vector. In the default case, it contains the six Cartesian
components of the target state in the Barycentric Celestial Reference System (`BCRS`) at the reference epoch.

When selected model parameters are estimated, they are appended to the same vector. Each trial value of
\(\boldsymbol{\theta}\) therefore defines one complete candidate orbit and parameter set for the observation arc.

## O-C Residual Vector

For one parameter vector, DiffOrb propagates the target and computes the modeled observations
\(\boldsymbol{h}(\boldsymbol{\theta})\). The residual is:

\[
\boldsymbol{r}(\boldsymbol{\theta}) =
\boldsymbol{y}_{\mathrm{obs}} -
\boldsymbol{h}(\boldsymbol{\theta})
\]

Here \(\boldsymbol{y}_{\mathrm{obs}}\) is the observed value. \(\boldsymbol{h}\) is the modeled value.

For optical observations, catalog debias corrections are applied to the reported angles before the residual is formed.
Optical observations give two angular residual components: right ascension and declination. Radar delay and radar
Doppler each give one scalar residual.

## Weighted Objective

If all observations had the same unit and the same reliability, the natural objective would be
\(\boldsymbol{r}^{\mathrm{T}}\boldsymbol{r}\). Real observation sets are different. Optical angles, radar delay, and
radar Doppler have different units and different uncertainty levels. The solve therefore minimizes a weighted objective:

\[
Q(\boldsymbol{\theta}) =
\boldsymbol{r}^{\mathrm{T}}
\boldsymbol{W}
\boldsymbol{r}
\]

\(\boldsymbol{W}\) is built from the adopted observation weights. Optical weights can come from a statistical model,
reported uncertainties, or user-specified uncertainties. Radar weights use the reported radar uncertainties. See
[Weighting And Debiasing Models](weighting-and-debiasing-models.md) for a more detailed description of the weighting
and optical debiasing models.

## Linearized Correction

The residual vector is nonlinear because propagation, light-time correction, and observation reduction depend on the
current parameter vector. The solver therefore linearizes the residuals around the current value
\(\boldsymbol{\theta}_0\):

\[
\boldsymbol{r}(\boldsymbol{\theta}_0 + \Delta\boldsymbol{\theta})
\simeq
\boldsymbol{r}(\boldsymbol{\theta}_0) +
\boldsymbol{J}_0 \Delta\boldsymbol{\theta}
\]

The Jacobian is:

\[
\boldsymbol{J}_0 =
\left.
{\partial \boldsymbol{r} \over \partial \boldsymbol{\theta}}
\right|_{\boldsymbol{\theta}_0}
\]

The Jacobian describes how a small change in the fitted parameters changes the residuals.

## Damped Least Squares

The linearized least-squares problem could be solved with a Gauss-Newton step. That can be efficient near the final
orbit. It can be unstable when the initial orbit is not close enough, when some parameters are weakly constrained, or
when different fitted parameters are strongly correlated.

DiffOrb therefore uses a damped Levenberg-Marquardt (`LM`) solve:

\[
\left(
\boldsymbol{J}_0^{\mathrm{T}}
\boldsymbol{W}
\boldsymbol{J}_0
+ \lambda \boldsymbol{D}_0
\right)
\Delta\boldsymbol{\theta}
=
-
\boldsymbol{J}_0^{\mathrm{T}}
\boldsymbol{W}
\boldsymbol{r}_0
\]

\(\lambda\) is the damping parameter. \(\boldsymbol{D}_0\) is a diagonal scaling matrix derived from the local weighted
normal matrix.

The solver tests each trial correction with the full nonlinear model. If the correction lowers
\(Q(\boldsymbol{\theta})\), the correction is accepted and the damping is reduced. If the correction does not lower
\(Q(\boldsymbol{\theta})\), the damping is increased and a smaller correction is tried.

After an accepted correction, the current parameter vector is updated:

\[
\boldsymbol{\theta}_0
\leftarrow
\boldsymbol{\theta}_0 + \Delta\boldsymbol{\theta}
\]

Near a good solution, the damping is small and the method behaves like Gauss-Newton. When the local linear model is not
reliable, larger damping limits the step.

## Jacobian From Automatic Differentiation

Traditional orbit-determination implementations often obtain the Jacobian from hand-derived partial derivatives,
variational equations, or finite differences. DiffOrb instead computes the Jacobian by automatic differentiation of the
same residual function used for prediction.

This keeps the derivatives tied to the implemented force model, light-time model, and observation model. When those
models change, the derivatives follow the same numerical path as the residuals.

## Outlier Rejection Around The Solve

Least squares is sensitive to outlying observations. DiffOrb therefore applies outlier rejection around differential
correction.

The weighted least-squares solve is the inner loop. Outlier rejection is the outer loop. After each least-squares
solution, the outlier set is updated, and the orbit is solved again until the outlier set no longer changes or a
configured limit is reached. See [Outlier Rejection](outlier-rejection.md) for a more detailed description of the
outlier rejection algorithm.

## Result Diagnostics

The result contains more than the fitted orbit. It records residual scatter, outlier information, covariance
information, iteration counts, and stop reasons. These fields answer different questions and should be read together.

## Read Next

- Read [Weighting And Debiasing Models](weighting-and-debiasing-models.md) for the weights in the objective.
- Read [Outlier Rejection](outlier-rejection.md) for the outer rejection loop.
- Read [Dynamical Models](dynamical-models.md) for the force model used during propagation.
- Use [Run Differential Correction From An Initial Orbit](../guides/run-differential-correction-from-an-initial-orbit.md)
  for one concrete solve.
- Use [Inspect Differential Correction Results](../guides/inspect-differential-correction-results.md) for result fields.
