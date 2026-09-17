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

DiffOrb solves the damped least-squares system without explicitly inverting the normal matrix. Parameter scaling keeps
the damping meaningful when fitted parameters use different physical units.

The solver evaluates each trial correction with the full nonlinear model and compares the actual loss reduction
with the reduction predicted by the linearized residuals. Insufficient improvement rejects the step and increases
damping. Good agreement permits a larger step on the next trial; other accepted steps retain the search scale.

When observation-time uncertainty is present, the optical weights depend on the predicted sky-plane rates. Weights
are recomputed at every accepted parameter vector and held fixed throughout its damping search. The Jacobian
differentiates the unweighted residuals; derivatives of the weight model are excluded from the local least-squares
system. Trial losses are always compared using the same weights.

After an accepted correction, the current parameter vector is updated:

\[
\boldsymbol{\theta}_0
\leftarrow
\boldsymbol{\theta}_0 + \Delta\boldsymbol{\theta}
\]

Near a good solution, the damping is small and the method behaves like Gauss-Newton. When the local linear model is not
reliable, larger damping limits the step.

## Convergence

DiffOrb uses a dimensionless correction norm and a separate residual root-mean-square (RMS) stagnation test. Either condition can end a fit successfully; the termination reason distinguishes them.

At each linearization point, let \(J_k\) be the residual Jacobian, \(W_k\) the frozen weight matrix, and \(\Delta\theta_{\mathrm{GN}}\) the undamped Gauss-Newton correction. The correction criterion is

\[
\mathrm{delnor}_k
= \sqrt{\frac{\Delta\theta_{\mathrm{GN}}^\mathsf{T}
J_k^\mathsf{T}W_kJ_k\Delta\theta_{\mathrm{GN}}}{n}}
< 10^{-3},
\]

where \(n\) includes all fitted state and model parameters. This threshold is fixed and is not a constructor option. This measures the correction in the full normal-matrix metric, including correlations between parameters. It does not require each physical parameter change to be small. The undamped correction is evaluated using a scaled SVD least-squares solve, so shrinking an LM trial through damping alone cannot satisfy the correction criterion. The trial must still be accepted, and its refreshed model must be finite, before returning `correction_converged`.

The second criterion compares RMS values at consecutive accepted points, with each point's refreshed weights and the same inlier mask. If

\[
R_{k+1} > 0.999R_k,
\]

the stagnation counter increases; otherwise it resets to zero. The minimum relative decrease is fixed at 0.1 percent, and the counter limit is fixed at six accepted steps. After six consecutive accepted steps with insufficient decrease, the fit returns `rms_stagnated`, unless the final RMS exceeds the preceding value by more than 10 percent, in which case it returns the failure reason `rms_increasing`. As in OrbFit, the counter includes both increases and plateaus, and the final comparison determines the exit status. Small increases can therefore be classified as stagnation. Rejected trials do not advance or reset this counter. Accepted-step logs report the refreshed RMS used by this test; rejected-trial logs and trust-region ratios use frozen weights.

This strategy follows the correction metric and RMS stopping logic in OrbFit 5.0.8's `diff_cor`, `sin_cor`, and `snorm` routines ([official source distribution](https://adams.dm.unipi.it/orbfit/OrbFit5.0.8.tar.gz)), adapted to accepted LM steps. Unlike OrbFit's two correction thresholds, DiffOrb applies the same settings to every fit before and after outlier rejection, including a final refit after a mask change. Each fit starts a new stagnation counter.

`rms_stagnated` means that the fit has stopped improving under this policy; it does not assert that `delnor` met its threshold. Neither successful stop reason establishes parameter identifiability: inspect covariance rank and conditioning separately. A zero iteration budget evaluates only the initial point and returns `max_iter_reached` for a finite model.

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

If the last permitted rejection pass changes the mask, DiffOrb performs one final fit with that mask. Returned
parameters, weights, covariance, and residual diagnostics therefore describe the same selected observations.

## Numerical Execution and JIT

The host and fully compiled execution modes use the same numerical method. Rejected observations are excluded by masks, and weights are refreshed only at accepted points.

`DCSolver` and `LeastSquares` default to `solver_jit=False`. This mode favors lower first-call compilation cost and supports live progress events. Residual, Jacobian, and propagation calculations still use JAX compilation.

With `solver_jit=True`, the complete fitting and rejection loops are compiled. This can improve throughput for repeated fits with the same shapes, but increases first-call compilation time and memory use. The same `solve` interface is used in both modes. At the lower-level least-squares interfaces, compiled solves can also be enclosed by `jax.jit` and `jax.vmap` when their callbacks are JAX-compatible. The default mode can emit live iteration events; the fully compiled mode emits final summaries after execution.

Differentiating the fitted solution is unsupported. The residual Jacobian within each fit still comes from automatic differentiation, with weights excluded from that differentiation.

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
