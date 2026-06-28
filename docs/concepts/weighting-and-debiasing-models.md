# Weighting And Debiasing Models

Weighting and debiasing are separate models. Debiasing changes optical measurements. Weighting changes how strongly
residuals count in the fit.

## Core Distinction

Debiasing removes known systematic error from optical astrometry. It changes the reported right ascension and
declination before residuals are computed.

Weighting describes random uncertainty. It does not change the measurement. It changes the inverse-variance weight used
by weighted least squares.

## Weighting Model

If one scalar residual has adopted one-sigma uncertainty \(\sigma\), its weight is:

\[
w = {1 \over \sigma^2}
\]

A smaller uncertainty gives a larger weight. A larger uncertainty gives a smaller weight.

Radar observations usually carry reported uncertainties, and DiffOrb uses them. Optical observations are less uniform.
The Astrometric Data Exchange Standard (`ADES`) can store optical uncertainties, but they are optional. Older `MPC1992`
optical records do not carry optical uncertainties.

DiffOrb supports three practical optical weight sources.

- The Vereš et al. statistical model, called `VFCC17` in DiffOrb.[^veres]
- Reported uncertainties stored in observation rows.
- Manual weights set by the user.

## Debiasing Model

Debiasing applies only to optical astrometry. It corrects systematic offsets linked to the star catalog used to reduce
the observation.

DiffOrb uses the Eggl et al. catalog debiasing model.[^eggl] The correction depends on observation epoch, catalog code,
and sky position. It is applied before the optical residual is formed.

## Where The Models Meet

Debiased optical measurements and adopted weights meet in differential correction. The residual uses the corrected
measurement. The weighted objective uses the adopted uncertainty.

Outlier rejection also depends on weights, because it tests whether fitted residuals are large relative to the adopted
error model.

## Read Next

- Read [Differential Correction](differential-correction.md) for the weighted objective.
- Read [Outlier Rejection](outlier-rejection.md) for how weights enter the rejection test.
- Use [Choose And Override Observation Weights](../guides/choose-and-override-observation-weights.md) for weight
  policies.
- Use [Inspect Optical Debias Corrections](../guides/inspect-optical-debias-corrections.md) for row-level debias output.
- Use the [Weights API](../api/weights.md) and [Debiasing API](../api/debiasing.md) for symbol-level details.

## References

[^veres]: Vereš, P., Farnocchia, D., Chesley, S. R., & Chamberlin, A. B. (2017). *Statistical analysis of astrometric
errors for the most productive asteroid surveys*. Icarus, 296, 139-149.
[^eggl]: Eggl, S., Farnocchia, D., Chamberlin, A. B., & Chesley, S. R. (2020). *Star catalog position and proper motion
corrections in asteroid astrometry II: The Gaia era*. Icarus, 339, 113596.
