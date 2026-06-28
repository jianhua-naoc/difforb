# Outlier Rejection

Least squares is sensitive to outlying observations. A few bad observations can move the fitted orbit by a large
amount. DiffOrb therefore uses the chi-square outlier rejection algorithm described by Carpino et al. (2003).[^carpino]

Outlier rejection is applied around differential correction. The weighted least-squares solve is the inner loop, and
outlier rejection is the outer loop. After each least-squares solution, DiffOrb updates the outlier set and solves the
orbit again. The process stops when the outlier set no longer changes or when the configured pass limit is reached.

The rejection test uses fitted residuals, the adopted observation weights, and the local residual covariance. A large
raw residual is not enough by itself. The residual must be large relative to the uncertainty used by the rejection test.

Optical observations are tested as two-component observations. Right ascension and declination are tested together.
Radar delay and radar Doppler are tested as one-component observations.

DiffOrb uses separate rejection and recovery thresholds. Recovery uses a lower threshold than rejection, so an
observation does not switch in and out of the fit because of a small numerical change near one threshold. By default,
the rejection and recovery thresholds are \(\chi^2_{\mathrm{rej}} = 8\) and \(\chi^2_{\mathrm{rec}} = 7\) for
two-dimensional optical observations, and \(\chi^2_{\mathrm{rej}} = 6\) and \(\chi^2_{\mathrm{rec}} = 5\) for
one-dimensional radar observations.

Users may choose tighter or looser thresholds, disable automatic rejection, or manually mark observations as inliers or
outliers. Outlier rejection is not a weight model. It excludes selected observations from the fit instead of changing
the least-squares loss shape.

## Read Next

- Read [Differential Correction](differential-correction.md) for the inner least-squares solve.
- Read [Weighting And Debiasing Models](weighting-and-debiasing-models.md) for the adopted uncertainties.
- Use [Configure Outlier Rejection For Orbit Determination](../guides/configure-outlier-rejection-for-orbit-determination.md)
  for policy setup.
- Use [Inspect Differential Correction Results](../guides/inspect-differential-correction-results.md) for rejected
  observations and rejection metrics.
- Use the [Outlier Rejection API](../api/outlier-rejection.md) for symbol-level details.

## References

[^carpino]: Carpino, M., Milani, A., & Chesley, S. R. (2003). *Error statistics of asteroid optical astrometric
observations*. Icarus, 166(2), 248-270.
