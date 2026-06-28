# Configure Outlier Rejection For Orbit Determination

This guide shows how to configure an outlier policy before an orbit-determination solve. In DiffOrb, this policy object is `InteractiveOutlierPolicy`.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- Prepare one local observation file with unchanged `input_index` values. See [Load Local ADES Observations](load-local-ades-observations.md).
- Choose the weight policy before you run a solve that uses automatic rejection. See [Choose And Override Observation Weights](choose-and-override-observation-weights.md).

For the automatic rejection model, read [Outlier Rejection](../concepts/outlier-rejection.md).

## 1. Create a chi-square rejecter

`Chi2OutlierRejecter` is the built-in automatic rejection rule.

Its constructor has threshold parameters:

- `chi2_rej_2d`: rejection threshold for two-component observations such as optical `RA/Dec`. The default is `8.0`.
- `chi2_rec_2d`: recovery threshold for two-component observations. The default is `7.0`.
- `chi2_rej_1d`: rejection threshold for one-component observations. The default is `6.0`.
- `chi2_rec_1d`: recovery threshold for one-component observations. The default is `5.0`.
- `progressive_alpha`: strength of the progressive threshold adjustment. The default is `0.25`.

Create the rejecter:

```python
from difforb.od import Chi2OutlierRejecter

rejecter = Chi2OutlierRejecter()
```

## 2. Create an outlier policy

`InteractiveOutlierPolicy` combines automatic rejection, manual inlier/outlier settings, and outer rejection loop configuration.

Its constructor has three inputs:

- `auto_rejecter`: the automatic rejection rule. Use the `rejecter` from the previous step. Use `None` if you only want manual inlier/outlier settings.
- `enable_auto_rejecter`: whether the automatic rule is enabled. Set it to `False` if you only want manual inlier/outlier settings.
- `max_iters`: the maximum number of iterations in the outer rejection loop.

Create the policy object:

```python
from difforb.od import InteractiveOutlierPolicy

outlier_policy = InteractiveOutlierPolicy(
    rejecter,
    enable_auto_rejecter=True,
    max_iters=3,
)
```

Keep this object. Pass it as `outlier_policy` when you run `DCSolver.solve(...)` in [Run Differential Correction From An Initial Orbit](run-differential-correction-from-an-initial-orbit.md), or when you run `ODSolver.solve(...)` in [Run Integrated Orbit Determination With ODSolver](run-integrated-orbit-determination-with-odsolver.md).

## 3. Add manual inlier/outlier settings

Manual settings use `input_index` values from the original observation file. They do not use row numbers from a sliced table. This is the same row key used by [row-level weight settings](choose-and-override-observation-weights.md).

```python
outlier_policy.force_outlier([350, 351])
outlier_policy.force_inlier(380)

outlier_policy.restore_manual(351)
```

These calls leave row `350` forced out, row `380` forced in, and row `351` back under the normal policy. The policy does
not expose a public list of manual rows. Check the final effect after a solve through `DCResult`.

Use `restore_manual()` with no argument to clear all manual settings.

## 4. Disable automatic rejection

Set `enable_auto_rejecter=False` when you want to use manual settings without automatic rejection.

```python
diagnostic_policy = InteractiveOutlierPolicy(
    rejecter,
    enable_auto_rejecter=False,
    max_iters=1,
)
```

This is useful when you compare weights or check a new force model. Automatic rejection will not exclude or restore
observations, but manual settings still apply.

## 5. Check the result after a solve

After a solve, read the outlier counts, the observations used by the final fit, and rejection metrics from `DCResult`.
See [Inspect Differential Correction Results](inspect-differential-correction-results.md).

## Common Mistakes

- Do not use dataframe row numbers after sorting as manual inlier/outlier indices. Use `input_index`.
- A later pass can restore an automatically rejected observation.
- A manual outlier setting wins over a manual inlier setting for the same row.
- Automatic rejection excludes or restores observations. It does not change the loss function.

## Next Steps

- Continue to [Run Differential Correction From An Initial Orbit](run-differential-correction-from-an-initial-orbit.md) to pass the policy to `DCSolver`.
- Continue to [Run Integrated Orbit Determination With ODSolver](run-integrated-orbit-determination-with-odsolver.md) to pass the policy to `ODSolver`.
- Continue to [Analyze Residuals By Station And Tracklet](analyze-residuals-by-station-and-tracklet.md) after a solve.
- Read [Weighting And Debiasing Models](../concepts/weighting-and-debiasing-models.md) before you change the uncertainties used by the solve.
- Use the [Outlier Rejection API](../api/outlier-rejection.md) for details on rejection policies.
