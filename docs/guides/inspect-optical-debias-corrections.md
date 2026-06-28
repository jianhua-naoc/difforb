# Inspect Optical Debias Corrections

This guide shows how to run the Eggl debias model on `/tmp/2025bc10-online-guide.psv` and check the row-level corrections in arcseconds.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- Run [Load Online Observations From MPC And JPL](load-online-observations-from-mpc-and-jpl.md) first.
- The example below uses `/tmp/2025bc10-online-guide.psv`.
- Install the `debias2018` data set in the DiffOrb data directory.
- Debiasing applies only to optical rows. Radar rows are not part of this correction layer.
- Read [Weighting And Debiasing Models](../concepts/weighting-and-debiasing-models.md) first if you want more detail on the debias model.

## 1. Run the Eggl model

`EgglDebiasPolicy` returns one bias array for optical rows. The values are in radians. Convert them to arcseconds if you want to check their size by eye.

```python
import numpy as np

from difforb.astrometry import EgglDebiasPolicy, load_local_observations

obs = load_local_observations("/tmp/2025bc10-online-guide.psv")
result = EgglDebiasPolicy().bias(obs)

optical_arcsec = np.rad2deg(result.optical_bias) * 3600.0
nonzero = np.where(np.any(np.abs(optical_arcsec) > 0.0, axis=1))[0]

print("NAME", obs.name)
print("N_OPTICAL", obs.num_optical)
print("N_NONZERO", len(nonzero))
print("FIRST_NONZERO_INDEX", int(nonzero[0]))
print("FIRST_NONZERO_CATALOG", obs.optical.catalog_codes[int(nonzero[0])])
print("FIRST_NONZERO_TIME", obs.optical.t[int(nonzero[0])].ut.iso_string)
print("FIRST_NONZERO_BIAS_ARCSEC", np.round(optical_arcsec[int(nonzero[0])], 6))
```

```text title="Output"
NAME 2025 BC10
N_OPTICAL 799
N_NONZERO 53
FIRST_NONZERO_INDEX 276
FIRST_NONZERO_CATALOG UCAC4
FIRST_NONZERO_TIME 2025-03-24 18:31:58.079
FIRST_NONZERO_BIAS_ARCSEC [0.053828 0.012516]
```

Some rows get a non-zero correction. Some rows stay at zero.

These row numbers come from the reference file saved on `2026-04-23`. If you save a new file later, the row numbers may change.

## 2. Check the first corrected rows

Use `input_index` to find the same rows in the original observation file.

```python
import numpy as np

from difforb.astrometry import EgglDebiasPolicy, load_local_observations

obs = load_local_observations("/tmp/2025bc10-online-guide.psv")
result = EgglDebiasPolicy().bias(obs)

optical_arcsec = np.rad2deg(result.optical_bias) * 3600.0
nonzero = np.where(np.any(np.abs(optical_arcsec) > 0.0, axis=1))[0][:5]

for idx in nonzero:
    print(
        "INPUT", int(obs.optical.input_indices[idx]),
        "TIME", obs.optical.t[idx].ut.iso_string,
        "CAT", obs.optical.catalog_codes[idx],
        "RA_BIAS", round(float(optical_arcsec[idx, 0]), 6),
        "DEC_BIAS", round(float(optical_arcsec[idx, 1]), 6),
    )
```

```text title="Output"
INPUT 276 TIME 2025-03-24 18:31:58.079 CAT UCAC4 RA_BIAS 0.053828 DEC_BIAS 0.012516
INPUT 279 TIME 2025-03-24 19:17:37.823 CAT UCAC4 RA_BIAS 0.053829 DEC_BIAS 0.012516
INPUT 285 TIME 2025-03-24 19:59:35.520 CAT UCAC4 RA_BIAS 0.05383 DEC_BIAS 0.012516
INPUT 288 TIME 2025-03-24 20:27:39.456 CAT UCAC4 RA_BIAS 0.053831 DEC_BIAS 0.012516
INPUT 293 TIME 2025-03-24 21:26:37.536 CAT UCAC4 RA_BIAS 0.053832 DEC_BIAS 0.012516
```

Rows with missing or unsupported catalog codes stay at zero.

## Common Mistakes

- Debias corrections are returned in radians. Convert to arcseconds before interpreting small values by eye.
- Debiasing applies only to optical rows. Do not expect radar rows to receive a catalog correction.
- A zero correction means no supported correction was applied to that row.

## Next Steps

- Continue to [Choose And Override Observation Weights](choose-and-override-observation-weights.md) when you also need to check the uncertainty side for `/tmp/2025bc10-online-guide.psv`.
- Return to [Load Local ADES Observations](load-local-ades-observations.md) or [Load Online Observations From MPC And JPL](load-online-observations-from-mpc-and-jpl.md) when you still need to adjust `/tmp/2025bc10-online-guide.psv`.
- Continue to [Orbit Determination Workflow](../workflows/orbit-determination-workflow.md) when you want to use the debias policy in a full solve.
- Use the [Debiasing API](../api/debiasing.md) for details on debias policies and results.
