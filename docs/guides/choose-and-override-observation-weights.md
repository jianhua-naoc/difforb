# Choose And Override Observation Weights

This guide shows how to set observation weights automatically or manually in DiffOrb. It uses `/tmp/2025bc10-online-guide.psv` to show how weight results are organized, compare automatic weights with reported uncertainties, and change the weights for selected rows.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- Run [Load Online Observations From MPC And JPL](load-online-observations-from-mpc-and-jpl.md) first.
- The example below uses `/tmp/2025bc10-online-guide.psv`.
- Install the `weights` data set in the DiffOrb data directory, or pass an explicit rule-table path to
  `VFCC17WeightPolicy(...)`.
- Read [Weighting And Debiasing Models](../concepts/weighting-and-debiasing-models.md) first if you want more detail on the weighting models.

## 1. Inspect weight blocks

DiffOrb provides several `WeightPolicy` classes for setting weights, such as `VFCC17WeightPolicy`, `ADESWeightPolicy`, and `InteractiveWeightPolicy`. They all use `weights(...)` to return a `WeightResult`. Like `ObservationData`, `WeightResult` keeps optical and radar data in separate arrays.

```python
from difforb.astrometry import load_local_observations, VFCC17WeightPolicy

obs = load_local_observations("/tmp/2025bc10-online-guide.psv")
result = VFCC17WeightPolicy().weights(obs)

print("OPTICAL_SHAPE", result.optical_uncertainties.shape)
print("RADAR_SHAPE", result.radar_uncertainties.shape)
print("OPTICAL_SOURCE_HEAD", result.optical_sources[:3])
print("RADAR_SOURCE_HEAD", result.radar_sources[:3])
```

```text title="Output"
OPTICAL_SHAPE (799, 2)
RADAR_SHAPE (8,)
OPTICAL_SOURCE_HEAD ['VFCC17' 'VFCC17' 'VFCC17']
RADAR_SOURCE_HEAD ['VFCC17' 'VFCC17' 'VFCC17']
```

Optical uncertainties have two columns. Radar uncertainties have one value per row.

## 2. Compare weight policies

`VFCC17WeightPolicy` gives statistical optical uncertainties. `ADESWeightPolicy` uses the uncertainties already stored in each row.

Read [Weighting And Debiasing Models](../concepts/weighting-and-debiasing-models.md) if you want more detail on the models behind `VFCC17WeightPolicy` and `ADESWeightPolicy`.

```python
import numpy as np

from difforb.astrometry import load_local_observations, ADESWeightPolicy, VFCC17WeightPolicy

obs = load_local_observations("/tmp/2025bc10-online-guide.psv")

vfcc = VFCC17WeightPolicy().weights(obs)
ades = ADESWeightPolicy().weights(obs)

for idx in [7, 8, 9]:
    vfcc_ra = float(np.rad2deg(vfcc.optical_uncertainties[idx, 0]) * 3600.0)
    ades_ra = float(np.rad2deg(ades.optical_uncertainties[idx, 0]) * 3600.0)
    print("ROW", idx, "VFCC17_RA", round(vfcc_ra, 3), "ADES_RA", round(ades_ra, 3))
```

```text title="Output"
ROW 7 VFCC17_RA 0.5 ADES_RA 0.384
ROW 8 VFCC17_RA 0.5 ADES_RA 0.765
ROW 9 VFCC17_RA 0.5 ADES_RA 0.382
```

These rows have reported ADES uncertainties, so the two policies do not match.

These row numbers come from the reference file saved on `2026-04-23`. If you save a new file later, the row numbers may change.

## 3. Use `InteractiveWeightPolicy`

Use `InteractiveWeightPolicy` when you want one default policy and then a few changes.

Create `InteractiveWeightPolicy` with these parameters:

- `default_policy` is required. It is the policy used for every row before any override is applied.
- `additional_policies` is optional. Add other policies here if you want `select_scheme(...)` to switch rows to them.

`InteractiveWeightPolicy` provides these interfaces:

- `set_manual_optical(...)` sets manual uncertainties for selected optical rows.
- `set_manual_radar(...)` sets a manual uncertainty for selected radar rows.
- `select_scheme(...)` switches selected rows to a registered `WeightPolicy` from `default_policy` or `additional_policies`.
- `restore_default_policy(...)` removes overrides and returns selected rows to the default policy. If you do not pass `input_index`, it clears every override.

These interfaces select rows by `input_index`. `input_index` is the row number in the original observation file. You can get it from `obs.optical.input_indices`, `obs.radar.input_indices`, or `obs.to_dataframe(sort_by="input")`, as shown in [Load Local ADES Observations](load-local-ades-observations.md).

### Set Manual Uncertainties

The example below uses `set_manual_optical(...)`. Use `set_manual_radar(...)` the same way for `obs.radar` rows, but pass one uncertainty value instead of `ra_unc` and `dec_unc`.

This example passes only `default_policy`, because manual uncertainties do not need any extra registered policy.

```python
import numpy as np

from difforb.astrometry import load_local_observations, InteractiveWeightPolicy, VFCC17WeightPolicy

obs = load_local_observations("/tmp/2025bc10-online-guide.psv")

interactive = InteractiveWeightPolicy(default_policy=VFCC17WeightPolicy())
interactive.set_manual_optical(
    [10],
    ra_unc=np.deg2rad(1.0 / 3600.0),
    dec_unc=np.deg2rad(1.5 / 3600.0),
)

result = interactive.weights(obs)

for idx in [9, 10]:
    ra = float(np.rad2deg(result.optical_uncertainties[idx, 0]) * 3600.0)
    dec = float(np.rad2deg(result.optical_uncertainties[idx, 1]) * 3600.0)
    print("ROW", idx, "SOURCE", result.optical_sources[idx], "RA", round(ra, 3), "DEC", round(dec, 3))
```

```text title="Output"
ROW 9 SOURCE VFCC17 RA 0.5 DEC 0.5
ROW 10 SOURCE MANUAL RA 1.0 DEC 1.5
```

Row `10` now uses the manual uncertainties. Row `9` still uses `VFCC17`.

### Select Another Policy

Use `select_scheme(...)` when you want matching rows to use another `WeightPolicy`.

This example selects all optical rows from station `W74` on `2025-03-03`. `day` and `day_end` define the time range. `optical` is the `obs.optical` table. `mask` selects the rows in that table that match both the time range and the station code. `selected_rows` are row positions inside the optical table. `indices` are `input_index` values from the original observation file.

This example passes `additional_policies=[ades]`, because `select_scheme(...)` can only switch rows to a policy that was registered when `InteractiveWeightPolicy` was created.

```python
import numpy as np

from difforb.astrometry import (
    load_local_observations,
    ADESWeightPolicy,
    InteractiveWeightPolicy,
    VFCC17WeightPolicy,
)
from difforb.core import Time

obs = load_local_observations("/tmp/2025bc10-online-guide.psv")

vfcc = VFCC17WeightPolicy()
ades = ADESWeightPolicy()
interactive = InteractiveWeightPolicy(default_policy=vfcc, additional_policies=[ades])

day = Time.from_ut_date(2025.0, 3.0, 3.0)
day_end = day + 1.0
optical = obs.optical
mask = np.asarray((optical.t >= day) & (optical.t < day_end)) & (optical.rx_codes == "W74")
selected_rows = np.flatnonzero(mask)
indices = optical.input_indices[selected_rows]

print("INDICES", indices)

interactive.select_scheme(indices, ades)
result = interactive.weights(obs)

for row in selected_rows:
    input_index = int(optical.input_indices[row])
    ra = float(np.rad2deg(result.optical_uncertainties[row, 0]) * 3600.0)
    dec = float(np.rad2deg(result.optical_uncertainties[row, 1]) * 3600.0)
    print(
        "ROW", int(row),
        "INPUT", input_index,
        "SOURCE", result.optical_sources[row],
        "RA", round(ra, 3),
        "DEC", round(dec, 3),
    )
```

```text title="Output"
INDICES [47 48 49 50 51]
ROW 47 INPUT 47 SOURCE ADES RA 0.037 DEC 0.042
ROW 48 INPUT 48 SOURCE ADES RA 0.038 DEC 0.046
ROW 49 INPUT 49 SOURCE ADES RA 0.035 DEC 0.046
ROW 50 INPUT 50 SOURCE ADES RA 0.034 DEC 0.046
ROW 51 INPUT 51 SOURCE ADES RA 0.036 DEC 0.041
```

## 4. Restore the default policy

When you no longer need the overrides, use `restore_default_policy(...)` to return rows to `default_policy`. If you call `restore_default_policy()` with no argument, it returns every row to `default_policy`.

```python
interactive.restore_default_policy([7, 8, 9, 10])
```

## Common Mistakes

- The override methods take `input_index` values from the original observation file, not row positions inside one table.
- `ADESWeightPolicy` can return `NaN` for rows with no reported uncertainty. Do not treat those rows as valid ADES-weight rows.
- `set_manual_optical(...)` expects radians, not arcseconds.
- `WeightResult` stores uncertainties. It also provides inverse-variance weights.

## Next Steps

- Continue to [Inspect Optical Debias Corrections](inspect-optical-debias-corrections.md) when you also need catalog corrections for `/tmp/2025bc10-online-guide.psv`.
- Return to [Load Local ADES Observations](load-local-ades-observations.md) or [Load Online Observations From MPC And JPL](load-online-observations-from-mpc-and-jpl.md) when you still need to adjust `/tmp/2025bc10-online-guide.psv`.
- Continue to [Orbit Determination Workflow](../workflows/orbit-determination-workflow.md) when you want to use the chosen policy in a full solve.
- Use the [Weights API](../api/weights.md) for details on weight policies and `WeightResult`.
