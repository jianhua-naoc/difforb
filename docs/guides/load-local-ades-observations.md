# Load Local ADES Observations

This guide shows how to reopen the local ADES PSV file written in [Load Online Observations From MPC And JPL](load-online-observations-from-mpc-and-jpl.md). It then shows how to view the tables and rows.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- Run [Load Online Observations From MPC And JPL](load-online-observations-from-mpc-and-jpl.md) first.
- The example below uses `/tmp/2025bc10-online-guide.psv`.
- `load_local_observations(...)` accepts ADES PSV files with the `.psv` suffix.

## 1. Load the saved file

Use `load_local_observations(...)` on the PSV file written by the online loader. The local loader rebuilds the mixed `ObservationData` bundle.

```python
from pathlib import Path

from difforb.astrometry import load_local_observations

saved_path = Path("/tmp/2025bc10-online-guide.psv")
obs = load_local_observations(str(saved_path))

print("NAME", obs.name)
print("N_OBS", len(obs))
print("N_OPTICAL", obs.num_optical)
print("N_RADAR", obs.num_radar)
print("T_START", obs.t_start.ut.iso_string)
print("T_END", obs.t_end.ut.iso_string)
```

```text title="Output"
NAME 2025 BC10
N_OBS 807
N_OPTICAL 799
N_RADAR 8
T_START 2025-01-28 13:31:33.500
T_END 2025-04-06 06:39:50.960
```

If you save a new file later, these numbers will match that file.

## 2. Check the modality tables

`ObservationData` keeps optical and radar observations in separate tables. Ground and space optical rows share the `optical` table and are distinguished by `observer_type`.

```python
from difforb.astrometry import load_local_observations

obs = load_local_observations("/tmp/2025bc10-online-guide.psv")

print("OPTICAL", obs.optical)
print("RADAR", obs.radar)
```

```text title="Output"
OPTICAL <OpticalObservationData n_obs=799 ut_start_jd=2460704.063582176 ut_end_jd=2460771.777673148>
RADAR <RadarObservationData n_obs=8 ut_start_jd=2460768.562500000 ut_end_jd=2460771.326388889>
```

`2025 BC10` has optical observations and radar observations.

## 3. Inspect input indices

Each modality table keeps its own `input_indices` array. These values are the row numbers from the original observation file.

```python
from difforb.astrometry import load_local_observations

obs = load_local_observations("/tmp/2025bc10-online-guide.psv")

print("OPTICAL_HEAD", obs.optical.input_indices[:5])
print("RADAR_HEAD", obs.radar.input_indices[:5])
```

```text title="Output"
OPTICAL_HEAD [0 1 2 3 4]
RADAR_HEAD [799 800 801 802 803]
```

The optical table starts at row `0`. The radar table starts at row `799`.

## 4. Show mixed row order

Use `obs.to_dataframe(sort_by="input")` when you want one mixed table in the original file order. The `input_index` column stores the row number from the original observation file.

```python
from difforb.astrometry import load_local_observations

obs = load_local_observations("/tmp/2025bc10-online-guide.psv")
df = obs.to_dataframe(sort_by="input")

subset = df.loc[
    df["input_index"].isin([0, 7, 799]),
    [
        "input_index",
        "t_iso",
        "obs_type",
        "obs_mode",
        "rx_code",
        "tx_code",
        "observer_type",
        "catalog_code",
        "radar_unit",
    ],
]
print(subset.to_string(index=False))
```

```text title="Output"
 input_index                   t_iso obs_type             obs_mode rx_code tx_code observer_type catalog_code radar_unit
           0 2025-01-28 13:31:33.500  optical                  CCD     F51     NaN  fixed-ground       Gaia3E        NaN
           7 2025-02-06 09:33:50.088  optical                  CCD     G96     NaN  fixed-ground        Gaia2        NaN
         799 2025-04-05 19:50:00.000    radar Radar Delay (Center)     -14     -14           NaN          NaN         us
```

These `input_index` values are the row numbers used later by weight and outlier overrides.

## 5. Load another local PSV file

Use the same API for any local ADES PSV file.

```python
from difforb.astrometry import load_local_observations

obs = load_local_observations("path/to/arc.psv")
```

## Common Mistakes

- `load_local_observations(...)` only supports `.psv` ADES PSV files. It does not read legacy MPC1992 files.
- One `ObservationData` bundle always holds one target.
- `ObservationData` is not row-indexable in mixed order. Use `to_dataframe()` when you need one mixed table.

## Next Steps

- Return to [Load Online Observations From MPC And JPL](load-online-observations-from-mpc-and-jpl.md) when you want to refresh `/tmp/2025bc10-online-guide.psv`.
- Continue to [Choose And Override Observation Weights](choose-and-override-observation-weights.md) when you want to set weights for rows from `/tmp/2025bc10-online-guide.psv`.
- Continue to [Inspect Optical Debias Corrections](inspect-optical-debias-corrections.md) when you want to check row-level catalog corrections from `/tmp/2025bc10-online-guide.psv`.
- Read [Weighting And Debiasing Models](../concepts/weighting-and-debiasing-models.md) for the model behind weights and catalog corrections.
- Use the [Observation Data API](../api/observation-data.md) for details on loaders and
  observation containers.
