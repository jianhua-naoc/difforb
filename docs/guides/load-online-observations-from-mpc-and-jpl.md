# Load Online Observations From MPC And JPL

This guide shows how to fetch online observations for `2025 BC10` from MPC and JPL and save them to a local ADES PSV file.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- This guide requires internet access to the MPC Observations API and the JPL small-body radar API.
- Online results can change. The output below is one reference run from `2026-04-23`.

## 1. Fetch online observations

Use `load_online_observations(...)` with a designation or name accepted by MPC. The same value is also sent to the JPL radar service.

```python
from difforb.astrometry import load_online_observations

obs = load_online_observations("2025 BC10")

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

## 2. Save the result to a local PSV file

Pass `save_path=` to save the fetched observations as ADES pipe-separated values (PSV). The path must use the `.psv` suffix.

```python
from pathlib import Path

from difforb.astrometry import load_online_observations

saved_path = Path("/tmp/2025bc10-online-guide.psv")
obs = load_online_observations("2025 BC10", save_path=str(saved_path))

print("FILE_EXISTS", saved_path.exists())
print("SUFFIX", saved_path.suffix)
print("N_OBS", len(obs))
```

```text title="Output"
FILE_EXISTS True
SUFFIX .psv
N_OBS 807
```

The next three guides use this same PSV file.

## Common Mistakes

- The online data can change, so the counts and date bounds can change too.
- `save_path=` requires a `.psv` filename.
- The returned `ObservationData` holds data for one target only.

## Next Steps

- Continue to [Load Local ADES Observations](load-local-ades-observations.md) to reopen `/tmp/2025bc10-online-guide.psv`.
- Continue to [Choose And Override Observation Weights](choose-and-override-observation-weights.md) to set weights for rows from `/tmp/2025bc10-online-guide.psv`.
- Continue to [Inspect Optical Debias Corrections](inspect-optical-debias-corrections.md) to inspect optical catalog corrections from `/tmp/2025bc10-online-guide.psv`.
- Continue to [Orbit Determination Workflow](../workflows/orbit-determination-workflow.md) when the online observations should feed a complete orbit-determination run.
- Use the [Observation Data API](../api/observation-data.md) for details on loaders and
  observation containers.
