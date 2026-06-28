# Configure Earth Orientation Data

This guide shows how to check the local Earth Orientation Parameter (`EOP`) data, update it when network access is
available, and verify the values used by one modern epoch.

For the data rules behind these values, read
[Earth Orientation Parameters](../concepts/earth-orientation-parameters.md).

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- No `SPK` kernel is required for this guide.
- Network access is required only when you update the local `EOP` file.

Check the DiffOrb data directory with:

```bash
python -m difforb.data dir
```

## 1. Check local EOP coverage

Use `load_default_eop_file()` when you want to check the local file without downloading anything. If the local file is
missing, this call raises `FileNotFoundError`.

```python
from difforb.core import Time
from difforb.core.eop import load_default_eop_file

eop = load_default_eop_file()

start = Time.from_utc_jd(float(eop.final_date_range[0]), 0.0)
end = Time.from_utc_jd(float(eop.final_date_range[1]), 0.0)

print("N_SAMPLES", len(eop.tt_jds))
print("FINAL_START_UTC", start.utc.iso_string)
print("FINAL_END_UTC", end.utc.iso_string)
```

```text title="Output"
N_SAMPLES 23461
FINAL_START_UTC 1962-01-01 00:00:00.000
FINAL_END_UTC 2026-03-26 00:00:00.000
```

The exact end date depends on the local file. If recent observations matter, make sure the file covers the dates you
need.

## 2. Update the default EOP file

Use `update_eop()` when the local file is missing or stale. This downloads the default IERS `EOP` file and refreshes
the cached table for the current Python process.

```python
from difforb.core.eop import update_eop

eop = update_eop()
print("UPDATED_FINAL_END_JD", float(eop.final_date_range[1]))
```

This step requires network access. If the network is blocked, keep the existing local file or update it outside the
current environment.

The same update can be run from the command line:

```bash
python -m difforb.data install eop --force
```

## 3. Verify one epoch

After the file loads, check one modern epoch. This confirms the `UT1 - UTC` correction, polar motion, and small
`dPsi/dEps` nutation-angle correction terms used by the time and Earth-rotation layers.

```python
import jax.numpy as jnp

from difforb.core import Time

t = Time.from_utc_date(2025, 1, 2, 3, 4, 5.678)

print("UT1_MINUS_UTC_S", f"{float((t.ut1.jd - t.utc.jd) * 86400.0):.6f}")
print("XPOLE_ARCSEC", f"{float(jnp.rad2deg(t.xpole) * 3600.0):.6f}")
print("YPOLE_ARCSEC", f"{float(jnp.rad2deg(t.ypole) * 3600.0):.6f}")
print("DELTA_LONGITUDE_MAS", f"{float(jnp.rad2deg(t.cor_delta_longitude) * 3600.0 * 1000.0):.6f}")
print("DELTA_OBLIQUITY_MAS", f"{float(jnp.rad2deg(t.cor_delta_obliquity) * 3600.0 * 1000.0):.6f}")
```

```text title="Output"
UT1_MINUS_UTC_S 0.046469
XPOLE_ARCSEC 0.142860
YPOLE_ARCSEC 0.304986
DELTA_LONGITUDE_MAS 0.755613
DELTA_OBLIQUITY_MAS -0.309439
```

These values are small but important. The polar-motion coordinates are shown in arcseconds. The `dPsi/dEps`
nutation-angle correction terms are shown in milliarcseconds.

## Common Mistakes

- Do not assume an offline environment can update the `EOP` file.
- Do not use a stale `EOP` file for recent high-precision ground geometry.
- Do not confuse `UT1 - UTC` with leap seconds. Leap seconds define `TAI - UTC`; `UT1 - UTC` comes from Earth rotation.
- Convert polar-motion coordinates and `dPsi/dEps` correction terms only for display. The runtime values are stored as angles.

## Next Steps

- Continue to [Create And Convert Time Objects](create-and-convert-time-objects.md) when you need basic time
  constructors.
- Continue to [Convert Between UTC, TT, TDB, UT1](convert-between-utc-tt-tdb-ut1.md) when you need time-scale offsets.
- Continue to [Get Earth Rotation Quantities And Matrices](get-earth-rotation-quantities-and-matrices.md) when you need
  the Earth-rotation matrices for one epoch.
- Continue to [Create A Ground Site And Get Its GCRS State](create-a-groundsite-and-get-its-gcrs-state.md) when you need
  ground-site geometry.
- Use the [Earth Orientation Parameters API](../api/eop.md) and [Time API](../api/time.md) for details on EOP data and
  time views.
