# Get Earth Rotation Quantities And Matrices

This guide shows how to read Earth-rotation quantities from a `Time` object, including polar-motion coordinates, the Earth Rotation Angle (`ERA`), and the `CIO`-based and equinox-based matrices used by the `ITRS -> GCRS` transformation.

For the model behind these quantities, read [Earth Rotation And Terrestrial Geometry](../concepts/earth-rotation-and-terrestrial-geometry.md).

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- No SPK kernel is required for this guide.
- The modern quantities in this guide use the default Earth Orientation Parameters (`EOP`) data loaded through
  `difforb.core.eop`. See [Configure Earth Orientation Data](configure-earth-orientation-data.md) when you need to check
  or update that file.

## 1. Read scalar quantities

Start from one epoch, then read the polar-motion coordinates, the `dPsi/dEps` nutation-angle correction terms from the
`EOP` file, and the `ERA`.

```python
import jax.numpy as jnp

from difforb.core import Time

t = Time.from_utc_date(2025, 1, 2, 3, 4, 5.678)

print("XPOLE_ARCSEC", round(float(jnp.rad2deg(t.xpole) * 3600.0), 6))
print("YPOLE_ARCSEC", round(float(jnp.rad2deg(t.ypole) * 3600.0), 6))
print("COR_DELTA_LONGITUDE_MAS", round(float(jnp.rad2deg(t.cor_delta_longitude) * 3600.0 * 1000.0), 6))
print("COR_DELTA_OBLIQUITY_MAS", round(float(jnp.rad2deg(t.cor_delta_obliquity) * 3600.0 * 1000.0), 6))
print("ERA_RAD", round(float(t.ERA), 12))
```


```text title="Output"
XPOLE_ARCSEC 0.14286
YPOLE_ARCSEC 0.304986
COR_DELTA_LONGITUDE_MAS 0.755613
COR_DELTA_OBLIQUITY_MAS -0.309439
ERA_RAD 2.578107797847
```

Use these fields as follows:

- `xpole` and `ypole` are the polar-motion coordinates from the `EOP` file, stored in radians.
- `cor_delta_longitude` and `cor_delta_obliquity` are the `dPsi/dEps` nutation-angle correction terms from the same `EOP` file, also in radians.
- `ERA` is the Earth rotation angle. It is a scalar angle, not a matrix.

## 2. Get the `CIO`-based matrices

DiffOrb uses the `CIO`-based path when evaluating ground-site positions in `GCRS`. The `Time` object exposes the matrix pieces directly.

- `inversed_polar_motion_matrix`: International Terrestrial Reference System (`ITRS`) to Terrestrial Intermediate Reference System (`TIRS`)
- `polar_motion_matrix`: `TIRS` to `ITRS`
- `gcrs_to_cirs_matrix`: Geocentric Celestial Reference System (`GCRS`) to Celestial Intermediate Reference System (`CIRS`)
- `cirs_to_gcrs_matrix`: `CIRS` to `GCRS`

```python
import jax.numpy as jnp

itrs_to_tirs = t.inversed_polar_motion_matrix
tirs_to_itrs = t.polar_motion_matrix
gcrs_to_cirs = t.gcrs_to_cirs_matrix
cirs_to_gcrs = t.cirs_to_gcrs_matrix

polar_roundtrip = itrs_to_tirs @ tirs_to_itrs
cio_roundtrip = gcrs_to_cirs @ cirs_to_gcrs

print("ITRS_TO_TIRS_ROW0", itrs_to_tirs[0])
print("CIRS_TO_GCRS_ROW0", cirs_to_gcrs[0])
print("POLAR_ROUNDTRIP_MAX", f"{float(jnp.abs(polar_roundtrip - jnp.eye(3)).max()):.3e}")
print("CIO_ROUNDTRIP_MAX", f"{float(jnp.abs(cio_roundtrip - jnp.eye(3)).max()):.3e}")
```


```text title="Output"
ITRS_TO_TIRS_ROW0 [ 1.00000000e+00  5.79983012e-11 -6.92603270e-07]
CIRS_TO_GCRS_ROW0 [ 9.99997047e-01 -8.50292043e-08  2.43017187e-03]
POLAR_ROUNDTRIP_MAX 2.220e-16
CIO_ROUNDTRIP_MAX 7.763e-20
```

The roundtrip checks are a quick sanity check. The two matrix pairs are inverses of each other to normal floating-point precision.

## 3. Get equinox matrices

DiffOrb also exposes the equinox-based pieces. Use them when you want to compare with classical references or with equinox-based formulas.

```python
precession_bias = t.precession_bias_matrix
nutation = t.nutation_matrix

print("PRECESSION_BIAS_ROW0", precession_bias[0])
print("NUTATION_ROW0", nutation[0])
```


```text title="Output"
PRECESSION_BIAS_ROW0 [ 0.99998142 -0.0055914  -0.00242929]
NUTATION_ROW0 [ 1.00000000e+00 -1.49506158e-06 -6.48087325e-07]
```

Use these matrix properties as follows:

- `precession_bias_matrix` maps `GCRS` to the mean equator and equinox of date.
- `nutation_matrix` maps the mean equator and equinox of date to the true equator and equinox of date.

## Common Mistakes

- `polar_motion_matrix` and `inversed_polar_motion_matrix` point in opposite directions.
- `ERA` is a scalar angle. It is not a `3 x 3` matrix.
- `xpole`, `ypole`, `cor_delta_longitude`, and `cor_delta_obliquity` are stored in radians. Convert them only for display.
- Modern Earth-rotation corrections depend on the default `EOP` data.

## Next Steps

- Continue to [Create A Ground Site And Get Its GCRS State](create-a-groundsite-and-get-its-gcrs-state.md) when you want a full ground-site conversion.
- Return to [Convert Between UTC, TT, TDB, UT1](convert-between-utc-tt-tdb-ut1.md) when you need the timescale side again.
- Return to [Configure Earth Orientation Data](configure-earth-orientation-data.md) when you need to check or update the
  local `EOP` file.
- Read [Earth Rotation And Terrestrial Geometry](../concepts/earth-rotation-and-terrestrial-geometry.md) for the model behind the `CIO`-based and equinox-based paths.
- Use the [Core API](../api/core.md) and [Time API](../api/time.md) for details on the returned
  Earth-rotation values and time views.
