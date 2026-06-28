# Create A Ground Site And Get Its GCRS State

This guide shows how to create a ground `Site` from common ground-observer inputs and evaluate the resulting site in canonical `GCRS` at a chosen epoch.

For the Earth-rotation quantities and matrices behind the conversion, read [Get Earth Rotation Quantities And Matrices](get-earth-rotation-quantities-and-matrices.md).

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- The `ITRS -> GCRS` site transformation uses the default EOP data.

## 1. Create from a code

`Site.from_code(...).require_ground()` is the shortest path when you already have an MPC or JPL observatory code.

```python
import jax.numpy as jnp

from difforb.body import Site

ground = Site.from_code("F51").require_ground()

print("GROUND_LON_DEG", float(jnp.rad2deg(ground.itrs_lon)))
```


```text title="Output"
GROUND_LON_DEG 203.74409
```

## 2. Create from a roving observer key

Use `Site.from_code("code @ lon_deg, lat_deg, alt_m")` when the observatory code is a roving ground code and the observation row supplies the WGS84 longitude, latitude, and ellipsoid height. The `@` payload is accepted only for codes that the observatory-code table marks as roving sites.

```python
import jax.numpy as jnp

from difforb.body import Site

roving = Site.from_code("247 @ 120.0, 30.0, 1000.0").require_ground()

print("ROVING_IS_GROUND", bool(roving.is_ground))
print("ROVING_IS_ROVING", bool(roving.is_roving_ground))
print("ROVING_LON_DEG", round(float(jnp.rad2deg(roving.itrs_lon)), 6))
print("ROVING_LAT_DEG", round(float(jnp.rad2deg(roving.ground_itrs.geodetic_lat)), 6))
print("ROVING_ALT_M", round(float(roving.ground_itrs.geodetic_alt), 3))
```


```text title="Output"
ROVING_IS_GROUND True
ROVING_IS_ROVING True
ROVING_LON_DEG 120.0
ROVING_LAT_DEG 30.0
ROVING_ALT_M 1000.0
```

## 3. Create from geodetic coordinates

Use `Site.from_geodetic(...)` when your source data already provides longitude, latitude, and ellipsoid height.

```python
import jax.numpy as jnp

from difforb.body import Site

ground = Site.from_geodetic(120.0, 30.0, 1000.0).require_ground()

print("GEODETIC_LON_DEG", round(float(jnp.rad2deg(ground.itrs_lon)), 6))
print("GEODETIC_LAT_DEG", round(float(jnp.rad2deg(ground.ground_itrs.geodetic_lat)), 6))
print("GEODETIC_ALT_M", round(float(ground.ground_itrs.geodetic_alt), 3))
```


```text title="Output"
GEODETIC_LON_DEG 120.0
GEODETIC_LAT_DEG 30.0
GEODETIC_ALT_M 1000.0
```

## 4. Create from geocentric constants

Use `Site.from_geocentric(...)` when your source data is already published as Minor Planet Center style geocentric constants `rho cos(phi')` and `rho sin(phi')`.

```python
import jax.numpy as jnp

from difforb.body import Site

ground = Site.from_geocentric(120.0, 0.75, 0.4330127019).require_ground()

print("GEOCENTRIC_LON_DEG", round(float(jnp.rad2deg(ground.itrs_lon)), 6))
print("GEOCENTRIC_POS_M", jnp.round(ground.ground_itrs.pos, 3))
```


```text title="Output"
GEOCENTRIC_LON_DEG 120.0
GEOCENTRIC_POS_M [-2391801.375  4142721.503  2761814.335]
```

## 5. Evaluate the site in canonical `GCRS`

All ground-site constructors return the same `Site` object type, so the `ground.state(...)` call is the same regardless of how the site was created. The method accepts a `Time` object and derives the required `UT1` quantities internally.

```python
from difforb.body import Site
from difforb.core import GCRS, Time

ground = Site.from_geodetic(120.0, 30.0, 1000.0).require_ground()
t = Time.from_utc_date(2025, 1, 2, 3, 4, 5.678)
site_gcrs = ground.state(t, frame=GCRS)
print("SITE_POS", site_gcrs.pos)
print("SITE_VEL", site_gcrs.vel)
```


```text title="Output"
SITE_POS [-1.42228188e-06 -3.69298055e-05  2.12007214e-05]
SITE_VEL [ 2.32676039e-04 -9.28549867e-06 -5.65110289e-07]
```

## Common Mistakes

- `Site.from_code(...)` accepts `@` coordinate payloads only for observatory codes marked as roving sites. Use `Site.from_geodetic(...)` for custom fixed ground coordinates.
- Space observer keys use `#` payloads and are outside this ground-site guide. Read [Observer Site Keys And Observer Types](../concepts/observer-site-classes-and-observer-types.md) for the full observer-key contract.
- `Site.from_geodetic(...)` expects longitude and latitude in degrees, and altitude in meters.
- `Site.from_geocentric(...)` expects geocentric observatory constants, not geodetic latitude or Cartesian coordinates.
- `Site.state(...)` takes a `Time` object; the needed `UT1` quantities are derived internally.
- `Site.state(..., frame=GCRS)` is the baseline output for fixed Earth sites.

## Next Steps

- Return to [Get Earth Rotation Quantities And Matrices](get-earth-rotation-quantities-and-matrices.md) when you want the matrix pieces behind this conversion.
- Continue to [Transform State Between Frames](transform-state-between-frames.md) when the site state needs a different reference frame.
- Continue to [Convert Between UTC, TT, TDB, UT1](convert-between-utc-tt-tdb-ut1.md) when you want to tighten the observing-time side of the workflow.
- Use the [Body API](../api/body.md) and [State API](../api/state.md) for details on `Site`
  and returned states.
