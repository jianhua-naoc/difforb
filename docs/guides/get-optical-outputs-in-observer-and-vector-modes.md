# Get Optical Outputs In Observer And Vector Modes

This guide shows how to get DiffOrb optical outputs for one propagated target, one observer, and one set of observation times. It first gets the observer-mode outputs, then gets the vector-mode outputs.

For the product correction levels, read [Ephemeris Products](../concepts/ephemeris-products.md). For the one-way
light-time model, read [Light-Time Model](../concepts/light-time-model.md).

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- You need a local planetary SPK kernel. Replace the placeholder path in the snippet with a local file such as
  `de441.bsp`.
- The target must already have a propagated trajectory before `optical_table(...)` or `vector_table(...)` can
  be called.
- The propagated interval must cover the observation times and the earlier emission times used by the light-time
  correction.
- Ground observers also use the default Earth Orientation Parameters (`EOP`) data loaded through `difforb.core.eop`.

## 1. Prepare inputs

This example uses one fixed `BCRS` state for `85472 Xizezong`, one ground observer at Xinglong Station (`327`), and
three daily `UTC` epochs at `10:00`.

```python
import jax.numpy as jnp

from difforb.body import Site, SmallBody
from difforb.core import BCRS, State, Time
from difforb.dynamics import DynamicSystem
from difforb.spk import set_default_ephemeris
from difforb.ephemeris import EphemerisGenerator
from difforb.integrator import NumericalIntegrator

planetary_kernel = "/path/to/your/de441.bsp"
set_default_ephemeris(planetary_kernel)

t0 = Time.from_tdb_jd(2461000.0, 0.5)
state0 = State(
    tdb=t0.tdb(),
    pos=[1.685775738339898, -1.336388854313325, -0.2144927004440800],
    vel=[0.008995712853117517, 0.006985684417802803, 0.004020851173846060],
    frame=BCRS,
)

body = SmallBody.create(state0)
force_model = DynamicSystem.from_standard_system().build_force_model()
integrator = NumericalIntegrator(method="IAS15", tol=1e-12)
body = body.propagate(
    t0.tdb(),
    Time.from_utc_date(2025, 12, 10).tdb(),
    force_model,
    integrator,
)

observer = Site.from_code("327").require_ground()
t_obs0 = Time.from_utc_date(2025, 11, 22, 10)
t_obs = t_obs0 + jnp.arange(0.0, 3.0, 1.0)

generator = EphemerisGenerator(body)
print(body.trajectory is not None)
```

```text title="Output"
True
```

Now the target, observer, and time grid are ready.

## 2. Get observer-mode outputs

Use `optical_table(...)` to get the observer-mode outputs.

```python
optical = generator.optical_table(t_obs, observer)

print("ASTROMETRIC0", float(optical.astrometric_ra[0]), float(optical.astrometric_dec[0]))
print("APPARENT0", float(optical.apparent_ra[0]), float(optical.apparent_dec[0]))
print("HORIZON0", float(optical.azimuth[0]), float(optical.elevation[0]))
print("GEOMETRY0", float(optical.delta[0]), float(optical.elongation[0]))
```

```text title="Output"
ASTROMETRIC0 299.83143836309944 -12.696409344776862
APPARENT0 300.1892926316854 -12.626111083062122
HORIZON0 213.35676005214424 30.540490936519447
GEOMETRY0 2.4887786016308406 59.64009403170461
```

The four output lines are, in order:

- astrometric `RA/Dec` in degrees
- apparent `RA/Dec` in degrees
- topocentric `azimuth/elevation` in degrees
- `delta` in `au` and `elongation` in degrees

For a ground observer, you can also turn on atmospheric refraction:

```python
optical_refracted = generator.optical_table(
    t_obs,
    observer,
    apply_refraction=True,
)

print("ELEVATION0_NOREFR", float(optical.elevation[0]))
print("ELEVATION0_REFR", float(optical_refracted.elevation[0]))
```

```text title="Output"
ELEVATION0_NOREFR 30.540490936519447
ELEVATION0_REFR 30.567791484788568
```

These are observer-mode quantities. `azimuth/elevation` and refraction only make sense for ground observers. For a space
observer, the main observer-mode outputs are `RA/Dec` and the auxiliary geometry values.

If you want the refraction model to use your own weather inputs, pass a `WeatherParams` object:

```python
from difforb.astrometry.reduction import WeatherParams

weather = WeatherParams(
    temperature=293.15,
    pressure=980.0,
    humidity=0.4,
    wavelength=0.65,
)

optical_custom_weather = generator.optical_table(
    t_obs,
    observer,
    apply_refraction=True,
    weather=weather,
)

print("ELEVATION0_STD", float(optical_refracted.elevation[0]))
print("ELEVATION0_CUSTOM", float(optical_custom_weather.elevation[0]))
```

```text title="Output"
ELEVATION0_STD 30.567791484788568
ELEVATION0_CUSTOM 30.56582290574301
```

Here `temperature` is in `K`, `pressure` is in `mb` (`hPa`), `humidity` is a fraction from `0` to `1`, and `wavelength`
is in `um`. The `weather` argument only matters when `apply_refraction=True` for a ground observer.

## 3. Get vector-mode outputs

Use `vector_table(...)` to get the vector-mode outputs.

```python
import numpy as np

vectors = generator.vector_table(t_obs, observer)

print("LT0", float(vectors.light_time[0]))
print("GEOM_POS0", np.asarray(vectors.geometric.pos[0], dtype=float))
print("ASTRO_POS0", np.asarray(vectors.astrometric.pos[0], dtype=float))
print("APP_POS0", np.asarray(vectors.apparent.pos[0], dtype=float))
```

```text title="Output"
LT0 0.01437398668065527
GEOM_POS0 [ 1.20789906 -2.10610479 -0.54693846]
ASTRO_POS0 [ 1.20777076 -2.10620599 -0.54699638]
APP_POS0 [ 1.20765762 -2.10627137 -0.54699444]
```

Here `light_time` is in days. The three position vectors are in `au`.

The `apparent` vector is not the same as the apparent `RA/Dec` from `optical_table(...)`. It includes the
one-way light-time correction and stellar aberration. It does not rotate the direction to the true equator and equinox
of date.

For the full model behind this difference,
see [Ephemeris Products](../concepts/ephemeris-products.md).

## Common Mistakes

- `body.trajectory` must be ready before either optical method can run.
- The propagated interval must cover both the observation times and the earlier emission times.
- `azimuth/elevation` only make sense for ground observers.
- `weather` only affects the result when `apply_refraction=True` for a ground observer.
- The `apparent` vector is not the same as the apparent `RA/Dec` from `optical_table(...)`.

## Next Steps

- Continue to [Propagate A SmallBody And Evaluate Dense Trajectories](propagate-a-smallbody-and-evaluate-dense-trajectories.md) if you want the propagation setup by itself.
- Read [Ephemeris Products](../concepts/ephemeris-products.md) if you want the definitions of geometric, astrometric,
  and apparent optical quantities.
- Use the [Ephemeris API](../api/ephemeris.md) for details on optical and vector tables.
