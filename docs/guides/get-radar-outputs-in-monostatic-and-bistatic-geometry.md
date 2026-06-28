# Get Radar Outputs In Monostatic And Bistatic Geometry

This guide shows how to get DiffOrb radar outputs for 2025 BC10 at DSS-14 receive epochs and compare them with JPL
radar astrometry records.

For the model behind these outputs, read [Light-Time Model](../concepts/light-time-model.md).

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- You need a local planetary SPK kernel. Replace the placeholder path in the snippet with a local file such as
  `de441.bsp`.
- The target must already have a propagated trajectory before `radar_table(...)` can be called.
- The propagated interval must cover the receive epoch and the earlier transmit and target epochs reached by the two-way
  light-time solution.
- `t_rec` is the receive epoch.
- The comparison values below come from the
  [NASA/JPL Small-Body Radar Astrometry API](https://ssd-api.jpl.nasa.gov/doc/sb_radar.html) query for `2025 BC10`.
- Receive epochs before `1962-01-01` are not supported.
- `tx_freq` must be given in `Hz`; the JPL radar table reports frequency in `MHz`.

## 1. Prepare inputs

Use a `BCRS` state for 2025 BC10 at `TDB JD 2460741.5`, Goldstone DSS-14 (`253`), and the 8.56 GHz transmit
frequency reported with the JPL radar records.

Replace the initial state with your own fitted orbit when you need to compare another orbit solution.

```python
from difforb.body import Site, SmallBody
from difforb.core import BCRS, State, Time
from difforb.dynamics import DynamicSystem
from difforb.spk import set_default_ephemeris
from difforb.ephemeris import EphemerisGenerator
from difforb.integrator import NumericalIntegrator

planetary_kernel = "/path/to/your/de441.bsp"
set_default_ephemeris(planetary_kernel)

t0 = Time.from_tdb_jd(2460741.0, 0.5)
state0 = State(
    tdb=t0.tdb(),
    pos=[-1.371967972861946, 0.1083567046426632, 0.0990169921209984],
    vel=[0.01066683288761389, -0.01158077078726909, -0.006522676597241548],
    frame=BCRS,
)

body = SmallBody.create(state0)
force_model = DynamicSystem.from_standard_system().build_force_model()
integrator = NumericalIntegrator(method="IAS15", tol=1e-12)
body = body.propagate(
    t0.tdb(),
    Time.from_utc_date(2025, 4, 6).tdb(),
    force_model,
    integrator,
)

rx = Site.from_code("253").require_ground()
tx_freq = 8.56e9

generator = EphemerisGenerator(body)
print(body.trajectory is not None)
```

```text title="Output"
True
```

Now the target and radar site are ready for the JPL receive epochs.

## 2. Compare monostatic delay

Use `radar_table(...)` with `rx` and `tx_freq`. If `tx` is omitted, DiffOrb uses `rx` as the transmitter, so
the geometry is monostatic.

```python
delay_epoch = Time.from_utc_date(2025, 4, 5, 19, 50, 0.0)
jpl_delay_us = 25268924.08

delay_prediction = generator.radar_table(
    delay_epoch,
    rx=rx,
    tx_freq=tx_freq,
)

print("MODEL_DELAY_US", round(float(delay_prediction.radar_delay), 3))
print("JPL_DELAY_US", jpl_delay_us)
print("DELAY_RESIDUAL_US", round(float(delay_prediction.radar_delay) - jpl_delay_us, 3))
print("MODEL_RANGE_AU", float(delay_prediction.radar_range))
```

```text title="Output"
MODEL_DELAY_US 25268924.179
JPL_DELAY_US 25268924.08
DELAY_RESIDUAL_US 0.099
MODEL_RANGE_AU 0.05063864114659815
```

`DELAY_RESIDUAL_US` is `model - JPL`. This value is tied to the initial orbit, force model, integrator tolerance, SPK
kernel, and JPL record used above.

The returned fields used here are:

- `radar_delay` in microseconds
- `radar_range` in `au`

Both values are two-way quantities reported at the receive epoch.

## 3. Compare monostatic Doppler

```python
doppler_epoch = Time.from_utc_date(2025, 4, 5, 19, 30, 0.0)
jpl_doppler_hz = -245905.543

doppler_prediction = generator.radar_table(
    doppler_epoch,
    rx=rx,
    tx_freq=tx_freq,
)

print("MODEL_DOPPLER_HZ", round(float(doppler_prediction.radar_doppler), 3))
print("JPL_DOPPLER_HZ", jpl_doppler_hz)
print("DOPPLER_RESIDUAL_HZ", round(float(doppler_prediction.radar_doppler) - jpl_doppler_hz, 3))
print("MODEL_RATE_AU_PER_D", float(doppler_prediction.radar_rate))
```

```text title="Output"
MODEL_DOPPLER_HZ -245905.726
JPL_DOPPLER_HZ -245905.543
DOPPLER_RESIDUAL_HZ -0.183
MODEL_RATE_AU_PER_D 0.004973978579719754
```

`DOPPLER_RESIDUAL_HZ` is also `model - JPL`.

The returned fields used here are:

- `radar_doppler` in `Hz`
- `radar_rate` in `au / day`

Both values are two-way quantities reported at the receive epoch.

## Common Mistakes

- `t_rec` is the receive epoch, not the transmit epoch.
- `tx_freq` must be given in `Hz`; multiply JPL `MHz` values by `1e6`.
- If `tx` is omitted, DiffOrb uses `rx`, so the call is monostatic.
- `radar_delay`, `radar_doppler`, `radar_range`, and `radar_rate` are all two-way quantities.
- Residuals depend on the orbit, force model, integrator tolerance, and SPK kernel.
- Receive epochs before `1962-01-01` are not supported.
- The propagated interval must cover the earlier epochs reached by the two-way light-time solution.

## Next Steps

- Read [Light-Time Model](../concepts/light-time-model.md) if you want the model behind these outputs.
- Continue to [Propagate A SmallBody And Evaluate Dense Trajectories](propagate-a-smallbody-and-evaluate-dense-trajectories.md) if you want the propagation setup by itself.
- Continue to [Run Differential Correction From An Initial Orbit](run-differential-correction-from-an-initial-orbit.md) if you need a fitted orbit before comparing residuals.
- Use the [Ephemeris API](../api/ephemeris.md) for details on radar tables.
