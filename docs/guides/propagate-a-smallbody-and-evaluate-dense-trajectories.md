# Propagate A SmallBody And Evaluate Dense Trajectories

This guide shows how to propagate a `SmallBody` over a chosen interval and query position and velocity from the dense trajectory inside the solved interval.

Use [Configure Force Models And Dynamic Systems](configure-force-models-and-dynamic-systems.md) first when you need a custom force model; the propagation path here uses the standard built-in force model.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- You need a local planetary SPK kernel. Replace the placeholder path in the snippet with a local file such as
  `de441.bsp`.
- The initial orbit below is already a canonical `BCRS` state.

For the model-level background, read [Numerical Integrators And Dense Trajectories](../concepts/numerical-integrators-and-dense-trajectories.md).

## 1. Build the initial `SmallBody`

Start from a canonical `BCRS` state.

```python
from difforb.body import SmallBody
from difforb.core import BCRS, State, Time

t0 = Time.from_tdb_date(2025, 1, 2)

state0 = State(
    tdb=t0.tdb(),
    pos=[1.685775738339898, -1.336388854313325, -0.2144927004440800],
    vel=[0.008995712853117517, 0.006985684417802803, 0.004020851173846060],
    frame=BCRS,
)

body = SmallBody.create(state0)
print(body)
```

```text title="Output"
<SmallBody shape=() epoch_jd=2460677.500000000 frame=BCRS mag_model=none trajectory=uninitialized>
```

The orbit is ready. The trajectory is not ready yet.

## 2. Build model and integrator

Load the planetary kernel. Then build the standard force model and one integrator.

```python
from difforb.dynamics import DynamicSystem
from difforb.spk import set_default_ephemeris
from difforb.integrator import NumericalIntegrator

planetary_kernel = "/path/to/your/de441.bsp"
set_default_ephemeris(planetary_kernel)

force_model = DynamicSystem.from_standard_system().build_force_model()
integrator = NumericalIntegrator(method="IAS15", tol=1e-12)

print(force_model)
print(integrator)
```

This path uses the standard major-body background and the `IAS15` integrator. If you want another integrator, change
`method` to `"DOPRI8"` or `"DOPRI5"`.

## 3. Propagate the orbit

Choose the end epoch and call `propagate(...)`.

```python
from difforb.core import Time

t_end = Time.from_tdb_date(2025, 2, 15)
body = body.propagate(t0.tdb(), t_end.tdb(), force_model, integrator)

print(body)
print(body.trajectory is not None)
```

```text title="Output"
<SmallBody shape=() epoch_jd=2460677.500000000 frame=BCRS mag_model=none trajectory=ready>
True
```

After this call, the dense trajectory is stored in `body.trajectory`.

## 4. Check whether one epoch is covered

Use `is_covered(...)` before you query.

```python
inside = Time.from_tdb_date(2025, 1, 20)
outside = Time.from_tdb_date(2025, 3, 1)

print(bool(body.trajectory.is_covered(inside.tdb().jd1, inside.tdb().jd2)))
print(bool(body.trajectory.is_covered(outside.tdb().jd1, outside.tdb().jd2)))
```

```text title="Output"
True
False
```

The first epoch is inside the solved interval. The second is outside.

## 5. Query the dense trajectory directly

Use `body.trajectory.evaluate(...)` when you want canonical `BCRS` position and velocity.

```python
pos_i, vel_i = body.trajectory.evaluate(inside.tdb().jd1, inside.tdb().jd2)

print(pos_i)
print(vel_i)
```

```text title="Output"
[ 1.8396007  -1.20465751 -0.14124085]
[0.00809158 0.00763366 0.00411113]
```

These are the interpolated Cartesian position and velocity at `2025-01-20 TDB`.

## 6. Query through `SmallBody`

Use `body.state(...)` when you want a full `State` object.

```python
state_i = body.state(inside.tdb(), frame=BCRS)

print(state_i.frame.name)
print(state_i.pos)
print(state_i.vel)
```

```text title="Output"
BCRS
[ 1.8396007  -1.20465751 -0.14124085]
[0.00809158 0.00763366 0.00411113]
```

The values match the direct trajectory interpolation. The difference is that `body.state(...)` returns a `State` object.

## Select an execution device

Pass a JAX device to place the propagation inputs, including force-model and ephemeris arrays, on that device. The numerical model remains FP64. The following uses a CPU and also runs on a Mac; use `jax.devices("cuda")[0]` in a Linux environment with a working JAX CUDA installation to select an NVIDIA GPU.

```python
import equinox as eqx
import jax

device = jax.devices("cpu")[0]
body_on_device = body.propagate(
    t0.tdb(), t_end.tdb(), force_model, integrator, device=device,
)

# Query inputs must use placement consistent with the trajectory.
arrays, metadata = eqx.partition(inside.tdb(), eqx.is_array)
query_on_device = eqx.combine(jax.device_put(arrays, device), metadata)
state_on_device = body_on_device.state(query_on_device)

assert state_on_device.pos.dtype.name == "float64"
assert state_on_device.pos.devices() == {device}
```

The original body and force model retain their arrays. Omitting `device` preserves the existing JAX placement behavior. No precision conversion or automatic transfer of subsequent query inputs is performed.

Propagation remains compatible with `jit`, `jacfwd`, and `jvp`. Capture the device in a closure or treat it as a static argument when compiling an enclosing function. Place the enclosing function's inputs consistently: an inner `device` argument does not override the enclosing computation's placement constraints. CPU and CUDA results should be compared with numerical tolerances, rather than requiring bitwise equality.

For ephemeris tables, also place the generator's Sun and Earth arrays and the query inputs on that device. `EphemerisGenerator` uses the registered default ephemeris and has no separate device option.

```python
from difforb.body.site import Site
from difforb.ephemeris.generator import EphemerisGenerator

generator = EphemerisGenerator(body_on_device)
observer = Site.from_code("568").require_ground()
arrays, metadata = eqx.partition((generator, inside, observer), eqx.is_array)
generator, query, observer = eqx.combine(jax.device_put(arrays, device), metadata)
table = generator.optical_table(query, observer)

assert table.astrometric_ra.dtype.name == "float64"
assert table.astrometric_ra.devices() == {device}
```

## Common Mistakes

- `body.trajectory` is `None` until you call `propagate(...)`.
- `body.state(...)` fails if the query epoch is outside the solved interval.
- `trajectory.evaluate(...)` takes split Julian date parts, not a `Time` object.
- `tol` is a shorthand tolerance. For `IAS15`, it fills `atol` when `atol` is not given. For `DOPRI8` and `DOPRI5`,
  it fills missing `rtol` and `atol`. Use explicit `rtol` and `atol` when you need separate relative and absolute
  tolerances.

## Next Steps

- Continue to [Configure Force Models And Dynamic Systems](configure-force-models-and-dynamic-systems.md) if you want a
  custom force model.
- Continue to [Ephemeris Products Workflow](../workflows/ephemeris-products-workflow.md) if you want observer products
  from the propagated orbit.
- Use the [Body API](../api/body.md), [Dynamics API](../api/dynamics.md), and [Integrator API](../api/integrator.md)
  for details on propagation objects.
