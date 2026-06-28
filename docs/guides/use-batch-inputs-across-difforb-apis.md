# Use Batch Inputs Across DiffOrb APIs

This guide shows how to use batch inputs with common DiffOrb APIs. It also shows how to inspect shapes, slice one row,
and choose point-wise or grid calls.

For the shape rules, read [Batch Inputs And Shapes](../concepts/batch-inputs-and-shapes.md).

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- The `Time`, `State`, and ground `Site` examples do not require an SPK kernel.
- The `SmallBody` and `EphemerisGenerator` examples require a local planetary SPK kernel. Replace the placeholder path
  with a local file such as `de441.bsp`.
- The site examples use observatory codes `568` and `253`.

## 1. Common batch-aware APIs

Common public batch-aware APIs include:

| Interface | Batch source | Output shape |
| --- | --- | --- |
| `Time.from_*` constructors | calendar or split-Julian-date inputs | broadcast input shape |
| `TimeDelta(...)` | scalar or array time intervals | broadcast input shape |
| `State(...)` and `State.from_array(...)` | `tdb`, `pos`, `vel`, or stacked arrays | leading state rows |
| `State.to(...)` | a batched state and one output frame | converted state batch |
| `KepElement(...)` and `KepElement.from_*` constructors | element arrays or batched `State` input | element batch |
| `KepElement.state()` | element batch | state batch |
| `Site.from_code(...)` | observatory-code arrays | site batch |
| `Site.from_geodetic(...)`, `Site.from_geocentric(...)`, or `Site.from_itrs(...)` | ground-site coordinate arrays | site batch |
| `Site.from_gcrs(...)` or `Site.from_state(...)` | space-observer `GCRS` states | site batch |
| `Site.state(...)` | site and epoch batches | site-time batch |
| `EphemerisBody.state(...)` | SPK body and `TDB` epoch batches | epoch batch |
| `SmallBody.create(...)` | batched `State` or `KepElement` input | target batch |
| `SmallBody.propagate(...)` | target batch and time bounds | propagated target batch |
| `SmallBody.state(...)` | target batch and query epochs | target-time batch |
| `EphemerisGenerator.vector_table(...)` | target, observer, and observation-time batches | observer-product batch |
| `EphemerisGenerator.optical_table(...)` | target, observer, and observation-time batches | optical-product batch |
| `EphemerisGenerator.radar_table(...)` | target, receiver or transmitter geometry, frequency, and receive-time batches | radar-product batch |
| `EphemerisGenerator.elements_table(...)` | target and `TDB` epoch batches | target-time batch |
| `EphemerisGenerator.find_apsides(...)` | target and search-interval batches | apsides-event batch |
| `EphemerisGenerator.find_close_approaches(...)` | target and search-interval batches | close-approach-event batch |

Orbit-determination APIs are not listed here as batch APIs. Orbit determination is not only a numerical calculation. It
also involves many decisions, such as data selection, outlier handling, model changes, staged fitting, and result review.
For that reason, DiffOrb treats orbit determination as a one-problem-at-a-time workflow, not as a batch-vectorized user
workflow.

## 2. Build a batched state

Pass lists or arrays into the usual constructors. In this example, `pos` stores three Cartesian positions, so its shape
is `(3, 3)`: three states, each with three components. The single velocity vector broadcasts to the same three states.

```python
import jax.numpy as jnp

from difforb.core import BCRS, State, Time

t = Time.from_tdb_date(
    [2025, 2025, 2025],
    [1, 1, 1],
    [2, 3, 4],
)

state = State(
    tdb=t.tdb(),
    pos=jnp.array([
        [1.0, 0.0, 0.0],
        [1.1, 0.1, 0.0],
        [1.2, 0.2, 0.1],
    ]),
    vel=[0.01, 0.02, 0.03],
    frame=BCRS,
)

print("TIME_SHAPE", t.shape)
print("STATE_SHAPE", state.shape)
print("POS_SHAPE", state.pos.shape)
print("VEL_SHAPE", state.vel.shape)
print("ROW1_SHAPE", state[1].shape)
print("ROW1_ARRAY", state[1].array)
```

```text title="Output"
TIME_SHAPE (3,)
STATE_SHAPE (3,)
POS_SHAPE (3, 3)
VEL_SHAPE (3, 3)
ROW1_SHAPE ()
ROW1_ARRAY [1.1  0.1  0.   0.01 0.02 0.03]
```

`STATE_SHAPE` is the batch shape. `POS_SHAPE` and `VEL_SHAPE` include the final Cartesian component axis.

`state[1]` selects one row. The result is one scalar state, so its shape is `()`.

## 3. Use one object with a batch input

You can use one object with a batched input. Here one ground site is evaluated at three epochs.

```python
from difforb.body import Site
from difforb.core import Time

site = Site.from_code("568").require_ground()
epochs = Time.from_utc_date(
    [2025, 2025, 2025],
    [1, 1, 1],
    [2, 3, 4],
)

site_states = site.state(epochs)

print("SITE_SHAPE", site.shape)
print("SITE_STATES_SHAPE", site_states.shape)
print("SITE_POS_SHAPE", site_states.pos.shape)
```

```text title="Output"
SITE_SHAPE ()
SITE_STATES_SHAPE (3,)
SITE_POS_SHAPE (3, 3)
```

The scalar site repeats across the three epochs. The result is a batch of three `GCRS` observer states.

## 4. Batch `SmallBody` targets

`SmallBody` keeps the batch shape of the initial orbit. Propagation keeps the same target batch.

```python
import jax.numpy as jnp

from difforb.core import BCRS, State, Time
from difforb.dynamics import DynamicSystem
from difforb.integrator import NumericalIntegrator
from difforb.body import SmallBody
from difforb.spk import set_default_ephemeris

set_default_ephemeris("/path/to/your/de441.bsp")

t0 = Time.from_tdb_date([2025, 2025], [1, 1], [2, 2])
orbits = State(
    tdb=t0.tdb(),
    pos=jnp.array([
        [1.685775738339898, -1.336388854313325, -0.2144927004440800],
        [1.735775738339898, -1.286388854313325, -0.1944927004440800],
    ]),
    vel=jnp.array([
        [0.008995712853117517, 0.006985684417802803, 0.004020851173846060],
        [0.008795712853117517, 0.007085684417802803, 0.004120851173846060],
    ]),
    frame=BCRS,
)

body = SmallBody.create(orbits)
print("BODY_SHAPE", body.shape)
print("BODY_TRAJECTORY_READY", body.trajectory is not None)

force_model = DynamicSystem.from_standard_system().build_force_model()
integrator = NumericalIntegrator(method="IAS15", tol=1e-12)
body = body.propagate(
    t0.tdb(),
    Time.from_tdb_date(2025, 1, 8).tdb(),
    force_model,
    integrator,
)

print("PROPAGATED_SHAPE", body.shape)
print("BODY_TRAJECTORY_READY", body.trajectory is not None)
```

```text title="Output"
BODY_SHAPE (2,)
BODY_TRAJECTORY_READY False
PROPAGATED_SHAPE (2,)
BODY_TRAJECTORY_READY True
```

The two rows are two target orbits. After propagation, `body.shape` is still `(2,)`.

## 5. Query targets by time

Point-wise mode matches target row `i` with time row `i`. `grid=True` evaluates every target at every query epoch. The
shape order is `(target, time)`.

```python
query_times = Time.from_tdb_date([2025, 2025], [1, 1], [4, 6]).tdb()
states = body.state(query_times)

grid_times = Time.from_tdb_date(
    [2025, 2025, 2025],
    [1, 1, 1],
    [3, 4, 5],
).tdb()
grid_states = body.state(grid_times, grid=True)

print("POINTWISE_STATE_SHAPE", states.shape)
print("POINTWISE_POS_SHAPE", states.pos.shape)
print("GRID_STATE_SHAPE", grid_states.shape)
print("GRID_STATE_POS_SHAPE", grid_states.pos.shape)
```

```text title="Output"
POINTWISE_STATE_SHAPE (2,)
POINTWISE_POS_SHAPE (2, 3)
GRID_STATE_SHAPE (2, 3)
GRID_STATE_POS_SHAPE (2, 3, 3)
```

`GRID_STATE_SHAPE` is `(2 targets, 3 times)`. The final `3` in `GRID_STATE_POS_SHAPE` is the Cartesian axis.

## 6. Build target-observer-time grids

`EphemerisGenerator` uses the same rule for observer products. With `grid=True`, the order is target, observer, then
time.

```python
from difforb.ephemeris import EphemerisGenerator
from difforb.body import Site
from difforb.core import Time

observer = Site.from_code(["568", "253"]).require_ground()
obs_times = Time.from_utc_date(
    [2025, 2025, 2025],
    [1, 1, 1],
    [3, 4, 5],
)

generator = EphemerisGenerator(body)
vector = generator.vector_table(obs_times, observer, grid=True)

print("VECTOR_TABLE_SHAPE", vector.shape)
print("GEOMETRIC_POS_SHAPE", vector.geometric.pos.shape)
print("LIGHT_TIME_SHAPE", vector.light_time.shape)
```

```text title="Output"
VECTOR_TABLE_SHAPE (2, 2, 3)
GEOMETRIC_POS_SHAPE (2, 2, 3, 3)
LIGHT_TIME_SHAPE (2, 2, 3)
```

`VECTOR_TABLE_SHAPE` is `(2 targets, 2 observers, 3 times)`. Vector fields add a final Cartesian axis.

## 7. Choose point-wise or grid calls

Point-wise mode matches site row `i` with time row `i`. `grid=True` asks for every combination of the input batches.

```python
from difforb.body import Site
from difforb.core import Time

sites = Site.from_code(["568", "253"]).require_ground()

paired_times = Time.from_utc_date(
    [2025, 2025],
    [1, 1],
    [2, 3],
)
paired = sites.state(paired_times)

grid_times = Time.from_utc_date(
    [2025, 2025, 2025],
    [1, 1, 1],
    [2, 3, 4],
)
grid = sites.state(grid_times, grid=True)

print("PAIRED_SHAPE", paired.shape)
print("GRID_SHAPE", grid.shape)
print("GRID_POS_SHAPE", grid.pos.shape)
```

```text title="Output"
PAIRED_SHAPE (2,)
GRID_SHAPE (2, 3)
GRID_POS_SHAPE (2, 3, 3)
```

The point-wise call pairs two sites with two epochs. The grid call evaluates two sites at three epochs, so the batch
shape is `(2, 3)`. This is the observer-time part of the target-observer-time order.

## Common Mistakes

- Do not count the final Cartesian dimension as a batch dimension.
- Pass `tdb=t.tdb()` into `State(...)`; do not pass the raw `Time` object.
- Keep `State.tdb.shape`, `State.pos.shape[:-1]`, and `State.vel.shape[:-1]` aligned.
- Use `grid=True` only when you want every combination of the input batches.
- Treat string identifiers, kernel paths, and solver settings as configuration unless a specific API documents them as
  batched data.

## Next Steps

- Read [Batch Inputs And Shapes](../concepts/batch-inputs-and-shapes.md) for the shared rules.
- Continue to [Create And Convert Time Objects](create-and-convert-time-objects.md) when you need more time-construction
  patterns.
- Continue to [Build State From Cartesian Data](build-state-from-cartesian-data.md) when you need more Cartesian state construction
  patterns.
- Use the [State API](../api/state.md) and [Time API](../api/time.md) for details on the batched
  objects used here.
