# Build State From Cartesian Data

This guide shows how to create a `State` from canonical position and velocity data, inspect its stacked array form, and repeat the same construction for batched `[x, y, z, vx, vy, vz]` rows.

For the model behind `State` and reference frames, read [Frames And State Representation](../concepts/frames-and-state-representation.md). Continue to [Transform State Between Frames](transform-state-between-frames.md) when you need frame conversion.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- No SPK kernel is required for this guide.

## 1. Create one canonical `BCRS` state

Use `State(...)` when your source data already has an epoch, a Cartesian position, a Cartesian velocity, and a reference frame.

```python
from difforb.core import BCRS, State, Time

t = Time.from_tdb_date(2025, 1, 2)

state = State(
    tdb=t.tdb(),
    pos=[1.685775738339898, -1.336388854313325, -0.2144927004440800],
    vel=[0.008995712853117517, 0.006985684417802803, 0.004020851173846060],
    frame=BCRS,
)

print("STATE_ARRAY", state.array)
print("DIST", float(state.dist))
print("LT", float(state.lt))
```


```text title="Output"
STATE_ARRAY [ 1.68577574 -1.33638885 -0.2144927   0.00899571  0.00698568  0.00402085]
DIST 2.1618931815545612
LT 0.012486053700677017
```

The stacked array is always ordered as `[x, y, z, vx, vy, vz]`.

## 2. Create from a stacked array

Use `State.from_array(...)` when your upstream source already stores Cartesian states in `[..., 6]` form.

```python
from difforb.core import BCRS, State, Time

t = Time.from_tdb_jd(2451545.0, 0.0)

state_from_array = State.from_array(
    tdb=t.tdb(),
    array=[1.0, -2.0, 0.5, 0.01, 0.02, -0.03],
    frame=BCRS,
)

print("ARRAY", state_from_array.array)
print("FRAME", state_from_array.frame.name)
```


```text title="Output"
ARRAY [ 1.   -2.    0.5   0.01  0.02 -0.03]
FRAME BCRS
```

## 3. Batch Cartesian inputs

`State` follows the broadcast shape carried by `tdb`, `pos`, and `vel`.

```python
import jax.numpy as jnp

from difforb.core import BCRS, State, Time

batched = State(
    tdb=Time.from_tdb_jd(
        jnp.array([2451545.0, 2451546.0]),
        jnp.array([0.0, 0.25]),
    ).tdb(),
    pos=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    vel=jnp.array([0.1, 0.2, 0.3]),
    frame=BCRS,
)

print("BATCH_SHAPE", batched.shape)
print("BATCH_VEL", batched.vel)
```


```text title="Output"
BATCH_SHAPE (2,)
BATCH_VEL [[0.1 0.2 0.3]
 [0.1 0.2 0.3]]
```

## Common Mistakes

- `pos` is in `au` and `vel` is in `au / day`.
- `State` stores the epoch in `TDB`.
- The stacked array order is position first, velocity second.

## Next Steps

- Continue to [Transform State Between Frames](transform-state-between-frames.md) when you need to change the reference frame of the state.
- Continue to [Create A SmallBody From State Or Elements](create-a-smallbody-from-state-or-elements.md) when the state should become a propagated object.
- Use the [State API](../api/state.md) for details on `State`, `Frame`, `Axes`, or `Origin`.
