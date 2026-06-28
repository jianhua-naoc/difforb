# Create And Convert Keplerian Elements

This guide shows how to create a `KepElement` from classical Keplerian elements, convert it to a Cartesian `State`, and recover elements from that state.

For the model behind the element representation, read [Keplerian Elements](../concepts/keplerian-elements.md).

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- The example below does not need an SPK kernel because the Cartesian state stays in `HELIO_ECLIP_J2000`.
- If you convert a state whose origin is not already heliocentric, `KepElement.from_state(...)` may need Sun or Earth ephemeris bodies for the origin translation.

## 1. Create elements from the classical set

`KepElement.from_classical(...)` accepts `tdb`, `a`, `e`, `inc`, `node`, `peri`, `m`, and `degrees`. Distance is in `au`. Angular inputs are in degrees by default. Pass `degrees=False` when the angular inputs are in radians.

```python
from difforb.core import KepElement, Time

t = Time.from_tdb_date(2025, 1, 2)

elements = KepElement.from_classical(
    tdb=t.tdb(),
    a=2.31,
    e=0.203,
    inc=9.30,
    node=67.70,
    peri=87.10,
    m=12.50,
)

print("ELEMENT_SHAPE", elements.shape)
print("A_AU", float(elements.a))
print("P_AU", float(elements.p))
```

```text title="Output"
ELEMENT_SHAPE ()
A_AU 2.31
P_AU 2.21480721
```

The object stores `p`, not `a`. The `a` value printed above is a derived property.

## 2. Convert elements to a Cartesian state

`elements.state()` returns the canonical Cartesian boundary for `KepElement`: a heliocentric ecliptic-of-`J2000.0` state.

```python
import numpy as np

state = elements.state()

print("STATE_FRAME", state.frame.name)
print("POS_AU", np.array2string(np.asarray(state.pos), precision=8))
print("VEL_AU_PER_D", np.array2string(np.asarray(state.vel), precision=8))
```

```text title="Output"
STATE_FRAME HELIO_ECLIP_J2000
POS_AU [-1.82636537  0.18703007  0.28833235]
VEL_AU_PER_D [-0.00225557 -0.01360199 -0.00050347]
```

This state is suitable for state-vector inspection or for later conversion into another supported frame. It is not the canonical `BCRS` storage used by `SmallBody`.

## 3. Recover elements from the state

Use `KepElement.from_state(...)` to convert a frame-aware Cartesian state back to elements. Because the `state` above is already `HELIO_ECLIP_J2000`, no ephemeris body is required.

```python
import jax.numpy as jnp

recovered = KepElement.from_state(state)

print("A_AU", float(recovered.a))
print("E", float(recovered.e))
print("INC_DEG", float(jnp.rad2deg(recovered.inc)))
print("NODE_DEG", float(jnp.rad2deg(recovered.node)))
print("PERI_DEG", float(jnp.rad2deg(recovered.peri)))
print("M_DEG", float(jnp.rad2deg(recovered.m)))
print("PERIOD_D", float(recovered.period))
```

```text title="Output"
A_AU 2.3099999999999996
E 0.20299999999999987
INC_DEG 9.300000000000015
NODE_DEG 67.7
PERI_DEG 87.10000000000001
M_DEG 12.499999999999995
PERIOD_D 1282.378997727952
```

Small numerical differences are normal because the round trip goes through floating-point Cartesian conversion and anomaly conversion.

## Use Radian Inputs

If your source data already uses radians, pass `degrees=False`.

```python
import jax.numpy as jnp

elements_rad = KepElement.from_classical(
    tdb=t.tdb(),
    a=2.31,
    e=0.203,
    inc=jnp.deg2rad(9.30),
    node=jnp.deg2rad(67.70),
    peri=jnp.deg2rad(87.10),
    m=jnp.deg2rad(12.50),
    degrees=False,
)
```

The stored object is equivalent to the degree-based construction above.

## Common Mistakes

- `KepElement.from_classical(...)` uses degrees for angular inputs unless `degrees=False`.
- `KepElement.state()` returns `HELIO_ECLIP_J2000`, not `BCRS`.
- `KepElement.from_state(...)` may require `sun=` or `earth=` when the input state must change origin before element extraction.
- Classical element angles can be ill-conditioned near circular or zero-inclination cases.

## Next Steps

- Continue to [Create A SmallBody From State Or Elements](create-a-smallbody-from-state-or-elements.md) when the element set should become a propagated object.
- Continue to [Transform State Between Frames](transform-state-between-frames.md) when you need to convert the Cartesian state to another supported frame.
- Use the [Core API](../api/core.md) and [State API](../api/state.md) for details on `KepElement` and `State`.
