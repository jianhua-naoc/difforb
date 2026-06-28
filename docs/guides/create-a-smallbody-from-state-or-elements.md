# Create A SmallBody From State Or Elements

This guide shows how to create a `SmallBody` from either a Cartesian state or Keplerian elements, inspect what gets stored internally, and check that the result is ready for later propagation.

Object creation stops before propagation or observer-table generation.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- Creating a `SmallBody` from a canonical `BCRS` state does not require an SPK kernel.
- Creating it from a heliocentric or geocentric state, or from `KepElement`, requires ephemeris bodies for the origin
  translation into canonical `BCRS`. A local planetary SPK such as `de441.bsp` provides those body states.

## 1. Create from BCRS state

If your input state is already canonical `BCRS`, the shortest path is direct construction through
`SmallBody.create(...)`.

```python
from difforb.body import SmallBody
from difforb.core import BCRS, State, Time

t = Time.from_tdb_date(2025, 1, 2)

state_input = State(
    tdb=t.tdb(),
    pos=[1.685775738339898, -1.336388854313325, -0.2144927004440800],
    vel=[0.008995712853117517, 0.006985684417802803, 0.004020851173846060],
    frame=BCRS,
)

body_from_state = SmallBody.create(state_input)

print("FROM_STATE_FRAME", body_from_state.orbit0.frame.name)
print("FROM_STATE_POS", body_from_state.orbit0.pos)
print("TRAJECTORY_NONE", body_from_state.trajectory is None)
```

```text title="Output"
FROM_STATE_FRAME BCRS
FROM_STATE_POS [ 1.68577574 -1.33638885 -0.2144927 ]
TRAJECTORY_NONE True
```

This tells you two important things:

- the stored orbit is canonical `BCRS`,
- propagation has not happened yet because `trajectory` is still `None`.

If your input state is not already `BCRS`, `SmallBody.create(...)` still accepts it. Pass `sun` and/or `earth` when
reference-frame conversion touches the `SUN` or `EARTH` origin so the object can be converted into canonical `BCRS`
storage.

## 2. Create from Keplerian elements

`KepElement` is the convenient front door when your source data is orbital elements rather than Cartesian states.

`KepElement.from_classical(...)` interprets angular inputs in degrees by default, so the example below passes `inc`,
`node`, `peri`, and `m` in degrees.

For element-only construction and round-trip conversion, see [Create And Convert Keplerian Elements](create-and-convert-keplerian-elements.md).

```python
from difforb.body import EphemerisBody, SmallBody
from difforb.core import KepElement, Time
from difforb.spk import set_default_ephemeris

set_default_ephemeris("/path/to/de441.bsp")

sun = EphemerisBody("sun")

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

state_from_elements = elements.state()
body_from_elements = SmallBody.create(elements, sun=sun)

print("ELEMENT_STATE_FRAME", state_from_elements.frame.name)
print("ELEMENT_STATE_POS", state_from_elements.pos)
print("BODY_FROM_ELEMENTS_FRAME", body_from_elements.orbit0.frame.name)
print("BODY_FROM_ELEMENTS_POS", body_from_elements.orbit0.pos)
```

```text title="Output"
ELEMENT_STATE_FRAME HELIO_ECLIP_J2000
ELEMENT_STATE_POS [-1.82636537  0.18703007  0.28833235]
BODY_FROM_ELEMENTS_FRAME BCRS
BODY_FROM_ELEMENTS_POS [-1.83208881  0.05232465  0.33714587]
```

The two outputs show different layers of the same orbit object:

- `elements.state()` returns the canonical Cartesian boundary of `KepElement`, which is `HELIO_ECLIP_J2000`,
- `SmallBody.create(elements, sun=sun)` then translates that heliocentric state into the internally stored canonical
  `BCRS` orbit.

If your input state is geocentric rather than heliocentric, pass `earth=EphemerisBody("earth")` as well when needed.

## 3. Verify stored elements

`KepElement.from_state(...)` is a good check when you want to confirm that the stored canonical orbit still corresponds
to the expected osculating elements.

```python
import jax.numpy as jnp

from difforb.core import KepElement

roundtrip = KepElement.from_state(body_from_elements.orbit0, sun=sun)

print("ROUNDTRIP_A", float(roundtrip.a))
print("ROUNDTRIP_E", float(roundtrip.e))
print("ROUNDTRIP_INC_DEG", float(jnp.rad2deg(roundtrip.inc)))
print("ROUNDTRIP_NODE_DEG", float(jnp.rad2deg(roundtrip.node)))
print("ROUNDTRIP_PERI_DEG", float(jnp.rad2deg(roundtrip.peri)))
print("ROUNDTRIP_M_DEG", float(jnp.rad2deg(roundtrip.m)))
```

```text title="Output"
ROUNDTRIP_A 2.3099999999999996
ROUNDTRIP_E 0.20299999999999987
ROUNDTRIP_INC_DEG 9.300000000000015
ROUNDTRIP_NODE_DEG 67.69999999999996
ROUNDTRIP_PERI_DEG 87.10000000000004
ROUNDTRIP_M_DEG 12.499999999999995
```

This shows that the `SmallBody` created from elements still represents the same osculating orbit after the canonical
conversion to `BCRS`.

## 4. Decide which constructor path to use

Use a `State` input when:

- your source already provides Cartesian vectors,
- you know the frame of those vectors,
- and you want exact control over the initial Cartesian boundary.

Use a `KepElement` input when:

- your source orbit is naturally expressed as elements,
- you want DiffOrb to convert the elements to a canonical state,
- or you want to use the same element-based path that later inspection code uses.

In both cases, the stored result is canonical `BCRS` on `body.orbit0`.

## Common Mistakes

- `SmallBody` stores canonical `BCRS` on `orbit0`, even if the input was heliocentric.
- `KepElement.from_classical(...)` interprets angular inputs in degrees by default.
- `body.trajectory` stays `None` until you call `propagate(...)`.
- Non-`BCRS` state inputs that touch `SUN` or `EARTH` need the corresponding ephemeris bodies.

## Next Steps

- Continue to [Build State From Cartesian Data](build-state-from-cartesian-data.md) if you need more control over the Cartesian input state before object creation.
- Continue to [Create And Convert Keplerian Elements](create-and-convert-keplerian-elements.md) if you want to inspect `KepElement` without constructing a `SmallBody`.
- Continue to [Propagate A SmallBody And Evaluate Dense Trajectories](propagate-a-smallbody-and-evaluate-dense-trajectories.md) when you are ready to integrate the orbit.
- Use the [Body API](../api/body.md), [State API](../api/state.md), and [Core API](../api/core.md) when you need
  details on `SmallBody`, `State`, or `KepElement`.
