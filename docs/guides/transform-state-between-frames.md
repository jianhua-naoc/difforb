# Transform State Between Frames

This guide shows how to convert one `State` between reference frames and roundtrip the result as a simple check that the conversion path is set up correctly.

Start with [Build State From Cartesian Data](build-state-from-cartesian-data.md) if you need the construction rules for canonical Cartesian inputs. For the model behind `State` and reference frames, read [Frames And State Representation](../concepts/frames-and-state-representation.md).

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- You need a local planetary SPK kernel for any conversion that touches the `SUN` or `EARTH` origin.
- Replace the placeholder kernel path in the snippet with a local file such as `de441.bsp`.
- See [Load SPK Kernels And Query Major Bodies](load-spk-kernels-and-query-major-bodies.md) if you need the SPK setup
  interface.

## 1. Convert one state and roundtrip it

Use `State.to(...)` for the general path, or one of the convenience methods such as `bcrs(...)`, `gcrs(...)`, or
`helio_eclip_j2000(...)` when the target reference frame is fixed.

```python
from difforb.body import EphemerisBody
from difforb.core import BCRS, HELIO_ECLIP_J2000, State, Time
from difforb.spk import set_default_ephemeris

set_default_ephemeris("/path/to/your/de441.bsp")

sun = EphemerisBody("sun")
t = Time.from_tdb_date(2025, 1, 2)

state = State(
    tdb=t.tdb(),
    pos=[1.685775738339898, -1.336388854313325, -0.2144927004440800],
    vel=[0.008995712853117517, 0.006985684417802803, 0.004020851173846060],
    frame=BCRS,
)

helio = state.to(HELIO_ECLIP_J2000, sun=sun)
back = helio.bcrs(sun=sun)

print("HELIO_FRAME", helio.frame.name)
print("ROUNDTRIP_MAX", float(abs(back.array - state.array).max()))
```

```text title="Output"
HELIO_FRAME HELIO_ECLIP_J2000
ROUNDTRIP_MAX 0.0
```

The `ROUNDTRIP_MAX` value is a simple sanity check. Small floating-point differences can appear on other platforms or
backend versions.

## 2. Choose the right conversion entry point

Use these patterns consistently:

- `state.to(frame, sun=sun, earth=earth)` when you already have the target frame object.
- `state.bcrs(...)`, `state.gcrs(...)`, `state.helio_icrs(...)`, `state.helio_j2000(...)`, or
  `state.helio_eclip_j2000(...)` when the target reference frame is fixed and the method name makes the call easier to
  read.

Pass `sun` or `earth` when the source or target reference frame uses the Sun or Earth as its origin.

## Common Mistakes

- Reference-frame conversion that touches `SUN` or `EARTH` needs the corresponding ephemeris bodies.
- `State.to(...)` returns a new `State` object in the target reference frame.
- Do not assume a conversion is body-free when the origin changes.

## Next Steps

- Continue to [Load SPK Kernels And Query Major Bodies](load-spk-kernels-and-query-major-bodies.md) when you need
  ephemeris-backed bodies for the conversion.
- Continue to [Create A Ground Site And Get Its GCRS State](create-a-groundsite-and-get-its-gcrs-state.md) when the
  converted state will feed observer geometry.
- Use the [State API](../api/state.md) for details on frame conversion methods and frame
  constants.
