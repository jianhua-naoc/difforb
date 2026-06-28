# Create A Space Site And Reuse Its GCRS State

This guide shows how to create a space `Site` from a supplied canonical `GCRS` state, then evaluate that observer through the `Time` interface used by the rest of the site API.

For the `GCRS` state-construction rules, read [Build State From Cartesian Data](build-state-from-cartesian-data.md).

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- The supplied `sat_gcrs` input must already be a canonical `GCRS` `State`.

## 1. Create from a `GCRS` state

`Site.from_state(...)` stores the canonical `GCRS` state that you provide.

```python
from difforb.body import Site
from difforb.core import GCRS, State, Time

t = Time.from_tdb_date(2025, 1, 2)
sat_gcrs = State(
    tdb=t.tdb(),
    pos=[1.0e-4, 2.0e-4, -3.0e-4],
    vel=[1.0e-6, -2.0e-6, 3.0e-6],
    frame=GCRS,
)

space = Site.from_state(sat_gcrs)
site_gcrs = space.state(t, frame=GCRS)

print("SPACE_SHAPE", space.shape)
print("SPACE_POS", site_gcrs.pos)
print("SPACE_VEL", site_gcrs.vel)
```


```text title="Output"
SPACE_SHAPE ()
SPACE_POS [ 0.0001  0.0002 -0.0003]
SPACE_VEL [ 1.e-06 -2.e-06  3.e-06]
```

## 2. Reuse the `GCRS` state

The space `Site` does not propagate the spacecraft. It stores the supplied canonical `GCRS` state and reuses it at the requested times.

```python
print("STORED_POS", space.gcrs_pos)
print("STORED_VEL", space.gcrs_vel)
```


```text title="Output"
STORED_POS [ 0.0001  0.0002 -0.0003]
STORED_VEL [ 1.e-06 -2.e-06  3.e-06]
```

## Common Mistakes

- `Site.from_state(...)` requires a `State` in `GCRS`.
- Use `Site.from_gcrs(...)` when you have position and velocity arrays rather than a `State`.
- A space `Site` stores the supplied state; it does not propagate an orbit.

## Next Steps

- Continue to [Transform State Between Frames](transform-state-between-frames.md) when the space-site state needs another reference frame.
- Continue to [Load SPK Kernels And Query Major Bodies](load-spk-kernels-and-query-major-bodies.md) when you want the site to interact with SPK-backed body states.
- Use the [Body API](../api/body.md) and [State API](../api/state.md) for details on `Site`
  and `GCRS` states.
