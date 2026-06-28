# Load SPK Kernels And Query Major Bodies

This guide shows how to set a default `SPK` kernel, inspect the major-body names it exposes, and query body states at one `TDB` epoch.

Use the state, site, and small-body guides after this one when you need downstream use of SPK-backed bodies.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- You need at least one local SPK kernel, such as a planetary kernel like `de441.bsp`.
- Querying major bodies uses `TDB` input epochs.

## 1. Set the default ephemeris

The simplest path is to configure one process-wide default ephemeris.

Replace the placeholder path with a real local kernel path before running the snippet.

```python
from difforb.spk import load_default_ephemeris, set_default_ephemeris

set_default_ephemeris("/path/to/de441.bsp")
eph = load_default_ephemeris()

print(eph.available_bodies[:12])
```

```text title="Output"
['VENUS BARYCENTER', 'VENUS', 'MERCURY BARYCENTER', 'MERCURY', 'EARTH BARYCENTER', 'EARTH', 'MOON', 'SOLAR SYSTEM BARYCENTER', 'SUN', 'PLUTO BARYCENTER', 'NEPTUNE BARYCENTER', 'URANUS BARYCENTER']
```

This list is the quickest way to check which names the loaded kernel graph actually exposes.

`set_default_ephemeris(...)` accepts either:

- one path string,
- a list of SPK paths when you want to merge several kernels into one ephemeris graph,
- or an already constructed `Ephemeris` object.

For example:

```python
set_default_ephemeris([
    "/path/to/de441.bsp",
    "/path/to/sb441-n16.bsp",
])
```

If you want to restrict segment loading to one explicit `TDB` coverage window, build scalar `TDBView` endpoints and
pass them through `load_window`:

```python
from difforb.core import Time
from difforb.spk import set_default_ephemeris

window_start = Time.from_tdb_jd(2460740.0, 0.0).tdb()
window_end = Time.from_tdb_jd(2460742.0, 0.0).tdb()

set_default_ephemeris(
    "/path/to/de441.bsp",
    load_window=(window_start, window_end),
)
```

DiffOrb preloads SPK ephemeris data from disk into memory to speed up later queries. This trades memory for speed. Use
`load_window` to load only the time span needed by your task.

If you do not pass `load_window`, DiffOrb does not filter by time. The full time span of each loaded kernel can be used
later. This is the simplest setup, but it can use much more memory when the kernel span is long and your task only
needs a short interval.

## 2. Build `EphemerisBody`

`EphemerisBody` resolves one body through the default ephemeris and caches the loaded segment path.

```python
from difforb.body import EphemerisBody
from difforb.core import HELIO_ICRS, Time

t = Time.from_tdb_date(2025, 1, 2)

earth = EphemerisBody("earth")
sun = EphemerisBody("sun")
mars = EphemerisBody("mars barycenter")

earth_bcrs = earth.state(t.tdb())
mars_helio = mars.state(t.tdb(), frame=HELIO_ICRS, sun=sun)

print("EARTH_BCRS_POS", earth_bcrs.pos)
print("MARS_HELIO_POS", mars_helio.pos)
print("EARTH_SEGMENTS", len(earth.segments))
```

```text title="Output"
EARTH_BCRS_POS [-0.20158253  0.87956107  0.38147617]
MARS_HELIO_POS [-0.53437915  1.37836443  0.64663682]
EARTH_SEGMENTS 2
```

Interpret these results as:

- `earth.state(...)` returns the Earth's canonical barycentric state in `BCRS`,
- `mars.state(..., frame=HELIO_ICRS, sun=sun)` first evaluates the native `BCRS` state, then translates it to
  heliocentric `ICRS`,
- `len(earth.segments)` shows how many merged SPK path segments are active for this body in the current ephemeris.

## 3. Match kernel names

DiffOrb uppercases the input name internally, but it does not invent aliases for you. In practice:

- `"earth"` works because `EARTH` is available,
- `"sun"` works because `SUN` is available,
- `"mars barycenter"` works because `MARS BARYCENTER` is available,
- `"mars"` is not a safe assumption unless that exact name appears in `available_bodies`.

The most reliable workflow is:

1. inspect `load_default_ephemeris().available_bodies`,
2. choose one exact exposed name,
3. reuse that spelling in `EphemerisBody(...)`.

## 4. Use explicit `Ephemeris`

If you do not want to rely on the process-wide default ephemeris, build an explicit `Ephemeris` object and pass it into
each body.

```python
from difforb.body import EphemerisBody
from difforb.core import Time
from difforb.spk import Ephemeris

window_start = Time.from_tdb_jd(2460740.0, 0.0).tdb()
window_end = Time.from_tdb_jd(2460742.0, 0.0).tdb()

eph = Ephemeris(
    "/path/to/de441.bsp",
    load_window=(window_start, window_end),
)
earth = EphemerisBody("earth", eph=eph)
```

This is useful when:

- you want several ephemeris configurations in one process,
- you are testing different kernel combinations side by side,
- or you want to avoid hidden global state in a larger application.

## Common Mistakes

- `EphemerisBody.state(...)` expects a `TDBView`, not a raw `Time` object.
- Use body names that actually appear in `available_bodies`.
- `set_default_ephemeris(...)` is process-local to the current Python process. Separate worker processes or test runners must each set their own default ephemeris.
- Reference-frame conversions that touch `SUN` or `EARTH` still need the corresponding ephemeris body objects.

## Next Steps

- Continue to [Transform State Between Frames](transform-state-between-frames.md) when you want to combine
  SPK-backed bodies with arbitrary Cartesian states.
- Continue to [Create A SmallBody From State Or Elements](create-a-smallbody-from-state-or-elements.md) when your target
  object is not a major body from SPK.
- Use the [SPK API](../api/spk.md) and [Body API](../api/body.md) for details on kernels and
  ephemeris-backed bodies.
