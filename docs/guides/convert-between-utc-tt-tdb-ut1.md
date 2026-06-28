# Convert Between UTC, TT, TDB, UT1

This guide shows how to expose one `Time` epoch in `UTC`, `TT`, `TDB`, and `UT1`, measure offsets between those timescales, and check when a topocentric location changes the `TDB` value.

Start with [Create And Convert Time Objects](create-and-convert-time-objects.md) if you need constructor patterns before timescale conversion.

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- No SPK kernel is required for this guide.
- `UT1` and the `TT <-> UT1` path depend on the default EOP data loaded through `difforb.core.eop`. See
  [Configure Earth Orientation Data](configure-earth-orientation-data.md) when you need to check or update that file.
- The topocentric `TDB - TT` example uses one ground `Site` location.

## 1. Start from one modern `UTC` epoch

Build one `Time` object in `UTC`, then ask for the scale-specific view you need.

```python
from difforb.body import Site
from difforb.core import Time

t = Time.from_utc_date(2025, 1, 2, 3, 4, 5.678)
site = Site.from_code("F51").require_ground()

print("UTC", t.utc.iso_string)
print("TAI", t.tai.iso_string)
print("TT", t.tt.iso_string)
print("UT1", t.ut1.iso_string)
print("TDB_GEO", t.tdb().iso_string)
print("TDB_TOPO", t.tdb(site.ground_itrs).iso_string)
```


```text title="Output"
UTC 2025-01-02 03:04:05.677
TAI 2025-01-02 03:04:42.677
TT 2025-01-02 03:05:14.862
UT1 2025-01-02 03:04:05.724
TDB_GEO 2025-01-02 03:05:14.861
TDB_TOPO 2025-01-02 03:05:14.861
```

Use these access patterns consistently:

- `.utc`, `.tai`, `.tt`, `.ut1` are properties.
- `.tdb(...)` is a method because the `TDB - TT` correction can depend on observer location.
- DiffOrb uses the same practical `TT <-> TDB` model as SOFA `iauDtdb`; `TDB -> TT` is obtained by inverting that same forward model.

The topocentric effect is smaller than the default three-decimal string format at this epoch, so you need to inspect the offset numerically if you care about the location dependence.

## 2. Measure the offsets explicitly

When you need to confirm which timescale offset is being applied, compare the Julian-date views directly.

```python
from difforb.body import Site
from difforb.core import Time

t = Time.from_utc_date(2025, 1, 2, 3, 4, 5.678)
site = Site.from_code("F51").require_ground()

geocenter_tdb = t.tdb()
topocenter_tdb = t.tdb(site.ground_itrs)

print("TAI_MINUS_UTC", float((t.tai.jd - t.utc.jd) * 86400.0))
print("TT_MINUS_TAI", float((t.tt.jd - t.tai.jd) * 86400.0))
print("UT1_MINUS_UTC", float((t.ut1.jd - t.utc.jd) * 86400.0))
print("TDB_MINUS_TT_GEO", float((geocenter_tdb.jd - t.tt.jd) * 86400.0))
print("TOPO_MINUS_GEO_TDB", float((topocenter_tdb.jd2 - geocenter_tdb.jd2) * 86400.0))
```


```text title="Output"
TAI_MINUS_UTC 37.00000047683716
TT_MINUS_TAI 32.183973491191864
UT1_MINUS_UTC 0.046469271183013916
TDB_MINUS_TT_GEO -4.023313522338867e-05
TOPO_MINUS_GEO_TDB -1.955563710964725e-06
```

Interpret these numbers as follows:

- `TAI - UTC` is the accumulated leap-second offset.
- `TT - TAI` is the fixed `32.184 s` offset, subject here to floating-point display noise.
- `UT1 - UTC` is the Earth-rotation correction from the EOP file.
- `TDB - TT` is a small relativistic correction.
- the topocentric `TDB` adjustment is even smaller and depends on the observer location.

## 3. Use topocentric `TDB`

For many dynamical operations, geocentric `TDB` is enough:

```python
tdb = t.tdb()
```

Use topocentric `TDB` when the operation is tied to one observing site:

```python
site = Site.from_code("F51").require_ground()
tdb_topo = t.tdb(site.ground_itrs)
```

Pass the underlying `ITRS` location, not the `Site` wrapper itself. The method expects `ITRS | None`.

This distinction matters most when you are building site states, solving observation geometry, or validating high-precision time handling around one observer.

## 4. Handle historical epochs correctly

`UTC` is only defined in DiffOrb on or after 1962-01-01. For earlier epochs, create the object from `UT1` or mixed `UT`.

```python
from difforb.core import Time

historical = Time.from_ut1_date(1950, 1, 1, 0, 0, 0.0)
print("HIST_UT1", historical.ut1.iso_string)
```


```text title="Output"
HIST_UT1 1950-01-01 00:00:00.000
```

Trying to read `historical.utc` raises an error whose key message is:

```text title="Error"
UTC is only defined for epochs on or after 1962-01-01.
```

If you need one batch that crosses the 1962 boundary, use mixed `UT` through `Time.from_ut_date(...)` and read it back with `.ut`.

## Common Mistakes

- `.tdb` is a method, not a property, because the topocentric correction can depend on location.
- Pass `site.ground_itrs` to `.tdb(...)`, not the `Site` object.
- Do not call `.utc` on pre-1962 epochs.
- Use `.ut` for mixed historical and modern batches that cross the `UTC` validity boundary.

## Next Steps

- Return to [Create And Convert Time Objects](create-and-convert-time-objects.md) if you need constructor patterns or arithmetic again.
- Continue to [Get Earth Rotation Quantities And Matrices](get-earth-rotation-quantities-and-matrices.md) when you want the `ERA`, polar-motion coordinates, or Earth-rotation matrices themselves.
- Continue to [Create A Ground Site And Get Its GCRS State](create-a-groundsite-and-get-its-gcrs-state.md) when you want to combine these times with observer geometry.
- Read [Time Scales And Epoch Storage](../concepts/time-scales-and-epoch-storage.md) for the deeper model behind `TT`, `UT1`, and `TDB`.
- Use the [Time API](../api/time.md) for details on time-scale view objects.
