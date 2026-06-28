# Create And Convert Time Objects

This guide shows how to create and inspect `Time` objects from calendar dates or split Julian dates, batch epochs, and shift them by uniform intervals.

For the scientific rules behind `TT`, `UTC`, `UT1`, and `TDB`, read [Time Scales And Epoch Storage](../concepts/time-scales-and-epoch-storage.md).

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- No SPK kernel is required for this guide.
- Modern `UT1` and related Earth-rotation quantities use the default EOP file loaded through `difforb.core.eop`. See
  [Configure Earth Orientation Data](configure-earth-orientation-data.md) when you need to check or update that file.

## 1. Create one `Time`

Choose the constructor that matches your input timescale. The object is stored internally in canonical `TT`, and the constructor name tells DiffOrb how to read the input.

DiffOrb uses a split Julian date. It stores a large component and a small remainder separately. This keeps time arithmetic and timescale conversion numerically stable.

- `.jd1` is the large component.
- `.jd2` is the small remainder.
- `.jd` is the summed Julian date for display or quick inspection.

```python
from difforb.core import Time

t = Time.from_utc_date(2025, 1, 2, 3, 4, 5.678)

print("UTC", t.utc.iso_string)
print("TT", t.tt.iso_string)
print("TT_JD1", float(t.tt.jd1))
print("TT_JD2", float(t.tt.jd2))
print("UTC_JD", float(t.utc.jd))
print("TT_JD", float(t.tt.jd))
```

```text title="Output"
UTC 2025-01-02 03:04:05.677
TT 2025-01-02 03:05:14.862
TT_JD1 2460678.0
TT_JD2 -0.3713557638888889
UTC_JD 2460677.6278434955
TT_JD 2460677.628644236
```

`Time.from_utc_date(...)` interprets the input as civil `UTC`.
`t.utc` is a view of the stored epoch in `UTC`.
`t.tt` is the same physical epoch viewed in `TT`.
The stored `TT` epoch is the pair `t.tt.jd1 + t.tt.jd2`.

Create one `Time` object. Then ask for the view you need.

## 2. Choose a constructor

Use these constructors when your source data arrives in different timescales:

```python
from difforb.core import Time

t_from_tt = Time.from_tt_jd(2451545.0, 0.0)
t_from_tdb = Time.from_tdb_jd(2451545.0, 0.0)
t_from_ut1 = Time.from_ut1_date(2025, 1, 2, 3, 4, 5.678)
t_from_ut = Time.from_ut_date(2025, 1, 2, 3, 4, 5.678)
```

The main families are:

- `from_*_date(...)`: your input is calendar-like year, month, day, hour, minute, second.
- `from_*_jd(...)`: your input is already a split Julian date pair `(jd1, jd2)`.

DiffOrb exposes constructors for:

- `TT`
- `TAI`
- `UTC`
- `UT1`
- mixed `UT`
- `TDB`

Use mixed `UT` only when your data model intentionally follows DiffOrb's legacy rule of `UT1` before 1962-01-01 and `UTC` on or after 1962-01-01.

If your input `TDB` epoch is topocentric rather than geocentric, pass `location=site.ground_itrs` to `Time.from_tdb_jd(...)` or `Time.from_tdb_date(...)`. This matches the same location-dependent rules used later by `t.tdb(location)`.

If your upstream code already follows a SOFA-like two-part Julian date convention, pass those two components directly to the matching `from_*_jd(...)` constructor. You do not need to collapse them into one float first.

## 3. Inspect calendar output

Every time-scale view provides interfaces for both calendar fields and format strings computed from its own split Julian date. These interfaces belong to the selected view itself, so `t.utc`, `t.tt`, `t.ut1`, and `t.tdb(...)` each expose their own calendar representation of the same physical instant.

### Calendar fields interface

Each time view exposes:

- `.ymdhms`: one tuple `(year, month, day, hour, min, sec)`
- `.year`, `.month`, `.day`, `.hour`, `.min`, `.sec`: convenience properties derived from `.ymdhms`

```python
from difforb.core import Time

t = Time.from_utc_date(2025, 1, 2, 3, 4, 5.678)

print("YEAR", int(t.tt.year))
print("MONTH", int(t.tt.month))
print("SECOND", float(t.tt.sec))
```

```text title="Output"
YEAR 2025
MONTH 1
SECOND 14.862
```

### Format strings interface

Call `format_string(template)` on the selected time view. For example, `t.utc.format_string(...)` formats the `UTC` calendar fields, while `t.tt.format_string(...)` formats the `TT` calendar fields of the same physical instant.

The default `iso_string` property is shorthand for `format_string("{YYYY}-{MM}-{DD} {hh}:{mm}:{ss.3}")`.

```python
from difforb.core import Time

t = Time.from_utc_date(2025, 1, 2, 3, 4, 5.678)

print("ISO", t.tt.iso_string)
print("FORMAT_PADDED", t.tt.format_string("{YYYY}/{MM}/{DD} {hh}:{mm}:{ss.6}"))
print("FORMAT_COMPACT", t.tt.format_string("{Y}-{M}-{D}T{h}:{m}:{s.2}"))
print("FORMAT_DATE_ONLY", t.tt.format_string("{YYYY}-{MM}-{DD}"))
```

```text title="Output"
ISO 2025-01-02 03:05:14.862
FORMAT_PADDED 2025/01/02 03:05:14.862000
FORMAT_COMPACT 2025-1-2T3:5:14.86
FORMAT_DATE_ONLY 2025-01-02
```

### Placeholder rules for `format_string(...)`

Placeholders are wrapped in braces such as `{YYYY}` or `{ss.6}`. Literal text outside braces is copied to the output unchanged.

| Placeholder | Meaning | Example |
| --- | --- | --- |
| `{YYYY}` | Year with at least 4 digits and zero padding when needed | `2025` |
| `{Y}` | Year without zero padding | `2025` |
| `{MM}` | Month with 2 digits | `01` |
| `{M}` | Month without zero padding | `1` |
| `{DD}` | Day of month with 2 digits | `02` |
| `{D}` | Day of month without zero padding | `2` |
| `{hh}` | Hour with 2 digits | `03` |
| `{h}` | Hour without zero padding | `3` |
| `{mm}` | Minute with 2 digits | `05` |
| `{m}` | Minute without zero padding | `5` |
| `{ss}` | Seconds with a 2-digit integer part and no fractional part | `14` |
| `{s}` | Seconds without zero padding and no fractional part | `14` |
| `{ss.N}` | Seconds with a 2-digit integer part and exactly `N` fractional digits, where `N >= 1` | `14.862000` |
| `{s.N}` | Seconds without zero padding in the integer part and exactly `N` fractional digits, where `N >= 1` | `14.86` |

The main formatting rules are:

- Use the doubled forms such as `{MM}` or `{hh}` when fixed-width output matters.
- Use the single-letter forms such as `{M}` or `{h}` when you want compact output without zero padding.
- `N` in `{ss.N}` and `{s.N}` must be an integer greater than or equal to `1`.
- Fractional seconds are truncated, not rounded. For example, a second value of `14.8629` formatted with `{ss.2}` becomes `14.86`.
- Invalid templates such as unmatched braces, empty placeholders, or unknown fields raise `ValueError`.

If the time view is batched, `format_string(...)` returns nested Python lists that match the batch shape.

```python
from difforb.core import Time

batch = Time.from_utc_date(
    [2025, 2025],
    [1, 1],
    [2, 3],
    [0, 12],
    [0, 0],
    [0.0, 30.0],
)

print(batch.utc.format_string("{YYYY}-{MM}-{DD} {hh}:{mm}:{ss}"))
```

```text title="Output"
['2025-01-02 00:00:00', '2025-01-03 12:00:30']
```

## 4. Build batched `Time` objects

`Time` accepts broadcastable array-like inputs. If your workflow naturally carries one epoch per row, pass lists or arrays directly.

```python
from difforb.core import Time

batch = Time.from_utc_date(
    [2025, 2025],
    [1, 1],
    [2, 3],
    [0, 12],
    [0, 0],
    [0.0, 30.0],
)

print("BATCH_UTC", batch.utc.iso_string)
print("BATCH_TT", batch.tt.iso_string)
print("SHAPE", batch.shape)
```


```text title="Output"
BATCH_UTC ['2025-01-02 00:00:00.000', '2025-01-03 12:00:30.000']
BATCH_TT ['2025-01-02 00:01:09.183', '2025-01-03 12:01:39.183']
SHAPE (2,)
```

This batch shape propagates through later APIs such as `State`, `Site`, `EphemerisBody`, and `SmallBody`.

## 5. Shift epochs with uniform intervals

Use numeric values when a plain number of days is enough. Use `TimeDelta` when you want the unit to be explicit.

```python
from difforb.core import TimeDelta, Time

t = Time.from_utc_date(2025, 1, 2, 3, 4, 5.678)

shifted_by_float = t + 1.5
shifted_by_delta = t + TimeDelta.from_seconds(90.0)
delta = shifted_by_delta - t

print("SHIFT_FLOAT_UTC", shifted_by_float.utc.iso_string)
print("SHIFT_DELTA_UTC", shifted_by_delta.utc.iso_string)
print("DELTA_SECONDS", float(delta.seconds))
```


```text title="Output"
SHIFT_FLOAT_UTC 2025-01-03 15:04:05.677
SHIFT_DELTA_UTC 2025-01-02 03:05:35.677
DELTA_SECONDS 89.99999999999969
```

These intervals are uniform SI intervals:

- `1.0` means exactly one day of `86400` SI seconds.
- `TimeDelta.from_seconds(90.0)` means exactly `90` SI seconds.

This is not calendar arithmetic. If you need civil-clock logic such as "same wall-clock time next month", handle that before you build the `Time` object.

## Common Mistakes

- Call the constructor that matches the input timescale. `from_utc_date(...)` and `from_tdb_date(...)` do not mean the same thing.
- The object stores canonical `TT` internally. Read back the view you need from `.utc`, `.tt`, `.ut1`, `.tai`, or `.tdb(...)`.
- When you care about numerical robustness or interoperability with SOFA-style code, keep using the split pair `.jd1` and `.jd2` instead of collapsing everything into `.jd`.
- Numeric addition and subtraction use uniform SI days, not calendar rules.
- `UTC` is not defined for epochs before 1962-01-01. Use `UT1` or mixed `UT` for historical epochs.

## Next Steps

- Continue to [Convert Between UTC, TT, TDB, UT1](convert-between-utc-tt-tdb-ut1.md) when you need precise timescale management from one epoch.
- Read [Time Scales And Epoch Storage](../concepts/time-scales-and-epoch-storage.md) for the scientific rules behind the views used here.
- Use the [Time API](../api/time.md) for details on `Time`, `TimeDelta`, and view objects.
