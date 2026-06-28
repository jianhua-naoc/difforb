# Time Scales And Epoch Storage

DiffOrb uses `Time` for one physical instant. It uses `State` for one dynamical state at one epoch. The same instant can be viewed as Terrestrial Time (`TT`), International Atomic Time (`TAI`), Coordinated Universal Time (`UTC`), Universal Time 1 (`UT1`), mixed `UT`, or Barycentric Dynamical Time (`TDB`). `Time` stores the instant in `TT`. `State` uses the same instant in `TDB`.[^kaplan][^sofa]

## Core Contract

`Time` is the main epoch container in DiffOrb. One `Time` object can show several time-scale views without changing the instant it represents. `State` uses the same idea at the dynamical layer, but it keeps one canonical epoch form for propagation and reference-frame conversion.

DiffOrb uses different canonical scales in these two layers. `TT` is the canonical scale for `Time`. `TDB` is the canonical scale for `State`.[^kaplan]

## Two-Part Julian Dates

DiffOrb stores epochs as split Julian dates: `jd = jd1 + jd2`.[^sofa] The large part stays in `jd1`. The small remainder stays in `jd2`. This reduces precision loss when a large Julian date is combined with a small offset or a time-scale correction.

DiffOrb keeps this split form through the whole conversion chain. If you exchange data with `SOFA`-style code, keep both parts. Do not collapse them into one floating-point Julian date.

## Time Scales Used By DiffOrb

In the modern International Astronomical Union (`IAU`) framework, some time scales are uniform atomic or coordinate scales, and some follow the Earth's real rotation.[^iau][^kaplan] DiffOrb follows that split. The current API exposes `TAI`, `TT`, `UTC`, `UT1`, mixed `UT`, and `TDB`. Geocentric Coordinate Time (`TCG`) and Barycentric Coordinate Time (`TCB`) matter in the formal background, but the current API does not expose them as standard `Time` views.

### Terrestrial Time

`TT` is the canonical storage scale of `Time`. In practice, DiffOrb uses the standard relation `TT = TAI + 32.184 s`.[^sofa] Civil `UTC` connects to `TT`. Earth-rotation `UT1` also connects to `TT`. `TDB` is derived from `TT`.

### International Atomic Time

`TAI` is the continuous atomic scale behind `TT`. In DiffOrb it mainly acts as the stable intermediate between `UTC` and `TT`.

### Coordinated Universal Time

`UTC` is the civil broadcast scale. DiffOrb defines it only for epochs on or after `1962-01-01`. For earlier epochs, use `UT1` instead.

`UTC` started to appear in 1961, but DiffOrb uses `1962-01-01` as the boundary because the high-precision Earth Orientation Parameter (`EOP`) data used for Earth-based timing starts there. This also matches the JPL Horizons convention. Horizons treats observer-table `UT` as `UT1` before `1962-01-01` and as `UTC` on and after `1962-01-01`.

Before 1972, `UTC` used linear rate adjustments instead of the modern leap-second system. DiffOrb keeps that historical distinction so converted observation times keep the right civil meaning.

### Universal Time 1 And Mixed UT

`UT1` is the Earth-rotation time scale. It follows the actual rotation angle of the Earth and is the `UT` scale used by Earth-rotation geometry.[^kaplan][^iers]

For modern dates, DiffOrb converts between `TT` and `UT1` with `EOP` data. When the needed `EOP` span is not available, it falls back to the historical `Delta T = TT - UT1` polynomial model summarized by Espenak and Meeus.[^deltat]

DiffOrb also exposes a mixed `UT` view. This is a library convention, not an `IAU` standard time scale. It means `UT1` before `1962-01-01` and `UTC` on and after `1962-01-01`.

### Barycentric Dynamical Time

`TDB` is the practical barycentric dynamical scale used for state epochs and ephemeris arguments. It is not the same as `TCB`. It is designed to stay close to `TT` in average rate.[^iau][^kaplan]

DiffOrb uses the same practical `TT -> TDB` model as `SOFA` `iauDtdb`.[^sofa] The inverse `TDB -> TT` path is built by fixed-point inversion of that same forward model.

In the topocentric form of the model, `TDB` can depend on observer position. For that reason, `Time.tdb(...)` accepts an `ITRS` location. If no location is supplied, DiffOrb uses the geocentric case.

## Why Time Stores TT While State Uses TDB

These canonical choices serve different jobs. `Time` needs one uniform scale that can connect Earth-based inputs and outputs such as `UTC`, `UT1`, and `EOP` to barycentric dynamical work. `TT` is the bridge for that job. `State` is already a dynamical Cartesian object, so `TDB` is the more natural time scale for its epoch.

## Connections To The Rest Of The Library

[Earth Rotation And Terrestrial Geometry](earth-rotation-and-terrestrial-geometry.md) uses `TT`, `UT1`, and `EOP` together because the `ITRS -> GCRS` transformation depends on both the Earth's real spin and the precession-nutation and frame-bias model.

[Frames And State Representation](frames-and-state-representation.md) starts one layer later. When a Cartesian `State` is transformed between reference frames, its epoch is already handled as `TDB`.

Higher-level objects follow the same split. Ground rows in `Site` start with Earth-based time and Earth rotation. `EphemerisBody`, `SmallBody`, space rows in `Site`, and generic `State` transforms use `TDB` epochs.

## Read Next

- Read [Frames And State Representation](frames-and-state-representation.md) for the state-vector and reference-frame rules.
- Read [Earth Rotation And Terrestrial Geometry](earth-rotation-and-terrestrial-geometry.md) for the `TT`/`UT1`/`EOP` chain behind terrestrial geometry.
- Read [Earth Orientation Parameters](earth-orientation-parameters.md) for the measured Earth-rotation data used by
  modern `UT1` and terrestrial geometry.
- Use [Configure Earth Orientation Data](../guides/configure-earth-orientation-data.md) when you need to check or update
  the local `EOP` file.
- Use [Create And Convert Time Objects](../guides/create-and-convert-time-objects.md) when you need to build, inspect,
  batch, or shift `Time` objects.
- Use [Convert Between UTC, TT, TDB, UT1](../guides/convert-between-utc-tt-tdb-ut1.md) when you want the concrete conversion interfaces.
- Use the [Time API](../api/time.md) for details on `Time`, `TimeDelta`, and time-scale views.

## References

[^iau]: International Astronomical Union. Resolutions adopted at the XXIVth and XXVIth General Assemblies, especially the 1991 and 2000 resolutions on reference systems and time scales. <https://www.iau.org/Iau/Iau/Publications/List-of-Resolutions.aspx>
[^kaplan]: Kaplan, G. H. *The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation And Terrestrial Geometrys: Explanation and Implementation*, especially the sections on `TT`, `TDB`, `UTC`, and `UT1`.
[^sofa]: Standards of Fundamental Astronomy. *SOFA Tools for Earth Attitude* and *SOFA Time Scale and Calendar Tools*, including the two-part Julian date convention and the practical `iauDtdb` model. <https://www.iausofa.org/>
[^iers]: International Earth Rotation and Reference Systems Service. *IERS Conventions (2010)*, especially the sections on `UT1`, Earth rotation, and Earth orientation parameters.
[^deltat]: Espenak, F., & Meeus, J. *Polynomial Expressions for Delta T*. NASA Eclipse Web Site. <https://eclipse.gsfc.nasa.gov/SEhelp/deltatpoly2004.html>
