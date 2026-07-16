# Earth Orientation Parameters

`EOP` are observed time series used to realize the orientation of Earth for high-precision time conversion and
ground-site geometry.[^iers-conventions] DiffOrb uses them for `UTC -> UT1` conversion and for the `ITRS -> GCRS`
transformation used by ground sites.

## What EOP Provides

Earth rotation is not fully represented by a uniform spin rate or by a precession-nutation model alone. `ERA` depends
on `UT1`, so DiffOrb needs observed `UT1 - UTC` values to connect civil time to the Earth's actual rotation angle. The
`CIP` also has time-dependent coordinates in `ITRS`, so DiffOrb needs polar-motion coordinates for the `ITRS -> TIRS`
step.

The precession-nutation model gives the conventional motion of the `CIP` in `GCRS`. It is not exact. Very small residual
terms remain, including unmodeled Free Core Nutation (`FCN`) and other model errors. `EOP` products provide observed
correction terms for those residuals; they do not mean that an observed axis itself is being "corrected."[^kaplan][^iers-ch5]

`EOP` supplies these measured values as a date-based series.[^iers-c04] The series has three jobs in DiffOrb:

- It connects `UTC` and `UT1` for modern dates.
- It gives polar-motion coordinates for the `ITRS -> TIRS` step of the `ITRS -> GCRS` transformation.
- It supplies observed correction terms for the modeled `CIP` coordinates in `GCRS`.

DiffOrb uses the C04 `dPsi/dEps` form.[^iers-c04] The `dPsi` and `dEps` values from the `EOP` table are added to the
model nutation angles before the corrected `CIP` vector and the `CIRS <-> GCRS` matrices are built. Their role is to
realize the observationally corrected `CIP` motion, not to change the precession polynomial by itself.

## Coverage And Freshness

`EOP` is observed data. It has a finite date range. It must be updated when recent Earth rotation matters.

A stale file can still be readable, but it may not describe the latest rotation of Earth. This matters most for recent
ground-based observations, radar geometry, and comparisons with other ephemeris services.

DiffOrb uses an explicit boundary policy for the `EOP` quantities that enter terrestrial geometry. For epochs before the
first covered `EOP` sample, the polar-motion coordinates and the observed `dPsi/dEps` correction terms are set to zero.
This omits polar motion and the observed correction terms for the modeled `CIP` coordinates. For epochs after the last
covered sample, those quantities stay at the final value in the loaded `EOP` table. DiffOrb does not extrapolate a new
polar-motion or correction-term model beyond the table.

The current default product is the IERS `EOP 20 C04 0h dPsi/dEps 1962-now` series. Future versions may add loaders for
other EOP products and correction forms.

With the current default product, DiffOrb defines `UTC` only on and after `1962-01-01`. Earlier epochs do not have a
`UTC` representation in DiffOrb; use `UT1` or mixed `UT` instead.

## Where It Matters

`EOP` matters when a calculation depends on the real orientation of Earth at a given date. Common cases include:

- Converting between `UTC` and `UT1`.
- Transforming ground-site coordinates between `ITRS` and `GCRS`, including topocentric optical observations and
  radar transmitters or receivers on Earth.

Space-based observations usually do not need EOP directly when the observer state is already expressed in a frame such
as `BCRS` or `GCRS`.

## Read Next

- Read [Time Scales And Epoch Storage](time-scales-and-epoch-storage.md) for the time-scale rules that connect
  atomic time, civil time, Earth-rotation time, and barycentric dynamical time.
- Read [Earth Rotation And Terrestrial Geometry](earth-rotation-and-terrestrial-geometry.md) for the full
  `ITRS -> GCRS` Earth-rotation model.
- Use [Configure Earth Orientation Data](../guides/configure-earth-orientation-data.md) when you need to check or
  update the local EOP file.
- Use [Get Earth Rotation Quantities And Matrices](../guides/get-earth-rotation-quantities-and-matrices.md) when you
  want the concrete Earth-rotation values used by one epoch.
- Use the [Earth Orientation Parameters API](../api/eop.md) and [Time API](../api/time.md) for details on EOP storage
  and time-scale views.

## References

[^iers-conventions]: International Earth Rotation and Reference Systems Service. *IERS Conventions (2010)*, especially
the sections on Earth orientation, polar motion, and celestial intermediate quantities.
[^iers-c04]: International Earth Rotation and Reference Systems Service. *EOP 20 C04 (IAU 2000A, dPsi, dEps), 0h UTC, 1962-now*. <https://datacenter.iers.org/versionMetadata.php?filename=latestVersionMeta%2F236_EOP_C04_20_dPsi_dEps_62-NOW_IAU2000236.txt>
[^kaplan]: Kaplan, G. H. *The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models: Explanation and Implementation*, especially Sections 5.4.4 and 6.5.1.
[^iers-ch5]: International Earth Rotation and Reference Systems Service. *IERS Conventions (2010)*, Chapter 5, "Transformation between the International Terrestrial Reference System and the Geocentric Celestial Reference System." <https://www.iers.org/IERS/EN/Publications/TechnicalNotes/tn36>
