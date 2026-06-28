# Frames And State Representation

DiffOrb uses `State` for Cartesian state vectors. A `State` stores one position, one velocity, one Barycentric Dynamical Time (`TDB`) epoch, and one reference frame. The reference frame tells DiffOrb how to interpret the numbers.

This model is used for dynamical states, ephemeris states, propagated small-body states, and observer states after they have been expressed in the Geocentric Celestial Reference System (`GCRS`) or another supported reference frame. It does not describe apparent place, topocentric place, or right ascension and declination. Those quantities are produced later by measurement-reduction code.

## Core Model

A state vector is not just six numbers. It is a position and velocity at one epoch in one reference frame.

A reference frame answers two questions:

- Where is the origin?
- How are the axes oriented?

Both parts matter. A vector relative to the Solar System Barycenter is not the same vector as one relative to the Earth or the Sun. A vector written with International Celestial Reference System (`ICRS`) axes is not the same numerical vector as one written with ecliptic-of-`J2000.0` axes.

Reference-frame conversion changes the numerical expression of the same physical state. It does not change the physical object or the epoch of the state.

## Supported Reference Frames

DiffOrb currently supports the following reference frames for `State` objects:

| Reference frame | Frame name in DiffOrb | Origin | Axis orientation |
| --- | --- | --- | --- |
| Barycentric Celestial Reference System | `BCRS` | Solar System Barycenter | International Celestial Reference System (`ICRS`) axes |
| Geocentric Celestial Reference System | `GCRS` | Earth center | `ICRS` axes |
| Heliocentric `ICRS` | `HELIO_ICRS` | Sun center | `ICRS` axes |
| Heliocentric `J2000.0` mean-equator frame | `HELIO_J2000` | Sun center | `J2000.0` mean-equator axes |
| Heliocentric ecliptic-of-`J2000.0` frame | `HELIO_ECLIP_J2000` | Sun center | `ICRS` axes rotated around the x-axis by a standard fixed obliquity angle (`epsilon`) of `84381.448` arcseconds, consistent with the Jet Propulsion Laboratory (`JPL`) Horizons ecliptic-of-`J2000.0` frame.[^horizons-frames][^horizons-api-eclip] |

## State Representation In DiffOrb

DiffOrb stores four pieces of information in a `State`:

- One epoch.
- One Cartesian position.
- One Cartesian velocity.
- One reference frame.

DiffOrb stores these values as follows:

- The epoch is stored as a `TDB` view.
- `pos` is in `au`.
- `vel` is in `au / day`.
- The last dimension of `pos` and `vel` has length `3`.
- Leading dimensions are batch dimensions.
- The stacked six-component order is `[x, y, z, vx, vy, vz]`.
- The reference frame is always stored with the state.

The `TDB`, `pos`, and `vel` values must have the same batch shape. A scalar state has no leading batch dimension. A batch of states keeps the same final vector dimension and adds leading batch dimensions.

In code, use the DiffOrb frame names in the table, such as `BCRS`, `GCRS`, and `HELIO_ECLIP_J2000`, when you create or convert a `State`. Each name refers to the origin and axis orientation shown in the same row.

See [Build State From Cartesian Data](../guides/build-state-from-cartesian-data.md) for the concrete construction interface.

## What A State Is Not

A `State` is not an observed place.

It is not an apparent place, an astrometric place, a topocentric place, or a pair of right ascension and declination values. Those names describe positions after extra modeling choices, such as observer location, light time, aberration, gravitational light deflection, Earth rotation, or atmospheric refraction.

DiffOrb keeps that separation explicit. `State` stores the geometric state vector used by dynamics and geometry. Optical and radar reduction later turn target and observer states into measured or predicted observables.

## Reference-Frame Conversion

Reference-frame conversion can include two different operations:

- An axis change rotates the position and velocity.
- An origin change shifts the position and velocity using the origin state at the same epoch.

Origin changes can require ephemeris data. For example, a conversion involving a Sun-centered frame needs the Sun state. A conversion involving a geocentric frame needs the Earth state.

The epoch remains the same physical epoch during reference-frame conversion. DiffOrb does not use reference-frame conversion to change the time scale or move the object forward in time.

See [Transform State Between Frames](../guides/transform-state-between-frames.md) for the concrete conversion interface.

## Boundary With Earth Rotation

The International Terrestrial Reference System (`ITRS`) is not the same kind of frame as the reference frames used directly by `State` conversion.

The `ITRS -> GCRS` transformation for ground-site coordinates depends on Terrestrial Time (`TT`), Universal Time 1 (`UT1`), Earth Orientation Parameters (`EOP`), polar motion, and Earth rotation. That model belongs to [Earth Rotation And Terrestrial Geometry](earth-rotation-and-terrestrial-geometry.md).

After a ground site has been expressed in `GCRS`, the resulting observer state follows the same state-vector and reference-frame model described here.

## Where This Model Is Used

`EphemerisBody` returns `State` objects with reference frames. `SmallBody` accepts or returns `State` objects when it connects propagation to Cartesian state vectors. `Site` evaluates fixed ground, roving ground, and space-based observers into the same `State` model before downstream reduction code uses them.

## Read Next

- Read [Time Scales And Epoch Storage](time-scales-and-epoch-storage.md) for the time-system rules behind `TDB` epochs.
- Read [Keplerian Elements](keplerian-elements.md) for the element representation that converts through `HELIO_ECLIP_J2000` states.
- Read [Earth Rotation And Terrestrial Geometry](earth-rotation-and-terrestrial-geometry.md) for the `ITRS -> GCRS` path used by ground sites.
- Read [Observer Site Keys And Observer Types](observer-site-classes-and-observer-types.md) for how observer keys connect to site geometry.
- Use [Build State From Cartesian Data](../guides/build-state-from-cartesian-data.md) to construct `State` objects.
- Use [Transform State Between Frames](../guides/transform-state-between-frames.md) for the concrete reference-frame conversion interfaces.
- Use the [State API](../api/state.md) for details on `State`, frames, axes, and origins.

## References

[^horizons-frames]: NASA/JPL Solar System Dynamics. *Horizons System Manual*, "Reference Frames" and "Ecliptic of Standard Epoch (J2000 or B1950)." <https://ssd.jpl.nasa.gov/horizons/manual.html>
[^horizons-api-eclip]: NASA/JPL Solar System Dynamics. *Horizons API Documentation*, user-specified heliocentric ecliptic osculating elements and the `ECLIP` parameter. <https://ssd-api.jpl.nasa.gov/doc/horizons.html>
