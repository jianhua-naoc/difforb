# Observer Site Keys And Observer Types

Small-body observations identify the receiver with an observer key. A key is both a station identity and, when needed, the source of the observer coordinates used by the reduction model.

`DiffOrb` uses one `Site` class for fixed ground, roving ground, and space-based observers. The observer type controls how `Site` resolves the key into a numerical state:

- Fixed ground keys are plain observatory codes such as `568`. The site position comes from the observatory-code table.
- Roving ground keys use `code @ lon_deg, lat_deg, alt_m` for observatory codes that the code table marks as roving sites. The coordinate payload is part of the station identity because the code alone does not distinguish one roving observing site from another.
- Space-based keys use `code # x_au, y_au, z_au`. The code is the station identity. The coordinate payload supplies the `GCRS` position needed for that observation, but it does not distinguish a different station identity.

## Observatory Code Tables

For optical observations, the standard source is the observatory-code list maintained by the Minor Planet Center (MPC) for the IAU.[^mpc-codes]

For radar observations, the JPL small-body radar astrometry page is the better source for radar-site geodetic coordinates.[^jpl-radar-sites] The MPC observatory-code list also contains most radar sites, but the JPL list gives better site positions, so `DiffOrb` uses the JPL radar observatory table for radar sites.

## Site State Model

`Site` stores a uniform batch of numerical fields. Ground rows store positions in the International Terrestrial Reference System (`ITRS`). Space rows store canonical positions in the Geocentric Celestial Reference System (`GCRS`), in astronomical units.

When a state is requested, `Site.state(...)` returns one `State` batch. Ground rows are converted from `ITRS` to `GCRS` through Earth rotation, then transformed to the requested frame if needed. Space rows use their stored `GCRS` payload directly. This keeps optical reduction, initial orbit determination, and differential correction on one observer-state path even when the input contains both terrestrial and space-based observations.

## Identity And Display

The optical observation table stores observer keys in `OpticalObservationData.rx_codes`. It does not keep a separate raw station code or separate observer-position columns.

Station grouping uses identity keys:

- A fixed ground key such as `568` groups as `568`.
- A roving key such as `247 @ 10, 20, 30` groups by the full key.
- A space key such as `C51 # 0.0001, 0.0002, 0.0003` groups as `C51`.

This distinction matters because roving coordinates identify the observing site, while space-observer coordinates are the instantaneous state data associated with an already distinct station code.

## Read Next

- Read [Earth Rotation And Terrestrial Geometry](earth-rotation-and-terrestrial-geometry.md) for the `ITRS -> GCRS` path used by Earth-based sites.
- Read [Frames And State Representation](frames-and-state-representation.md) for the state-vector and reference-frame model used by evaluated site states.
- Use the [Body API](../api/body.md) and [State API](../api/state.md) for details on site objects and returned observer states.

## References

[^mpc-codes]: Minor Planet Center. *Observatory Codes*. <https://www.minorplanetcenter.net/iau/lists/ObsCodesF.html>
[^jpl-radar-sites]: JPL Solar System Dynamics. *Small-Body Radar Astrometry*. <https://ssd.jpl.nasa.gov/sb/radar.html>
