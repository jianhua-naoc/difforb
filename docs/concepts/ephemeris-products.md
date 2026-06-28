# Ephemeris Products

DiffOrb builds ephemeris products from a fitted orbit or from a user-supplied orbit. These products share one idea: a
target is propagated first, then product-specific quantities are read from the same dense trajectory.

## Product Families

DiffOrb exposes several product families.

- Optical tables give right ascension, declination, topocentric angles, distance, phase angle, elongation, and
  modeled magnitude.
- Radar tables give delay, Doppler, range, and range rate.
- Vector tables give relative target-observer states.
- Element tables give osculating elements at requested epochs.
- Apsides tables give periapsis and apoapsis events.
- Close-approach tables give close-approach epochs, distances, and relative speeds.

## Corrections And References

Vector products can have three correction levels.

- A geometric vector evaluates target and observer at the receive epoch and applies no light-time correction.
- An astrometric vector solves one-way light time. The observer is evaluated at the receive epoch, and the target is
  evaluated at the emission epoch. The light-time solution can include Shapiro delay.[^shapiro]
- An apparent vector starts from the astrometric vector and applies stellar aberration.[^urban]

Optical products have astrometric and apparent angles. They do not expose a geometric optical sky position.
A direction with no light-time correction is usually not a useful optical observable.

The optical apparent level applies more corrections than the vector apparent level. It includes gravitational light
deflection by the Sun, stellar aberration, rotation to the true equator and equinox of date, and optional refraction for
ground observers.[^urban]

Radar products use the two-way light-time model described by Yeomans et al. (1992).[^yeomans] The delay can
include relativistic delay, solar-corona delay, and tropospheric delay.[^shapiro][^muhleman-anderson][^standish][^urban]
Radar Doppler is the derivative of the converged two-way delay with respect to the receive epoch.

Element, apsides, and close-approach products are derived from the propagated target trajectory. They do not apply
observer light-time corrections.

## Relation To JPL Horizons

DiffOrb provides Horizons-like optical, radar, vector, element, apsides, and close-approach products. The optical and
vector products use geometric, astrometric, and apparent concepts in the same broad sense as JPL Horizons.

One difference matters for direct comparison. DiffOrb's Earth-based apparent `RA/Dec` uses the modern equator-of-date
rotation described in [Earth Rotation And Terrestrial Geometry](earth-rotation-and-terrestrial-geometry.md). JPL
Horizons documents that its default Earth-based apparent `RA/Dec` uses the legacy `IAU 76/80` true-of-date system. The
right-ascension origin in that legacy system is offset by about `53 mas` from the modern `IAU 2006/2000A`
of-date origin.[^horizons]

## Read Next

- Read [Numerical Integrators And Dense Trajectories](numerical-integrators-and-dense-trajectories.md) for the
  trajectory shared by the products.
- Read [Light-Time Model](light-time-model.md) for one-way light time, two-way light time, and radar Doppler.
- Read [Photocenter Correction](photocenter-correction.md) for the optical center-of-light correction used in comet
  orbit fitting.
- Use [Ephemeris Products Workflow](../workflows/ephemeris-products-workflow.md) for an end-to-end product path.
- Use the [Ephemeris API](../api/ephemeris.md) for table fields and product calls.

## References

[^shapiro]: Shapiro, I. I. (1964). *Fourth Test of General Relativity*. Physical Review Letters, 13(26), 789-791.
<https://doi.org/10.1103/PhysRevLett.13.789>
[^urban]: Urban, S. E., & Seidelmann, P. K. (eds.). *Explanatory Supplement to the Astronomical Almanac*, especially
the chapters on astrometric and apparent place, relativity, and tropospheric delay.
[^yeomans]: Yeomans, D. K., Campbell, D. B., Chodas, P. W., Giorgini, J. D., & Ostro, S. J. (1992). *Asteroid and
Comet Orbits Using Radar Data*. The Astronomical Journal, 103(1), 303-317.
[^muhleman-anderson]: Muhleman, D. O., & Anderson, J. D. (1981). *Solar wind electron densities from Viking
dual-frequency radio measurements*. The Astrophysical Journal, 247, 1093-1101. NASA NTRS record:
<https://ntrs.nasa.gov/citations/19810061604>
[^standish]: Standish, E. M., & Williams, J. G. *Orbital Ephemerides of the Sun, Moon, and Planets*, in *Explanatory
Supplement to the Astronomical Almanac*, especially Section 8.7.6.
[^horizons]: JPL Solar System Dynamics. *Horizons System Manual*, especially the sections on Earth true equator and
equinox of date, apparent `RA/Dec`, and the documented `-53 mas` offset between the legacy `IAU 76/80` and modern
`IAU 2006/2000A` of-date right-ascension origins. <https://ssd.jpl.nasa.gov/horizons/manual.html>
