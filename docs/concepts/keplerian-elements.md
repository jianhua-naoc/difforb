# Keplerian Elements

DiffOrb uses `KepElement` for osculating Keplerian elements at one Barycentric Dynamical Time (`TDB`) epoch. The object is an alternate representation of an orbit state, not a propagator and not a replacement for frame-aware Cartesian `State` objects.

`KepElement` is useful when the input or output product is naturally element-based, such as a Horizons element query, an initial orbit table, or an inspection product after propagation or orbit determination. Dynamics, observer geometry, and most frame transformations still operate on Cartesian states.

## Element Representation

DiffOrb stores a `KepElement` as:

| Attribute | Meaning | Unit |
| --- | --- | --- |
| `tdb` | Element epoch | `TDB` |
| `p` | Semi-latus rectum | `au` |
| `e` | Eccentricity | dimensionless |
| `inc` | Inclination | radians |
| `node` | Longitude of ascending node | radians |
| `peri` | Argument of perihelion | radians |
| `m` | Mean anomaly | radians |

The stored six-element array is ordered as `[p, e, inc, node, peri, m]`. Leading dimensions are batch dimensions, following the same scalar-and-batch convention described in [Batch Inputs And Shapes](batch-inputs-and-shapes.md).

The semi-major axis `a` is a derived property, not the stored distance element. `KepElement.from_classical(...)` accepts the classical set `(a, e, inc, node, peri, m)` and converts it to the stored semi-latus rectum using `p = a * (1 - e**2)`. Angle inputs to `from_classical(...)` are in degrees by default; the constructed object stores radians.

## Epoch And Cartesian Boundary

The epoch of a `KepElement` is stored as `TDB`. This matches the time scale used by `State` and by Solar-System dynamics in DiffOrb.

The Cartesian boundary of `KepElement` is fixed:

- `KepElement.state()` returns a `State` in `HELIO_ECLIP_J2000`.
- Position is in `au`.
- Velocity is in `au / day`.
- The frame origin is the Sun.
- The axis orientation is the heliocentric ecliptic-of-`J2000.0` convention used by the DiffOrb `HELIO_ECLIP_J2000` frame.

The reverse conversion follows the same boundary. `KepElement.from_state(...)` first converts the input `State` to `HELIO_ECLIP_J2000`, then extracts the element set. If the input state has an origin that requires Sun or Earth ephemerides for the frame conversion, pass the corresponding `EphemerisBody` objects.

## Relationship To State And SmallBody

`KepElement` describes an osculating two-body orbit matching one state at one epoch. Under perturbations, the corresponding elements change with time. A `KepElement` by itself is not an integrated trajectory.

`SmallBody.create(...)` accepts a `KepElement`, but `SmallBody` stores its initial orbit as canonical `BCRS`. That means there are two distinct boundaries:

- `KepElement.state()` converts elements to a heliocentric `HELIO_ECLIP_J2000` state.
- `SmallBody.create(elements, sun=sun)` converts that heliocentric state to the `BCRS` state stored on `body.orbit0`.

Keep this distinction explicit when comparing printed states or debugging frame conversions.

## Derived Quantities

`KepElement` exposes derived quantities for common inspection tasks:

- `a` returns the semi-major axis in `au`; it returns `inf` for the parabolic case.
- `v` returns the true anomaly in radians by solving the appropriate Kepler equation branch.
- `period` returns the two-body orbital period in days for elliptic cases and `inf` for non-periodic cases.
- `perit_jd` returns the perihelion time as a Julian Date for elliptic cases and `nan` for non-periodic cases.

These quantities are derived from the stored element set and the solar gravitational parameter used by the element conversion routines.

## Singular Cases

The classical Keplerian angles are singular or poorly conditioned for some geometries. Near zero inclination, the line of nodes is not well defined. Near zero eccentricity, the argument of perihelion and true anomaly are not individually well defined. Round-trip conversion can preserve the physical Cartesian state while producing angle values that are numerically unstable or convention-dependent in these cases.

Use `KepElement` when the classical element convention is the intended representation. Use Cartesian `State` objects when the orbit is close to a classical-element singularity or when downstream code only needs geometry and dynamics.

## Read Next

- Read [Frames And State Representation](frames-and-state-representation.md) for the Cartesian state and frame contracts used by `KepElement.state()`.
- Use [Create And Convert Keplerian Elements](../guides/create-and-convert-keplerian-elements.md) for the concrete construction and round-trip conversion path.
- Use [Create A SmallBody From State Or Elements](../guides/create-a-smallbody-from-state-or-elements.md) when an element set should become a propagated `SmallBody`.
- Use the [Core API](../api/core.md) for details on `KepElement`.
