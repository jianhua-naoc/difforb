# Dynamical Models

DiffOrb propagates one target small body in the Barycentric Celestial Reference System (`BCRS`) as a function of
Barycentric Dynamical Time (`TDB`). A dynamical model gives the acceleration used by that propagation.

## Core Model

DiffOrb only integrates the target small body. It does not integrate the Sun, planets, Moon, Pluto, or massive asteroid
perturbers. Those perturbing bodies are read from SPK-format ephemerides.

## Ephemeris-Backed Bodies

DiffOrb can load one SPK kernel or a list of SPK kernels. The kernels may store segments with different centers.
DiffOrb finds and combines the needed path segments so a body state can be evaluated with respect to the Solar System
Barycenter.

Many DE and asteroid SPK kernels store Chebyshev coefficients for position. DiffOrb evaluates the position at the
requested `TDB` epoch. It then obtains velocity and acceleration from the same position interpolation by automatic
differentiation. This keeps position, velocity, and acceleration tied to one interpolation model.

## Gravitational Terms

DiffOrb supports additive gravitational force terms.[^urban]

- Newtonian point-mass gravity gives the inverse-square acceleration from selected bodies.
- Point-mass parameterized post-Newtonian (`PPN`) gravity adds the relativistic correction used for Solar System
  propagation.
- The second zonal harmonic (`J2`) models the leading oblateness effect of the Sun or Earth.

Users can choose which bodies use Newtonian gravity and which bodies use `PPN` gravity. Common setups use `PPN` for
the Sun and major planets, and Newtonian gravity for selected massive asteroids.

## Non-Gravitational Terms

For small bodies, DiffOrb uses empirical non-gravitational acceleration models for cometary outgassing, the Yarkovsky
effect, and radiation pressure.[^marsden][^yarkovsky][^radiation] These models are written in one heliocentric
radial-transverse-normal (`RTN`) form:

\[
\mathbf{a}_{\mathrm{ng}} =
g(r)\left(A_1 \hat{\mathbf{r}} + A_2 \hat{\mathbf{t}} + A_3 \hat{\mathbf{n}}\right).
\]

Here \( r \) is the heliocentric distance. The unit vectors \( \hat{\mathbf{r}} \), \( \hat{\mathbf{t}} \), and
\( \hat{\mathbf{n}} \) are the orbital radial, transverse, and normal directions. The parameters \( A_i \)
\((i=1,2,3)\) are fitted acceleration parameters at a heliocentric distance of `1 au`. `A1` is radial, `A2` is
transverse, and `A3` is normal. \( g(r) \) is a heliocentric-distance law.

For cometary outgassing, DiffOrb uses the Marsden cometary outgassing model.[^marsden] In this model, \( g(r) \)
represents the sublimation rate as a function of heliocentric distance. The default \( g(r) \) corresponds to water-ice
sublimation. Its shape parameters can be changed to represent sublimation processes of other materials, such as sodium
or forsterite sublimation.[^sekanina-2015][^sekanina-2014]

By setting

\[
g(r)=\left({1\,\mathrm{au}\over r}\right)^2,
\]

the same `RTN` form can model the Yarkovsky effect as the purely transverse acceleration \( A_2 g(r) \), and radiation
pressure as the purely radial acceleration \( A_1 g(r) \).[^farnocchia-2013]

These are empirical acceleration models for orbit determination. They are not complete physical models of gas flow,
thermal emission, or surface scattering.

## Model Boundary

The force model defines acceleration. It does not define observations. Optical and radar observations use the
propagated target state, site states, and light-time models in later layers.

The force model also does not choose the numerical method. That belongs to the integrator layer.

## Read Next

- Read [Numerical Integrators And Dense Trajectories](numerical-integrators-and-dense-trajectories.md) for the
  propagation layer that uses the dynamical model.
- Read [Light-Time Model](light-time-model.md) for one-way optical light time, two-way radar light time, and radar
  Doppler.
- Use [Configure Force Models And Dynamic Systems](../guides/configure-force-models-and-dynamic-systems.md) when you
  need the concrete force-model setup.
- Use the [Dynamics API](../api/dynamics.md) for symbol-level details.

## References

[^urban]: Urban, S. E., & Seidelmann, P. K. (eds.). *Explanatory Supplement to the Astronomical Almanac*, especially
the sections on Solar System equations of motion.
[^marsden]: Marsden, B. G., Sekanina, Z., & Yeomans, D. K. (1973). *Comets and nongravitational forces. V*. The
Astronomical Journal, 78, 211-225. <https://doi.org/10.1086/111402>
[^yarkovsky]: Vokrouhlicky, D., Bottke, W. F., Chesley, S. R., Scheeres, D. J., & Statler, T. S. (2015). *The
Yarkovsky and YORP Effects*. In P. Michel, F. DeMeo, & W. Bottke (eds.), *Asteroids IV*, 509-531.
<https://doi.org/10.2458/azu_uapress_9780816532131-ch027>
[^radiation]: Vokrouhlicky, D., & Milani, A. (2000). *Direct solar radiation pressure on the orbits of small
near-Earth asteroids: observable effects?* Astronomy and Astrophysics, 362, 746-755.
[^sekanina-2015]: Sekanina, Z., & Kracht, R. (2015). *Strong Erosion-Driven Nongravitational Effects in Orbital
Motions of the Kreutz Sungrazing System's Dwarf Comets*. The Astrophysical Journal, 801, 135.
<https://doi.org/10.1088/0004-637X/801/2/135>
[^sekanina-2014]: Sekanina, Z., & Kracht, R. (2014). *Disintegration of Comet C/2012 S1 (ISON) Shortly Before
Perihelion: Evidence from Independent Data Sets*. arXiv. <https://doi.org/10.48550/arXiv.1404.5968>
[^farnocchia-2013]: Farnocchia, D., Chesley, S. R., Vokrouhlicky, D., Milani, A., Spoto, F., & Bottke, W. F. (2013).
*Near Earth Asteroids with measurable Yarkovsky effect*. Icarus, 224(1), 1-13.
<https://doi.org/10.1016/j.icarus.2013.02.004>
