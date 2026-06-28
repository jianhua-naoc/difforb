import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

ARCSEC_TO_DEG = jnp.array(1. / 3600., dtype=jnp.float64)
DEG_TO_ARCSEC = jnp.array(3600., dtype=jnp.float64)
RAD_TO_ARCSEC = jnp.array(1. / jnp.pi * 180 * 3600., dtype=jnp.float64)

J2000 = jnp.array(2451545.0, dtype=jnp.float64)
DAY_S = jnp.array(86400.0, dtype=jnp.float64)
JULIAN_CENTURY = jnp.array(36525.0, dtype=jnp.float64)
MJD_START = jnp.array(2400000.5, dtype=jnp.float64)
LC = jnp.array(1.48082686741e-8, dtype=jnp.float64)

# Ref: NASA JPL Horizon (from IAU 2015)
R_SUN = jnp.array(0.004650467260962157, dtype=jnp.float64)  # AU, solar radius

# Ref: DE441 tech comments
AU_M = jnp.array(149597870700.0, dtype=jnp.float64)
AU_KM = jnp.array(149597870.7, dtype=jnp.float64)
C_KM_SEC = jnp.array(2.9979245800000000e5, dtype=jnp.float64)
C = C_KM_SEC / AU_KM * DAY_S  # AU/DAY
INV_C = jnp.reciprocal(C)
INV_C2 = jnp.reciprocal(C * C)
J2_SUN = jnp.array(2.1961391516529825e-7, dtype=jnp.float64)
# AU^3/DAY^2
GM_SUN = jnp.array(2.9591220828411951e-04, dtype=jnp.float64)
