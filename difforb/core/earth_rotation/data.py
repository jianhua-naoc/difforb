"""Coefficient tables for Earth-rotation models.

This module loads the constant series used by the IAU 2000/2006 short-term model and the Vondrak et al. (2011) long-term model. The arrays are shared by :mod:`difforb.core.earth_rotation.iau`, :mod:`difforb.core.earth_rotation.vondrak`, and :mod:`difforb.core.earth_rotation.unified`.
"""

import jax.numpy as jnp

from difforb.core.config import get_data_path
from difforb.core.constants import J2000, JULIAN_CENTURY

# =========================================================================
# IAU 2000/2006 Series Data
# =========================================================================
DEFAULT_IAU_NUTATION_DATA_FILENAME = "iau_nutation_series"
M_lunisolar = jnp.load(str(get_data_path("iau_nutation_series/M_lunisolar.npy", dataset="earth-rotation-series")))
M_planetary = jnp.load(str(get_data_path("iau_nutation_series/M_planetary.npy", dataset="earth-rotation-series")))
psi_coff_lunisolar = jnp.load(str(get_data_path("iau_nutation_series/psi_c_lunisolar.npy", dataset="earth-rotation-series"))) / 1e7
psi_coff_planetary = jnp.load(str(get_data_path("iau_nutation_series/psi_c_planetary.npy", dataset="earth-rotation-series"))) / 1e7
epsilon_coff_lunisolar = jnp.load(str(get_data_path("iau_nutation_series/epsilon_c_lunisolar.npy", dataset="earth-rotation-series"))) / 1e7
epsilon_coff_planetary = jnp.load(str(get_data_path("iau_nutation_series/epsilon_c_planetary.npy", dataset="earth-rotation-series"))) / 1e7

C_lunisolar = epsilon_coff_lunisolar[:, 0]
C_dot_lunisolar = epsilon_coff_lunisolar[:, 1]
S_prime_lunisolar = epsilon_coff_lunisolar[:, 2]

C_planetary = epsilon_coff_planetary[:, 0]
S_prime_planetary = epsilon_coff_planetary[:, 1]

S_lunisolar = psi_coff_lunisolar[:, 0]
S_dot_lunisolar = psi_coff_lunisolar[:, 1]
C_prime_lunisolar = psi_coff_lunisolar[:, 2]

S_planetary = psi_coff_planetary[:, 0]
C_prime_planetary = psi_coff_planetary[:, 1]

DEFAULT_IAU_CIO_LOCATOR_DATA_FILENAME = "iau_cio_locator_series"
coff_fundamental_args = jnp.load(str(get_data_path("iau_cio_locator_series/coff_fundamental_args.npy", dataset="earth-rotation-series")))
coff_sin = jnp.load(str(get_data_path("iau_cio_locator_series/coff_sin.npy", dataset="earth-rotation-series")))
coff_cos = jnp.load(str(get_data_path("iau_cio_locator_series/coff_cos.npy", dataset="earth-rotation-series")))
coff_sin_0, coff_sin_1, coff_sin_2, coff_sin_3, coff_sin_4 = (
    coff_sin[:33, :].squeeze(), coff_sin[33:36, :].squeeze(), coff_sin[36:61, :].squeeze(),
    coff_sin[61:65, :].squeeze(), coff_sin[65:, :].squeeze()
)
coff_cos_0, coff_cos_1, coff_cos_2, coff_cos_3, coff_cos_4 = (
    coff_cos[:33, :].squeeze(), coff_cos[33:36, :].squeeze(), coff_cos[36:61, :].squeeze(),
    coff_cos[61:65, :].squeeze(), coff_cos[65:, :].squeeze()
)

EPSILON_0_ARCSEC: float = 84381.406

# Switch boundary for IAU 2006
SHORT_TERM_START = (2378131.5 - J2000) / JULIAN_CENTURY  # 1799-01-01
SHORT_TERM_END = (2525323.5 - J2000) / JULIAN_CENTURY  # 2202-01-01

# =========================================================================
# Vondrak 2011 long-term precession data.
# All polynomial coefficients and periodic amplitudes are in arcseconds.
# =========================================================================

# Vondrak (2011) Table 2: periodic terms for ``CIP`` coordinates.
VONDRAK_CIP_PERIODS = jnp.array(
    [256.75, 708.15, 274.20, 241.45, 2309.00, 492.20, 396.10, 288.90, 231.10, 1610.00, 620.00, 157.87, 220.30, 1200.00])
VONDRAK_CIP_X_COS_COEFFS = jnp.array(
    [-819.940624, -8444.676815, 2600.009459, 2755.175630, -167.659835, 871.855056, 44.769698, -512.313065, -819.415595,
     -538.071099, -189.793622, -402.922932, 179.516345, -9.814756])
VONDRAK_CIP_X_SIN_COEFFS = jnp.array(
    [81491.287984, 787.163481, 1251.296102, -1257.950837, -2966.799730, 639.744522, 131.600209, -445.040117, 584.522874,
     -89.756563, 524.429630, -13.549067, -210.157124, -44.919798])
VONDRAK_CIP_Y_COS_COEFFS = jnp.array(
    [75004.344875, 624.033993, 1251.136893, -1102.212834, -2660.664980, 699.291817, 153.167220, -950.865637, 499.754645,
     -145.188210, 558.116553, -23.923029, -165.405086, 9.344131])
VONDRAK_CIP_Y_SIN_COEFFS = jnp.array(
    [1558.515853, 7774.939698, -2219.534038, -2523.969396, 247.850422, -846.485643, -1393.124055, 368.526116,
     749.045012, 444.704518, 235.934465, 374.049623, -171.330180, -22.899655])

# Vondrak (2011) Table 9: periodic terms for the ``CIO`` locator.
VONDRAK_CIO_LOCATOR_PERIODS = jnp.array(
    [256.75, 402.79, 708.15, 288.92, 274.20, 537.22, 241.45, 729.81, 483.00, 438.22, 128.38, 1552.00, 2022.00, 230.44])
VONDRAK_CIO_LOCATOR_COS_COEFFS = jnp.array(
    [861.759585, -3534.781660, -1757.969632, -379.971514, 808.400066, 528.646661, 566.991239, -164.251097, 239.102099,
     -239.146933, -61.768986, -279.716974, -96.750819, -57.265608])
VONDRAK_CIO_LOCATOR_SIN_COEFFS = jnp.array(
    [17367.906013, -206.865955, 937.453020, 794.788562, 101.350197, -509.801031, -302.310637, -538.092166, 383.848135,
     -373.925805, -344.946642, -85.660616, -132.781674, 38.452480])

# Vondrak (2011) Table 3: periodic terms for mean obliquity of the ecliptic.
VONDRAK_MEAN_OBLIQUITY_PERIODS = jnp.array(
    [409.90, 396.15, 537.22, 402.90, 417.15, 288.92, 4043.00, 306.00, 277.00, 203.00])
VONDRAK_MEAN_OBLIQUITY_COS_COEFFS = jnp.array(
    [753.872780, -247.805823, 379.471484, -53.880558, -90.109153, -353.600190, -63.115353, -28.248187, 17.703387,
     38.911307])
VONDRAK_MEAN_OBLIQUITY_SIN_COEFFS = jnp.array(
    [-1704.720302, -862.308358, 447.832178, -889.571909, 190.402846, -56.564991, -296.222622, -75.859952, 67.473503,
     3.014055])

# Vondrak (2011) Table 1: periodic terms for ecliptic-pole coordinates.
VONDRAK_ECLIPTIC_POLE_PERIODS = jnp.array([708.15, 2309.00, 1620.00, 492.20, 1183.00, 622.00, 882.00, 547.00])
VONDRAK_ECLIPTIC_POLE_X_COS_COEFFS = jnp.array(
    [-5486.751211, -17.127623, -617.517403, 413.442940, 78.614193, -180.732815, -87.676083, 46.140315])
VONDRAK_ECLIPTIC_POLE_X_SIN_COEFFS = jnp.array(
    [667.666730, -2354.886252, -428.152441, 376.202861, 184.778874, 335.321713, -185.138669, -120.972830])
VONDRAK_ECLIPTIC_POLE_Y_COS_COEFFS = jnp.array(
    [-684.661560, 2446.283880, 399.671049, -356.652376, -186.387003, -316.800070, 198.296071, 101.135679])
VONDRAK_ECLIPTIC_POLE_Y_SIN_COEFFS = jnp.array(
    [-5523.863691, -549.747450, -310.998056, 421.535876, -36.776172, -145.278396, -34.744450, 22.885731])
