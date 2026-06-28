import jax
import jax.numpy as jnp
import numpy as np

from difforb.core.constants import J2000, JULIAN_CENTURY
from difforb.core.earth_rotation.vondrak import (
    vondrak_cip_xyz_single,
    vondrak_cio_locator_single,
    vondrak_mean_obliquity_single,
    vondrak_mean_poles_single,
    vondrak_precession_bias_matrix_single,
)
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)

VONDRAK_APPENDIX_A5_TT_JD1 = 1219339.0
VONDRAK_APPENDIX_A5_TT_JD2 = 0.078
VONDRAK_APPENDIX_A5_LABEL = "-1374-05-03 13:52:19.2 TT"

VONDRAK_APPENDIX_A5_MEAN_ECLIPTIC_POLE = jnp.array([
    0.00041724785764001342,
    -0.40495491104576162693,
    0.91433656053126552350,
], dtype=jnp.float64)

VONDRAK_APPENDIX_A5_MEAN_EQUATOR_POLE = jnp.array([
    -0.29437643797369031532,
    -0.11719098023370257855,
    0.94847708824082091796,
], dtype=jnp.float64)

VONDRAK_APPENDIX_A5_PRECESSION_BIAS_MATRIX = jnp.array([
    [0.68473392912753224372, 0.66647788221176470103, 0.29486722236305384992],
    [-0.66669476463873305255, 0.73625641199831485100, -0.11595079385100924091],
    [-0.29437652267952261218, -0.11719099075396051880, 0.94847706065103424635],
], dtype=jnp.float64)

VONDRAK_MEAN_OBLIQUITY_PERIODS = np.array(
    [409.90, 396.15, 537.22, 402.90, 417.15, 288.92, 4043.00, 306.00, 277.00, 203.00],
    dtype=np.float64,
)
VONDRAK_MEAN_OBLIQUITY_COS_COEFFS = np.array(
    [753.872780, -247.805823, 379.471484, -53.880558, -90.109153, -353.600190, -63.115353, -28.248187, 17.703387,
     38.911307],
    dtype=np.float64,
)
VONDRAK_MEAN_OBLIQUITY_SIN_COEFFS = np.array(
    [-1704.720302, -862.308358, 447.832178, -889.571909, 190.402846, -56.564991, -296.222622, -75.859952, 67.473503,
     3.014055],
    dtype=np.float64,
)
VONDRAK_CIO_LOCATOR_PERIODS = np.array(
    [256.75, 402.79, 708.15, 288.92, 274.20, 537.22, 241.45, 729.81, 483.00, 438.22, 128.38, 1552.00, 2022.00, 230.44],
    dtype=np.float64,
)
VONDRAK_CIO_LOCATOR_COS_COEFFS = np.array(
    [861.759585, -3534.781660, -1757.969632, -379.971514, 808.400066, 528.646661, 566.991239, -164.251097, 239.102099,
     -239.146933, -61.768986, -279.716974, -96.750819, -57.265608],
    dtype=np.float64,
)
VONDRAK_CIO_LOCATOR_SIN_COEFFS = np.array(
    [17367.906013, -206.865955, 937.453020, 794.788562, 101.350197, -509.801031, -302.310637, -538.092166, 383.848135,
     -373.925805, -344.946642, -85.660616, -132.781674, 38.452480],
    dtype=np.float64,
)


def _appendix_table3_mean_obliquity(tt_jd1, tt_jd2):
    t = ((tt_jd1 - float(J2000)) + tt_jd2) / float(JULIAN_CENTURY)
    obliquity_poly = 84028.206305 + t * (0.3624445 + t * (-0.00004039 + t * (-110e-9)))
    obliquity_args = 2.0 * np.pi * t / VONDRAK_MEAN_OBLIQUITY_PERIODS
    obliquity_periodic = np.sum(
        VONDRAK_MEAN_OBLIQUITY_COS_COEFFS * np.cos(obliquity_args)
        + VONDRAK_MEAN_OBLIQUITY_SIN_COEFFS * np.sin(obliquity_args)
    )
    return np.deg2rad((obliquity_poly + obliquity_periodic) / 3600.0)


def _table9_cio_locator(tt_jd1, tt_jd2):
    t = ((tt_jd1 - float(J2000)) + tt_jd2) / float(JULIAN_CENTURY)
    cio_locator_poly = 3566.723572 + t * (-414.3015011 + t * (0.00085448 + t * (365e-9)))
    cio_locator_args = 2.0 * np.pi * t / VONDRAK_CIO_LOCATOR_PERIODS
    cio_locator_periodic = np.sum(
        VONDRAK_CIO_LOCATOR_COS_COEFFS * np.cos(cio_locator_args)
        + VONDRAK_CIO_LOCATOR_SIN_COEFFS * np.sin(cio_locator_args)
    )
    return np.deg2rad((cio_locator_poly + cio_locator_periodic) / 3600.0)


def test_vondrak_mean_poles_single_against_appendix_a5():
    t = ((VONDRAK_APPENDIX_A5_TT_JD1 - J2000) + VONDRAK_APPENDIX_A5_TT_JD2) / JULIAN_CENTURY

    actual_mean_equator_pole, actual_mean_ecliptic_pole = vondrak_mean_poles_single(t)

    mean_equator_max_abs_diff = jnp.max(jnp.abs(actual_mean_equator_pole - VONDRAK_APPENDIX_A5_MEAN_EQUATOR_POLE))
    mean_ecliptic_max_abs_diff = jnp.max(jnp.abs(actual_mean_ecliptic_pole - VONDRAK_APPENDIX_A5_MEAN_ECLIPTIC_POLE))

    print(
        "[vondrak_mean_poles_single] "
        f"label={VONDRAK_APPENDIX_A5_LABEL:<22} "
        f"equator_max_abs_diff={float(mean_equator_max_abs_diff):+.12e} "
        f"ecliptic_max_abs_diff={float(mean_ecliptic_max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual_mean_equator_pole,
        VONDRAK_APPENDIX_A5_MEAN_EQUATOR_POLE,
        atol=5e-15,
        rtol=0.0,
        msg="Vondrak mean equator pole mismatch for Appendix A.5 test case",
    )
    assert_allclose(
        actual_mean_ecliptic_pole,
        VONDRAK_APPENDIX_A5_MEAN_ECLIPTIC_POLE,
        atol=5e-15,
        rtol=0.0,
        msg="Vondrak mean ecliptic pole mismatch for Appendix A.5 test case",
    )


def test_vondrak_mean_obliquity_single_against_appendix_a5():
    t = ((VONDRAK_APPENDIX_A5_TT_JD1 - J2000) + VONDRAK_APPENDIX_A5_TT_JD2) / JULIAN_CENTURY

    actual = vondrak_mean_obliquity_single(t)
    expected = _appendix_table3_mean_obliquity(VONDRAK_APPENDIX_A5_TT_JD1, VONDRAK_APPENDIX_A5_TT_JD2)
    diff_rad = actual - expected

    print(
        "[vondrak_mean_obliquity_single] "
        f"label={VONDRAK_APPENDIX_A5_LABEL:<22} "
        f"diff={float(diff_rad):+.12e} rad"
    )

    assert_allclose(
        actual,
        expected,
        atol=5e-15,
        rtol=0.0,
        msg="Vondrak mean obliquity mismatch for Vondrak (2011) Eq.10/Table 3 test case",
    )


def test_vondrak_precession_bias_matrix_single_against_appendix_a5():
    t = ((VONDRAK_APPENDIX_A5_TT_JD1 - J2000) + VONDRAK_APPENDIX_A5_TT_JD2) / JULIAN_CENTURY

    actual = vondrak_precession_bias_matrix_single(t)
    max_abs_diff = jnp.max(jnp.abs(actual - VONDRAK_APPENDIX_A5_PRECESSION_BIAS_MATRIX))

    print(
        "[vondrak_precession_bias_matrix_single] "
        f"label={VONDRAK_APPENDIX_A5_LABEL:<22} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        VONDRAK_APPENDIX_A5_PRECESSION_BIAS_MATRIX,
        atol=5e-15,
        rtol=0.0,
        msg="Vondrak precession-bias matrix mismatch for Appendix A.5 test case",
    )


def test_vondrak_cip_xyz_single_against_appendix_a5():
    t = ((VONDRAK_APPENDIX_A5_TT_JD1 - J2000) + VONDRAK_APPENDIX_A5_TT_JD2) / JULIAN_CENTURY

    actual_x, actual_y, actual_z = vondrak_cip_xyz_single(t)
    actual = jnp.array([actual_x, actual_y, actual_z], dtype=jnp.float64)
    expected = VONDRAK_APPENDIX_A5_PRECESSION_BIAS_MATRIX[2]
    max_abs_diff = jnp.max(jnp.abs(actual - expected))

    print(
        "[vondrak_cip_xyz_single] "
        f"label={VONDRAK_APPENDIX_A5_LABEL:<22} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        expected,
        atol=5e-15,
        rtol=0.0,
        msg="Vondrak CIP xyz mismatch for Appendix A.5 test case",
    )


def test_vondrak_cio_locator_single_against_table9():
    t = ((VONDRAK_APPENDIX_A5_TT_JD1 - J2000) + VONDRAK_APPENDIX_A5_TT_JD2) / JULIAN_CENTURY

    actual = vondrak_cio_locator_single(t)
    expected = _table9_cio_locator(VONDRAK_APPENDIX_A5_TT_JD1, VONDRAK_APPENDIX_A5_TT_JD2)
    diff_rad = actual - expected

    print(
        "[vondrak_cio_locator_single] "
        f"label={VONDRAK_APPENDIX_A5_LABEL:<22} "
        f"diff={float(diff_rad):+.12e} rad"
    )

    assert_allclose(
        actual,
        expected,
        atol=5e-15,
        rtol=0.0,
        msg="Vondrak CIO locator mismatch for Vondrak (2011) Eq.26/Table 9 test case",
    )
