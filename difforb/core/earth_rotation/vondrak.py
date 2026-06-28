"""Long-term Earth-rotation kernels based on the Vondrak et al. model.

This module provides single-epoch kernels for the long-term precession model of Vondrak, Capitaine, and Wallace (2011). The routines cover mean obliquity, mean pole directions, the Celestial Intermediate Pole (``CIP``), the Celestial Intermediate Origin (``CIO``) locator, and an equinox-based precession-bias rotation matrix used outside the shorter IAU 2006/2000A validity interval.

All time arguments are Julian centuries since J2000 in ``TT``. Periodic coefficients and secular terms are imported from :mod:`difforb.core.earth_rotation.data`.
"""
import jax
from jax import Array
from jaxtyping import Float
from typing import Tuple
from difforb.utils import arcsec_to_rad
from difforb.core.earth_rotation.data import *

jax.config.update("jax_enable_x64", True)


def vondrak_mean_obliquity_single(t: Float[Array, ""]) -> Float[Array, ""]:
    """
    Compute the long-term mean obliquity of the ecliptic.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, ""]
        Mean obliquity in radians for the mean equator and ecliptic of date.

    References
    ----------
    1. Vondrak, J., Capitaine, N., & Wallace, P. T. (2011). New precession expressions, valid for long time intervals. Eq.10.
    """
    obliquity_poly = 84028.206305 + t * (0.3624445 + t * (-0.00004039 + t * (-110e-9)))
    obliquity_args = 2 * jnp.pi * t / VONDRAK_MEAN_OBLIQUITY_PERIODS
    obliquity_periodic = jnp.sum(VONDRAK_MEAN_OBLIQUITY_COS_COEFFS * jnp.cos(obliquity_args) +
                                 VONDRAK_MEAN_OBLIQUITY_SIN_COEFFS * jnp.sin(obliquity_args))

    return arcsec_to_rad(obliquity_poly + obliquity_periodic)


def vondrak_mean_poles_single(t: Float[Array, ""]) -> Tuple[Float[Array, "3"], Float[Array, "3"]]:
    """
    Compute the mean equator and mean ecliptic pole vectors.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Tuple[Float[Array, "3"], Float[Array, "3"]]
        ``(mean_equator_pole_vector, mean_ecliptic_pole_vector)`` as unit vectors in the J2000.0 mean equator and equinox system.

    Notes
    -----
    The first vector is the long-term mean pole of the Earth's equator. The second vector is rotated from the ecliptic model coefficients into the same J2000.0 frame so both poles can be combined to build equinox-based rotation matrices.

    References
    ----------
    1. Vondrak, J., Capitaine, N., & Wallace, P. T. (2011). New precession expressions, valid for long time intervals. Eq.9.
    """
    # Step 1: Build the mean equator pole in the J2000.0 system.
    cip_x_poly = 5453.282155 + t * (0.4252841 + t * (-0.00037173 + t * (-152e-9)))
    cip_y_poly = -73750.930350 + t * (-0.7675452 + t * (-0.00018725 + t * (231e-9)))

    cip_args = 2 * jnp.pi * t / VONDRAK_CIP_PERIODS
    sin_cip_args, cos_cip_args = jnp.sin(cip_args), jnp.cos(cip_args)

    mean_equator_pole_x = arcsec_to_rad(
        cip_x_poly + jnp.sum(VONDRAK_CIP_X_COS_COEFFS * cos_cip_args + VONDRAK_CIP_X_SIN_COEFFS * sin_cip_args))
    mean_equator_pole_y = arcsec_to_rad(
        cip_y_poly + jnp.sum(VONDRAK_CIP_Y_COS_COEFFS * cos_cip_args + VONDRAK_CIP_Y_SIN_COEFFS * sin_cip_args))
    mean_equator_pole_z = jnp.sqrt(jnp.maximum(1.0 - mean_equator_pole_x ** 2 - mean_equator_pole_y ** 2, 0.))

    mean_equator_pole_vector = jnp.array([mean_equator_pole_x, mean_equator_pole_y, mean_equator_pole_z])

    # Step 2: Build the mean ecliptic pole.
    ecliptic_x_poly = 5851.607687 + t * (-0.1189000 + t * (-0.00028913 + t * (101e-9)))
    ecliptic_y_poly = -1600.886300 + t * (1.1689818 + t * (-0.00000020 + t * (-437e-9)))

    ecliptic_args = 2 * jnp.pi * t / VONDRAK_ECLIPTIC_POLE_PERIODS
    sin_ecliptic_args, cos_ecliptic_args = jnp.sin(ecliptic_args), jnp.cos(ecliptic_args)

    ecliptic_pole_x = arcsec_to_rad(ecliptic_x_poly + jnp.sum(
        VONDRAK_ECLIPTIC_POLE_X_COS_COEFFS * cos_ecliptic_args + VONDRAK_ECLIPTIC_POLE_X_SIN_COEFFS * sin_ecliptic_args))
    ecliptic_pole_y = arcsec_to_rad(ecliptic_y_poly + jnp.sum(
        VONDRAK_ECLIPTIC_POLE_Y_COS_COEFFS * cos_ecliptic_args + VONDRAK_ECLIPTIC_POLE_Y_SIN_COEFFS * sin_ecliptic_args))
    ecliptic_pole_z = jnp.sqrt(jnp.maximum(1.0 - ecliptic_pole_x ** 2 - ecliptic_pole_y ** 2, 0.))

    mean_obliquity_j2000 = arcsec_to_rad(EPSILON_0_ARCSEC)
    sin_eps0, cos_eps0 = jnp.sin(mean_obliquity_j2000), jnp.cos(mean_obliquity_j2000)

    mean_ecliptic_pole_vector = jnp.array([
        ecliptic_pole_x,
        -ecliptic_pole_y * cos_eps0 - ecliptic_pole_z * sin_eps0,
        -ecliptic_pole_y * sin_eps0 + ecliptic_pole_z * cos_eps0
    ])

    return mean_equator_pole_vector, mean_ecliptic_pole_vector


def vondrak_cip_xyz_single(t: Float[Array, ""]) -> Tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """
    Compute the long-term Celestial Intermediate Pole (``CIP``) coordinates.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]
        ``(x, y, z)`` components of the ``CIP`` unit vector in ``GCRS``.

    Notes
    -----
        The Vondrak pole is first evaluated in the mean J2000.0 system and then shifted to ``GCRS`` with the small frame-bias rotation used by the IERS conventions.

    References
    ----------
    1. Vondrak, J., Capitaine, N., & Wallace, P. T. (2011). New precession expressions, valid for long time intervals. Appendix A.4.
    """
    mean_equator_pole_vector, _ = vondrak_mean_poles_single(t)
    mean_equator_pole_x, mean_equator_pole_y, mean_equator_pole_z = mean_equator_pole_vector[0], \
        mean_equator_pole_vector[1], mean_equator_pole_vector[2]

    # Apply the IERS frame bias from the mean J2000.0 system to ``GCRS``.
    bias_offset_x = arcsec_to_rad(-0.016617)
    bias_offset_y = arcsec_to_rad(-0.0068192)
    bias_offset_z = arcsec_to_rad(-0.0146)

    cip_x_gcrs = mean_equator_pole_x - mean_equator_pole_y * bias_offset_z + mean_equator_pole_z * bias_offset_x
    cip_y_gcrs = mean_equator_pole_x * bias_offset_z + mean_equator_pole_y + mean_equator_pole_z * bias_offset_y
    cip_z_gcrs = -mean_equator_pole_x * bias_offset_x - mean_equator_pole_y * bias_offset_y + mean_equator_pole_z

    return cip_x_gcrs, cip_y_gcrs, cip_z_gcrs


def vondrak_cio_locator_single(t: Float[Array, ""]) -> Float[Array, ""]:
    """
    Compute the long-term Celestial Intermediate Origin (``CIO``) locator ``s``.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, ""]
        ``CIO`` locator in radians.

    Notes
    -----
    The polynomial and periodic coefficients from Vondrak et al. are evaluated directly in arcseconds and converted to radians only at the end. This keeps the implementation consistent with the published coefficient tables.

    References
    ----------
    1. Vondrak, J., Capitaine, N., & Wallace, P. T. (2011). New precession expressions, valid for long time intervals. Eq.26.
    """
    cio_locator_poly = 3566.723572 + t * (-414.3015011 + t * (0.00085448 + t * (365e-9)))
    cio_locator_args = 2 * jnp.pi * t / VONDRAK_CIO_LOCATOR_PERIODS
    cio_locator_periodic = jnp.sum(VONDRAK_CIO_LOCATOR_COS_COEFFS * jnp.cos(cio_locator_args) +
                                   VONDRAK_CIO_LOCATOR_SIN_COEFFS * jnp.sin(cio_locator_args))

    # These values are in arcseconds, so convert only once at the end.
    return arcsec_to_rad(cio_locator_poly + cio_locator_periodic)


def vondrak_precession_bias_matrix_single(t: Float[Array, ""]) -> Float[Array, "3 3"]:
    """
    Compute the long-term precession-bias rotation matrix.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "3 3"]
        Orthogonal ``3 x 3`` matrix that rotates ``GCRS`` vectors to the mean equator and equinox of date.

    Notes
    -----
        The matrix basis is built from the mean equator pole, the mean ecliptic pole, and their cross product. A first-order frame-bias correction is then applied so the result is consistent with the ``GCRS`` realization used by the surrounding Earth-rotation package.

    References
    ----------
    1. Vondrak, J., Capitaine, N., & Wallace, P. T. (2011). New precession expressions, valid for long time intervals. Appendix A.4.
    """
    mean_equator_pole_vector, mean_ecliptic_pole_vector = vondrak_mean_poles_single(t)

    # Build the equinox direction from the two pole vectors.
    equinox_vector = jnp.cross(mean_equator_pole_vector, mean_ecliptic_pole_vector)
    equinox_vector = equinox_vector / jnp.linalg.norm(equinox_vector)

    # Build the orthogonal Y axis.
    y_axis_vector = jnp.cross(mean_equator_pole_vector, equinox_vector)

    # Assemble the base precession matrix.
    precession_matrix = jnp.stack([equinox_vector, y_axis_vector, mean_equator_pole_vector])

    # Apply the IERS frame bias from mean J2000.0 to ``GCRS``.
    bias_offset_x = arcsec_to_rad(-0.016617)
    bias_offset_y = arcsec_to_rad(-0.0068192)
    bias_offset_z = arcsec_to_rad(-0.0146)

    bias_matrix = jnp.array([[1.0, bias_offset_z, -bias_offset_x], [-bias_offset_z, 1.0, -bias_offset_y],
                             [bias_offset_x, bias_offset_y, 1.0]
                             ])

    return precession_matrix @ bias_matrix
