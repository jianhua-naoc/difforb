"""Unified Earth-rotation kernels across short and long time spans.

This module routes scalar Earth-rotation computations between the short-term IAU 2006/2000A model and the long-term Vondrak et al. (2011) model. The switch interval is defined by ``SHORT_TERM_START`` and ``SHORT_TERM_END`` from :mod:`difforb.core.earth_rotation.data`.

All epochs are Julian centuries since J2000 in ``TT``, except :func:`earth_rotation_angle_single`, which uses split Julian dates in ``UT1``. These functions are the single-epoch kernels wrapped by the vectorized public API in :mod:`difforb.core.earth_rotation`.
"""
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float
from typing import Tuple
from difforb.core.constants import J2000
from difforb.utils import R1_single, R2_single, R3_single, arcsec_to_rad
from .data import SHORT_TERM_START, SHORT_TERM_END
from difforb.core.earth_rotation.iau import iau_mean_obliquity_single, iau_cip_xyz_single, \
    iau_cio_locator_single, iau_precession_bias_matrix_single, iau_nutation_matrix_single, \
    iau_bias_precession_nutation_matrix_single
from difforb.core.earth_rotation.vondrak import vondrak_mean_obliquity_single, vondrak_cip_xyz_single, \
    vondrak_cio_locator_single, vondrak_precession_bias_matrix_single

jax.config.update("jax_enable_x64", True)


# =========================================================================
# Unified Routers (Time-based switching)
# =========================================================================
def mean_obliquity_single(t: Float[Array, ""]) -> Float[Array, ""]:
    """
    Compute mean obliquity.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, ""]
        Mean obliquity in radians for the mean equator and ecliptic of date.

    Notes
    -----
    The IAU mean-obliquity model is used from 1799-01-01 through 2202-01-01, and the Vondrak et al. (2011) long-term model is used outside that interval.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.6.39.
    2. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. Eq.5.12.
    3. Vondrak, J., Capitaine, N., & Wallace, P. T. (2011). New precession expressions, valid for long time intervals. Eq.10.
    """
    is_short_term = (t >= SHORT_TERM_START) & (t <= SHORT_TERM_END)
    return jnp.where(is_short_term, iau_mean_obliquity_single(t), vondrak_mean_obliquity_single(t))


def cip_xyz_single(t: Float[Array, ""], cor_delta_obliquity: Float[Array, ""], cor_delta_longitude: Float[Array, ""]) -> \
        Tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """
    Compute the Celestial Intermediate Pole (``CIP``) coordinates.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.
    cor_delta_obliquity, cor_delta_longitude : Float[Array, ""]
        Additive corrections to the model nutation angles in radians.

    Returns
    -------
    Tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]
        ``(x, y, z)`` components of the ``CIP`` unit vector in ``GCRS``.

    Notes
    -----
    Within 1799-01-01 to 2202-01-01, it uses the IAU 2006/2000A ``CIP`` model with the supplied nutation corrections. Outside that interval, it switches to the Vondrak et al. (2011) long-term model.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.76.
    2. Vondrak, J., Capitaine, N., & Wallace, P. T. (2011). New precession expressions, valid for long time intervals. Appendix A.4.
    """
    is_short_term = (t >= SHORT_TERM_START) & (t <= SHORT_TERM_END)
    x_s, y_s, z_s = iau_cip_xyz_single(t, cor_delta_obliquity, cor_delta_longitude)
    x_l, y_l, z_l = vondrak_cip_xyz_single(t)
    return (jnp.where(is_short_term, x_s, x_l), jnp.where(is_short_term, y_s, y_l), jnp.where(is_short_term, z_s, z_l))


def cio_locator_single(t: Float[Array, ""], x: Float[Array, ""], y: Float[Array, ""]) -> Float[Array, ""]:
    """
    Compute the Celestial Intermediate Origin (``CIO``) locator ``s``.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.
    x, y : Float[Array, ""]
        ``x`` and ``y`` components of the ``CIP`` unit vector in ``GCRS``.

    Returns
    -------
    Float[Array, ""]
        ``CIO`` locator in radians.

    Notes
    -----
    Within 1799-01-01 to 2202-01-01, it uses the IAU 2006/2000A ``CIO`` model. Outside that interval, it switches to the Vondrak et al. (2011) long-term model.

    References
    ----------
    1. https://github.com/CS-SI/Orekit/blob/develop/src/main/resources/assets/org/orekit/IERS-conventions/2003/tab5.2c.txt#L47.
    2. SOFA ``iauS06``.
    3. Vondrak, J., Capitaine, N., & Wallace, P. T. (2011). New precession expressions, valid for long time intervals. Eq.26.
    """
    is_short_term = (t >= SHORT_TERM_START) & (t <= SHORT_TERM_END)
    return jnp.where(is_short_term, iau_cio_locator_single(t, x, y), vondrak_cio_locator_single(t))


# =========================================================================
# Unified Transformation Matrices
# =========================================================================
def gcrs_to_cirs_matrix_single(t: Float[Array, ""], cor_delta_obliquity: Float[Array, ""],
                               cor_delta_longitude: Float[Array, ""]) -> Float[Array, "3 3"]:
    """
    Compute the rotation matrix from ``GCRS`` to the Celestial Intermediate Reference System (``CIRS``) for ``CIO``-based transformation.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.
    cor_delta_obliquity, cor_delta_longitude : Float[Array, ""]
        Additive corrections to the model nutation angles in radians.

    Returns
    -------
    Float[Array, "3 3"]
        Orthogonal ``3 x 3`` rotation matrix from the ``GCRS`` to the ``CIRS``.

    Notes
    -----
        Within 1799-01-01 to 2202-01-01, it uses the IAU 2006/2000A ``CIP`` and ``CIO`` models. Outside that interval, it switches to the Vondrak et al. (2011) long-term model.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.73, 7.75.
    """
    x, y, z = cip_xyz_single(t, cor_delta_obliquity, cor_delta_longitude)
    s = cio_locator_single(t, x, y)
    a = 1. / (1. + z)
    x2, y2, axy = x * x, y * y, a * x * y
    R_sigma = jnp.array([[1. - a * x2, -axy, -x],
                         [-axy, 1. - a * y2, -y], [x, y, 1. - a * (x2 + y2)]
                         ])
    return R3_single(-s) @ R_sigma


def cirs_to_gcrs_matrix_single(t: Float[Array, ""], cor_delta_obliquity: Float[Array, ""],
                               cor_delta_longitude: Float[Array, ""]) -> Float[Array, "3 3"]:
    """
    Compute the rotation matrix from the Celestial Intermediate Reference System (``CIRS``) to ``GCRS`` for ``CIO``-based transformation.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.
    cor_delta_obliquity, cor_delta_longitude : Float[Array, ""]
        Additive corrections to the model nutation angles in radians.

    Returns
    -------
    Float[Array, "3 3"]
        Orthogonal ``3 x 3`` rotation matrix from the ``CIRS`` to the ``GCRS``.

    Notes
    -----
        Within 1799-01-01 to 2202-01-01, it uses the IAU 2006/2000A ``CIP`` and ``CIO`` models. Outside that interval, it switches to the Vondrak et al. (2011) long-term model.

    References
    ----------
    1. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models.
        Eq.6.18.
    """
    x, y, z = cip_xyz_single(t, cor_delta_obliquity, cor_delta_longitude)
    s = cio_locator_single(t, x, y)
    a = 1. / (1. + z)
    x2, y2, axy = x * x, y * y, a * x * y
    R_sigma = jnp.array([
        [1. - a * x2, -axy, x],
        [-axy, 1. - a * y2, y], [-x, -y, 1. - a * (x2 + y2)]
    ])
    return R_sigma @ R3_single(s)


def precession_bias_matrix_single(t: Float[Array, ""]) -> Float[Array, "3 3"]:
    """
    Compute the precession-bias rotation matrix from ``GCRS`` to the mean equator and equinox of date for equinox-based transformation.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "3 3"]
        Orthogonal ``3 x 3`` rotation matrix from the ``GCRS`` to the mean equator and equinox of date.

    Notes
    -----
    The IAU precession-bias model is used from 1799-01-01 through 2202-01-01, and the Vondrak et al. (2011) long-term precession model is used outside that interval.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.6.26.
    2. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models.
    Sec. 5.4.1.
    3. Vondrak, J., Capitaine, N., & Wallace, P. T. (2011). New precession expressions, valid for long time intervals. Appendix A.4.
    """
    is_short_term = (t >= SHORT_TERM_START) & (t <= SHORT_TERM_END)
    return jnp.where(is_short_term, iau_precession_bias_matrix_single(t), vondrak_precession_bias_matrix_single(t))


def nutation_matrix_single(t: Float[Array, ""]) -> Float[Array, "3 3"]:
    """
    Compute the nutation rotation matrix from the mean equator and equinox of date to the true equator and equinox of date for
    equinox-based transformation.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "3 3"]
        Orthogonal ``3 x 3`` rotation matrix from the mean equator and equinox of date to the true equator and equinox of date.

    Notes
    -----
    Within 1799-01-01 to 2202-01-01, it returns the IAU 2000A nutation matrix with the implemented IAU 2006-compatible adjustments. Outside that interval it only returns the identity matrix.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.6.39-6.41.
    2. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models.
        Eq.5.20-5.21.
    """
    is_short_term = (t >= SHORT_TERM_START) & (t <= SHORT_TERM_END)
    return jnp.where(is_short_term, iau_nutation_matrix_single(t), jnp.eye(3))


def bias_precession_nutation_matrix_single(t: Float[Array, ""]) -> Float[Array, "3 3"]:
    """
    Compute the bias-precession-nutation rotation matrix from ``GCRS`` to the true equator and equinox of date for equinox-based transformation.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "3 3"]
        Orthogonal ``3 x 3`` rotation matrix from ``GCRS`` to the true equator and equinox of date.

    Notes
    -----
    Within 1799-01-01 to 2202-01-01, it returns the full IAU 2006/2000A bias-precession-nutation matrix. Outside that interval, it switches to the Vondrak et al. (2011) long-term precession-bias model without nutation model.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.67-7.68.
    2. Vondrak, J., Capitaine, N., & Wallace, P. T. (2011). New precession expressions, valid for long time intervals. Appendix A.4.
    """
    is_short_term = (t >= SHORT_TERM_START) & (t <= SHORT_TERM_END)
    return jnp.where(is_short_term, iau_bias_precession_nutation_matrix_single(t),
                     vondrak_precession_bias_matrix_single(t))


# =========================================================================
# Polar Motion & Earth Rotation (Valid for all times natively)
# =========================================================================
def tio_locator_single(t: Float[Array, ""]) -> Float[Array, ""]:
    """
    Compute the Terrestrial Intermediate Origin (TIO) locator ``s'``.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, ""]
        ``TIO`` locator in radians.

    References
    ----------
    1. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. p 63.
    """
    return arcsec_to_rad(-0.000047 * t)


def polar_motion_matrix_single(t: Float[Array, ""], xp: Float[Array, ""], yp: Float[Array, ""]) -> Float[Array, "3 3"]:
    """
    Compute the polar-motion rotation matrix from the Terrestrial Intermediate Reference System (``TIRS``) to ``ITRS``.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.
    xp, yp : Float[Array, ""]
        Coordinates X and Y of the Celestial Intermediate Pole (``CIP``) in radians.

    Returns
    -------
    Float[Array, "3 3"]
        Orthogonal ``3 x 3`` rotation matrix from the ``TIRS`` to ``ITRS``.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.77.
    """
    s_prime = tio_locator_single(t)
    return R1_single(-yp) @ R2_single(-xp) @ R3_single(s_prime)


def inversed_polar_motion_matrix_single(t: Float[Array, ""], xp: Float[Array, ""], yp: Float[Array, ""]) -> Float[
    Array, "3 3"]:
    """
    Compute the inverse polar-motion rotation matrix from ``ITRS`` to the Terrestrial Intermediate Reference System (``TIRS``).

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.
    xp, yp : Float[Array, ""]
        Coordinates X and Y of the Celestial Intermediate Pole (``CIP``) in radians.

    Returns
    -------
    Float[Array, "3 3"]
        Orthogonal ``3 x 3`` rotation matrix from the ``ITRS`` to ``TIRS``.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.138.
    """
    s_prime = tio_locator_single(t)
    return R3_single(-s_prime) @ R2_single(xp) @ R1_single(yp)


def earth_rotation_angle_single(ut1_jd1: Float[Array, ""], ut1_jd2: Float[Array, ""]) -> Float[Array, ""]:
    """
    Compute the Earth Rotation Angle (ERA).

    Parameters
    ----------
    ut1_jd1, ut1_jd2 : Float[Array, ""]
        Split Julian date of the ``UT1`` epoch.

    Returns
    -------
    Float[Array, ""]
        Earth rotation angle in radians.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.6.59.
    """
    # Reorder the split like SOFA/ERFA to reduce sensitivity to how the
    # Julian date is apportioned between the two inputs.
    d1 = jnp.minimum(ut1_jd1, ut1_jd2)
    d2 = jnp.maximum(ut1_jd1, ut1_jd2)
    t = d1 + (d2 - J2000)
    # Reduce the summed fractional day back into ``[0, 1)`` so that
    # equivalent two-part splits such as ``(d, 0.25)`` and ``(d-0.5, 0.75)``
    # map to the same ERA argument up to floating-point roundoff.
    f = jnp.mod(jnp.mod(d1, 1.0) + jnp.mod(d2, 1.0), 1.0)
    theta = 2 * jnp.pi * (f + 0.7790572732640 + 0.00273781191135448 * t)
    return jnp.mod(theta, 2 * jnp.pi)
