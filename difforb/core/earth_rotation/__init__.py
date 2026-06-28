"""Internal vectorized wrappers for the Earth-rotation model.

These wrappers are used by object-level time and frame transformations. User code normally reaches this model through :class:`difforb.core.time.Time`, site-state conversion, or state-frame conversion objects.
"""

import jax
from jax import Array
from jaxtyping import Float
from typing import Tuple
from difforb.core.batch import safe_dispatch
from difforb.core.earth_rotation.unified import (
    mean_obliquity_single, cip_xyz_single, cio_locator_single,
    gcrs_to_cirs_matrix_single, cirs_to_gcrs_matrix_single,
    precession_bias_matrix_single, nutation_matrix_single, bias_precession_nutation_matrix_single,
    tio_locator_single, polar_motion_matrix_single, inversed_polar_motion_matrix_single, earth_rotation_angle_single
)

__all__ = []

jax.config.update("jax_enable_x64", True)


@jax.jit
def mean_obliquity(t: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Compute mean obliquity.

    Parameters
    ----------
    t : Float[Array, "..."]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "..."]
        Mean obliquity in radians with the broadcast shape of ``t``.

    Notes
    -----
    The IAU mean-obliquity model is used from 1799-01-01 through 2202-01-01, and the Vondrak et al. (2011) long-term model is used outside that interval.
    Vectorize :func:`difforb.core.earth_rotation.unified.mean_obliquity_single`.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.6.39.
    2. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. Eq.5.12.
    3. Vondrak, J., Capitaine, N., & Wallace, P. T. (2011). New precession expressions, valid for long time intervals. Eq.10.
    """
    return safe_dispatch(mean_obliquity_single, (0,), t)


@jax.jit
def tio_locator(t: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Compute the Terrestrial Intermediate Origin (TIO) locator ``s'``.

    Parameters
    ----------
    t : Float[Array, "..."]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "..."]
        ``TIO`` locator in radians with the broadcast shape of ``t``.

    Notes
    -----
    Vectorize :func:`difforb.core.earth_rotation.unified.tio_locator_single`.

    References
    ----------
    1. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. p.63.
    """
    return safe_dispatch(tio_locator_single, (0,), t)


@jax.jit
def polar_motion_matrix(t: Float[Array, "..."], xp: Float[Array, "..."], yp: Float[Array, "..."]) -> Float[Array, "... 3 3"]:
    """
    Compute the polar-motion rotation matrix.

    Parameters
    ----------
    t : Float[Array, "..."]
        Julian centuries since J2000 in ``TT``.
    xp, yp : Float[Array, "..."]
        Polar-motion coordinates in radians.

    Returns
    -------
    Float[Array, "... 3 3"]
        Broadcast orthogonal ``(..., 3, 3)`` rotation matrix from ``ITRS`` to the Terrestrial Intermediate Reference System (``TIRS``).

    Notes
    -----
    Vectorize :func:`difforb.core.earth_rotation.unified.polar_motion_matrix_single`.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.77.
    """
    return safe_dispatch(polar_motion_matrix_single, (0, 0, 0), t, xp, yp)


@jax.jit
def inversed_polar_motion_matrix(t: Float[Array, "..."], xp: Float[Array, "..."], yp: Float[Array, "..."]) -> Float[
    Array, "... 3 3"]:
    """
    Compute the inverse polar-motion rotation matrix.

    Parameters
    ----------
    t : Float[Array, "..."]
        Julian centuries since J2000 in ``TT``.
    xp, yp : Float[Array, "..."]
        Polar-motion coordinates in radians.

    Returns
    -------
    Float[Array, "... 3 3"]
        Broadcast orthogonal ``(..., 3, 3)`` rotation matrix from the Terrestrial Intermediate Reference System (``TIRS``) to ``ITRS``.

    Notes
    -----
    Vectorize :func:`difforb.core.earth_rotation.unified.inversed_polar_motion_matrix_single`.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.138.
    """
    return safe_dispatch(inversed_polar_motion_matrix_single, (0, 0, 0), t, xp, yp)


@jax.jit
def earth_rotation_angle(ut1_jd1: Float[Array, "..."], ut1_jd2: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Compute the Earth Rotation Angle (ERA).

    Parameters
    ----------
    ut1_jd1, ut1_jd2 : Float[Array, "..."]
        Split Julian date of the ``UT1`` epoch.

    Returns
    -------
    Float[Array, "..."]
        Earth rotation angle in radians with the broadcast shape of the input epochs.

    Notes
    -----
    Vectorize :func:`difforb.core.earth_rotation.unified.earth_rotation_angle_single`.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.6.59.
    """
    return safe_dispatch(earth_rotation_angle_single, (0, 0), ut1_jd1, ut1_jd2)


@jax.jit
def cip_xyz(t: Float[Array, "..."], cor_delta_obliquity: Float[Array, "..."], cor_delta_longitude: Float[Array, "..."]) -> Tuple[
    Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """
    Compute the Celestial Intermediate Pole (``CIP``) coordinates.

    Parameters
    ----------
    t : Float[Array, "..."]
        Julian centuries since J2000 in ``TT``.
    cor_delta_obliquity, cor_delta_longitude : Float[Array, "..."]
        Additive corrections to the model nutation angles in radians.

    Returns
    -------
    Tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]
        Broadcast ``(x, y, z)`` components of the ``CIP`` unit vector in ``GCRS``.

    Notes
    -----
    Within 1799-01-01 to 2202-01-01, it uses the IAU 2006/2000A ``CIP`` model with the supplied nutation corrections. Outside that interval, it switches to the Vondrak et al. (2011) long-term model.
    Vectorize :func:`difforb.core.earth_rotation.unified.cip_xyz_single`.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.76.
    2. Vondrak, J., Capitaine, N., & Wallace, P. T. (2011). New precession expressions, valid for long time intervals. Appendix A.4.
    """
    return safe_dispatch(cip_xyz_single, (0, 0, 0), t, cor_delta_obliquity, cor_delta_longitude)


@jax.jit
def cio_locator(t: Float[Array, "..."], x: Float[Array, "..."], y: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Compute the Celestial Intermediate Origin (``CIO``) locator ``s``.

    Parameters
    ----------
    t : Float[Array, "..."]
        Julian centuries since J2000 in ``TT``.
    x, y : Float[Array, "..."]
        ``x`` and ``y`` components of the ``CIP`` unit vector in ``GCRS``.

    Returns
    -------
    Float[Array, "..."]
        ``CIO`` locator in radians with the broadcast shape of ``t``.

    Notes
    -----
    Within 1799-01-01 to 2202-01-01, it uses the IAU 2006/2000A ``CIO`` model. Outside that interval, it switches to the Vondrak et al. (2011) long-term model.
    Vectorize :func:`difforb.core.earth_rotation.unified.cio_locator_single`.

    References
    ----------
    1. https://github.com/CS-SI/Orekit/blob/develop/src/main/resources/assets/org/orekit/IERS-conventions/2003/tab5.2c.txt#L47.
    2. SOFA ``iauS06``.
    3. Vondrak, J., Capitaine, N., & Wallace, P. T. (2011). New precession expressions, valid for long time intervals. Eq.26.
    """
    return safe_dispatch(cio_locator_single, (0, 0, 0), t, x, y)


@jax.jit
def gcrs_to_cirs_matrix(t: Float[Array, "..."], cor_delta_obliquity: Float[Array, "..."],
                        cor_delta_longitude: Float[Array, "..."]) -> Float[Array, "... 3 3"]:
    """
    Compute the rotation matrix from ``GCRS`` to the Celestial Intermediate Reference System (``CIRS``) for ``CIO``-based transformation.

    Parameters
    ----------
    t : Float[Array, "..."]
        Julian centuries since J2000 in ``TT``.
    cor_delta_obliquity, cor_delta_longitude : Float[Array, "..."]
        Additive corrections to the model nutation angles in radians.

    Returns
    -------
    Float[Array, "... 3 3"]
        Broadcast orthogonal ``(..., 3, 3)`` rotation matrix from the ``GCRS`` to the ``CIRS``.

    Notes
    -----
    Within 1799-01-01 to 2202-01-01, it uses the IAU 2006/2000A ``CIP`` and ``CIO`` models. Outside that interval, it switches to the Vondrak et al. (2011) long-term model.
    Vectorize :func:`difforb.core.earth_rotation.unified.gcrs_to_cirs_matrix_single`.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.73, 7.75.
    """
    return safe_dispatch(gcrs_to_cirs_matrix_single, (0, 0, 0), t, cor_delta_obliquity, cor_delta_longitude)


@jax.jit
def cirs_to_gcrs_matrix(t: Float[Array, "..."], cor_delta_obliquity: Float[Array, "..."],
                        cor_delta_longitude: Float[Array, "..."]) -> Float[Array, "... 3 3"]:
    """
    Compute the rotation matrix from the Celestial Intermediate Reference System (``CIRS``) to ``GCRS`` for ``CIO``-based transformation.

    Parameters
    ----------
    t : Float[Array, "..."]
        Julian centuries since J2000 in ``TT``.
    cor_delta_obliquity, cor_delta_longitude : Float[Array, "..."]
        Additive corrections to the model nutation angles in radians.

    Returns
    -------
    Float[Array, "... 3 3"]
        Broadcast orthogonal ``(..., 3, 3)`` rotation matrix from the ``CIRS`` to the ``GCRS``.

    Notes
    -----
    Within 1799-01-01 to 2202-01-01, it uses the IAU 2006/2000A ``CIP`` and ``CIO`` models. Outside that interval, it switches to the Vondrak et al. (2011) long-term model.
    Vectorize :func:`difforb.core.earth_rotation.unified.cirs_to_gcrs_matrix_single`.

    References
    ----------
    1. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models.
        Eq.6.18.
    """
    return safe_dispatch(cirs_to_gcrs_matrix_single, (0, 0, 0), t, cor_delta_obliquity, cor_delta_longitude)


@jax.jit
def precession_bias_matrix(t: Float[Array, "..."]) -> Float[Array, "... 3 3"]:
    """
    Compute the precession-bias rotation matrix from ``GCRS`` to the mean equator and equinox of date for equinox-based transformation.

    Parameters
    ----------
    t : Float[Array, "..."]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "... 3 3"]
        Broadcast orthogonal ``(..., 3, 3)`` rotation matrix from the ``GCRS`` to the mean equator and equinox of date.

    Notes
    -----
    The IAU precession-bias model is used from 1799-01-01 through 2202-01-01, and the Vondrak et al. (2011) long-term precession model is used outside that interval.
    Vectorize :func:`difforb.core.earth_rotation.unified.precession_bias_matrix_single`.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.6.26.
    2. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models.
    Sec. 5.4.1.
    3. Vondrak, J., Capitaine, N., & Wallace, P. T. (2011). New precession expressions, valid for long time intervals. Appendix A.4.
    """
    return safe_dispatch(precession_bias_matrix_single, (0,), t)


@jax.jit
def nutation_matrix(t: Float[Array, "..."]) -> Float[Array, "... 3 3"]:
    """
    Compute the nutation rotation matrix from the mean equator and equinox of date to the true equator and equinox of date for
    equinox-based transformation.

    Parameters
    ----------
    t : Float[Array, "..."]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "... 3 3"]
        Broadcast orthogonal ``(..., 3, 3)`` rotation matrix from the mean equator and equinox of date to the true equator and equinox of date.

    Notes
    -----
    Within 1799-01-01 to 2202-01-01, it returns the IAU 2000A nutation matrix with the implemented IAU 2006-compatible adjustments. Outside that interval it only returns the identity matrix.
    Vectorize :func:`difforb.core.earth_rotation.unified.nutation_matrix_single`.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.6.39-6.41.
    2. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models.
        Eq.5.20-5.21.
    """
    return safe_dispatch(nutation_matrix_single, (0,), t)


@jax.jit
def bias_precession_nutation_matrix(t: Float[Array, "..."]) -> Float[Array, "... 3 3"]:
    """
    Compute the bias-precession-nutation rotation matrix from ``GCRS`` to the true equator and equinox of date for equinox-based transformation.

    Parameters
    ----------
    t : Float[Array, "..."]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "... 3 3"]
        Broadcast orthogonal ``(..., 3, 3)`` rotation matrix from ``GCRS`` to the true equator and equinox of date.

    Notes
    -----
    Within 1799-01-01 to 2202-01-01, it returns the full IAU 2006/2000A bias-precession-nutation matrix. Outside that interval, it switches to the Vondrak et al. (2011) long-term precession-bias model.
    Vectorize :func:`difforb.core.earth_rotation.unified.bias_precession_nutation_matrix_single`.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.67-7.68.
    2. Vondrak, J., Capitaine, N., & Wallace, P. T. (2011). New precession expressions, valid for long time intervals. Appendix A.4.
    """
    return safe_dispatch(bias_precession_nutation_matrix_single, (0,), t)
