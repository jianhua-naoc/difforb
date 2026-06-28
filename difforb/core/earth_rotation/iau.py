"""IAU precession, nutation, and ``CIO``/``CIP`` kernels.

This module provides scalar building blocks for the IAU 2006 precession model, the IAU 2000A nutation series, and derived Earth-rotation quantities such as the Celestial Intermediate Pole (``CIP``) coordinates, the Celestial Intermediate Origin (``CIO``) locator, and several equinox-based rotation matrices.

All time arguments are Julian centuries since J2000 in ``TT``. Coefficient tables and pretabulated series multipliers are imported from :mod:`difforb.core.earth_rotation.data`. These routines are single-epoch kernels that can be wrapped by higher-level batch dispatch code in the surrounding Earth-rotation package.
"""
import jax
from jaxtyping import Float
from jax import Array
from typing import Tuple
from difforb.utils import arcsec_to_rad, R1_single, R3_single
from difforb.core.earth_rotation.data import *

jax.config.update("jax_enable_x64", True)


def iau_precession_angles_single(t: Float[Array, ""]) -> Tuple[
    Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """
    Compute the IAU 2006 Fukushima-Williams bias-precession angles.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]]
        ``(gamma_bar, phi_bar, psi_bar, epsilon_a)`` in radians. These are the precession-bias angles and mean obliquity used by the equinox-based IAU 2006 formulation.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Table 6.3, Eq.7.69-7.72.
    """
    gamma_bar = (-0.052928 + (
            10.556378 + (0.4932044 + (-0.00031238 + (-0.000002788 + 0.0000000260 * t) * t) * t) * t) * t)
    phi_bar = (84381.412819 + (
            -46.811016 + (0.0511268 + (0.00053289 + (-0.000000440 + (-0.0000000176) * t) * t) * t) * t) * t)
    psi_bar = (-0.041775 + (
            5038.481484 + (1.5584175 + (-0.00018522 + (-0.000026452 + (-0.0000000148) * t) * t) * t) * t) * t)
    epsilon_a = iau_mean_obliquity_single(t)
    return arcsec_to_rad(gamma_bar), arcsec_to_rad(phi_bar), arcsec_to_rad(psi_bar), epsilon_a


def iers_lunisolar_fargs_single(t: Float[Array, ""]) -> Float[Array, "5"]:
    """
    Compute the five IERS 2003 lunisolar fundamental arguments.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "5"]
        Arguments in radians ordered as mean anomaly of the Moon, mean anomaly of the Sun, mean argument of latitude of the Moon, mean elongation of the Moon from the Sun, and mean longitude of the Moon's ascending node.

    References
    ----------
    1. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. Eq.5.19.
    """
    # IERS 2003 lunisolar fundamental args:
    # --l: mean anomaly of the Moon
    # --l': mean anomaly of the Sun
    # --F: mean argument of latitude of the Moon
    # --D: mean elongation of the Moon from the Sun
    # --Omega: mean longitude of the Moon's mean ascending node
    tm = jnp.array([1., t, t ** 2, t ** 3, t ** 4])
    fargs_arcsec = jnp.array([[485868.249036, 1717915923.2178, 31.8792, 0.051635, -0.00024470],
                              [1287104.79305, 129596581.0481, -0.5532, 0.000136, -0.00001149],
                              [335779.526232, 1739527262.8478, -12.7512, -0.001037, 0.00000417],
                              [1072260.70369, 1602961601.2090, -6.3706, 0.006593, -0.00003169],
                              [450160.398036, -6962890.5431, 7.4722, 0.007702, -0.00005939],
                              ]) @ tm
    return arcsec_to_rad(fargs_arcsec) % (2 * jnp.pi)


def iers_planetary_fargs_single(t: Float[Array, ""]) -> Float[Array, "7"]:
    """
    Compute the IERS 2003 planetary fundamental arguments used here.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "7"]
        Mean heliocentric ecliptic longitudes of Mercury through Uranus in radians. Neptune and the approximate general precession longitude are appended separately by :func:`mixed_fargs_single`.

    References
    ----------
    1. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. Eq.5.17.
    2. SOFA ``iauFame03``, ``iauFave03``, ``iauFae03``, ``iauFama03``, ``iauFaju03``, ``iauFasa03``, and ``iauFaur03``.
    """
    tm = jnp.array([1., t])
    fargs_rad = jnp.array(
        [[4.402608842, 2608.7903141574], [3.176146697, 1021.3285546211], [1.753470314, 628.3075849991],
         [6.203480913, 334.0612426700], [0.599546497, 52.9690962641], [0.874016757, 21.3299104960],
         [5.481293872, 7.4781598567]
         ]) @ tm
    return fargs_rad % (2 * jnp.pi)


def iers_app_fargs_single(t: Float[Array, ""]) -> Float[Array, "1"]:
    """
    Compute the IERS 2003 approximate general precession longitude.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "1"]
        Approximate general precession longitude ``p_A`` in radians.

    References
    ----------
    1. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. Eq.5.18.
    2. SOFA ``iauFapa03``.
    """
    tm = jnp.array([t, t * t])
    fargs_rad = jnp.array([[0.024381750, 0.00000538691]]) @ tm
    return fargs_rad % (2 * jnp.pi)


def mhb_lunisolar_fargs_single(t: Float[Array, ""]) -> Float[Array, "5"]:
    """
    Compute the MHB2000 lunisolar fundamental arguments.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "5"]
        Arguments in radians ordered as ``l``, ``l'``, ``F``, ``D``, and
        ``Omega`` in the SOFA MHB2000 convention.

    Notes
    -----
    The returned ordering matches the planetary nutation coefficient tables in
    :mod:`difforb.core.earth_rotation.data`.

    References
    ----------
    1. SOFA ``iauNut00a``.
    2. ftp//maia.usno.navy.mil/conv2000/chapter5/IAU2000A.
    """
    tm = jnp.array([1., t])
    fargs_rad = jnp.array([[2.35555598, 8328.6914269554], [6.24006013, 628.301955], [1.627905234, 8433.466158131],
                           [5.198466741, 7771.3771468121], [2.18243920, -33.757045]
                           ]) @ tm
    return fargs_rad % (2 * jnp.pi)


def mixed_fargs_single(t: Float[Array, ""]) -> Float[Array, "14"]:
    """
    Assemble the mixed fundamental-argument vector for planetary nutation.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "14"]
        Argument vector in radians ordered as Mercury through Uranus
        heliocentric longitudes, Neptune longitude, approximate general
        precession longitude, and the five MHB2000 lunisolar arguments.

    Notes
    -----
    SOFA continues the approach of MHB2000, using different fundamental arguments to compute the planetary nutation. We have also continued this practice.

    References
    ----------
    1. SOFA ``iauNut00a``.
    """
    lunisolar_fargs = mhb_lunisolar_fargs_single(t)
    app_fargs = iers_app_fargs_single(t)
    planetary_fargs = iers_planetary_fargs_single(t)
    neptune_fargs = jnp.array([[5.321159000, 3.8127774000]]) @ jnp.array([1., t])
    return jnp.concatenate((planetary_fargs, neptune_fargs, app_fargs, lunisolar_fargs), axis=0)


def phi_single(t: Float[Array, ""]) -> Float[Array, "1365"]:
    """
    Evaluate the phase arguments for the nutation series terms.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "1365"]
        Nutation phase angles in radians. The first 678 entries correspond to the lunisolar series and the remaining 687 entries correspond to the planetary series.

    References
    ----------
    1. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. Eq.5.16.
    2. SOFA ``iauNut00a``.
    """
    return jnp.concatenate((M_lunisolar @ iers_lunisolar_fargs_single(t),
                            M_planetary @ mixed_fargs_single(t)))


def sin_cos_phi_single(t: Float[Array, ""]) -> Tuple[
    Float[Array, "678"], Float[Array, "678"], Float[Array, "687"], Float[Array, "687"]]:
    """
    Compute trigonometric factors for the nutation phase angles.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Tuple[Float[Array, "678"], Float[Array, "678"], Float[Array, "687"], Float[Array, "687"]]
        ``(sin_phi_lunisolar, cos_phi_lunisolar, sin_phi_planetary,
        cos_phi_planetary)`` with the 678-term lunisolar block and the
        687-term planetary block split explicitly.
    """
    phi_arg = phi_single(t)
    phi_l, phi_p = phi_arg[:678], phi_arg[678:]
    return jnp.sin(phi_l), jnp.cos(phi_l), jnp.sin(phi_p), jnp.cos(phi_p)


def delta_obliquity_single(t: Float[Array, ""], sin_phi_l: Float[Array, "678"], cos_phi_l: Float[Array, "678"],
                           sin_phi_p: Float[Array, "687"], cos_phi_p: Float[Array, "687"]) -> Float[Array, ""]:
    """
    Compute the nutation contribution to the change of the obliquity.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.
    sin_phi_l, cos_phi_l : Float[Array, "678"]
        Trigonometric factors for the lunisolar phase angles.
    sin_phi_p, cos_phi_p : Float[Array, "687"]
        Trigonometric factors for the planetary phase angles.

    Returns
    -------
    Float[Array, ""]
        Nutation in obliquity ``delta_epsilon`` in radians, including the small IAU 2006 compatibility correction applied to the IAU 2000A series.

    References
    ----------
    1. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. Eq.5.15.
    2. Wallace, P. T., & Capitaine, N. (2006). Precession-nutation procedures consistent with IAU 2006 resolutions. Eq.5.
    """
    term_l = (C_lunisolar + C_dot_lunisolar * t) * cos_phi_l + S_prime_lunisolar * sin_phi_l
    delta_epsilon_lunisolar = jnp.sum(term_l)
    term_p = C_planetary * cos_phi_p + S_prime_planetary * sin_phi_p
    delta_epsilon_planetary = jnp.sum(term_p)
    delta_arcsec = delta_epsilon_lunisolar + delta_epsilon_planetary
    delta_rad = arcsec_to_rad(delta_arcsec)
    # Apply the IAU 2006 scale adjustment that keeps the IAU 2000A nutation series consistent with the updated precession model.
    f = - 2.7774e-6 * t
    delta_rad += f * delta_rad
    return delta_rad


def delta_longitude_single(t: Float[Array, ""], sin_phi_l: Float[Array, "678"], cos_phi_l: Float[Array, "678"],
                           sin_phi_p: Float[Array, "687"], cos_phi_p: Float[Array, "687"]) -> Float[Array, ""]:
    """
    Compute the nutation contribution to the change of the position of the equinox along the ecliptic longitude.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.
    sin_phi_l, cos_phi_l : Float[Array, "678"]
        Trigonometric factors for the lunisolar phase angles.
    sin_phi_p, cos_phi_p : Float[Array, "687"]
        Trigonometric factors for the planetary phase angles.

    Returns
    -------
    Float[Array, ""]
        Nutation in longitude ``delta_psi`` in radians, including the IAU 2006 compatibility correction applied to the IAU 2000A series.

    Notes
    -----
    This routine shares the same phase-angle decomposition as :func:`delta_obliquity_single` and is typically called with the same cached trigonometric arrays.

    References
    ----------
    1. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. Eq.5.15.
    2. Wallace, P. T., & Capitaine, N. (2006). Precession-nutation procedures consistent with IAU 2006 resolutions. Eq.5.
    """
    term_l = (S_lunisolar + S_dot_lunisolar * t) * sin_phi_l + C_prime_lunisolar * cos_phi_l
    delta_psi_lunisolar = jnp.sum(term_l)
    term_p = S_planetary * sin_phi_p + C_prime_planetary * cos_phi_p
    delta_psi_planetary = jnp.sum(term_p)
    delta_arcsec = delta_psi_lunisolar + delta_psi_planetary
    delta_rad = arcsec_to_rad(delta_arcsec)
    # Apply the small longitude offset and secular scale term adopted for the IAU 2006/2000A combined precession-nutation model.
    f = -2.7774e-6 * t
    delta_rad += (0.4697e-6 + f) * delta_rad
    return delta_rad


def iau_mean_obliquity_single(t: Float[Array, ""]) -> Float[Array, ""]:
    """
    Compute the IAU 2006 mean obliquity of the ecliptic.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, ""]
        Mean obliquity in radians for the mean equator and equinox of date.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.6.39.
    2. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. Eq.5.12.
    """
    obl_arcsec = ((((
                            -0.0000000434 * t - 0.000000576) * t + 0.00200340) * t - 0.0001831) * t - 46.836769) * t + EPSILON_0_ARCSEC
    return arcsec_to_rad(obl_arcsec)


def iau_cip_xyz_single(t: Float[Array, ""], cor_delta_obliquity: Float[Array, ""],
                       cor_delta_longitude: Float[Array, ""]) -> Tuple[
    Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """
    Compute the Celestial Intermediate Pole (``CIP``) unit-vector components.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.
    cor_delta_obliquity : Float[Array, ""]
        Additive correction to the model nutation in obliquity, in radians.
    cor_delta_longitude : Float[Array, ""]
        Additive correction to the model nutation in longitude, in radians.

    Returns
    -------
    Tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]
        ``(x, y, z)`` components of the ``CIP`` unit vector in ``GCRS``.

    Notes
    -----
    The returned components satisfy ``x**2 + y**2 + z**2 = 1`` up to floating point error. Setting both ``cor_delta_obliquity`` and ``cor_delta_longitude`` corrections to zero yields the pure IAU 2006/2000A model pole.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.76.
    """
    # Evaluate the precession-bias angles and the nutation angles, then map them onto the CIP unit vector.
    gamma_bar, phi_bar, psi_bar, epsilon_a = iau_precession_angles_single(t)
    sin_phi_l, cos_phi_l, sin_phi_p, cos_phi_p = sin_cos_phi_single(t)
    delta_epsilon = delta_obliquity_single(t, sin_phi_l, cos_phi_l, sin_phi_p, cos_phi_p) + cor_delta_obliquity
    delta_psi = delta_longitude_single(t, sin_phi_l, cos_phi_l, sin_phi_p, cos_phi_p) + cor_delta_longitude

    epsilon, psi = epsilon_a + delta_epsilon, psi_bar + delta_psi
    sin_eps, cos_eps = jnp.sin(epsilon), jnp.cos(epsilon)
    sin_psi, cos_psi = jnp.sin(psi), jnp.cos(psi)
    sin_gamma, cos_gamma = jnp.sin(gamma_bar), jnp.cos(gamma_bar)
    sin_phi, cos_phi = jnp.sin(phi_bar), jnp.cos(phi_bar)

    scal1 = sin_eps * sin_psi
    scal2 = sin_eps * cos_psi * cos_phi - cos_eps * sin_phi

    x = scal1 * cos_gamma - scal2 * sin_gamma
    y = scal1 * sin_gamma + scal2 * cos_gamma
    z = jnp.sqrt(1. - x * x - y * y)
    return x, y, z


def iau_cio_locator_single(t: Float[Array, ""], x: Float[Array, ""], y: Float[Array, ""]) -> Float[Array, ""]:
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
        ``CIO`` locator ``s`` in radians.

    References
    ----------
    1. https://github.com/CS-SI/Orekit/blob/develop/src/main/resources/assets/org/orekit/IERS-conventions/2003/tab5.2c.txt#L47.
    2. SOFA ``iauS06``.
    """
    cio_loc_poly_part = 94e-6 + (3808.65e-6 + (-122.68e-6 + (-72574.11e-6 + (27.98e-6 + 15.62e-6 * t) * t) * t) * t) * t

    # Fundamental args for cio locator computation:
    # -- five IERS 2003 lunisolar fundamental arguments
    # -- IERS 2003 mean heliocentric ecliptic longitudes of planets Venus and Earth
    # -- IERS 2003 approximate general precession longitude
    fargs = jnp.concatenate(
        (iers_lunisolar_fargs_single(t), iers_planetary_fargs_single(t)[jnp.array([1, 2])], iers_app_fargs_single(t)))
    combined_fargs = coff_fundamental_args @ fargs

    part_0 = jnp.sum(coff_sin_0 * jnp.sin(combined_fargs[:33]) + coff_cos_0 * jnp.cos(combined_fargs[:33]))
    part_1 = jnp.sum(coff_sin_1 * jnp.sin(combined_fargs[33:36]) + coff_cos_1 * jnp.cos(combined_fargs[33:36]))
    part_2 = jnp.sum(coff_sin_2 * jnp.sin(combined_fargs[36:61]) + coff_cos_2 * jnp.cos(combined_fargs[36:61]))
    part_3 = jnp.sum(coff_sin_3 * jnp.sin(combined_fargs[61:65]) + coff_cos_3 * jnp.cos(combined_fargs[61:65]))
    part_4 = jnp.sum(coff_sin_4 * jnp.sin(combined_fargs[65:]) + coff_cos_4 * jnp.cos(combined_fargs[65:]))

    cio_loc_nopoly_part = part_0 + (part_1 + (part_2 + (part_3 + part_4 * t) * t) * t) * t
    return arcsec_to_rad(cio_loc_poly_part + cio_loc_nopoly_part) - x * y / 2.


def iau_precession_bias_matrix_single(t: Float[Array, ""]) -> Float[Array, "3 3"]:
    """
    Compute the IAU 2006 precession-bias rotation matrix from ``GCRS`` to the mean equator and equinox of date for equinox-based transformation.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "3 3"]
        Orthogonal ``3 x 3`` rotation matrix assembled from the Fukushima-Williams precession-bias angles.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.6.26.
    2. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models.
    Sec. 5.4.1.
    """
    gamma_bar, phi_bar, psi_bar, epsilon_a = iau_precession_angles_single(t)
    return R1_single(-epsilon_a) @ R3_single(-psi_bar) @ R1_single(phi_bar) @ R3_single(gamma_bar)


def iau_nutation_matrix_single(t: Float[Array, ""]) -> Float[Array, "3 3"]:
    """
    Compute the IAU 2000A nutation matrix with IAU 2006 compatibility terms from the mean equator and equinox of date to the true equator and equinox of date for
    equinox-based transformation.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "3 3"]
        Orthogonal ``3 x 3`` nutation rotation matrix derived from ``delta_psi`` and ``delta_epsilon`` nutation angles.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.6.39-6.41.
    2. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. Eq.5.20-5.21.
    """
    sin_phi_l, cos_phi_l, sin_phi_p, cos_phi_p = sin_cos_phi_single(t)
    epsilon_a = iau_mean_obliquity_single(t)
    delta_epsilon = delta_obliquity_single(t, sin_phi_l, cos_phi_l, sin_phi_p, cos_phi_p)
    epsilon = epsilon_a + delta_epsilon
    delta_psi = delta_longitude_single(t, sin_phi_l, cos_phi_l, sin_phi_p, cos_phi_p)

    sin_ea, cos_ea = jnp.sin(epsilon_a), jnp.cos(epsilon_a)
    sin_e, cos_e = jnp.sin(epsilon), jnp.cos(epsilon)
    sin_dpsi, cos_dpsi = jnp.sin(delta_psi), jnp.cos(delta_psi)

    return jnp.array([[cos_dpsi, -sin_dpsi * cos_ea, -sin_dpsi * sin_ea],
                      [sin_dpsi * cos_e, cos_dpsi * cos_e * cos_ea + sin_e * sin_ea,
                       cos_dpsi * cos_e * sin_ea - sin_e * cos_ea],
                      [sin_dpsi * sin_e, cos_dpsi * sin_e * cos_ea - cos_e * sin_ea,
                       cos_dpsi * sin_e * sin_ea + cos_e * cos_ea]
                      ])


def iau_bias_precession_nutation_matrix_single(t: Float[Array, ""]) -> Float[Array, "3 3"]:
    """
    Compute the combined IAU 2006/2000A bias-precession-nutation matrix from ``GCRS`` to the true equator and equinox of date for equinox-based transformation.

    Parameters
    ----------
    t : Float[Array, ""]
        Julian centuries since J2000 in ``TT``.

    Returns
    -------
    Float[Array, "3 3"]
        Orthogonal ``3 x 3`` rotation matrix that combines frame bias, precession, and nutation.

    Notes
    -----
        This matrix is equal to ``iau_nutation_matrix_single(t) @ iau_precession_bias_matrix_single(t)``.

    References
    ----------
    1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.67-7.68.
    """
    gamma_bar, phi_bar, psi_bar, epsilon_a = iau_precession_angles_single(t)
    sin_phi_l, cos_phi_l, sin_phi_p, cos_phi_p = sin_cos_phi_single(t)
    delta_epsilon = delta_obliquity_single(t, sin_phi_l, cos_phi_l, sin_phi_p, cos_phi_p)
    delta_psi = delta_longitude_single(t, sin_phi_l, cos_phi_l, sin_phi_p, cos_phi_p)

    return R1_single(-(epsilon_a + delta_epsilon)) @ R3_single(-(psi_bar + delta_psi)) @ R1_single(phi_bar) @ R3_single(
        gamma_bar)
