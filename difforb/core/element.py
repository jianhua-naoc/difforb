"""Keplerian elements and Cartesian conversions.

This module stores Keplerian elements and converts them to and from Cartesian state vectors. The main element set is ``(p, e, inc, node, peri, m)`` at a ``TDB`` epoch. Angles are in radians, distance is in ``au``, and velocity is in ``au / day``.
"""

from typing import TYPE_CHECKING, Tuple
import jax
from difforb.core.constants import GM_SUN
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float

from difforb.core.batch import BatchableObject, safe_dispatch
from difforb.core.time.timescale import TDBView, TTView
from difforb.core.validate import validate_timeview
from difforb.report.text import build_repr, format_float_array, format_shape
from difforb.report.display_units import orbit_element_specs, repr_fields_from_specs

if TYPE_CHECKING:
    from difforb.body.ephbody import EphemerisBody
    from difforb.core.state.state import State


def kep_to_cart_single(p: Float[Array, ""], e: Float[Array, ""], inc_rad: Float[Array, ""],
                       node_rad: Float[Array, ""],
                       peri_rad: Float[Array, ""], v_rad: Float[Array, ""]) -> Tuple[
    Float[Array, "3"], Float[Array, "3"]]:
    """Convert one Keplerian element set to a Cartesian state.

    Parameters
    ----------
    p : Float[Array, ""]
        Semi-latus rectum in ``au``.
    e : Float[Array, ""]
        Eccentricity.
    inc_rad : Float[Array, ""]
        Inclination in radians.
    node_rad : Float[Array, ""]
        Longitude of ascending node in radians.
    peri_rad : Float[Array, ""]
        Argument of perihelion in radians.
    v_rad : Float[Array, ""]
        True anomaly in radians.

    Returns
    -------
    tuple[Float[Array, "3"], Float[Array, "3"]]
        Position and velocity in heliocentric ecliptic J2000. Position is in ``au``. Velocity is in ``au / day``.

    References
    ----------
    1. Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 10.
    """
    cosinc, sininc = jnp.cos(inc_rad), jnp.sin(inc_rad)
    cosnode, sinnode = jnp.cos(node_rad), jnp.sin(node_rad)
    cosperi, sinperi = jnp.cos(peri_rad), jnp.sin(peri_rad)
    cosv, sinv = jnp.cos(v_rad), jnp.sin(v_rad)

    scal1 = 1. + e * cosv
    scal2 = jnp.sqrt(GM_SUN / p)
    pos_pqw = jnp.array([p * cosv / scal1, p * sinv / scal1, 0.])
    vel_pqw = jnp.array([-scal2 * sinv, scal2 * (e + cosv), 0.])
    rot_mat = jnp.array([
        [cosnode * cosperi - sinnode * sinperi * cosinc, -cosnode * sinperi - sinnode * cosperi * cosinc,
         sinnode * sininc],
        [sinnode * cosperi + cosnode * sinperi * cosinc,
         -sinnode * sinperi + cosnode * cosperi * cosinc, -cosnode * sininc],
        [sinperi * sininc, cosperi * sininc, cosinc]
    ])
    pos = rot_mat @ pos_pqw
    vel = rot_mat @ vel_pqw
    return pos, vel


def v_to_m_single(v_rad: Float[Array, ""], e: Float[Array, ""]) -> Float[Array, ""]:
    """Convert one true anomaly to a mean anomaly.

    Parameters
    ----------
    v_rad : Float[Array, ""]
        True anomaly in radians.
    e : Float[Array, ""]
        Eccentricity.

    Returns
    -------
    Float[Array, ""]
        Mean anomaly in radians.

    Notes
    -----
    The conversion handles circular, elliptical, parabolic, and hyperbolic cases.

    References
    ----------
    1. For elliptical case: 吴连大, 《人造卫星与空间碎片的轨道和探测》, p.44-45.
    2. For other cases: Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 5.
    """
    is_circular = jnp.isclose(e, 0.)
    is_elliptical = jnp.logical_and(e > 0., e < 1.)
    is_hyperbolic = e > 1.

    def circular(args: Tuple[Array, Array]) -> Array:
        v, e = args
        return v

    def elliptical(args: Tuple[Array, Array]) -> Array:
        v, e = args
        scal1 = jnp.sqrt(1. - e ** 2)
        sinE = scal1 * jnp.sin(v) / (1. + e * jnp.cos(v))
        E = v - 2. * jnp.arctan2(
            e * jnp.sin(v),
            1 + scal1 + e * jnp.cos(v)
        )
        m = E - e * sinE
        m = jnp.fmod(m, jnp.pi * 2)
        m = jnp.where(m < 0., m + jnp.pi * 2, m)
        E = jnp.fmod(E, jnp.pi * 2)
        return m

    def hyperbolic(args: Tuple[Array, Array]) -> Array:
        v, e = args
        # Use the half-angle form for good stability near the asymptote.
        tan_v2 = jnp.tan(v / 2.0)
        tan_H2 = jnp.sqrt((e - 1.0) / (e + 1.0)) * tan_v2
        H = 2.0 * jnp.arctanh(tan_H2)
        m = e * jnp.sinh(H) - H
        return m

    def parabolic(args: Tuple[Array, Array]) -> Array:
        v, e = args
        E = jnp.tan(v / 2)
        m = E + (E ** 3) / 3
        return m

    args = (v_rad, e)
    m = jax.lax.cond(
        is_circular, circular, lambda _: jax.lax.cond(is_elliptical, elliptical,
                                                      lambda _: jax.lax.cond(is_hyperbolic, hyperbolic,
                                                                             parabolic,
                                                                             args), args), args
    )
    return m


def m_to_v_single(m_rad: Float[Array, ""], e: Float[Array, ""]) -> Float[Array, ""]:
    """Convert one mean anomaly to a true anomaly.

    Parameters
    ----------
    m_rad : Float[Array, ""]
        Mean anomaly in radians.
    e : Float[Array, ""]
        Eccentricity.

    Returns
    -------
    Float[Array, ""]
        True anomaly in radians.

    Notes
    -----
    The conversion handles circular, elliptical, parabolic, and hyperbolic cases.

    References
    ----------
    1. For elliptical Kepler equation solving: Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 2.
    2. For hyperbolic Kepler equation solving: Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 4.
    3. For parabolic Kepler equation solving: https://github.com/CelesTrak/fundamentals-of-astrodynamics/blob/main/software/python/src/valladopy/astro/twobody/newton.py#L161.
    4. For elliptical conversion from eccentric anomaly to true anomaly: 吴连大, 《人造卫星与空间碎片的轨道和探测》, p.44-45.
    5. For hyperbolic conversion from hyperbolic anomaly to true anomaly: https://github.com/CelesTrak/fundamentals-of-astrodynamics/blob/main/software/python/src/valladopy/astro/twobody/newton.py#L156
    6. For parabolic conversion from Barker variable to true anomaly: https://github.com/CelesTrak/fundamentals-of-astrodynamics/blob/main/software/python/src/valladopy/astro/twobody/newton.py#L165
    """
    is_circular = jnp.isclose(e, 0.)
    is_elliptical = jnp.logical_and(e > 0., e < 1.)
    is_hyperbolic = e > 1.

    def circular(args: Tuple[Array, Array]) -> Array:
        m, e = args
        return m

    def elliptical(args: Tuple[Array, Array]) -> Array:
        m, e = args
        E = jnp.where(((-jnp.pi < m) & (m < 0)) | (m > jnp.pi), m - e, m + e)

        def solve_kep_equ_step(carry, _):
            E, m, e = carry
            new_E = E + (m - E + e * jnp.sin(E)) / (1 - e * jnp.cos(E))
            return (new_E, m, e), None

        (E, _, _), _ = jax.lax.scan(solve_kep_equ_step, (E, m, e), None, length=15)
        v = E + 2. * jnp.arctan2(e * jnp.sin(E), 1. + jnp.sqrt(1. - e ** 2) - e * jnp.cos(E))
        return v

    def hyperbolic(args: Tuple[Array, Array]) -> Array:
        m, e = args
        m_abs = jnp.abs(m)

        # Step 1: Build a stable initial guess for the hyperbolic anomaly.
        H0 = jnp.where(
            m_abs > e,
            jnp.arcsinh(m_abs / e),
            jnp.where(
                e < 1.1,
                jnp.cbrt(6.0 * m_abs),
                m_abs / e
            )
        )

        # Step 2: Solve the hyperbolic Kepler equation with Newton iteration.
        tolerance = 1e-15
        max_iters = 30

        def cond_fn(val):
            H_n, H_prev, iters = val
            return (jnp.abs(H_n - H_prev) >= tolerance) & (iters < max_iters)

        def body_fn(val):
            H_n, _, iters = val

            sh = jnp.sinh(H_n)
            ch = jnp.cosh(H_n)

            f = e * sh - H_n - m_abs
            fp = e * ch - 1.0

            delta = f / fp

            # Keep extreme random tests away from NaN.
            delta = jnp.clip(delta, -15.0, 15.0)

            H_next = H_n - delta
            return (H_next, H_n, iters + 1)

        init_val = (H0, H0 + 2.0 * tolerance, jnp.array(0, dtype=jnp.int32))
        final_val = jax.lax.while_loop(cond_fn, body_fn, init_val)
        H_final = final_val[0]

        # Restore the sign from the odd symmetry of the solution.
        H_final = jnp.where(m < 0., -H_final, H_final)

        # Step 3: Convert the hyperbolic anomaly to true anomaly.
        sinv = (jnp.sqrt(e ** 2 - 1.0) * jnp.sinh(H_final)) / (e * jnp.cosh(H_final) - 1.0)
        cosv = (e - jnp.cosh(H_final)) / (e * jnp.cosh(H_final) - 1.0)
        v = jnp.arctan2(sinv, cosv)

        return v

    def parabolic(args: Tuple[Array, Array]) -> Array:
        m, e = args
        s = 0.5 * (0.5 * jnp.pi - jnp.arctan(1.5 * m))
        w = jnp.arctan(jnp.tan(s) ** (1. / 3.))
        E = 2. / jnp.tan(2 * w)
        v = 2. * jnp.arctan(E)
        return v

    args = (m_rad, e)
    v = jax.lax.cond(
        is_circular, circular, lambda _: jax.lax.cond(is_elliptical, elliptical,
                                                      lambda _: jax.lax.cond(is_hyperbolic, hyperbolic,
                                                                             parabolic,
                                                                             args), args), args
    )

    return v


def cart_to_kep_single(pos: Float[Array, "3"], vel: Float[Array, "3"]) -> dict[str, Float[Array, ""]]:
    """Convert one Cartesian state to Keplerian elements.

    Parameters
    ----------
    pos : Float[Array, "3"]
        Position in ``au``.
    vel : Float[Array, "3"]
        Velocity in ``au / day``.

    Returns
    -------
    dict[str, Float[Array, ""]]
        Element values ``p``, ``e``, ``inc``, ``node``, ``peri``, and ``v``. Angles are in radians.

    References
    ----------
    1. Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 9.
    """
    pos_mag = jnp.linalg.norm(pos)
    vel_mag = jnp.linalg.norm(vel)
    vel_mag2 = vel_mag ** 2
    h_vec = jnp.cross(pos, vel)
    h_mag = jnp.linalg.norm(h_vec)
    n_vec = jnp.array([-h_vec[1], h_vec[0], 0.])
    n_mag = jnp.linalg.norm(n_vec)
    e_vec = ((vel_mag2 - GM_SUN / pos_mag) * pos - jnp.dot(pos, vel) * vel) / GM_SUN
    e_mag = jnp.linalg.norm(e_vec)

    p = h_mag ** 2 / GM_SUN

    inc = jnp.arccos(jnp.clip(h_vec[2] / h_mag, -1., 1.))
    node = jnp.arccos(jnp.clip(n_vec[0] / n_mag, -1., 1.))
    node = jnp.where(n_vec[1] < 0, 2 * jnp.pi - node, node)

    peri = jnp.arccos(jnp.clip(
        jnp.dot(n_vec, e_vec) / (n_mag * e_mag)
        , -1., 1.))
    peri = jnp.where(e_vec[2] < 0, 2 * jnp.pi - peri, peri)

    v = jnp.arccos(jnp.clip(
        jnp.dot(e_vec, pos) / (pos_mag * e_mag)
        , -1., 1.))
    v = jnp.where(jnp.dot(pos, vel) < 0, 2 * jnp.pi - v, v)

    return {
        "p": p,
        "e": e_mag,
        "inc": inc,
        "node": node,
        "peri": peri,
        "v": v
    }


def kep_to_cart(p: Float[Array, "..."], e: Float[Array, "..."], inc_rad: Float[Array, "..."],
                node_rad: Float[Array, "..."],
                peri_rad: Float[Array, "..."], v_rad: Float[Array, "..."]) -> Tuple[
    Float[Array, "... 3"], Float[Array, "... 3"]]:
    """Convert Keplerian elements to Cartesian states.

    Parameters
    ----------
    p : Float[Array, "..."]
        Semi-latus rectum in ``au``.
    e : Float[Array, "..."]
        Eccentricity.
    inc_rad : Float[Array, "..."]
        Inclination in radians.
    node_rad : Float[Array, "..."]
        Longitude of ascending node in radians.
    peri_rad : Float[Array, "..."]
        Argument of perihelion in radians.
    v_rad : Float[Array, "..."]
        True anomaly in radians.

    Returns
    -------
    tuple[Float[Array, "... 3"], Float[Array, "... 3"]]
        Position and velocity in heliocentric ecliptic J2000.

    Notes
    -----
    Vectorize :func:`kep_to_cart_single`.

    References
    ----------
    1. Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 10.
    """
    return safe_dispatch(kep_to_cart_single, (0, 0, 0, 0, 0, 0), p, e, inc_rad, node_rad, peri_rad, v_rad)


def v_to_m(v_rad: Float[Array, "..."], e: Float[Array, "..."]) -> Float[Array, "..."]:
    """Convert true anomaly to mean anomaly.

    Parameters
    ----------
    v_rad : Float[Array, "..."]
        True anomaly in radians.
    e : Float[Array, "..."]
        Eccentricity.

    Returns
    -------
    Float[Array, "..."]
        Mean anomaly in radians.

    Notes
    -----
    Vectorize :func:`v_to_m_single`.

    References
    ----------
    1. For elliptical case: 吴连大, 《人造卫星与空间碎片的轨道和探测》, p.44-45.
    2. For other cases: Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 5.
    """
    return safe_dispatch(v_to_m_single, (0, 0), v_rad, e)


def m_to_v(m_rad: Float[Array, "..."], e: Float[Array, "..."]) -> Float[Array, "..."]:
    """Convert mean anomaly to true anomaly.

    Parameters
    ----------
    m_rad : Float[Array, "..."]
        Mean anomaly in radians.
    e : Float[Array, "..."]
        Eccentricity.

    Returns
    -------
    Float[Array, "..."]
        True anomaly in radians.

    Notes
    -----
    Vectorize :func:`m_to_v_single`.

    References
    ----------
    1. For elliptical Kepler equation solving: Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 2.
    2. For hyperbolic Kepler equation solving: Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 4.
    3. or parabolic Kepler equation solving: https://github.com/CelesTrak/fundamentals-of-astrodynamics/blob/main/software
    /python/src/valladopy/astro/twobody/newton.py#L161.
    4. For elliptical conversion from eccentric anomaly to true anomaly: 吴连大, 《人造卫星与空间碎片的轨道和探测》, p.44-45.
    5. For hyperbolic conversion from hyperbolic anomaly to true anomaly:
    https://github.com/CelesTrak/fundamentals-of-astrodynamics/blob/main/software/python/src/valladopy/astro/twobody/newton.py#L156
    6. For parabolic conversion from Barker variable to true anomaly:
    https://github.com/CelesTrak/fundamentals-of-astrodynamics/blob/main/software/python/src/valladopy/astro/twobody/newton.py#L165
    """
    return safe_dispatch(m_to_v_single, (0, 0), m_rad, e)


def cart_to_kep(pos: Float[Array, "... 3"], vel: Float[Array, "... 3"]) -> dict[str, Float[Array, ""]]:
    """Convert Cartesian states to Keplerian elements.

    Parameters
    ----------
    pos : Float[Array, "... 3"]
        Position in ``au``.
    vel : Float[Array, "... 3"]
        Velocity in ``au / day``.

    Returns
    -------
    dict[str, Float[Array, ""]]
        Element values ``p``, ``e``, ``inc``, ``node``, ``peri``, and ``v``.

    Notes
    -----
    Vectorize :func:`cart_to_kep_single`.

    References
    ----------
    1. Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 9, pp. 115-116.
    """
    return safe_dispatch(cart_to_kep_single, (1, 1), pos, vel)


class KepElement(BatchableObject):
    """Osculating Keplerian elements.

    The stored elements are ``(p, e, inc, node, peri, m)`` at a ``TDB`` epoch. Here ``p`` is the semi-latus rectum. All angles are
    stored in radians.

    Parameters
    ----------
    tdb : TDBView
        Epoch of the osculating elements in ``TDB``.
    p : Float[ArrayLike, "..."]
        Semi-latus rectum in ``au``.
    e : Float[ArrayLike, "..."]
        Eccentricity.
    inc : Float[ArrayLike, "..."]
        Inclination in radians.
    node : Float[ArrayLike, "..."]
        Longitude of ascending node in radians.
    peri : Float[ArrayLike, "..."]
        Argument of perihelion in radians.
    m : Float[ArrayLike, "..."]
        Mean anomaly in radians.

    See Also
    --------
    from_classical
        Build from the classical ``(a, e, i, node, peri, M)`` elements set.
    from_true_anomaly
        Build from ``(p, e, i, node, peri, v)`` elements set.
    from_equinoctial_elements
        Build from equinoctial elements.
    from_helio_eclip_j2000
        Build from a heliocentric JPL Horizons ecliptic-of-J2000 state vector.

    Examples
    --------
    >>> from difforb.core.element import KepElement
    >>> from difforb.core.time.timescale import Time
    >>> tdb = Time.from_tdb_date(2025, 1, 1).tdb()
    >>> elem = KepElement.from_classical(tdb, 2.0, 0.1, 5.0, 30.0, 40.0, 10.0)
    >>> elem.array.shape
    (6,)
    """
    tdb: TDBView
    p: Float[Array, "..."]
    e: Float[Array, "..."]
    inc: Float[Array, "..."]
    node: Float[Array, "..."]
    peri: Float[Array, "..."]
    m: Float[Array, "..."]

    def __init__(self, tdb: TDBView, p: Float[ArrayLike, "..."], e: Float[ArrayLike, "..."],
                 inc: Float[ArrayLike, "..."],
                 node: Float[ArrayLike, "..."], peri: Float[ArrayLike, "..."], m: Float[ArrayLike, "..."]):
        """Initialize heliocentric ecliptic J2000 Keplerian elements.

        Parameters
        ----------
        tdb : TDBView
            Epoch in ``TDB``.
        p : Float[ArrayLike, "..."]
            Semi-latus rectum in ``au``.
        e : Float[ArrayLike, "..."]
            Eccentricity.
        inc : Float[ArrayLike, "..."]
            Inclination in radians.
        node : Float[ArrayLike, "..."]
            Longitude of ascending node in radians.
        peri : Float[ArrayLike, "..."]
            Argument of perihelion in radians.
        m : Float[ArrayLike, "..."]
            Mean anomaly in radians.
        """
        self.tdb = tdb
        self.p = jnp.asarray(p, dtype=float)
        self.e = jnp.asarray(e, dtype=float)
        self.inc = jnp.asarray(inc, dtype=float)
        self.node = jnp.asarray(node, dtype=float)
        self.peri = jnp.asarray(peri, dtype=float)
        self.m = jnp.asarray(m, dtype=float)

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("shape", format_shape(self.shape)),
                ("epoch_jd", format_float_array(self.tdb.jd, precision=9, scientific=False, signed=False)),
                *repr_fields_from_specs(self, orbit_element_specs(self)),
            ],
        )

    @property
    def a(self) -> Float[Array, "..."]:
        """Semi-major axis in ``au``. Returns ``inf`` for parabolic cases."""
        return self.p / (1. - self.e * self.e)

    @property
    def v(self) -> Float[Array, "..."]:
        """True anomaly in radians.

        References
        ----------
        1. For elliptical Kepler equation solving: Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 2.
        2. For hyperbolic Kepler equation solving: Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 4.
        3. For parabolic Kepler equation solving: https://github.com/CelesTrak/fundamentals-of-astrodynamics/blob/main/software/python/src/valladopy/astro/twobody/newton.py#L161.
        4. For elliptical conversion from eccentric anomaly to true anomaly: 吴连大, 《人造卫星与空间碎片的轨道和探测》, p.44-45.
        5. For hyperbolic conversion from hyperbolic anomaly to true anomaly: https://github.com/CelesTrak/fundamentals-of-astrodynamics/blob/main/software/python/src/valladopy/astro/twobody/newton.py#L156
        6. For parabolic conversion from Barker variable to true anomaly: https://github.com/CelesTrak/fundamentals-of-astrodynamics/blob/main/software/python/src/valladopy/astro/twobody/newton.py#L165
        """
        return m_to_v(self.m, self.e)

    @property
    def period(self) -> Float[Array, "..."]:
        """Orbital period in days. Returns ``inf`` for non-periodic cases."""
        is_periodic = self.e < 1.
        period = jnp.where(is_periodic, jnp.pi * 2. * jnp.sqrt(self.a ** 3 / GM_SUN), jnp.inf)
        return period

    @property
    def perit_jd(self) -> Float[Array, "..."]:
        """Perihelion time in Julian Date. Returns ``nan`` for non-periodic cases."""
        is_periodic = self.e < 1.
        perit_jd = jnp.where(is_periodic, self.tdb.jd + (2 * jnp.pi - self.m) / (jnp.pi * 2 / self.period),
                             jnp.nan)
        return perit_jd

    @classmethod
    def from_classical(cls, tdb: TDBView, a: Float[ArrayLike, "..."], e: Float[ArrayLike, "..."],
                       inc: Float[ArrayLike, "..."],
                       node: Float[ArrayLike, "..."], peri: Float[ArrayLike, "..."],
                       m: Float[ArrayLike, "..."],
                       degrees: bool = True) -> 'KepElement':
        """Build from the classical ``(a, e, i, node, peri, M)`` elements set.

        Parameters
        ----------
        tdb : TDBView
            Epoch in ``TDB``.
        a : Float[ArrayLike, "..."]
            Semi-major axis in ``au``.
        e : Float[ArrayLike, "..."]
            Eccentricity.
        inc : Float[ArrayLike, "..."]
            Inclination.
        node : Float[ArrayLike, "..."]
            Longitude of ascending node.
        peri : Float[ArrayLike, "..."]
            Argument of perihelion.
        m : Float[ArrayLike, "..."]
            Mean anomaly.
        degrees : bool, default=True
            If ``True``, angle inputs are in degrees else in radians.

        Returns
        -------
        KepElement
            Element object stored as ``(p, e, inc, node, peri, m)``.
        """
        validate_timeview(tdb, TDBView, 'tdb')
        a = jnp.asarray(a, dtype=float)
        e = jnp.asarray(e, dtype=float)
        p = a * (1. - e ** 2)

        factor = jnp.pi / 180. if degrees else 1.
        return cls(tdb, p, e,
                   jnp.asarray(inc, dtype=float) * factor,
                   jnp.asarray(node, dtype=float) * factor,
                   jnp.asarray(peri, dtype=float) * factor,
                   jnp.asarray(m, dtype=float) * factor)

    @classmethod
    def from_true_anomaly(cls, tdb: TDBView, p: Float[ArrayLike, "..."], e: Float[ArrayLike, "..."],
                          inc: Float[ArrayLike, "..."],
                          node: Float[ArrayLike, "..."], peri: Float[ArrayLike, "..."],
                          v: Float[ArrayLike, "..."],
                          degrees: bool = True) -> 'KepElement':
        """Build from ``(p, e, i, node, peri, v)`` elements set.

        Parameters
        ----------
        tdb : TDBView
            Epoch in ``TDB``.
        p : Float[ArrayLike, "..."]
            Semi-latus rectum in ``au``.
        e : Float[ArrayLike, "..."]
            Eccentricity.
        inc : Float[ArrayLike, "..."]
            Inclination.
        node : Float[ArrayLike, "..."]
            Longitude of ascending node.
        peri : Float[ArrayLike, "..."]
            Argument of perihelion.
        v : Float[ArrayLike, "..."]
            True anomaly.
        degrees : bool, default=True
            If ``True``, angle inputs are in degrees.

        Returns
        -------
        KepElement
            Element object with mean anomaly derived from ``v``.

        References
        ----------
        1. For elliptical case: 吴连大, 《人造卫星与空间碎片的轨道和探测》, p.44-45.
        2. For other cases: Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 5.
        """
        validate_timeview(tdb, TDBView, 'tdb')
        v = jnp.asarray(v, dtype=float)
        e = jnp.asarray(e, dtype=float)

        factor = jnp.pi / 180. if degrees else 1.
        v_rad = v * factor

        m_rad = v_to_m(v_rad, e)

        return cls(tdb,
                   jnp.asarray(p, dtype=float), e,
                   jnp.asarray(inc, dtype=float) * factor,
                   jnp.asarray(node, dtype=float) * factor,
                   jnp.asarray(peri, dtype=float) * factor,
                   m_rad)

    @classmethod
    def from_equinoctial_elements(cls, tt: TTView, a: Float[ArrayLike, "..."], g: Float[ArrayLike, "..."],
                                  f: Float[ArrayLike, "..."],
                                  k: Float[ArrayLike, "..."], h: Float[ArrayLike, "..."],
                                  M: Float[ArrayLike, "..."]) -> 'KepElement':
        """Build from equinoctial elements.

        Parameters
        ----------
        tt : TTView
            Epoch in ``TT``.
        a : Float[ArrayLike, "..."]
            Semi-major axis in ``au``.
        g : Float[ArrayLike, "..."]
            ``e * sin(peri + node)`` term.
        f : Float[ArrayLike, "..."]
            ``e * cos(peri + node)`` term.
        k : Float[ArrayLike, "..."]
            ``tan(inc / 2) * sin(node)`` term.
        h : Float[ArrayLike, "..."]
            ``tan(inc / 2) * cos(node)`` term.
        M : Float[ArrayLike, "..."]
            Sum ``node + peri + m`` in radians.

        Returns
        -------
        KepElement
            Equivalent element set at the corresponding ``TDB`` epoch.

        References
        ----------
        1. https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/EquinoctalElements-modified.pdf.
        """
        validate_timeview(tt, TTView, 'tt')
        a = jnp.asarray(a, dtype=float)
        g = jnp.asarray(g, dtype=float)
        f = jnp.asarray(f, dtype=float)
        k = jnp.asarray(k, dtype=float)
        h = jnp.asarray(h, dtype=float)
        M = jnp.asarray(M, dtype=float)

        double_pi = 2 * jnp.pi
        e = jnp.sqrt(f * f + g * g)
        inc = (jnp.atan2(2 * jnp.sqrt(h * h + k * k), 1. - h * h - k * k)) % double_pi
        peri = jnp.atan2(g * h - f * k, f * h + g * k) % double_pi
        node = jnp.atan2(k, h) % double_pi
        m = (M - jnp.atan2(g, f)) % double_pi
        return cls.from_classical(tdb=tt.time.tdb(),
                                  a=a, e=e, inc=inc, peri=peri, node=node, m=m, degrees=False)

    @classmethod
    def from_state(
            cls,
            state: 'State',
            *,
            sun: 'EphemerisBody | None' = None,
            earth: 'EphemerisBody | None' = None,
    ) -> 'KepElement':
        """Build from one frame-aware Cartesian state.

        Parameters
        ----------
        state : State
            Input state at a ``TDB`` epoch. The state is converted internally to canonical ``HELIO_ECLIP_J2000`` before element extraction.
        sun : EphemerisBody, optional
            Sun ephemeris body used if ``state`` must be shifted from ``SSB`` or ``EARTH`` to the heliocentric origin.
        earth : EphemerisBody, optional
            Earth ephemeris body used if ``state`` must be shifted from ``EARTH`` to another origin before the final heliocentric conversion.

        Returns
        -------
        KepElement
            Osculating element set for ``state`` in the library canonical element convention.

        Raises
        ------
        ValueError
            If converting ``state`` to canonical ``HELIO_ECLIP_J2000`` requires the Sun or Earth and the corresponding ephemeris body is not available.

        Notes
        -----
        The canonical Cartesian boundary of :class:`KepElement` is heliocentric JPL Horizons ecliptic-of-J2000. This method converts the input state to that frame first and then applies the Cartesian-to-element mapping.

        References
        ----------
        1. Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 9, pp. 115-116.
        """
        from difforb.core.state.frame import HELIO_ECLIP_J2000

        state = state.to(HELIO_ECLIP_J2000, sun=sun, earth=earth)
        pos = state.pos
        vel = state.vel

        results = cart_to_kep(pos, vel)
        p = results['p']
        e = results['e']
        inc = results['inc']
        node = results['node']
        peri = results['peri']
        v = results['v']

        return cls.from_true_anomaly(tdb=state.tdb, p=p, e=e, inc=inc, node=node, peri=peri, v=v, degrees=False)

    @classmethod
    def from_array(cls, tdb: TDBView, array: Float[ArrayLike, "... 6"]):
        """Build from a stacked element array.

        Parameters
        ----------
        tdb : TDBView
            Epoch in ``TDB``.
        array : Float[ArrayLike, "... 6"]
            Array with shape ``(..., 6)`` ordered as ``[..., p, e, inc, node, peri, m]``.

        Returns
        -------
        KepElement
            Element object of the receiving class.
        """
        validate_timeview(tdb, TDBView, 'tdb')
        array = jnp.asarray(array, dtype=float)
        return cls(tdb=tdb, p=array[..., 0], e=array[..., 1], inc=array[..., 2], node=array[..., 3],
                   peri=array[..., 4],
                   m=array[..., 5])

    @property
    def array(self) -> Float[Array, "N 6"]:
        """Return the stacked element array.

        Returns
        -------
        Float[Array, "N 6"]
            Array ordered as ``[..., p, e, inc, node, peri, m]``.
        """
        return jnp.stack([self.p, self.e, self.inc, self.node, self.peri, self.m], axis=-1)

    def state(self) -> 'State':
        """Convert elements to the canonical Cartesian state.

        Returns
        -------
        State
            Heliocentric JPL Horizons ecliptic-of-J2000 state at the same epoch.

        Notes
        -----
        This is the canonical Cartesian boundary of :class:`KepElement`. The returned state uses the ``HELIO_ECLIP_J2000`` frame of :mod:`difforb.core.state`.

        References
        ----------
        1. Vallado, D. A. (2022). Fundamentals of Astrodynamics and Applications. Algorithm 10.
        """
        from difforb.core.state.frame import HELIO_ECLIP_J2000
        from difforb.core.state.state import State
        pos, vel = kep_to_cart(self.p, self.e, self.inc, self.node, self.peri, self.v)

        return State(
            pos=pos,
            vel=vel,
            tdb=self.tdb,
            frame=HELIO_ECLIP_J2000,
        )

    @property
    def shape(self):
        """Return the batch shape."""
        return self.p.shape
