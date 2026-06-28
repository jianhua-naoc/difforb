"""Solar and terrestrial ``J2`` force terms.

This module separates the two built-in zonal quadrupole models because their pole conventions differ. The solar term uses a fixed inertial pole, defaulting to the ecliptic-of-J2000 pole used by the OrbFit solar oblateness approximation. The terrestrial term evaluates the ``ITRS`` z-axis at the propagation epoch and can be limited to a near-Earth distance range.
"""

from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, Float

from difforb.body.ephbody import EphemerisBody
from difforb.core.constants import AU_M, J2_SUN, R_SUN
from difforb.core.geo import ITRF
from difforb.core.state.axes import Axes, axes_to_icrs_rotation
from difforb.core.time.timescale import Time
from difforb.dynamics.force_model.base import Force
from difforb.utils import R3_single


J2_EARTH = jnp.array(1.0826267e-3, dtype=jnp.float64)
R_EARTH = jnp.array(ITRF.a, dtype=jnp.float64) / AU_M
EARTH_J2_MAX_DISTANCE = jnp.array(0.1, dtype=jnp.float64)
ECLIPTIC_J2000_POLE_ICRS = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64) @ axes_to_icrs_rotation(Axes.ECLIP_J2000)
ITRS_POLE = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)


@jax.jit
def compute_j2_acceleration_vec(
        pos: Float[Array, "3"],
        pos_center: Float[Array, "3"],
        mu: Float[ArrayLike, ""],
        j2: Float[ArrayLike, ""],
        radius: Float[ArrayLike, ""],
        pole_unit_vec: Float[Array, "3"],
) -> Float[Array, "3"]:
    """Compute acceleration from one positive ``J2`` zonal quadrupole.

    Parameters
    ----------
    pos : Float[Array, "3"]
        Integrated body position in ``BCRS``, in ``au``.
    pos_center : Float[Array, "3"]
        Center-body position in ``BCRS``, in ``au``.
    mu : Float[ArrayLike, ""]
        Center-body gravitational parameter in ``au^3 / day^2``.
    j2 : Float[ArrayLike, ""]
        Positive dimensionless second zonal harmonic ``J2``. Do not pass the signed spherical-harmonic coefficient ``C20``.
    radius : Float[ArrayLike, ""]
        Center-body reference radius in ``au``.
    pole_unit_vec : Float[Array, "3"]
        Unit vector of the center-body pole in ``BCRS`` axes.

    Returns
    -------
    Float[Array, "3"]
        ``J2`` acceleration in ``au / day^2``.
    """
    r_vec = pos - pos_center
    r2 = jnp.dot(r_vec, r_vec)
    r_inv = jax.lax.rsqrt(r2)
    r_inv2 = r_inv * r_inv
    r_inv5 = r_inv2 * r_inv2 * r_inv

    z = jnp.dot(r_vec, pole_unit_vec)
    factor = 1.5 * j2 * mu * (radius ** 2) * r_inv5
    term1 = (5.0 * (z ** 2) * r_inv2 - 1.0) * r_vec
    term2 = 2.0 * z * pole_unit_vec
    return factor * (term1 - term2)


class SolarJ2Perturbation(Force):
    """Solar oblateness perturbation with a fixed inertial pole.

    The default pole follows the OrbFit solar ``J2`` approximation: the pole is normal to the ecliptic-of-J2000 plane and expressed in the ``BCRS``/``ICRS`` axes used by the propagation state.
    """
    body: EphemerisBody
    j2: Float[Array, ""]
    radius: Float[Array, ""]
    pole_unit_vec: Float[Array, "3"]

    def __init__(
            self,
            body: EphemerisBody,
            j2: Float[ArrayLike, ""] = J2_SUN,
            radius: Float[ArrayLike, ""] = R_SUN,
            pole_unit_vec: Float[ArrayLike, "3"] | None = None,
    ):
        """Initialize a solar ``J2`` perturbation.

        Parameters
        ----------
        body : EphemerisBody
            Solar ephemeris body that supplies the center position and gravitational parameter.
        j2 : Float[ArrayLike, ""], default=J2_SUN
            Positive dimensionless solar ``J2``.
        radius : Float[ArrayLike, ""], default=R_SUN
            Solar reference radius in ``au``.
        pole_unit_vec : Float[ArrayLike, "3"] or None, optional
            Fixed solar pole unit vector in ``BCRS`` axes. If omitted, use the ecliptic-of-J2000 pole expressed in ``ICRS`` axes, matching the OrbFit approximation that neglects solar-spin-axis tilt to the ecliptic.
        """
        self.body = body
        self.j2 = jnp.asarray(j2, dtype=float)
        self.radius = jnp.asarray(radius, dtype=float)
        if pole_unit_vec is None:
            pole_unit_vec = ECLIPTIC_J2000_POLE_ICRS
        pole_unit_vec = jnp.asarray(pole_unit_vec, dtype=float)
        self.pole_unit_vec = pole_unit_vec / jnp.linalg.norm(pole_unit_vec)

    def __call__(
            self,
            tdb_jd1: Float[Array, ""],
            tdb_jd2: Float[Array, ""],
            state: Tuple[Float[Array, "3"], Float[Array, "3"]],
            args: Any = None,
    ) -> Float[Array, "3"]:
        """Evaluate the solar ``J2`` acceleration at one ``TDB`` epoch.

        Parameters
        ----------
        tdb_jd1, tdb_jd2 : Float[Array, ""]
            Two parts of the ``TDB`` Julian Date.
        state : tuple[Float[Array, "3"], Float[Array, "3"]]
            Integrated body state ``(pos, vel)`` in ``BCRS``. Position is in ``au``.
        args : Any, optional
            Extra propagator data. This force does not use it.

        Returns
        -------
        Float[Array, "3"]
            Solar ``J2`` acceleration in ``au / day^2``.
        """
        pos, _ = state
        pos_center = self.body._bcrs_pos_jd(tdb_jd1, tdb_jd2)
        return compute_j2_acceleration_vec(pos, pos_center, self.body.gm, self.j2, self.radius, self.pole_unit_vec)

    @property
    def shape(self):
        """Return the batch shape."""
        return ()


class EarthJ2Perturbation(Force):
    """Terrestrial ``J2`` perturbation with an epoch-dependent Earth pole.

    By default, the pole is the ``ITRS`` z-axis transformed through polar motion, Earth rotation, and the ``CIRS`` to ``GCRS`` rotation at the requested epoch. A fixed pole can be supplied for tests or simplified sensitivity calculations.
    """
    body: EphemerisBody
    j2: Float[Array, ""]
    radius: Float[Array, ""]
    max_distance: Float[Array, ""]
    fixed_pole_unit_vec: Float[Array, "3"]
    use_fixed_pole: bool = eqx.field(static=True)

    def __init__(
            self,
            body: EphemerisBody,
            j2: Float[ArrayLike, ""] = J2_EARTH,
            radius: Float[ArrayLike, ""] = R_EARTH,
            max_distance: Float[ArrayLike, ""] = EARTH_J2_MAX_DISTANCE,
            fixed_pole_unit_vec: Float[ArrayLike, "3"] | None = None,
    ):
        """Initialize an Earth ``J2`` perturbation.

        Parameters
        ----------
        body : EphemerisBody
            Earth ephemeris body that supplies the center position and gravitational parameter.
        j2 : Float[ArrayLike, ""], default=J2_EARTH
            Positive dimensionless terrestrial ``J2``.
        radius : Float[ArrayLike, ""], default=R_EARTH
            Terrestrial equatorial reference radius in ``au``.
        max_distance : Float[ArrayLike, ""], default=0.1
            Maximum geocentric distance for applying the term, in ``au``.
        fixed_pole_unit_vec : Float[ArrayLike, "3"] or None, optional
            Fixed Earth pole unit vector in ``BCRS`` axes. If omitted, the pole is evaluated at each ``TDB`` epoch from ``ITRS`` through the Earth-rotation matrices.
        """
        self.body = body
        self.j2 = jnp.asarray(j2, dtype=float)
        self.radius = jnp.asarray(radius, dtype=float)
        self.max_distance = jnp.asarray(max_distance, dtype=float)
        self.use_fixed_pole = fixed_pole_unit_vec is not None
        if fixed_pole_unit_vec is None:
            self.fixed_pole_unit_vec = jnp.zeros(3, dtype=float)
        else:
            fixed_pole_unit_vec = jnp.asarray(fixed_pole_unit_vec, dtype=float)
            self.fixed_pole_unit_vec = fixed_pole_unit_vec / jnp.linalg.norm(fixed_pole_unit_vec)

    def get_current_pole(self, tdb_jd1: Float[Array, ""], tdb_jd2: Float[Array, ""]) -> Float[Array, "3"]:
        """Return the Earth pole unit vector in ``BCRS`` axes at one ``TDB`` epoch.

        Parameters
        ----------
        tdb_jd1, tdb_jd2 : Float[Array, ""]
            Two parts of the ``TDB`` Julian Date.

        Returns
        -------
        Float[Array, "3"]
            Unit pole vector in the same axes as the propagated ``BCRS`` state.
        """
        if self.use_fixed_pole:
            return self.fixed_pole_unit_vec
        time = Time.from_tdb_jd(tdb_jd1, tdb_jd2)
        pole_tirs = time.inversed_polar_motion_matrix @ ITRS_POLE
        pole_cirs = R3_single(-time.ERA) @ pole_tirs
        pole_gcrs = time.cirs_to_gcrs_matrix @ pole_cirs
        return pole_gcrs / jnp.linalg.norm(pole_gcrs)

    def __call__(
            self,
            tdb_jd1: Float[Array, ""],
            tdb_jd2: Float[Array, ""],
            state: Tuple[Float[Array, "3"], Float[Array, "3"]],
            args: Any = None,
    ) -> Float[Array, "3"]:
        """Evaluate the Earth ``J2`` acceleration at one ``TDB`` epoch.

        Parameters
        ----------
        tdb_jd1, tdb_jd2 : Float[Array, ""]
            Two parts of the ``TDB`` Julian Date.
        state : tuple[Float[Array, "3"], Float[Array, "3"]]
            Integrated body state ``(pos, vel)`` in ``BCRS``. Position is in ``au``.
        args : Any, optional
            Extra propagator data. This force does not use it.

        Returns
        -------
        Float[Array, "3"]
            Earth ``J2`` acceleration in ``au / day^2``. It is zero outside ``max_distance``.
        """
        pos, _ = state
        pos_center = self.body._bcrs_pos_jd(tdb_jd1, tdb_jd2)
        dist = jnp.linalg.norm(pos - pos_center)
        j2_acc = compute_j2_acceleration_vec(
            pos, pos_center, self.body.gm, self.j2, self.radius, self.get_current_pole(tdb_jd1, tdb_jd2)
        )
        return jnp.where(dist < self.max_distance, j2_acc, jnp.zeros_like(j2_acc))

    @property
    def shape(self):
        """Return the batch shape."""
        return ()
