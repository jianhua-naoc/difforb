"""ITRS site coordinates and Earth-rotation conversion helpers.

This module stores ground-site positions in ``ITRS`` and evaluates them as states in ``GCRS``. The main inputs are geodetic or geocentric site coordinates in meters and radians. The main output is a ``GCRS`` Cartesian state in ``au`` and ``au / day``.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array
from jaxtyping import Float
import equinox as eqx
from typing import TYPE_CHECKING, Tuple

from typing_extensions import ClassVar, TypeVar

from difforb.core.constants import AU_M, DAY_S
from difforb.utils import R3_single
from difforb.core.batch import safe_dispatch, safe_cartesian_dispatch, BatchableObject
from difforb.core.state.frame import GCRS, Frame
from difforb.core.state.origins import Origin
from difforb.core.state.state import State
from difforb.report.text import build_repr, format_float_array, format_shape

if TYPE_CHECKING:
    from difforb.body.ephbody import EphemerisBody
    from difforb.core.time.timescale import Time

jax.config.update("jax_enable_x64", True)

I = TypeVar("I", bound='ITRS')


def correct_solid_tide_displacement(site_state: State) -> State:
    """Apply the degree-2 solid Earth tide displacement to one canonical ``GCRS`` site state.

    Parameters
    ----------
    site_state : State
        Site state in canonical ``GCRS``.

    Returns
    -------
    State
        Corrected site state in canonical ``GCRS``.

    Raises
    ------
    ValueError
        If ``site_state.frame`` is not ``GCRS``.
    """
    from difforb.body.ephbody import EphemerisBody
    if site_state.frame != GCRS:
        raise ValueError("Solid tide displacement requires a site state in canonical ``GCRS``.")
    t = site_state.tdb
    sun = EphemerisBody('sun')
    earth = EphemerisBody('earth')
    moon = EphemerisBody('moon')
    delta_pos = jnp.zeros_like(site_state.pos)
    r_earth = 6378136.6 / AU_M
    h2 = 0.6078  # nominal degree 2 Love number
    l2 = 0.0847  # nominal degree 2 Shida number
    geo_pos = earth._bcrs_pos_jd(t.jd1, t.jd2)
    geo2site_pos = site_state.pos - geo_pos
    geo2site_dist = jnp.linalg.norm(geo2site_pos, axis=-1, keepdims=True)
    geo2site_uv = geo2site_pos / geo2site_dist
    term0 = r_earth ** 4 / earth.gm
    for obj in [sun, moon]:
        geo2obj_pos = obj._bcrs_pos_jd(t.jd1, t.jd2) - geo_pos
        geo2obj_dist = jnp.linalg.norm(geo2obj_pos, axis=-1, keepdims=True)
        geo2obj_uv = geo2obj_pos / geo2obj_dist
        dot = jnp.sum(geo2obj_uv * geo2site_uv, axis=-1, keepdims=True)
        term1 = term0 * obj.gm / geo2obj_dist ** 3
        term2 = h2 * geo2site_uv * (
                3 * dot ** 2 - 1
        ) / 2.
        term3 = 3 * l2 * dot * (geo2obj_uv - dot * geo2site_uv)
        delta_pos = delta_pos + term1 * (term2 + term3)
    cor_state = State(site_state.tdb, site_state.pos + delta_pos, site_state.vel, GCRS)
    return cor_state


def itrs_to_gcrs_single(itrs_pos: Float[Array, "3"], W_T: Float[Array, "3 3"], era: Float[Array, ""],
                        C_T: Float[Array, "3 3"],
                        C_T_deriv: Float[Array, "3 3"]) -> Tuple[Float[Array, "3"], Float[Array, "3"]]:
    """Convert one ``ITRS`` position to one ``GCRS`` state.

    Parameters
    ----------
    itrs_pos : Float[Array, "3"]
        ``ITRS`` Cartesian position in meters.
    W_T : Float[Array, "3 3"]
        Polar-motion matrix from ``ITRS`` to the Terrestrial Intermediate Reference System (``TIRS``).
    era : Float[Array, ""]
        Earth rotation angle in radians.
    C_T : Float[Array, "3 3"]
        Matrix from the Celestial Intermediate Reference System (``CIRS``) to ``GCRS``.
    C_T_deriv : Float[Array, "3 3"]
        Time derivative of ``C_T`` in units of ``1 / day``.

    Returns
    -------
    tuple[Float[Array, "3"], Float[Array, "3"]]
        ``GCRS`` position in ``au`` and ``GCRS`` velocity in ``au / day``.

    References
    ----------
    1. Sean Urban and P. Kenneth Seidelmann, Explanatory Supplement to the Astronomical Almanac, 2012, Sec. 7.4.3.
    2. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. Sec. 6.4.
    """
    # -------------------------------------------------------------------------
    # Step 1: Rotate the ``ITRS`` position into ``TIRS``.
    # -------------------------------------------------------------------------
    tirs_pos = W_T @ itrs_pos

    # -------------------------------------------------------------------------
    # Step 2: Rotate the ``TIRS`` position into ``CIRS`` with the Earth rotation angle.
    # -------------------------------------------------------------------------
    era_mat = R3_single(-era)
    cirs_pos = era_mat @ tirs_pos
    sinbeta = jnp.sin(era)
    cosbeta = jnp.cos(era)
    rmat = jnp.array([
        [-sinbeta, -cosbeta, 0.],
        [cosbeta, -sinbeta, 0.],
        [0., 0., 0.]
    ])
    cirs_vel = (2 * jnp.pi * 1.00273781191135448 / 86400.) * (rmat @ tirs_pos)

    # -------------------------------------------------------------------------
    # Step 3: Convert the ``CIRS`` state from SI units to ``au``-based units.
    # -------------------------------------------------------------------------
    cirs_pos = cirs_pos / AU_M
    cirs_vel = cirs_vel * DAY_S / AU_M

    # -------------------------------------------------------------------------
    # Step 4: Rotate the ``CIRS`` state into ``GCRS`` and add the derivative term.
    # -------------------------------------------------------------------------
    gcrs_pos = C_T @ cirs_pos
    # Coriolis term (Ref #1 7.4.3.4)
    gcrs_vel = C_T @ cirs_vel + C_T_deriv @ cirs_pos
    return gcrs_pos, gcrs_vel


class ITRS(BatchableObject):
    """Location in ``ITRS``.

    This class stores Cartesian site coordinates in ``ITRS`` in meters and a longitude in radians.

    Parameters
    ----------
    pos : Float[ArrayLike, "... 3"]
        Geocentric equatorial rectangular coordinates in meters.
    lon : Float[ArrayLike, "..."]
        Site longitude in radians.
    """
    a: ClassVar[float] = 0.  # equatorial radius in meter
    f_inv: ClassVar[float] = 0.  # flattening
    pos: Float[Array, "... 3"]
    lon: Float[Array, "..."]

    def __init__(self, pos: Float[ArrayLike, '...'],
                 lon: Float[ArrayLike, '...'], ) -> None:
        """Initialize an ``ITRS`` location from position and longitude.

        Parameters
        ----------
        pos : Float[ArrayLike, "... 3"]
            Geocentric equatorial rectangular coordinates in meters.
        lon : Float[ArrayLike, "..."]
            Site longitude in radians.
        """
        self.pos = jnp.asarray(pos)
        self.lon = jnp.asarray(lon)
        if self.pos.shape[-1] != 3:
            raise ValueError(f"ITRS position last dimension must be 3, got {self.pos.shape}")

    @property
    def geodetic_lat(self) -> Array:
        """Geodetic latitude in radians."""
        x = self.pos[..., 0]
        y = self.pos[..., 1]
        z = self.pos[..., 2]

        f = jnp.where(self.f_inv != 0., 1.0 / self.f_inv, 0.)

        e2 = 2.0 * f - f * f
        ep2 = e2 / (1.0 - e2)
        b = self.a * (1.0 - f)

        p = jnp.sqrt(x ** 2 + y ** 2)

        theta = jnp.arctan2(z * self.a, p * b)

        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)

        phi = jnp.arctan2(
            z + ep2 * b * sin_theta ** 3,
            p - e2 * self.a * cos_theta ** 3
        )
        return phi

    @property
    def geocentric_lat(self) -> Array:
        """Geocentric latitude in radians."""
        p = jnp.sqrt(self.pos[..., 0] ** 2 + self.pos[..., 1] ** 2)
        return jnp.arctan2(self.pos[..., 2], p)

    @property
    def geodetic_alt(self) -> Array:
        """Ellipsoidal height in meters."""
        p = jnp.sqrt(self.pos[..., 0] ** 2 + self.pos[..., 1] ** 2)
        phi = self.geodetic_lat
        sin_phi = jnp.sin(phi)

        f = jnp.where(self.f_inv != 0., 1.0 / self.f_inv, 0.)
        e2 = 2.0 * f - f * f

        N = self.a / jnp.sqrt(1.0 - e2 * sin_phi ** 2)

        h = p / jnp.cos(phi) - N
        return h

    @property
    def geocentric_dist(self) -> Array:
        """Geocentric distance in meters."""
        return jnp.linalg.norm(self.pos, axis=-1)

    @classmethod
    def from_geodetic(cls: 'type[I]', lon: Float[ArrayLike, '...'], lat: Float[ArrayLike, '...'],
                      alt: Float[ArrayLike, '...']) -> 'I':
        """Build an ``ITRS`` location from geodetic coordinates.

        Parameters
        ----------
        lon : Float[ArrayLike, "..."]
            Geodetic longitude in degrees.
        lat : Float[ArrayLike, "..."]
            Geodetic latitude in degrees.
        alt : Float[ArrayLike, "..."]
            Ellipsoid height in meters.

        Returns
        -------
        ``ITRS``
            Site position in ``ITRS``.

        References
        ----------
        1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.131.
        """
        lon = jnp.asarray(lon)
        lat = jnp.asarray(lat)
        alt = jnp.asarray(alt)
        lon, lat, alt = jnp.broadcast_arrays(lon, lat, alt)

        lon_rad = jnp.deg2rad(lon)
        lat_rad = jnp.deg2rad(lat)
        coslat = jnp.cos(lat_rad)
        sinlat = jnp.sin(lat_rad)
        coslon = jnp.cos(lon_rad)
        sinlon = jnp.sin(lon_rad)
        scal0 = (cls.f_inv - 1) / cls.f_inv
        scal1 = scal0 * scal0
        C = 1. / jnp.sqrt(coslat * coslat + scal1 * sinlat * sinlat)
        S = scal1 * C

        scal2 = (cls.a * C + alt) * coslat
        itrs_pos = jnp.stack([scal2 * coslon,
                              scal2 * sinlon,
                              (cls.a * S + alt) * sinlat], axis=-1)
        return cls(itrs_pos, lon_rad)

    @classmethod
    def from_geocentric(cls: 'type[I]', lon: Float[ArrayLike, '...'],
                        parallax_const1: Float[ArrayLike, '...'],
                        parallax_const2: Float[ArrayLike, '...']) -> 'I':
        """Build an ``ITRS`` location from geocentric constants.

        Parameters
        ----------
        lon : Float[ArrayLike, "..."]
            Geocentric longitude in degrees.
        parallax_const1 : Float[ArrayLike, "..."]
            ``rho cos(phi')`` where ``phi'`` is the geocentric latitude and ``rho`` is the geocentric distance in Earth radii.
        parallax_const2 : Float[ArrayLike, "..."]
            ``rho sin(phi')`` where ``phi'`` is the geocentric latitude and ``rho`` is the geocentric distance in Earth radii.

        Returns
        -------
        ``ITRS``
            Site position in ``ITRS``.

        Notes
        -----
        The parallax constants follow the Minor Planet Center observatory code definition (https://www.minorplanetcenter.net/iau/lists/ObsCodesF.html).

        References
        ----------
        1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.131.
        """
        lon = jnp.asarray(lon)
        parallax_const1 = jnp.asarray(parallax_const1)
        parallax_const2 = jnp.asarray(parallax_const2)
        lon, parallax_const1, parallax_const2 = jnp.broadcast_arrays(lon, parallax_const1, parallax_const2)

        lon_rad = jnp.deg2rad(lon)
        scal1 = parallax_const1 * cls.a
        scal2 = parallax_const2 * cls.a
        coslon = jnp.cos(lon_rad)
        sinlon = jnp.sin(lon_rad)
        itrs_pos = jnp.stack([scal1 * coslon, scal1 * sinlon, scal2], axis=-1)
        return cls(itrs_pos, lon_rad)

    @eqx.filter_jit
    def state(
            self,
            t: 'Time',
            frame: Frame = GCRS,
            *,
            sun: "EphemerisBody | None" = None,
            earth: "EphemerisBody | None" = None,
            grid: bool = False,
    ) -> State:
        """Convert the location in ``ITRS`` to one requested frame.

        Parameters
        ----------
        t : Time
            Time.
        frame : Frame, default=``GCRS``
            Target output frame.
        sun : EphemerisBody, optional
            Sun ephemeris body used when ``frame`` touches the ``SUN`` origin.
        earth : EphemerisBody, optional
            Earth ephemeris body used when converting from the canonical ``GCRS`` origin to another origin.
        grid : bool, default=False
            If ``False``, broadcast location and time inputs together. If ``True``, build the Cartesian product of location and time inputs.

        Returns
        -------
        State
            Location in ``frame`` at the corresponding ``TDB`` epoch.

        Raises
        ------
        ValueError
            If converting the canonical ``GCRS`` state to ``frame`` requires the Sun or Earth and the corresponding ephemeris body is not available.

        References
        ----------
        1. Sean Urban and P. Kenneth Seidelmann, Explanatory Supplement to the Astronomical Almanac, 2012, Sec. 7.4.3.
        2. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. Sec. 6.4.
        """
        # Estimate the time derivative of C_T with a central difference.
        t_delta = 1e-2 / DAY_S
        t_before = t - t_delta
        C_T_before = t_before.cirs_to_gcrs_matrix
        t_after = t + t_delta
        C_T_after = t_after.cirs_to_gcrs_matrix
        C_T_deriv = (C_T_after - C_T_before) / (2. * t_delta)
        W_T = t.inversed_polar_motion_matrix
        ERA = t.ERA
        C_T = t.cirs_to_gcrs_matrix

        # C_T, C_T_deriv = compute_C_T_and_C_T_deriv(t.jd)

        if not grid:
            gcrs_pos, gcrs_vel = safe_dispatch(itrs_to_gcrs_single, (1, 2, 0, 2, 2), self.pos, W_T, ERA, C_T,
                                               C_T_deriv)
            tdb = t.tdb(self, grid=grid)
        else:
            gcrs_pos, gcrs_vel = safe_cartesian_dispatch(itrs_to_gcrs_single, ((1,), (self.pos,)),
                                                         ((2, 0, 2, 2), (W_T, ERA, C_T, C_T_deriv)))
            tdb = t.tdb(self, grid=grid)

        state = State(tdb, gcrs_pos, gcrs_vel, GCRS)
        if frame == GCRS:
            return state
        if frame.origin is not Origin.EARTH and earth is None:
            from difforb.body.ephbody import EphemerisBody
            earth = EphemerisBody("earth")
        if frame.origin is Origin.SUN and sun is None:
            from difforb.body.ephbody import EphemerisBody
            sun = EphemerisBody("sun")
        return state.to(frame, sun=sun, earth=earth)

    @property
    def shape(self):
        """Return the batch shape."""
        return self.pos.shape[:-1]

    def __repr__(self) -> str:
        fields = [
            ("shape", format_shape(self.shape)),
            ("lon_deg", format_float_array(jnp.rad2deg(self.lon))),
        ]
        try:
            fields.extend(
                [
                    ("lat_deg", format_float_array(jnp.rad2deg(self.geodetic_lat))),
                    ("alt_m", format_float_array(self.geodetic_alt)),
                ]
            )
        except Exception:
            fields.append(("pos_m", format_float_array(self.pos)))
        return build_repr(self.__class__.__name__, fields)


class WGS84(ITRS):
    """Location in ``WGS84``. ``WGS84`` is one implementation of ``ITRS``."""
    a: ClassVar[float] = 6378137.
    f_inv: ClassVar[float] = 298.257223563


class ITRF(ITRS):
    """Location in ``ITRF``. ``ITRF`` is one implementation of ``ITRS``."""
    a: ClassVar[float] = 6378136.6
    f_inv: ClassVar[float] = 298.25642
