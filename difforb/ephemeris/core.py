"""Core ephemeris table builders.

This module defines the table containers returned by the ephemeris layer and the single-case builders behind them. The generated products include optical tables, radar tables, Cartesian vector tables, osculating element tables, apsides tables, and close-approach tables.

Observation times are given in ``UT1``, ``UTC``, or mixed ``UT`` where needed. Dynamical states are evaluated in ``TDB`` and are usually expressed in ``ICRS``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, Bool, Int
import equinox as eqx

from difforb.astrometry.reduction.optical import compute_astrometric_vector_single, correct_light_bending_single, \
    compute_geometric_vector_single, correct_stellar_aberration_single
from difforb.astrometry.reduction.radar import compute_radar_obs_single, compute_radar_obs_transmit_single
from difforb.astrometry.reduction.lt import LightTimeContext
from difforb.astrometry.reduction.refraction import WeatherParams, auer_standish_refraction_single
from difforb.body.ephbody import EphemerisBody
from difforb.body.site import Site
from difforb.body.smallbody import SmallBody
from difforb.core.batch import BatchableObject
from difforb.core.element import KepElement
from difforb.core.state.frame import BCRS, HELIO_ECLIP_J2000
from difforb.core.state.relative import RelativeState
from difforb.core.time.timescale import Time, TDBView
from difforb.report.display_units import (
    OPTICAL_TABLE_SPECS,
    STATE_REPR_SPECS,
    prefixed_repr_fields_from_specs,
    repr_fields_from_specs, RADAR_TABLE_SPECS,
)
from difforb.report.text import build_repr, format_count, format_float_array, format_shape, format_string_array
from difforb.utils import car2sph, R3_single


# ==========================================
# 1. Core Data Container
# ==========================================


class VectorTable(BatchableObject):
    """Geometric, astrometric, and apparent vector table.

    Parameters
    ----------
    t_obs : Time
        Observation epoch at the observer.
    geometric : RelativeState
        Geometric relative state in fixed ``ICRS`` axes. The target and observer are both evaluated at the observed epoch, with no light-time correction.
    astrometric : RelativeState
        Astrometric relative state in fixed ``ICRS`` axes. This state uses the solved down-leg light time. The light-time solver also includes the Sun Shapiro delay.
    apparent : RelativeState
        Apparent relative state in fixed ``ICRS`` axes. This state starts from ``astrometric`` and then applies stellar aberration. It does not apply solar light bending or any rotation to an equator-of-date frame.
    light_time : Float[Array, "..."]
        One-way down-leg light time in days. This is the same light time used by ``astrometric`` and ``apparent``.
    """
    t_obs: Time
    geometric: RelativeState  # Geometric relative state in fixed ``ICRS`` axes at the observed epoch.
    astrometric: RelativeState  # Relative state in fixed ``ICRS`` axes with solved down-leg light time.
    apparent: RelativeState  # Relative state in fixed ``ICRS`` axes with stellar aberration.
    light_time: Float[Array, "..."]  # Light time [day]

    @property
    def shape(self):
        return self.light_time.shape

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("shape", format_shape(self.shape)),
                ("epoch_jd", format_float_array(self.t_obs.jd, precision=9, scientific=False, signed=False)),
                ("light_time_day", format_float_array(self.light_time)),
                *prefixed_repr_fields_from_specs(self.geometric, STATE_REPR_SPECS, "geometric"),
                *prefixed_repr_fields_from_specs(self.astrometric, STATE_REPR_SPECS, "astrometric"),
                *prefixed_repr_fields_from_specs(self.apparent, STATE_REPR_SPECS, "apparent"),
            ],
        )


class OpticalTable(BatchableObject):
    """Optical table.

    Parameters
    ----------
    t_obs : Time
        Observation epoch at the observer.
    astrometric_ra, astrometric_dec : Float[Array, "..."]
        Astrometric right ascension and declination in degrees. These angles come from the solved down-leg direction in ``ICRS``. The light-time solver also includes the Sun Shapiro delay.
    apparent_ra, apparent_dec : Float[Array, "..."]
        Apparent right ascension and declination in degrees. These angles start from the astrometric direction, then apply solar light bending and stellar aberration, and then rotate to the true equator and equinox of date. For ground observers, they also include atmospheric refraction when ``apply_refraction=True``.
    azimuth, elevation : Float[Array, "..."]
        Topocentric azimuth (from north to east) and elevation in degrees. These values come from the apparent topocentric direction. For ground
        observers, they also include atmospheric refraction when ``apply_refraction=True``.
    delta : Float[Array, "..."]
        Target-observer distance in ``au``.
    r : Float[Array, "..."]
        Target-Sun distance in ``au``.
    phase_angle : Float[Array, "..."]
        Sun-target-observer phase angle in degrees.
    elongation : Float[Array, "..."]
        Solar elongation in degrees.
    mag : Float[Array, "..."]
        Modeled apparent magnitude.
    """
    t_obs: Time
    # --- Optical Measurement ---
    astrometric_ra: Float[Array, "..."]  # Astrometric right ascension in ``ICRS`` [deg].
    astrometric_dec: Float[Array, "..."]  # Astrometric declination in ``ICRS`` [deg].
    apparent_ra: Float[Array, "..."]  # Apparent right ascension in the true equator and equinox of date [deg].
    apparent_dec: Float[Array, "..."]  # Apparent declination in the true equator and equinox of date [deg].
    azimuth: Float[Array, "..."]  # Topocentric azimuth from north to east [deg].
    elevation: Float[Array, "..."]  # Topocentric apparent elevation [deg].
    # --- Auxiliary Value ---
    delta: Float[Array, "..."]  # Target-Site Distance [AU]
    r: Float[Array, "..."]  # Target-Sun Distance [AU]
    phase_angle: Float[Array, "..."]  # Phase Angles (Sun-Target-Observer) [deg]
    elongation: Float[Array, "..."]  # Elongation [deg]
    # --- Magnitude ---
    mag: Float[Array, "..."]

    @property
    def shape(self):
        return self.astrometric_ra.shape

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("shape", format_shape(self.shape)),
                ("epoch_jd", format_float_array(self.t_obs.jd, precision=9, scientific=False, signed=False)),
                *repr_fields_from_specs(self, OPTICAL_TABLE_SPECS),
            ],
        )


class RadarTable(BatchableObject):
    """Radar table.

    Parameters
    ----------
    t : Time
        Reference epoch supplied by the caller. Its endpoint is set by ``epoch_at``.
    epoch_at : {"receive", "transmit"}
        Signal-path endpoint represented by ``t``.
    radar_delay : Float[Array, "..."]
        Two-way light time in microseconds.
    radar_doppler : Float[Array, "..."]
        Two-way Doppler shift in ``Hz``.
    radar_range : Float[Array, "..."]
        Two-way range in ``au``.
    radar_rate : Float[Array, "..."]
        Two-way range rate in ``au / day``.
    tx_azimuth, tx_elevation : Float[Array, "..."]
        Transmitter pointing azimuth and elevation in degrees at the transmit epoch. Space transmitter rows are ``NaN``.
    rx_azimuth, rx_elevation : Float[Array, "..."]
        Receiver pointing azimuth and elevation in degrees at the receive epoch. Space receiver rows are ``NaN``.
    """
    t: Time
    epoch_at: str = eqx.field(static=True)
    # --- Radar Measurement ---
    radar_delay: Float[Array, "..."]  # [us]
    radar_doppler: Float[Array, "..."]  # [Hz]
    radar_range: Float[Array, "..."]  # [au]
    radar_rate: Float[Array, "..."]  # [au/day]
    tx_azimuth: Float[Array, "..."]  # [deg]
    tx_elevation: Float[Array, "..."]  # [deg]
    rx_azimuth: Float[Array, "..."]  # [deg]
    rx_elevation: Float[Array, "..."]  # [deg]

    @property
    def shape(self):
        return self.radar_delay.shape

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("shape", format_shape(self.shape)),
                ("epoch_at", self.epoch_at),
                ("epoch_tt_jd", format_float_array(self.t.tt.jd, precision=9, scientific=False, signed=False)),
                *repr_fields_from_specs(self, RADAR_TABLE_SPECS),
            ],
        )


def _format_event_types(event_type) -> str:
    arr = np.asarray(event_type)
    mapper = np.vectorize(lambda x: "periapsis" if int(x) == 0 else "apoapsis", otypes=[object])
    return format_string_array(mapper(arr), quote=False)


class ApsidesTable(BatchableObject):
    """Apsides events.

    Parameters
    ----------
    is_valid : Bool[Array, "..."]
        Valid-event mask.
    t_apsides : TDBView
        Event epochs in ``TDB``.
    event_type : Int[Array, "..."]
        Event types. ``0`` means periapsis and ``1`` means apoapsis.
    distance : Float[Array, "..."]
        Apsides distances in ``au``.

    Notes
    -----
    This table has a fixed output length. If the actual number of events is smaller than the allocated length, the remaining slots are padding. ``is_valid`` marks valid events and invalid padding events.
    """
    is_valid: Bool[Array, "..."]
    t_apsides: TDBView
    event_type: Int[Array, "..."]  # 0 = Periapsis, 1 = Apoapsis
    distance: Float[Array, "..."]  # Apsides Distance [AU]

    @property
    def valid(self) -> 'ApsidesTable':
        return self[self.is_valid]

    @property
    def periapsis(self) -> 'ApsidesTable':
        mask = self.is_valid & (self.event_type == 0)
        return self[mask]

    @property
    def apoapsis(self) -> 'ApsidesTable':
        mask = self.is_valid & (self.event_type == 1)
        return self[mask]

    @property
    def shape(self):
        return self.is_valid.shape

    def __repr__(self) -> str:
        valid_mask = np.asarray(self.is_valid)
        return build_repr(
            self.__class__.__name__,
            [
                ("shape", format_shape(self.shape)),
                ("valid", format_count(np.sum(valid_mask), valid_mask.size)),
                ("t_jd", format_float_array(self.t_apsides.jd, precision=9, scientific=False, signed=False)),
                ("event_type", _format_event_types(self.event_type)),
                ("distance_au", format_float_array(self.distance)),
            ],
        )


class CloseApproachTable(BatchableObject):
    """Close-approach events.

    Parameters
    ----------
    is_valid : Bool[Array, "..."]
        Valid-event mask.
    t_close : TDBView
        Event epochs in ``TDB``.
    distance : Float[Array, "..."]
        Close-approach distances in ``au``.
    relative_velocity : Float[Array, "..."]
        Relative speeds in ``au / day``.

    Notes
    -----
    This table has a fixed output length. If the actual number of events is smaller than the allocated length, the remaining slots are padding. ``is_valid`` marks valid events and invalid padding events.
    """
    is_valid: Bool[Array, "..."]
    t_close: TDBView
    distance: Float[Array, "..."]
    relative_velocity: Float[Array, "..."]

    @property
    def valid(self) -> 'CloseApproachTable':
        return self[self.is_valid]

    @property
    def shape(self):
        return self.is_valid.shape

    def __repr__(self) -> str:
        valid_mask = np.asarray(self.is_valid)
        return build_repr(
            self.__class__.__name__,
            [
                ("shape", format_shape(self.shape)),
                ("valid", format_count(np.sum(valid_mask), valid_mask.size)),
                ("t_jd", format_float_array(self.t_close.jd, precision=9, scientific=False, signed=False)),
                ("distance_au", format_float_array(self.distance)),
                ("relative_velocity_au_per_d", format_float_array(self.relative_velocity)),
            ],
        )


# ==========================================
# 2. Vector Table
# ==========================================

def generate_vector_table_single(t_obs: Time, observer: Site, target: SmallBody, sun: EphemerisBody,
                                 earth: EphemerisBody) -> VectorTable:
    """Build one vector table.

    Parameters
    ----------
    t_obs : Time
        Observation epoch at the observer.
    observer : Site
        Observer site.
    target : SmallBody
        Target body with the propagated trajectory.
    sun : EphemerisBody
        Sun ephemeris body.
    earth : EphemerisBody
        Earth ephemeris body.

    Returns
    -------
    VectorTable
        Geometric, astrometric, and apparent vectors with one-way light time.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.
    """
    astrometry_context = LightTimeContext(sun=sun, earth=earth)
    # ==========================================
    # 1. Build the geometric vector.
    # ==========================================
    geom_path = compute_geometric_vector_single(t_obs, observer, target)
    t_obs_tdb = geom_path.end.tdb
    geom_state = RelativeState(t_obs_tdb, geom_path.pos, geom_path.vel)
    # ==========================================
    # 2. Build the astrometric vector.
    # ==========================================
    astro_path = compute_astrometric_vector_single(t_obs, observer, target, astrometry_context)
    lt = astro_path.lt
    astro_state = RelativeState(t_obs_tdb, astro_path.pos, astro_path.vel)
    # ==========================================
    # 3. Build the apparent vector.
    # ==========================================
    aberrated_pos = correct_stellar_aberration_single(astro_path)
    app_state = RelativeState(t_obs_tdb, aberrated_pos, astro_path.vel)

    return VectorTable(t_obs=t_obs, geometric=geom_state, astrometric=astro_state, apparent=app_state, light_time=lt)


def generate_vector_table_single_reorder(target: SmallBody, observer: Site, t_obs: Time,
                                         sun: EphemerisBody,
                                         earth: EphemerisBody) -> VectorTable:
    """Reorder ``generate_vector_table_single`` arguments for batch dispatch."""
    return generate_vector_table_single(t_obs, observer, target, sun, earth)


# ==========================================
# 3. Optical Table
# ==========================================

def generate_optical_table_single(t_obs: Time, observer: Site, target: SmallBody, apply_refraction: bool,
                                  weather: WeatherParams,
                                  sun: EphemerisBody, earth: EphemerisBody) -> OpticalTable:
    """Build one optical table.

    Parameters
    ----------
    t_obs : Time
        Observation epoch at the observer.
    observer : Site
        Observer site.
    target : SmallBody
        Target body with the propagated trajectory.
    apply_refraction : bool
        If ``True``, apply atmospheric refraction for ground observers.
    weather : WeatherParams
        Weather model used by the refraction correction.
    sun : EphemerisBody
        Sun ephemeris body.
    earth : EphemerisBody
        Earth ephemeris body.

    Returns
    -------
    OpticalTable
        Optical observables and auxiliary geometric values.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.
    """
    astrometry_context = LightTimeContext(sun=sun, earth=earth)
    # ==========================================
    # 1. Build the astrometric direction.
    # ==========================================
    astro_path = compute_astrometric_vector_single(t_obs, observer, target, astrometry_context)
    astro_vec = astro_path.pos
    astro_ra, astro_dec = car2sph(astro_vec)
    astro_ra, astro_dec = jnp.rad2deg(astro_ra), jnp.rad2deg(astro_dec)

    # ==========================================
    # 2. Build the apparent direction.
    # ==========================================
    bent_pos = correct_light_bending_single(sun, astro_path)
    bent_path = eqx.tree_at(lambda p: p.pos, astro_path, bent_pos)
    aberrated_pos = correct_stellar_aberration_single(bent_path)

    # For ground sites, also build topocentric azimuth and elevation. Space
    # rows are masked to ``NaN`` after the calculation.
    ground_itrs = observer.ground_itrs
    cirs_pos = t_obs.gcrs_to_cirs_matrix @ aberrated_pos
    tirs_pos = R3_single(t_obs.ERA) @ cirs_pos
    itrs_pos = t_obs.polar_motion_matrix @ tirs_pos

    sin_lat, cos_lat = jnp.sin(ground_itrs.geodetic_lat), jnp.cos(ground_itrs.geodetic_lat)
    sin_lon, cos_lon = jnp.sin(ground_itrs.lon), jnp.cos(ground_itrs.lon)
    enu_mat = jnp.array(
        [[-sin_lon, cos_lon, 0, ], [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
         [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]])
    enu_pos = enu_mat @ itrs_pos
    azimuth = jnp.arctan2(enu_pos[0], enu_pos[1]) % (2. * jnp.pi)
    elevation = jnp.arctan2(enu_pos[2], jnp.sqrt(enu_pos[0] ** 2 + enu_pos[1] ** 2))
    zenith_true = jnp.pi / 2. - elevation
    if apply_refraction:
        zenith_obs = zenith_true
        for _ in range(3):
            xi = auer_standish_refraction_single(zenith_obs, ground_itrs.geodetic_lat, ground_itrs.geodetic_alt,
                                                 weather)
            zenith_obs = zenith_true - xi
        elevation = jnp.pi / 2. - zenith_obs
        cel_obs = jnp.cos(elevation)
        enu_pos_refracted = jnp.array([jnp.sin(azimuth) * cel_obs, jnp.cos(azimuth) * cel_obs, jnp.sin(elevation)])
        itrs_pos_refracted = enu_mat.T @ enu_pos_refracted
        tirs_pos_refracted = t_obs.inversed_polar_motion_matrix @ itrs_pos_refracted
        cirs_pos_refracted = R3_single(-t_obs.ERA) @ tirs_pos_refracted
        refracted_pos = t_obs.cirs_to_gcrs_matrix @ cirs_pos_refracted
        aberrated_pos = jnp.where(observer.is_ground, refracted_pos, aberrated_pos)
    azimuth = jnp.where(observer.is_ground, jnp.rad2deg(azimuth), jnp.full_like(astro_ra, jnp.nan))
    elevation = jnp.where(observer.is_ground, jnp.rad2deg(elevation), jnp.full_like(astro_dec, jnp.nan))

    # Rotate from ``ICRS`` to true-equator coordinates of date.
    NPB_mat = t_obs.nutation_matrix @ t_obs.precession_bias_matrix
    apparent_pos_tod = NPB_mat @ aberrated_pos  # Position wrt true equator and equinox of date
    apparent_ra, apparent_dec = car2sph(apparent_pos_tod)
    apparent_ra, apparent_dec = jnp.rad2deg(apparent_ra), jnp.rad2deg(apparent_dec)

    # ==========================================
    # 3. Build the auxiliary geometry values.
    # ==========================================
    delta = jnp.linalg.norm(astro_vec)
    target_state = astro_path.start
    obs_state = astro_path.end
    sun2target_pos = target_state.pos - sun._bcrs_pos_jd(target_state.tdb.jd1, target_state.tdb.jd2)
    r = jnp.linalg.norm(sun2target_pos)
    sun2obs_pos = obs_state.pos - sun._bcrs_pos_jd(obs_state.tdb.jd1, obs_state.tdb.jd2)
    cos_phase_angle = jnp.dot(sun2obs_pos - sun2target_pos, -sun2target_pos) / (delta * r)
    safe_cos_phase_angle = jnp.clip(cos_phase_angle, -1., 1.)
    phase_angle_rad = jnp.arccos(safe_cos_phase_angle)
    phase_angle = jnp.rad2deg(phase_angle_rad)
    cos_elong = jnp.dot(astro_vec, -sun2obs_pos) / (delta * jnp.linalg.norm(sun2obs_pos))
    safe_cos_elong = jnp.clip(cos_elong, -1., 1.)
    elong_rad = jnp.arccos(safe_cos_elong)
    elong = jnp.rad2deg(elong_rad)

    # ==========================================
    # 4. Build the magnitude value.
    # ==========================================
    if target.mag_model is not None:
        mag = target.mag_model.compute_mag(r, delta, phase_angle_rad)
    else:
        mag = jnp.full_like(delta, jnp.nan)

    return OpticalTable(t_obs=t_obs, astrometric_ra=astro_ra, astrometric_dec=astro_dec,
                        apparent_ra=apparent_ra, apparent_dec=apparent_dec, azimuth=azimuth, elevation=elevation,
                        delta=delta, r=r,
                        phase_angle=phase_angle,
                        elongation=elong, mag=mag)


def generate_optical_table_single_reorder(target: SmallBody, observer: Site, t_obs: Time,
                                          weather: WeatherParams, apply_refraction: bool,
                                          sun: EphemerisBody,
                                          earth: EphemerisBody) -> OpticalTable:
    """Reorder ``generate_optical_table_single`` arguments for batch dispatch."""
    return generate_optical_table_single(t_obs, observer, target, apply_refraction, weather, sun,
                                         earth)


# ==========================================
# 4. Radar Table
# ==========================================

def _radar_table_from_observation(t: Time, epoch_at: str, radar_obs) -> RadarTable:
    """Build one radar table from a reduced radar observation."""
    return RadarTable(
        t=t,
        epoch_at=epoch_at,
        radar_delay=radar_obs.delay,
        radar_range=radar_obs.range,
        radar_doppler=radar_obs.doppler_shift,
        radar_rate=radar_obs.rate,
        tx_azimuth=radar_obs.tx_azimuth,
        tx_elevation=radar_obs.tx_elevation,
        rx_azimuth=radar_obs.rx_azimuth,
        rx_elevation=radar_obs.rx_elevation,
    )


def generate_radar_table_single(t: Time, rx: Site, tx: Site, tx_freq: float,
                                target: SmallBody, sun: EphemerisBody, earth: EphemerisBody,
                                epoch_at: str = "receive") -> RadarTable:
    """Build one radar table.

    Parameters
    ----------
    t : Time
        Reference epoch. If ``epoch_at="receive"``, this is the receive epoch at the receiver site. If ``epoch_at="transmit"``, this is the transmit epoch at the transmitter site.
    rx : Site
        Receiver site.
    tx : Site
        Transmitter site. If ``None``, use ``rx``.
    tx_freq : float
        Transmit frequency in ``Hz``.
    target : SmallBody
        Target body with the propagated trajectory.
    sun : EphemerisBody
        Sun ephemeris body.
    earth : EphemerisBody
        Earth ephemeris body.
    epoch_at : {"receive", "transmit"}, default="receive"
        Signal-path endpoint represented by ``t``.

    Returns
    -------
    RadarTable
        Two-way radar observables and transmitter/receiver pointing angles.

    Raises
    ------
    ValueError
        If ``epoch_at`` is not ``"receive"`` or ``"transmit"``.
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.
    """
    if epoch_at not in ("receive", "transmit"):
        raise ValueError("`epoch_at` must be 'receive' or 'transmit'.")
    radar_context = LightTimeContext(sun=sun, earth=earth, atmos_cor_enable=True, corona_cor_enable=True)
    if tx is None:
        tx = rx
    if epoch_at == "receive":
        radar_obs = compute_radar_obs_single(t, rx, tx, tx_freq, target, radar_context)
    else:
        radar_obs = compute_radar_obs_transmit_single(t, rx, tx, tx_freq, target, radar_context)
    return _radar_table_from_observation(t, epoch_at, radar_obs)


def generate_radar_table_single_reorder(target: SmallBody, rx: Site, tx: Site, tx_freq: float,
                                        t: Time, sun: EphemerisBody,
                                        earth: EphemerisBody, epoch_at: str) -> RadarTable:
    """Reorder ``generate_radar_table_single`` arguments for batch dispatch."""
    return generate_radar_table_single(t, rx, tx, tx_freq, target, sun, earth, epoch_at)


# ==========================================
# 5. Elements Table
# ==========================================

def generate_elements_single(tdb: TDBView, target: SmallBody, sun: EphemerisBody) -> KepElement:
    """Build osculating elements at one epoch.

    Parameters
    ----------
    tdb : TDBView
        Epoch in ``TDB``.
    target : SmallBody
        Target body with the propagated trajectory.
    sun : EphemerisBody
        Sun ephemeris body used to shift the origin.

    Returns
    -------
    KepElement
        Heliocentric ecliptic J2000 osculating elements.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.
    """
    target_helio_eclip_state = target.state(tdb, frame=HELIO_ECLIP_J2000, sun=sun)
    return KepElement.from_state(target_helio_eclip_state, sun=sun)


def generate_elements_single_reorder(target, tdb: TDBView, sun: EphemerisBody) -> KepElement:
    """Reorder ``generate_elements_single`` arguments for batch dispatch."""
    return generate_elements_single(tdb, target, sun)


# ==========================================
# 6. Find Apsides & Close Approach Events
# ==========================================

def find_distance_extrema_single(target: SmallBody, center: EphemerisBody,
                                 t_start: TDBView, t_end: TDBView, max_events: int,
                                 extrema_type: str = 'min', max_distance: float = jnp.inf):
    """Find distance extrema between two bodies.

    Parameters
    ----------
    target : SmallBody
        Target body with the propagated trajectory.
    center : EphemerisBody
        Center body.
    t_start, t_end : TDBView
        Search interval in ``TDB``.
    max_events : int
        Maximum number of returned candidate events.
    extrema_type : str, default='min'
        Which extrema to keep. Supported values are ``'min'``, ``'max'``, and ``'both'``.
    max_distance : float, default=inf
        Maximum allowed distance in ``au`` for minimum-distance candidates.

    Returns
    -------
    tuple[Bool[Array, "..."], TDBView, Int[Array, "..."]]
        Valid-event mask, refined event epochs, and event types. ``0`` means minimum distance and ``1`` means maximum distance.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.
    """
    # ==========================================
    # 1. Build the coarse grid and locate sign changes.
    # ==========================================
    N_pts = 2000
    jd1_grid = jnp.linspace(t_start.jd1, t_end.jd1, N_pts)
    jd2_grid = jnp.linspace(t_start.jd2, t_end.jd2, N_pts)
    target_pos, target_vel = target._bcrs_pv_jd(jd1_grid, jd2_grid)
    center_pos, center_vel = center._bcrs_pv_jd(jd1_grid, jd2_grid)
    geom_pos = target_pos - center_pos
    geom_vel = target_vel - center_vel
    dist = jnp.linalg.norm(geom_pos, axis=-1)
    pdotv = jnp.sum(geom_pos * geom_vel, axis=-1)

    # Treat numerically tiny distance-derivative samples as exact stationary points.
    # This keeps true apsides on the search boundaries from being missed just because
    # the sampled ``pdotv`` lands at a small non-zero roundoff value like ``-1e-14``.
    pdotv_sign_tol = 1e-12
    sign_pdotv = jnp.where(
        pdotv > pdotv_sign_tol,
        1.0,
        jnp.where(pdotv < -pdotv_sign_tol, -1.0, 0.0),
    )
    diff_sign = sign_pdotv[1:] - sign_pdotv[:-1]

    # Close-approach searches must apply the distance gate before ``max_events`` truncation.
    # Otherwise early, distant minima can consume all output slots and hide later valid encounters.
    candidate_size = N_pts - 1 if extrema_type == 'min' else max_events
    min_idx = jnp.where(diff_sign > 0, size=candidate_size, fill_value=-1)[0]
    max_idx = jnp.where(diff_sign < 0, size=candidate_size, fill_value=-1)[0]

    if extrema_type == 'min':
        all_idx = min_idx
        all_type = jnp.zeros_like(all_idx)
    elif extrema_type == 'max':
        all_idx, all_type = max_idx, jnp.ones_like(max_idx)
    else:
        all_idx = jnp.concatenate([min_idx, max_idx])
        all_type = jnp.concatenate([jnp.zeros_like(min_idx), jnp.ones_like(max_idx)])

    sort_keys = jnp.where(all_idx == -1, jnp.inf, all_idx)
    sort_order = jnp.argsort(sort_keys)
    final_idx = all_idx[sort_order]
    final_type = all_type[sort_order]
    is_valid = final_idx != -1

    idx0 = jnp.maximum(final_idx, 0)
    idx1 = jnp.minimum(final_idx + 1, N_pts - 1)

    # ==========================================
    # 2. Refine the candidate times with the secant method.
    # ==========================================
    def _compute_pdotv_single(jd1, jd2):
        target_pos, target_vel = target._bcrs_pv_jd(jd1, jd2)
        center_pos, center_vel = center._bcrs_pv_jd(jd1, jd2)
        return jnp.dot(target_pos - center_pos, target_vel - center_vel)

    def _refine_root_single(jd1_after, jd2_after, jd1_before, jd2_before, valid):
        left_jd = jnp.minimum(jd1_after + jd2_after, jd1_before + jd2_before)
        right_jd = jnp.maximum(jd1_after + jd2_after, jd1_before + jd2_before)

        def split_jd(jd):
            jd1 = jnp.floor(jd)
            return jd1, jd - jd1

        def _compute_pdotv_jd(jd):
            jd1, jd2 = split_jd(jd)
            return _compute_pdotv_single(jd1, jd2)

        pdotv_left = _compute_pdotv_jd(left_jd)
        pdotv_right = _compute_pdotv_jd(right_jd)
        current_jd = right_jd
        current_pdotv = pdotv_right

        init_state = (left_jd, right_jd, pdotv_left, pdotv_right, current_jd, current_pdotv, 0, valid)

        def cond_fun(val):
            left_jd, right_jd, _pdotv_left, _pdotv_right, _current_jd, current_pdotv, step, valid = val
            return valid & ((right_jd - left_jd) > 1e-10) & (jnp.abs(current_pdotv) > 1e-16) & (step < 80)

        def body_fun(val):
            left_jd, right_jd, pdotv_left, pdotv_right, _current_jd, _current_pdotv, step, valid = val
            df = pdotv_right - pdotv_left
            secant_jd = right_jd - pdotv_right * (right_jd - left_jd) / jnp.where(jnp.abs(df) < 1e-16, 1e-16, df)
            midpoint_jd = 0.5 * (left_jd + right_jd)
            use_midpoint = (secant_jd <= left_jd) | (secant_jd >= right_jd) | ~jnp.isfinite(secant_jd)
            current_jd = jnp.where(use_midpoint, midpoint_jd, secant_jd)
            current_pdotv = _compute_pdotv_jd(current_jd)

            # Keep every trial inside the original sign-change bracket. This prevents
            # distance-extrema refinement from querying propagated or SPK states outside
            # the requested search interval.
            replace_right = (pdotv_left == 0.0) | (current_pdotv == 0.0) | ((pdotv_left > 0.0) != (current_pdotv > 0.0))
            next_left_jd = jnp.where(replace_right, left_jd, current_jd)
            next_right_jd = jnp.where(replace_right, current_jd, right_jd)
            next_pdotv_left = jnp.where(replace_right, pdotv_left, current_pdotv)
            next_pdotv_right = jnp.where(replace_right, current_pdotv, pdotv_right)
            return next_left_jd, next_right_jd, next_pdotv_left, next_pdotv_right, current_jd, current_pdotv, step + 1, valid

        _, _, _, _, final_jd, *_ = jax.lax.while_loop(cond_fun, body_fun, init_state)
        final_jd1, final_jd2 = split_jd(final_jd)
        return jnp.where(valid, final_jd1, jd1_after), jnp.where(valid, final_jd2, jd2_after)

    final_jd1, final_jd2 = jax.vmap(_refine_root_single)(jd1_grid[idx0], jd2_grid[idx0],
                                                         jd1_grid[idx1], jd2_grid[idx1], is_valid)
    time = Time.from_tdb_jd(final_jd1, final_jd2, eop=t_start.time.eop, gregorian_start=t_start.time.gregorian_start)
    tdb = time.tdb()

    if extrema_type == 'min':
        target_pos, _ = target._bcrs_pv_jd(tdb.jd1, tdb.jd2)
        center_pos, _ = center._bcrs_pv_jd(tdb.jd1, tdb.jd2)
        refined_dist = jnp.linalg.norm(target_pos - center_pos, axis=-1)
        is_valid = is_valid & (refined_dist <= max_distance)

    final_order = jnp.argsort(jnp.where(is_valid, tdb.jd, jnp.inf))[:max_events]
    return is_valid[final_order], tdb[final_order], final_type[final_order]


def find_apsides_single(t_start: TDBView, t_end: TDBView, target: SmallBody, center: EphemerisBody,
                        max_events: int) -> ApsidesTable:
    """Build one apsides table.

    Parameters
    ----------
    t_start, t_end : TDBView
        Search interval in ``TDB``.
    target : SmallBody
        Target body with the propagated trajectory.
    center : EphemerisBody
        Center body.
    max_events : int
        Maximum number of returned events.

    Returns
    -------
    ApsidesTable
        Periapsis and apoapsis events in the search interval.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.
    """
    is_valid, tdb, type = find_distance_extrema_single(target, center, t_start, t_end, max_events, 'both')
    center_pos = center._bcrs_pos_jd(tdb.jd1, tdb.jd2)
    dist = jnp.linalg.norm(target.state(tdb, frame=BCRS).pos - center_pos, axis=-1)
    return ApsidesTable(is_valid, tdb, type, dist)


def find_apsides_single_reorder(target: SmallBody, t_start: TDBView, t_end: TDBView, center: EphemerisBody,
                                max_events: int) -> ApsidesTable:
    """Reorder ``find_apsides_single`` arguments for batch dispatch."""
    return find_apsides_single(t_start, t_end, target, center, max_events)


def find_close_approaches_single(t_start: TDBView, t_end: TDBView, target: SmallBody, center: EphemerisBody,
                                 max_distance: float, max_events: int) -> CloseApproachTable:
    """Build one close-approach table.

    Parameters
    ----------
    t_start, t_end : TDBView
        Search interval in ``TDB``.
    target : SmallBody
        Target body with the propagated trajectory.
    center : EphemerisBody
        Center body.
    max_distance : float
        Maximum close-approach distance in ``au``.
    max_events : int
        Maximum number of returned events.

    Returns
    -------
    CloseApproachTable
        Minimum-distance events within the requested distance limit.

    Raises
    ------
    RuntimeError
        If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.
    """
    is_valid, t_close, type = find_distance_extrema_single(
        target,
        center,
        t_start,
        t_end,
        max_events,
        extrema_type='min',
        max_distance=max_distance,
    )
    target_state = target.state(t_close, frame=BCRS)
    center_state = center.state(t_close, frame=BCRS)
    dist = jnp.linalg.norm(target_state.pos - center_state.pos, axis=-1)
    rel_vel = jnp.linalg.norm(target_state.vel - center_state.vel, axis=-1)
    is_valid = is_valid & (dist <= max_distance)
    sort_keys = jnp.where(is_valid, t_close.jd, jnp.inf)
    final_order = jnp.argsort(sort_keys)
    is_valid = is_valid[final_order]
    t_close = t_close[final_order]
    dist = jnp.where(is_valid, dist[final_order], jnp.nan)
    rel_vel = jnp.where(is_valid, rel_vel[final_order], jnp.nan)
    return CloseApproachTable(is_valid, t_close, dist, rel_vel)


def _find_close_approaches_single_reorder(target: SmallBody, t_start: TDBView, t_end: TDBView, center: EphemerisBody,
                                          max_distance: float, max_events: int) -> CloseApproachTable:
    """Reorder ``find_close_approaches_single`` arguments for batch dispatch."""
    return find_close_approaches_single(t_start, t_end, target, center, max_distance, max_events)
