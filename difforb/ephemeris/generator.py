"""High-level ephemeris generator.

This module wraps the single-case builders in :mod:`difforb.ephemeris.core` and exposes batch-aware public methods. Each method can use point-wise broadcasting or Cartesian-product dispatch through the ``grid`` option.
"""

from functools import partial

import warnings
from jax import Array
from jaxtyping import Float
import equinox as eqx

from difforb.astrometry.reduction.refraction import WeatherParams
from difforb.body.ephbody import EphemerisBody
from difforb.body.site import Site
from difforb.body.smallbody import SmallBody
from difforb.core.element import KepElement
from difforb.core.time.timescale import Time, TDBView
from difforb.core.validate import validate_timeview
from difforb.ephemeris.core import (OpticalTable, RadarTable, VectorTable,
                                    generate_optical_table_single_reorder, generate_radar_table_single_reorder, \
                                    generate_vector_table_single_reorder, generate_elements_single_reorder, ApsidesTable, \
                                    find_apsides_single_reorder, CloseApproachTable, _find_close_approaches_single_reorder)
from difforb.core.batch import safe_dispatch, safe_cartesian_dispatch
from difforb.report.text import build_repr, format_shape

warnings.filterwarnings(
    "ignore",
    message=r".*A JAX array is being set as static!.*",
    category=UserWarning,
)


class EphemerisGenerator(eqx.Module):
    """High-level generator for ephemeris tables."""
    target: SmallBody
    sun: EphemerisBody = eqx.field(static=True)
    earth: EphemerisBody = eqx.field(static=True)

    def __init__(self, target: SmallBody):
        """Initialize the ephemeris generator.

        Parameters
        ----------
        target : SmallBody
            Target body with the propagated trajectory.
        """
        self.target = target
        self.sun = EphemerisBody('sun')
        self.earth = EphemerisBody('earth')

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("target_type", self.target.__class__.__name__),
                ("target_shape", format_shape(getattr(self.target, "shape", ()))),
            ],
        )

    @eqx.filter_jit
    def optical_table(
            self,
            t_obs: Time,
            observer: Site,
            apply_refraction: bool = False,
            weather: WeatherParams = WeatherParams(),
            grid: bool = False,
    ) -> OpticalTable:
        """Build optical tables.

        Parameters
        ----------
        t_obs : Time
            Observation epochs at the observer.
        observer : Site
            Observer sites.
        apply_refraction : bool, default=False
            If ``True``, apply atmospheric refraction for ground observers.
        weather : WeatherParams, default=WeatherParams()
            Weather model used by the refraction correction.
        grid : bool, default=False
            If ``False``, use point-wise broadcasting. If ``True``, use the Cartesian product of the input batches.

        Returns
        -------
        OpticalTable
            Optical tables for the target. The astrometric angles use the solved down-leg direction in ``ICRS``. The apparent angles add solar light bending and stellar aberration, then rotate to the true equator and equinox of date. For ground observers, azimuth, elevation, and the apparent angles also include atmospheric refraction when ``apply_refraction=True``.

        Raises
        ------
        RuntimeError
            If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

        Notes
        -----
        Vectorize :func:`difforb.ephemeris.core.generate_optical_table_single`.
        """
        wrapper = partial(generate_optical_table_single_reorder, apply_refraction=apply_refraction, sun=self.sun,
                          earth=self.earth)
        if not grid:
            return safe_dispatch(wrapper, (0, 0, 0, 0), self.target, observer, t_obs, weather)
        else:
            return safe_cartesian_dispatch(wrapper, ((0,), (self.target,)), ((0,), (observer,)),
                                           ((0, 0), (t_obs, weather)))

    @eqx.filter_jit
    def radar_table(
            self,
            t: Time,
            rx: Site,
            tx: Site = None,
            tx_freq: Float[Array, "..."] = 0.,
            epoch_at: str = "receive",
            grid: bool = False,
    ) -> RadarTable:
        """Build radar tables.

        Parameters
        ----------
        t : Time
            Reference epochs. If ``epoch_at="receive"``, these are receive epochs at the receiver site. If ``epoch_at="transmit"``, these are transmit epochs at the transmitter site. Epochs before 1962-01-01 are not supported.
        rx : Site
            Receiver sites.
        tx : Site, optional
            Transmitter sites. If ``None``, use ``rx``.
        tx_freq : Float[Array, "..."], default=0
            Transmit frequencies in ``Hz``.
        epoch_at : {"receive", "transmit"}, default="receive"
            Signal-path endpoint represented by ``t``.
        grid : bool, default=False
            If ``False``, use point-wise broadcasting. If ``True``, use the Cartesian product of the input batches.

        Returns
        -------
        RadarTable
            Radar tables for the target.

        Raises
        ------
        ValueError
            Raised when ``epoch_at`` is not ``"receive"`` or ``"transmit"``.
        RuntimeError
            Raised when any reference epoch is earlier than 1962-01-01, or when the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

        Notes
        -----
        Vectorize :func:`difforb.ephemeris.core.generate_radar_table_single`.
        """
        wrapper = partial(generate_radar_table_single_reorder, sun=self.sun,
                          earth=self.earth, epoch_at=epoch_at)
        if not grid:
            return safe_dispatch(wrapper, (0, 0, 0, 0, 0), self.target, rx, tx, tx_freq,
                                 t)
        else:
            return safe_cartesian_dispatch(wrapper, ((0,), (self.target,)), ((0,), (rx,)),
                                           ((0, 0), (tx, tx_freq)), ((0,), (t,)))

    @eqx.filter_jit
    def vector_table(self, t_obs: Time, observer: 'Site', grid: bool = False) -> VectorTable:
        """Build vector tables.

        Parameters
        ----------
        t_obs : Time
            Observation epochs at the observer.
        observer : Site
            Observer sites.
        grid : bool, default=False
            If ``False``, use point-wise broadcasting. If ``True``, use the Cartesian product of the input batches.

        Returns
        -------
        VectorTable
            Geometric, astrometric, and apparent vectors for the target. The astrometric state uses the solved down-leg light time. The apparent state adds stellar aberration on top of the astrometric state, but it does not apply solar light bending or rotation to an equator-of-date frame.

        Raises
        ------
        TypeError
            Raised when ``t_obs`` is not a supported Earth-rotation time scale.
        RuntimeError
            If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

        Notes
        -----
        Vectorize :func:`difforb.ephemeris.core.generate_vector_table_single`.
        """
        validate_timeview(t_obs, Time, 't_obs')
        wrapper = partial(generate_vector_table_single_reorder, sun=self.sun, earth=self.earth)
        if not grid:
            return safe_dispatch(wrapper, (0, 0, 0), self.target, observer, t_obs)
        else:
            return safe_cartesian_dispatch(wrapper, ((0,), (self.target,)), ((0,), (observer,)), ((0,), (t_obs,)))

    @eqx.filter_jit
    def elements_table(self, tdb: TDBView, grid: bool = False) -> KepElement:
        """Build osculating element tables.

        Parameters
        ----------
        tdb : TDBView
            Epochs in ``TDB``.
        grid : bool, default=False
            If ``False``, use point-wise broadcasting. If ``True``, use the Cartesian product of the input batches.

        Returns
        -------
        KepElement
            Osculating Keplerian elements in the JPL Horizons ecliptic-of-J2000 reference system.

        Raises
        ------
        TypeError
            Raised when ``t`` is not in ``TDB``.
        RuntimeError
            If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

        Notes
        -----
        Vectorize :func:`difforb.ephemeris.core.generate_elements_single`.
        """
        validate_timeview(tdb, TDBView, 'tdb')
        wrapper = partial(generate_elements_single_reorder, sun=self.sun)
        if not grid:
            return safe_dispatch(wrapper, (0, 0), self.target, tdb)
        else:
            return safe_cartesian_dispatch(wrapper, ((0,), (self.target,)), ((0,), (tdb,)))

    @eqx.filter_jit
    def find_apsides(self, t_start: TDBView, t_end: TDBView, center: EphemerisBody, max_events: int = 5,
                     grid: bool = False) -> ApsidesTable:
        """Find apsides events.

        Parameters
        ----------
        t_start, t_end : TDBView
            Search interval in ``TDB``.
        center : EphemerisBody
            Center body.
        max_events : int, default=5
            Maximum number of returned events.
        grid : bool, default=False
            If ``False``, use point-wise broadcasting. If ``True``, use the Cartesian product of the input batches.

        Returns
        -------
        ApsidesTable
            Periapsis and apoapsis events for the target.

        Raises
        ------
        TypeError
            Raised when ``t_start`` or ``t_end`` is not in ``TDB``.
        RuntimeError
            If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

        Notes
        -----
        Vectorize :func:`difforb.ephemeris.core.find_apsides_single`.
        """
        validate_timeview(t_start, TDBView, 't_start')
        validate_timeview(t_end, TDBView, 't_end')
        wrapper = partial(find_apsides_single_reorder, center=center, max_events=max_events)
        if not grid:
            return safe_dispatch(wrapper, (0, 0, 0), self.target, t_start, t_end)
        else:
            return safe_cartesian_dispatch(wrapper, ((0,), (self.target,)), ((0, 0), (t_start, t_end)))

    @eqx.filter_jit
    def find_close_approaches(self, t_start: TDBView, t_end: TDBView, center: EphemerisBody, max_distance: float = 0.5,
                              max_events: int = 5, grid: bool = False) -> CloseApproachTable:
        """Find close-approach events.

        Parameters
        ----------
        t_start, t_end : TDBView
            Search interval in ``TDB``.
        center : EphemerisBody
            Center body.
        max_distance : float, default=0.5
            Maximum close-approach distance in ``au``.
        max_events : int, default=5
            Maximum number of returned events.
        grid : bool, default=False
            If ``False``, use point-wise broadcasting. If ``True``, use the Cartesian product of the input batches.

        Returns
        -------
        CloseApproachTable
            Close-approach events for the target.

        Raises
        ------
        TypeError
            Raised when ``t_start`` or ``t_end`` is not in ``TDB``.
        RuntimeError
            If the target trajectory is not initialized or the requested epoch is outside the propagated coverage.

        Notes
        -----
        Vectorize :func:`difforb.ephemeris.core.find_close_approaches_single`.
        """
        validate_timeview(t_start, TDBView, 't_start')
        validate_timeview(t_end, TDBView, 't_end')
        wrapper = partial(_find_close_approaches_single_reorder, center=center, max_distance=max_distance,
                          max_events=max_events)
        if not grid:
            return safe_dispatch(wrapper, (0, 0, 0), self.target, t_start, t_end)
        else:
            return safe_cartesian_dispatch(wrapper, ((0,), (self.target,)), ((0, 0), (t_start, t_end)))
