"""Object-level time API for ``time``.

This module defines the split-Julian-date time container used by the ``time`` package. It stores epochs internally in ``TT`` and exposes view objects for ``TT``, ``TDB``, ``TAI``, mixed ``UT``, ``UT1``, and ``UTC``.

Low-level transforms come from :mod:`difforb.core.time.tai`, :mod:`difforb.core.time.utc`, :mod:`difforb.core.time.ut1`, and :mod:`difforb.core.time.tdb`. Calendar conversion comes from :mod:`difforb.core.time.utils`.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float
import equinox as eqx
import warnings

from difforb.core.batch import BatchableObject, is_array
from difforb.core.constants import J2000, JULIAN_CENTURY
from difforb.core.earth_rotation import precession_bias_matrix, nutation_matrix, polar_motion_matrix, \
    inversed_polar_motion_matrix, gcrs_to_cirs_matrix, cirs_to_gcrs_matrix, earth_rotation_angle
from difforb.core.eop import load_default_eop_file
from difforb.core.eop.container import EarthOrientationData
from difforb.core.geo import ITRS
from difforb.core.time.timedelta import TimeDelta
from difforb.core.time.tdb import tt_to_tdb, tdb_to_tt
from difforb.core.time.ut1 import tt_to_ut1, ut1_to_tt
from difforb.core.time.tai import tai_to_tt, tt_to_tai
from difforb.core.time.utc import UTC_START_JD, utc_to_tai, tai_to_utc, julian_date_for_utc, calendar_date_for_utc
from difforb.core.time.utils import renormalize_split_jd, calendar_date, julian_date, ut1_fraction, GREGORIAN_START_JD
from difforb.report.text import build_repr, format_shape, format_float_array, format_string_array
from difforb.utils import arcsec_to_rad

jax.config.update("jax_enable_x64", True)

warnings.filterwarnings(
    "ignore",
    message=r".*A JAX array is being set as static!.*",
    category=UserWarning,
)


class Time(BatchableObject):
    """Class for time representations.

    This class stores each epoch internally as a normalized ``TT`` epoch. To preserve precision in time arithmetic and time-scale conversion, the internal ``TT`` epoch is stored as a split Julian date, with the large Julian-day part and the small fractional part kept separately.

    Parameters
    ----------
    _tt_jd1 : Float[ArrayLike, "..."]
        Large component of the Julian date in ``TT``.
    _tt_jd2 : Float[ArrayLike, "..."]
        Small remainder component of the Julian date in ``TT``.
    gregorian_start : Float[ArrayLike, ""], default=GREGORIAN_START_JD
        Julian date at which computed calendar fields switch from the Julian calendar to the Gregorian calendar.
    """

    _tt_jd1: Float[Array, "..."]
    _tt_jd2: Float[Array, "..."]
    eop: EarthOrientationData = eqx.field(static=True)
    gregorian_start: float = eqx.field(static=True)

    def __init__(self, tt_jd1: Float[ArrayLike, "..."], tt_jd2: Float[ArrayLike, "..."], *, eop: EarthOrientationData | None,
                 gregorian_start: Float[ArrayLike, ""] = GREGORIAN_START_JD):
        """Initialize a time container from a ``TT`` split Julian date."""
        _tt_jd1, _tt_jd2 = jnp.asarray(tt_jd1, dtype=float), jnp.asarray(tt_jd2, dtype=float)
        self._tt_jd1, self._tt_jd2 = renormalize_split_jd(_tt_jd1, _tt_jd2)
        self.eop = eop or load_default_eop_file()
        self.gregorian_start = float(np.asarray(gregorian_start, dtype=float).reshape(-1)[0])

    def __add__(self, other):
        """Shift the epoch forward by a uniform time interval.

        Parameters
        ----------
        other : TimeDelta or ArrayLike
            Interval added to the epoch. Numeric inputs are interpreted as day offsets and are internally converted to
            :class:`difforb.core.time.timedelta.TimeDelta`.

        Returns
        -------
        Time
            Time container shifted by the requested uniform interval.

        Notes
        -----
        The addition is performed on the internally stored ``TT`` split Julian date. Numeric inputs therefore represent
        uniform day offsets, with ``1 day = 86400 SI seconds``.
        """
        if isinstance(other, TimeDelta):
            delta = other
        elif is_array(other) or np.isscalar(other):
            delta = TimeDelta.from_days(other)
        else:
            return NotImplemented

        tt_jd1 = self._tt_jd1 + delta.jd1
        tt_jd2 = self._tt_jd2 + delta.jd2
        return self.__class__(tt_jd1, tt_jd2, eop=self.eop, gregorian_start=self.gregorian_start)

    def __radd__(self, other):
        """Shift the epoch forward when the numeric day offset appears on the left."""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract an epoch or a uniform time interval.

        Parameters
        ----------
        other : Time, TimeDelta, or ArrayLike
            Operand subtracted from the epoch. If ``other`` is a :class:`Time`, the result is a
            :class:`difforb.core.time.timedelta.TimeDelta`. Numeric inputs are interpreted as day offsets and are
            internally converted to :class:`difforb.core.time.timedelta.TimeDelta`.

        Returns
        -------
        Time or TimeDelta
            Shifted epoch when subtracting a uniform interval, or a uniform interval when subtracting another epoch.

        Notes
        -----
        The subtraction is performed on the internally stored ``TT`` split Julian date. Numeric inputs therefore
        represent uniform day offsets, with ``1 day = 86400 SI seconds``.
        """
        if isinstance(other, Time):
            return TimeDelta(self._tt_jd1 - other._tt_jd1, self._tt_jd2 - other._tt_jd2)

        if isinstance(other, TimeDelta):
            delta = other
        elif is_array(other) or np.isscalar(other):
            delta = TimeDelta.from_days(other)
        else:
            return NotImplemented

        tt_jd1 = self._tt_jd1 - delta.jd1
        tt_jd2 = self._tt_jd2 - delta.jd2
        return self.__class__(tt_jd1, tt_jd2, eop=self.eop, gregorian_start=self.gregorian_start)

    def __eq__(self, other):
        """Compare whether two time containers represent the same epoch."""
        if not isinstance(other, Time):
            return NotImplemented
        return (self._tt_jd1 + self._tt_jd2) == (other._tt_jd1 + other._tt_jd2)

    def __ne__(self, other):
        """Compare whether two time containers represent different epochs."""
        if not isinstance(other, Time):
            return NotImplemented
        return (self._tt_jd1 + self._tt_jd2) != (other._tt_jd1 + other._tt_jd2)

    def __lt__(self, other):
        """Compare whether this epoch is earlier than another epoch."""
        if not isinstance(other, Time):
            return NotImplemented
        return (self._tt_jd1 + self._tt_jd2) < (other._tt_jd1 + other._tt_jd2)

    def __le__(self, other):
        """Compare whether this epoch is earlier than or equal to another epoch."""
        if not isinstance(other, Time):
            return NotImplemented
        return (self._tt_jd1 + self._tt_jd2) <= (other._tt_jd1 + other._tt_jd2)

    def __gt__(self, other):
        """Compare whether this epoch is later than another epoch."""
        if not isinstance(other, Time):
            return NotImplemented
        return (self._tt_jd1 + self._tt_jd2) > (other._tt_jd1 + other._tt_jd2)

    def __ge__(self, other):
        """Compare whether this epoch is later than or equal to another epoch."""
        if not isinstance(other, Time):
            return NotImplemented
        return (self._tt_jd1 + self._tt_jd2) >= (other._tt_jd1 + other._tt_jd2)

    @property
    def shape(self):
        """Return the broadcast batch shape carried by the time object."""
        return self._tt_jd1.shape

    def __repr__(self) -> str:
        """Return a compact summary of the internally stored ``TT`` epoch."""
        tt_view = self.tt
        return build_repr(
            self.__class__.__name__,
            [
                ("shape", format_shape(self.shape)),
                ("tt_jd", format_float_array(tt_view.jd, precision=9, scientific=False, signed=False)),
                ("tt_iso", format_string_array(tt_view.iso_string)),
            ],
        )

    @classmethod
    def from_tt_jd(cls, tt_jd1: Float[ArrayLike, "..."], tt_jd2: Float[ArrayLike, "..."], eop: EarthOrientationData | None = None,
                   gregorian_start: Float[ArrayLike, "..."] = GREGORIAN_START_JD):
        """Build a time container from a ``TT`` split Julian date."""
        return cls(tt_jd1, tt_jd2, eop=eop, gregorian_start=gregorian_start)

    @classmethod
    def from_tt_date(cls, year: Float[ArrayLike, "..."],
                     month: Float[ArrayLike, "..."], day: Float[ArrayLike, "..."],
                     hour: Float[ArrayLike, "..."] = 0.,
                     min: Float[ArrayLike, "..."] = 0.,
                     sec: Float[ArrayLike, "..."] = 0., eop: EarthOrientationData | None = None,
                     gregorian_start: Float[ArrayLike, ""] = GREGORIAN_START_JD):
        """Build a time container from a ``TT`` calendar date."""
        tt_jd1, tt_jd2 = julian_date(year, month, day, hour, min, sec, gregorian_start)
        return cls(tt_jd1, tt_jd2, eop=eop, gregorian_start=gregorian_start)

    @classmethod
    def from_tai_jd(cls, tai_jd1: Float[ArrayLike, "..."], tai_jd2: Float[ArrayLike, "..."], eop: EarthOrientationData | None =
    None,
                    gregorian_start: Float[ArrayLike, "..."] = GREGORIAN_START_JD):
        """Build a time container from a ``TAI`` split Julian date."""
        tai_jd1 = jnp.asarray(tai_jd1, dtype=float)
        tai_jd2 = jnp.asarray(tai_jd2, dtype=float)
        tt_jd1, tt_jd2 = tai_to_tt(tai_jd1, tai_jd2)
        return cls(tt_jd1, tt_jd2, eop=eop, gregorian_start=gregorian_start)

    @classmethod
    def from_tai_date(cls, year: Float[ArrayLike, "..."],
                      month: Float[ArrayLike, "..."], day: Float[ArrayLike, "..."],
                      hour: Float[ArrayLike, "..."] = 0.,
                      min: Float[ArrayLike, "..."] = 0.,
                      sec: Float[ArrayLike, "..."] = 0., eop: EarthOrientationData | None = None,
                      gregorian_start: Float[ArrayLike, ""] = GREGORIAN_START_JD):
        """Build a time container from a ``TAI`` calendar date."""
        tai_jd1, tai_jd2 = julian_date(year, month, day, hour, min, sec, gregorian_start)
        return cls.from_tai_jd(tai_jd1, tai_jd2, eop=eop, gregorian_start=gregorian_start)

    @classmethod
    def from_utc_jd(cls, utc_jd1: Float[ArrayLike, "..."], utc_jd2: Float[ArrayLike, "..."], eop: EarthOrientationData | None =
    None,
                    gregorian_start: Float[ArrayLike, ""] = GREGORIAN_START_JD):
        """Build a time container from a ``UTC`` split quasi-Julian date."""
        utc_jd1 = jnp.asarray(utc_jd1, dtype=float)
        utc_jd2 = jnp.asarray(utc_jd2, dtype=float)
        utc_jd2 = eqx.error_if(utc_jd2, jnp.any(utc_jd1 + utc_jd2 < UTC_START_JD),
                               "UTC is only defined for epochs on or after 1962-01-01. "
                               "For epochs before 1962-01-01, please use ``UT1`` timescale. "
                               "If both pre-1962 and after-1962 epochs should be contained, please use ``UT`` timescale.")
        tai_jd1, tai_jd2 = utc_to_tai(utc_jd1, utc_jd2)
        tt_jd1, tt_jd2 = tai_to_tt(tai_jd1, tai_jd2)
        return cls(tt_jd1, tt_jd2, eop=eop, gregorian_start=gregorian_start)

    @classmethod
    def from_utc_date(cls, year: Float[ArrayLike, "..."],
                      month: Float[ArrayLike, "..."], day: Float[ArrayLike, "..."],
                      hour: Float[ArrayLike, "..."] = 0.,
                      min: Float[ArrayLike, "..."] = 0.,
                      sec: Float[ArrayLike, "..."] = 0., eop: EarthOrientationData | None = None,
                      gregorian_start: Float[ArrayLike, ""] = GREGORIAN_START_JD):
        """Build a time container from a ``UTC`` calendar date."""
        utc_jd1, utc_jd2 = julian_date_for_utc(year, month, day, hour, min, sec)
        return cls.from_utc_jd(utc_jd1, utc_jd2, eop=eop, gregorian_start=gregorian_start)

    @classmethod
    def from_ut_jd(cls, ut_jd1: Float[ArrayLike, "..."], ut_jd2: Float[ArrayLike, "..."], eop: EarthOrientationData | None = None,
                   gregorian_start: Float[ArrayLike, ""] = GREGORIAN_START_JD):
        """Build a time container from a mixed ``UT`` split Julian date.

        Parameters
        ----------
        ut_jd1, ut_jd2 : Float[ArrayLike, "..."]
            Split Julian date of the mixed ``UT`` epoch. Epochs before 1962-01-01 are interpreted as ``UT1``. Epochs on
            and after 1962-01-01 are interpreted as ``UTC`` quasi-Julian dates.
        eop : EarthOrientationData, optional
            Earth orientation data used when the input epoch is interpreted as ``UT1``.
        gregorian_start : Float[ArrayLike, ""], default=GREGORIAN_START_JD
            Julian date at which the calendar fields switch from the Julian calendar to the Gregorian calendar.

        Returns
        -------
        Time
            Time container that stores the matching epoch internally in ``TT``.

        Notes
        -----
        Mixed ``UT`` follows the legacy DiffOrb convention: it represents ``UT1`` before 1962-01-01 and ``UTC`` on and
        after 1962-01-01. The boundary test uses the mixed-``UT`` Julian date itself and is independent of the coverage
        range of the loaded ``EOP`` file.
        """
        eop = eop or load_default_eop_file()

        ut_jd1 = jnp.asarray(ut_jd1, dtype=float)
        ut_jd2 = jnp.asarray(ut_jd2, dtype=float)
        ut_jd1, ut_jd2 = jnp.broadcast_arrays(ut_jd1, ut_jd2)
        ut_jd1, ut_jd2 = renormalize_split_jd(ut_jd1, ut_jd2)

        is_utc = (ut_jd1 + ut_jd2) >= UTC_START_JD

        tt_jd1_ut1, tt_jd2_ut1 = ut1_to_tt(ut_jd1, ut_jd2, eop)

        safe_utc_jd1 = jnp.where(is_utc, ut_jd1, UTC_START_JD)
        safe_utc_jd2 = jnp.where(is_utc, ut_jd2, 0.0)
        tai_jd1_utc, tai_jd2_utc = utc_to_tai(safe_utc_jd1, safe_utc_jd2)
        tt_jd1_utc, tt_jd2_utc = tai_to_tt(tai_jd1_utc, tai_jd2_utc)

        tt_jd1 = jnp.where(is_utc, tt_jd1_utc, tt_jd1_ut1)
        tt_jd2 = jnp.where(is_utc, tt_jd2_utc, tt_jd2_ut1)
        tt_jd1, tt_jd2 = renormalize_split_jd(tt_jd1, tt_jd2)
        return cls(tt_jd1, tt_jd2, eop=eop, gregorian_start=gregorian_start)

    @classmethod
    def from_ut_date(cls, year: Float[ArrayLike, "..."],
                     month: Float[ArrayLike, "..."], day: Float[ArrayLike, "..."],
                     hour: Float[ArrayLike, "..."] = 0.,
                     min: Float[ArrayLike, "..."] = 0.,
                     sec: Float[ArrayLike, "..."] = 0., eop: EarthOrientationData | None = None,
                     gregorian_start: Float[ArrayLike, ""] = GREGORIAN_START_JD):
        """Build a time container from a mixed ``UT`` calendar date.

        Parameters
        ----------
        year, month, day, hour, min, sec : Float[ArrayLike, "..."]
            Calendar fields of the mixed ``UT`` epoch. Inputs that overflow one nominal day are allowed. Epochs before
            1962-01-01 are interpreted as ``UT1`` calendar dates. Epochs on and after 1962-01-01 are interpreted as
            ``UTC`` calendar dates.
        eop : EarthOrientationData, optional
            Earth orientation data used when the input epoch is interpreted as ``UT1``.
        gregorian_start : Float[ArrayLike, ""], default=GREGORIAN_START_JD
            Julian date at which the calendar fields switch from the Julian calendar to the Gregorian calendar.

        Returns
        -------
        Time
            Time container that stores the matching epoch internally in ``TT``.

        Notes
        -----
        The 1962-01-01 split is a civil-calendar convention, so the branch selection is evaluated with the Gregorian
        calendar regardless of ``gregorian_start``. Once the branch is chosen, the ``UT1`` path uses
        :func:`difforb.core.time.utils.julian_date` with the requested ``gregorian_start``, while the ``UTC`` path uses
        :func:`difforb.core.time.utc.julian_date_for_utc`.
        """
        year = jnp.asarray(year, dtype=jnp.float64)
        month = jnp.asarray(month, dtype=jnp.float64)
        day = jnp.asarray(day, dtype=jnp.float64)
        hour = jnp.asarray(hour, dtype=jnp.float64)
        min = jnp.asarray(min, dtype=jnp.float64)
        sec = jnp.asarray(sec, dtype=jnp.float64)
        year, month, day, hour, min, sec = jnp.broadcast_arrays(year, month, day, hour, min, sec)

        probe_jd1, probe_jd2 = julian_date(year, month, day, hour, min, sec, GREGORIAN_START_JD)
        is_utc = (probe_jd1 + probe_jd2) >= UTC_START_JD

        ut1_jd1, ut1_jd2 = julian_date(year, month, day, hour, min, sec, gregorian_start)

        safe_year = jnp.where(is_utc, year, 1962.0)
        safe_month = jnp.where(is_utc, month, 1.0)
        safe_day = jnp.where(is_utc, day, 1.0)
        safe_hour = jnp.where(is_utc, hour, 0.0)
        safe_min = jnp.where(is_utc, min, 0.0)
        safe_sec = jnp.where(is_utc, sec, 0.0)
        utc_jd1, utc_jd2 = julian_date_for_utc(safe_year, safe_month, safe_day, safe_hour, safe_min, safe_sec)

        ut_jd1 = jnp.where(is_utc, utc_jd1, ut1_jd1)
        ut_jd2 = jnp.where(is_utc, utc_jd2, ut1_jd2)
        return cls.from_ut_jd(ut_jd1, ut_jd2, eop=eop, gregorian_start=gregorian_start)

    @classmethod
    def from_ut1_jd(cls, ut1_jd1: Float[ArrayLike, "..."], ut1_jd2: Float[ArrayLike, "..."],
                    eop: EarthOrientationData | None = None,
                    gregorian_start:
                    Float[ArrayLike, ""] = GREGORIAN_START_JD):
        """Build a time container from a ``UT1`` split Julian date."""
        ut1_jd1 = jnp.asarray(ut1_jd1, dtype=float)
        ut1_jd2 = jnp.asarray(ut1_jd2, dtype=float)
        eop = eop or load_default_eop_file()
        tt_jd1, tt_jd2 = ut1_to_tt(ut1_jd1, ut1_jd2, eop)
        return cls(tt_jd1, tt_jd2, eop=eop, gregorian_start=gregorian_start)

    @classmethod
    def from_ut1_date(cls, year: Float[ArrayLike, "..."],
                      month: Float[ArrayLike, "..."], day: Float[ArrayLike, "..."],
                      hour: Float[ArrayLike, "..."] = 0.,
                      min: Float[ArrayLike, "..."] = 0.,
                      sec: Float[ArrayLike, "..."] = 0., eop: EarthOrientationData | None = None,
                      gregorian_start: Float[ArrayLike, ""] = GREGORIAN_START_JD):
        """Build a time container from a ``UT1`` calendar date."""
        ut1_jd1, ut1_jd2 = julian_date(year, month, day, hour, min, sec, gregorian_start)
        return cls.from_ut1_jd(ut1_jd1, ut1_jd2, eop=eop, gregorian_start=gregorian_start)

    @classmethod
    def from_tdb_jd(cls, tdb_jd1: Float[ArrayLike, "..."], tdb_jd2: Float[ArrayLike, "..."],
                    eop: EarthOrientationData | None = None,
                    location: ITRS | None = None, grid: bool = False, gregorian_start:
            Float[ArrayLike, ""] = GREGORIAN_START_JD):
        """
        Build a time container from a ``TDB`` split Julian date.

        Parameters
        ----------
        tdb_jd1, tdb_jd2 : Float[ArrayLike, "..."]
            Split Julian date of the ``TDB`` epoch.
        eop : EarthOrientationData
            Earth orientation data used during the ``TDB`` to ``TT`` inversion.
        location : ``ITRS`` or None, optional
            Observer location used by the topocentric ``TDB - TT`` correction. If omitted, use the geocenter.
        grid : bool, default=False
            If ``False``, pair location and time inputs point-wise using normal broadcasting. If ``True``, use the
            Cartesian product of the location and time batches, following :func:`difforb.core.time.tdb.tdb_to_tt`.
        gregorian_start : Float[ArrayLike, ""], default=GREGORIAN_START_JD
            Julian date at which derived calendar fields switch from the Julian calendar to the Gregorian calendar.
        """
        tdb_jd1 = jnp.asarray(tdb_jd1, dtype=float)
        tdb_jd2 = jnp.asarray(tdb_jd2, dtype=float)
        if location is not None:
            lon = location.lon
            u = jnp.linalg.norm(location.pos[..., :2], axis=-1) / 1000.
            v = location.pos[..., 2] / 1000.
        else:
            lon, u, v = jnp.zeros_like(tdb_jd1), jnp.zeros_like(tdb_jd1), jnp.zeros_like(tdb_jd1)
        eop = eop or load_default_eop_file()
        tt_jd1, tt_jd2 = tdb_to_tt(lon, u, v, tdb_jd1, tdb_jd2, eop, grid)
        return cls(tt_jd1, tt_jd2, eop=eop, gregorian_start=gregorian_start)

    @classmethod
    def from_tdb_date(cls, year: Float[ArrayLike, "..."],
                      month: Float[ArrayLike, "..."], day: Float[ArrayLike, "..."],
                      hour: Float[ArrayLike, "..."] = 0.,
                      min: Float[ArrayLike, "..."] = 0.,
                      sec: Float[ArrayLike, "..."] = 0., eop: EarthOrientationData | None = None, location: ITRS | None = None,
                      grid: bool = False,
                      gregorian_start: Float[ArrayLike, ""] = GREGORIAN_START_JD):
        """Build a time container from a ``TDB`` calendar date.

        Parameters
        ----------
        year, month, day, hour, min, sec : Float[ArrayLike, "..."]
            Calendar fields of the ``TDB`` epoch.
        eop : EarthOrientationData, optional
            Earth orientation data used during the ``TDB`` to ``TT`` inversion.
        location : ``ITRS`` or None, optional
            Observer location used by the topocentric ``TDB - TT`` correction. If omitted, use the geocenter.
        grid : bool, default=False
            If ``False``, pair location and time inputs point-wise using normal broadcasting. If ``True``, use the
            Cartesian product of the location and time batches, following :func:`difforb.core.time.tdb.tdb_to_tt`.
        gregorian_start : Float[ArrayLike, ""], default=GREGORIAN_START_JD
            Julian date at which the calendar fields switch from the Julian calendar to the Gregorian calendar.
        """
        tdb_jd1, tdb_jd2 = julian_date(year, month, day, hour, min, sec, gregorian_start)
        return cls.from_tdb_jd(tdb_jd1, tdb_jd2, eop=eop, location=location, grid=grid,
                               gregorian_start=gregorian_start)

    @property
    def tt(self):
        """View the stored epoch in ``TT``.

        ``TT`` is the proper time of an Earth-based observer and is directly related to the atomic timescale ``TAI``. In DiffOrb it is the bridge between Earth-rotation timescales such as ``UTC`` and ``UT1``, and the dynamical timescale ``TDB``.
        """
        return TTView(self._tt_jd1, self._tt_jd2, self)

    @property
    def tai(self):
        """View the stored epoch in ``TAI``.

        ``TAI`` is a continuous atomic timescale. In DiffOrb it mainly serves as the intermediate timescale between ``UTC`` and ``TT``.
        """
        tai_jd1, tai_jd2 = tt_to_tai(self._tt_jd1, self._tt_jd2)
        tai_jd1, tai_jd2 = renormalize_split_jd(tai_jd1, tai_jd2)
        return TAIView(tai_jd1, tai_jd2, self)

    def tdb(self, location: ITRS | None = None, grid: bool = False):
        """View the stored epoch in ``TDB``.

        ``TDB`` is the barycentric relativistic timescale used for solar-system dynamics and ephemeris access.

        Parameters
        ----------
        location : ``ITRS`` or None, optional
            Observer location used by the topocentric ``TDB - TT`` correction. If omitted, use the geocenter.
        grid : bool, default=False
            If ``False``, pair location and time inputs point-wise using normal broadcasting. If ``True``, use the
            Cartesian product of the location and time batches, following :func:`difforb.core.time.tdb.tt_to_tdb`.
        """
        if location is not None:
            lon = location.lon
            u = jnp.linalg.norm(location.pos[..., :2], axis=-1) / 1000.
            v = location.pos[..., 2] / 1000.
        else:
            lon, u, v = jnp.zeros_like(self._tt_jd1), jnp.zeros_like(self._tt_jd1), jnp.zeros_like(self._tt_jd1)
        ut1_jd1, ut1_jd2 = tt_to_ut1(self._tt_jd1, self._tt_jd2, self.eop)
        ut1_frac = ut1_fraction(ut1_jd1, ut1_jd2)
        tdb_jd1, tdb_jd2 = tt_to_tdb(lon, u, v, self._tt_jd1, self._tt_jd2, ut1_frac, grid)
        tdb_jd1, tdb_jd2 = renormalize_split_jd(tdb_jd1, tdb_jd2)
        return TDBView(tdb_jd1, tdb_jd2, self)

    @property
    def ut1(self):
        """View the stored epoch in ``UT1``.

        ``UT1`` is the Earth-rotation timescale that follows the actual rotation angle of the Earth.
        """
        ut1_jd1, ut1_jd2 = tt_to_ut1(self._tt_jd1, self._tt_jd2, self.eop)
        ut1_jd1, ut1_jd2 = renormalize_split_jd(ut1_jd1, ut1_jd2)
        return UT1View(ut1_jd1, ut1_jd2, self)

    @property
    def utc(self):
        """View the stored epoch in ``UTC``.

        DiffOrb only defines ``UTC`` for epochs on and after 1962-01-01. Earlier epochs should use ``UT1`` instead. Mixed
        batches that cross the 1962 boundary should use :class:`UTView` which could be accessed by ``.ut`` property method.
        """
        tai_jd1, tai_jd2 = tt_to_tai(self._tt_jd1, self._tt_jd2)
        utc_jd1, utc_jd2, is_valid = tai_to_utc(tai_jd1, tai_jd2)
        eqx.error_if(utc_jd2, jnp.any(~is_valid),
                     "UTC is only defined for epochs on or after 1962-01-01. "
                     "For epochs before 1962-01-01, please get `UT1` timescale by `.ut1`. "
                     "If both pre-1962 and after-1962 epochs should be contained, please use ``UT`` timescale by `.ut`.")
        return UTCView(utc_jd1, utc_jd2, self)

    @property
    def ut(self):
        """View the stored epoch in mixed ``UT``.

        It represents ``UT1`` before 1962-01-01 and ``UTC`` on and after 1962-01-01.
        """
        ut1_jd1, ut1_jd2 = tt_to_ut1(self._tt_jd1, self._tt_jd2, self.eop)
        tai_jd1, tai_jd2 = tt_to_tai(self._tt_jd1, self._tt_jd2)
        utc_jd1, utc_jd2, is_utc = tai_to_utc(tai_jd1, tai_jd2)
        ut_jd1 = jnp.where(is_utc, utc_jd1, ut1_jd1)
        ut_jd2 = jnp.where(is_utc, utc_jd2, ut1_jd2)
        return UTView(ut_jd1, ut_jd2, self)

    @property
    def xpole(self) -> Float[Array, "..."]:
        """
        Polar-motion coordinate ``xp`` from the Earth Orientation Parameter (EOP) file, in radians.

        Notes
        -----
        For epochs before the covered ``EOP`` range, ``xpole`` is zero. For future epochs beyond the ``EOP``-file coverage, ``xpole`` stays at the final predicted value.
        """
        xpole = self.eop.xpole(self._tt_jd1, self._tt_jd2)
        return arcsec_to_rad(xpole)

    @property
    def ypole(self) -> Float[Array, "..."]:
        """
        Polar-motion coordinate ``yp`` from the Earth Orientation Parameter (EOP) file, in radians.

        Notes
        -----
        For epochs before the covered ``EOP`` range, ``ypole`` is zero. For future epochs beyond the ``EOP``-file coverage, ``ypole`` stays at the final predicted value.
        """
        ypole = self.eop.ypole(self._tt_jd1, self._tt_jd2)
        return arcsec_to_rad(ypole)

    @property
    def cor_delta_longitude(self) -> Float[Array, "..."]:
        """
        Return the additive ``dPsi`` correction to model nutation in longitude from the Earth Orientation Parameter (EOP) file, in radians.

        Notes
        -----
        For epochs before the covered ``EOP`` range, ``cor_delta_longitude`` is zero. For future epochs beyond the ``EOP``-file coverage, ``cor_delta_longitude`` stays at the final predicted value.
        """
        cor_delta_longitude = self.eop.cor_delta_longitude(self._tt_jd1, self._tt_jd2)
        return arcsec_to_rad(cor_delta_longitude)

    @property
    def cor_delta_obliquity(self) -> Float[Array, "..."]:
        """
        Return the additive ``dEps`` correction to model nutation in obliquity from the Earth Orientation Parameter (EOP) file, in radians.

        Notes
        -----
        For epochs before the covered ``EOP`` range, ``cor_delta_obliquity`` is zero. For future epochs beyond the ``EOP``-file coverage, ``cor_delta_obliquity`` stays at the final predicted value.
        """
        cor_delta_obliquity = self.eop.cor_delta_obliquity(self._tt_jd1, self._tt_jd2)
        return arcsec_to_rad(cor_delta_obliquity)

    @property
    def precession_bias_matrix(self) -> Float[Array, "... 3 3"]:
        """
        Return the unified precession-bias rotation matrix from ``GCRS`` to the mean equator and equinox of date for equinox-based transformation.

        Returns
        -------
        Float[Array, "... 3 3"]
            Orthogonal ``(..., 3 x 3)`` rotation matrix from the ``GCRS`` to the mean equator and equinox of date.

        Notes
        -----
        The IAU precession-bias model is used from 1799-01-01 through 2202-01-01, and the Vondrak et al. (2011) long-term
        precession model is used outside that interval.

        References
        ----------
        1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq. 6.26.
        2. Vondrák, J., et al. (2011). New precession expressions, valid for long time intervals. Appendix A.4.
        """
        tt_jd_j2000 = ((self._tt_jd1 - J2000) + self._tt_jd2) / JULIAN_CENTURY
        return precession_bias_matrix(tt_jd_j2000)

    @property
    def nutation_matrix(self) -> Float[Array, "... 3 3"]:
        """
        Return the unified nutation rotation matrix from the mean equator and equinox of date to the true equator and equinox
        of date for equinox-based transformation.

        Returns
        -------
        Float[Array, "... 3 3"]
            Orthogonal ``(..., 3 x 3)`` rotation matrix from the mean equator and equinox of date to the true equator and equinox of date.

        Notes
        -----
        Within 1799-01-01 to 2202-01-01, it returns the IAU 2000A nutation matrix with the implemented IAU 2006-compatible
        adjustments. Outside that interval it only returns the identity matrix.

        References
        ----------
        Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq. 6.39-6.41.
        """
        tt_jd_j2000 = ((self._tt_jd1 - J2000) + self._tt_jd2) / JULIAN_CENTURY
        return nutation_matrix(tt_jd_j2000)

    @property
    def polar_motion_matrix(self) -> Float[Array, "... 3 3"]:
        """
        Return the IAU polar-motion rotation matrix from the Terrestrial Intermediate Reference System (``TIRS``) to ``ITRS``.

        Returns
        -------
        Float[Array, "... 3 3"]
            Orthogonal ``(..., 3 x 3)`` rotation matrix from the ``TIRS`` to ``ITRS``.

        References
        ----------
        1. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. Eq.7.77.
        """
        tt_jd_j2000 = ((self._tt_jd1 - J2000) + self._tt_jd2) / JULIAN_CENTURY
        return polar_motion_matrix(tt_jd_j2000, self.xpole, self.ypole)

    @property
    def inversed_polar_motion_matrix(self) -> Float[Array, "... 3 3"]:
        """
        Return the inverse IAU polar-motion rotation matrix from ``ITRS`` to the Terrestrial Intermediate Reference System (``TIRS``).

        Returns
        -------
        Float[Array, "... 3 3"]
            Orthogonal ``(..., 3 x 3)`` rotation matrix from the ``ITRS`` to ``TIRS``.

        References
        ----------
        1. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models. Eq.7.138.
        """
        tt_jd_j2000 = ((self._tt_jd1 - J2000) + self._tt_jd2) / JULIAN_CENTURY
        return inversed_polar_motion_matrix(tt_jd_j2000, self.xpole, self.ypole)

    @property
    def ERA(self) -> Float[Array, "..."]:
        """
        Return the Earth rotation angle in radians.

        References
        ----------
        1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.6.59.
        """
        ut1_jd1, ut1_jd2 = tt_to_ut1(self._tt_jd1, self._tt_jd2, self.eop)
        return earth_rotation_angle(ut1_jd1, ut1_jd2)

    @property
    def gcrs_to_cirs_matrix(self) -> Float[Array, "... 3 3"]:
        """
        Return the rotation matrix from ``GCRS`` to the Celestial Intermediate Reference System (``CIRS``) for ``CIO``-based transformation.

        Returns
        -------
        Float[Array, "... 3 3"]
            Orthogonal ``(..., 3 x 3)`` rotation matrix from the ``GCRS`` to the ``CIRS``.

        Notes
        -----
        Within 1799-01-01 to 2202-01-01, it uses the IAU 2006/2000A ``CIP`` and ``CIO`` models. Outside that interval, it switches to the Vondrak et al. (2011) long-term model.

        References
        ----------
        1. Urban, S. E., & Seidelmann, P. K. (2012). Explanatory Supplement to the Astronomical Almanac. Eq.7.73, 7.75.
        """
        tt_jd_j2000 = ((self._tt_jd1 - J2000) + self._tt_jd2) / JULIAN_CENTURY
        return gcrs_to_cirs_matrix(tt_jd_j2000, self.cor_delta_obliquity, self.cor_delta_longitude)

    @property
    def cirs_to_gcrs_matrix(self) -> Float[Array, "... 3 3"]:
        """
        Return the rotation matrix from the Celestial Intermediate Reference System (``CIRS``) to ``GCRS`` for ``CIO``-based transformation.

        Returns
        -------
        Float[Array, "... 3 3"]
            Orthogonal ``(..., 3 x 3)`` rotation matrix from the ``CIRS`` to the ``GCRS``.

        Notes
        -----
        Within 1799-01-01 to 2202-01-01, it uses the IAU 2006/2000A ``CIP`` and ``CIO`` models. Outside that interval, it switches to the Vondrak et al. (2011) long-term model.

        References
        ----------
        1. Kaplan, G. H. (2005). The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models.
        Eq.6.18.
        """
        tt_jd_j2000 = ((self._tt_jd1 - J2000) + self._tt_jd2) / JULIAN_CENTURY
        return cirs_to_gcrs_matrix(tt_jd_j2000, self.cor_delta_obliquity, self.cor_delta_longitude)


class TimeView(BatchableObject):
    """Read-only time-scale view built from a :class:`Time` object.

    Each view stores one split Julian date pair in its own timescale and keeps a pointer to the parent :class:`Time` object for shared configuration such as ``EOP`` and the Gregorian switch epoch.
    """
    jd1: Float[Array, "..."]
    jd2: Float[Array, "..."]
    time: Time
    _DEFAULT_STRING_TEMPLATE: ClassVar[str] = "{YYYY}-{MM}-{DD} {hh}:{mm}:{ss.3}"

    def __init__(self, jd1: Float[Array, "..."], jd2: Float[Array, "..."], time: Time):
        """Initialize a time-scale view."""
        self.jd1 = jd1
        self.jd2 = jd2
        self.time = time

    @property
    def jd(self):
        """Return the full Julian date in this timescale."""
        return self.jd1 + self.jd2

    @property
    def ymdhms(self) -> tuple[
        Float[Array, "..."],
        Float[Array, "..."],
        Float[Array, "..."],
        Float[Array, "..."],
        Float[Array, "..."],
        Float[Array, "..."],
    ]:
        """Return calendar fields computed from the split Julian date."""
        if isinstance(self, UTCView):
            return calendar_date_for_utc(self.jd1, self.jd2)
        if isinstance(self, UTView):
            is_utc = (self.jd1 + self.jd2) >= UTC_START_JD
            safe_utc_jd1 = jnp.where(is_utc, self.jd1, UTC_START_JD)
            safe_utc_jd2 = jnp.where(is_utc, self.jd2, 0.0)
            utc_fields = calendar_date_for_utc(safe_utc_jd1, safe_utc_jd2)
            ut1_fields = calendar_date(self.jd1, self.jd2, self.time.gregorian_start)
            return tuple(jnp.where(is_utc, utc_field, ut1_field) for utc_field, ut1_field in zip(utc_fields, ut1_fields))
        else:
            return calendar_date(self.jd1, self.jd2, self.time.gregorian_start)

    @property
    def year(self) -> Float[Array, "..."]:
        """Return the calendar year computed from the stored epoch."""
        return self.ymdhms[0]

    @property
    def month(self) -> Float[Array, "..."]:
        """Return the calendar month computed from the stored epoch."""
        return self.ymdhms[1]

    @property
    def day(self) -> Float[Array, "..."]:
        """Return the calendar day-of-month computed from the stored epoch."""
        return self.ymdhms[2]

    @property
    def hour(self) -> Float[Array, "..."]:
        """Return the hour-of-day computed from the stored epoch."""
        return self.ymdhms[3]

    @property
    def min(self) -> Float[Array, "..."]:
        """Return the minute-of-hour computed from the stored epoch."""
        return self.ymdhms[4]

    @property
    def sec(self) -> Float[Array, "..."]:
        """Return the second-of-minute computed from the stored epoch."""
        return self.ymdhms[5]

    @staticmethod
    def _parse_format_template(template: str) -> list[tuple[str, str]]:
        """
        Parse a string-format template into literal and placeholder tokens.

        Returns
        -------
        list[tuple[str, str]]
            Sequence of ``("literal", text)`` and ``("placeholder", field)`` tokens.
        """
        tokens = []
        idx = 0
        while idx < len(template):
            char = template[idx]
            if char == "{":
                end_idx = template.find("}", idx + 1)
                if end_idx == -1:
                    raise ValueError(f"Unmatched '{{' in time format template: {template!r}")
                placeholder = template[idx + 1:end_idx]
                if not placeholder:
                    raise ValueError(f"Empty placeholder in time format template: {template!r}")
                tokens.append(("placeholder", placeholder))
                idx = end_idx + 1
                continue
            if char == "}":
                raise ValueError(f"Unmatched '}}' in time format template: {template!r}")

            literal_start = idx
            while idx < len(template) and template[idx] not in "{}":
                idx += 1
            tokens.append(("literal", template[literal_start:idx]))
        return tokens

    @staticmethod
    def _format_year_component(year: int) -> str:
        """Format a calendar year with a minimum width of four digits."""
        if year < 0:
            return f"-{abs(year):04d}"
        return f"{year:04d}"

    @staticmethod
    def _format_second_component(second: float, precision: int | None = None, zero_pad: bool = True) -> str:
        """
        Format the second field with optional truncated fractional digits.

        Parameters
        ----------
        second : float
            Stored second component of the epoch.
        precision : int or None, default=None
            Number of fractional digits to keep. ``None`` returns only the integer second field.
        zero_pad : bool, default=True
            If ``True``, left-pad the integer second field to two digits.

        Returns
        -------
        str
            Formatted second string.
        """
        second_value = float(second)
        second_int = int(np.trunc(second_value))
        integer_text = f"{second_int:02d}" if zero_pad else str(second_int)
        if precision is None:
            return integer_text

        scale = 10 ** precision
        frac_value = second_value - second_int
        frac_digits = int(np.trunc(frac_value * scale))
        frac_digits = min(max(frac_digits, 0), scale - 1)
        return f"{integer_text}.{frac_digits:0{precision}d}"

    @classmethod
    def _render_placeholder(cls, placeholder: str, year: int, month: int, day: int,
                            hour: int, minute: int, second: float) -> str:
        """
        Render one placeholder from calendar fields.

        Parameters
        ----------
        placeholder : str
            Placeholder token without braces.
        year, month, day, hour, minute : int
            Integer calendar components.
        second : float
            Second component, possibly with a fractional part.

        Returns
        -------
        str
            Rendered placeholder value.

        Raises
        ------
        ValueError
            If the placeholder is unknown or uses an invalid second precision.
        """
        if placeholder == "YYYY":
            return cls._format_year_component(year)
        if placeholder == "Y":
            return str(year)
        if placeholder == "MM":
            return f"{month:02d}"
        if placeholder == "M":
            return str(month)
        if placeholder == "DD":
            return f"{day:02d}"
        if placeholder == "D":
            return str(day)
        if placeholder == "hh":
            return f"{hour:02d}"
        if placeholder == "h":
            return str(hour)
        if placeholder == "mm":
            return f"{minute:02d}"
        if placeholder == "m":
            return str(minute)
        if placeholder == "ss":
            return cls._format_second_component(second)
        if placeholder == "s":
            return cls._format_second_component(second, zero_pad=False)
        if placeholder.startswith("ss."):
            precision_str = placeholder[3:]
            if not precision_str.isdigit():
                raise ValueError(f"Invalid second-precision placeholder: {placeholder!r}")
            precision = int(precision_str)
            if precision < 1:
                raise ValueError(f"Invalid second-precision placeholder: {placeholder!r}")
            return cls._format_second_component(second, precision=precision)
        if placeholder.startswith("s."):
            precision_str = placeholder[2:]
            if not precision_str.isdigit():
                raise ValueError(f"Invalid second-precision placeholder: {placeholder!r}")
            precision = int(precision_str)
            if precision < 1:
                raise ValueError(f"Invalid second-precision placeholder: {placeholder!r}")
            return cls._format_second_component(second, precision=precision, zero_pad=False)
        raise ValueError(f"Unknown time format placeholder: {placeholder!r}")

    def format_string(self, template: str = _DEFAULT_STRING_TEMPLATE) -> str | list[str]:
        """
        Format epochs from calendar fields derived from the stored split Julian date.

        Parameters
        ----------
        template : str, default="{YYYY}-{MM}-{DD} {hh}:{mm}:{ss.3}"
            Output template. Supported placeholders are:

            - ``{YYYY}``: 4-digit year. Pad with leading zeros if needed.
            - ``{Y}``: year without zero padding.
            - ``{MM}``: 2-digit month. Pad with a leading zero if needed.
            - ``{M}``: month without zero padding.
            - ``{DD}``: 2-digit day of month. Pad with a leading zero if needed.
            - ``{D}``: day of month without zero padding.
            - ``{hh}``: 2-digit hour. Pad with a leading zero if needed.
            - ``{h}``: hour without zero padding.
            - ``{mm}``: 2-digit minute. Pad with a leading zero if needed.
            - ``{m}``: minute without zero padding.
            - ``{ss}``: seconds with a 2-digit integer part. Pad with a leading zero if needed.
            - ``{s}``: seconds without zero padding in the integer part.
            - ``{ss.N}``: seconds with fixed fractional precision ``N >= 1`` and a 2-digit integer part. Pad the integer part with a leading zero if needed.
            - ``{s.N}``: seconds with fixed fractional precision ``N >= 1`` and no zero padding in the integer part.

        Returns
        -------
        str or list[str]
            Scalar strings for scalar epochs, or nested lists of strings that match the batch shape of the time object.

        Raises
        ------
        ValueError
            If ``template`` contains unmatched braces, unknown placeholders, or malformed ``{ss.N}`` fields.

        Notes
        -----
        Formatting computes the calendar fields once from the stored split Julian date. This preserves the hybrid
        Julian/Gregorian calendar convention associated with ``gregorian_start``. Fractional seconds are truncated rather
        than rounded.
        """
        tokens = self._parse_format_template(template)
        flat_y, flat_m, flat_d, flat_h, flat_min, flat_s = (
            np.asarray(field).ravel() for field in self.ymdhms
        )

        flat_string_list = []
        for y, m, d, h, mn, s in zip(flat_y, flat_m, flat_d, flat_h, flat_min, flat_s):
            year = int(np.trunc(y))
            month = int(np.trunc(m))
            day = int(np.trunc(d))
            hour = int(np.trunc(h))
            minute = int(np.trunc(mn))
            second = float(s)

            parts = []
            for token_type, token_value in tokens:
                if token_type == "literal":
                    parts.append(token_value)
                else:
                    parts.append(
                        self._render_placeholder(token_value, year, month, day, hour, minute, second)
                    )
            flat_string_list.append("".join(parts))

        string_array = np.array(flat_string_list, dtype=object).reshape(self.shape)
        return string_array.tolist()

    @property
    def iso_string(self) -> str | list[str]:
        """
        Return the default ISO-like timestamp view for display.

        Returns
        -------
        str or list[str]
            Scalar strings for scalar epochs, or nested lists of strings that match the batch shape of the time object.

        Notes
        -----
        This property is equivalent to ``format_string("{YYYY}-{MM}-{DD} {hh}:{mm}:{ss.3}")``. The strings are built
        from calendar fields computed from the stored split Julian date and therefore preserve the configured hybrid
        Julian/Gregorian convention.
        """
        return self.format_string(self._DEFAULT_STRING_TEMPLATE)

    @property
    def shape(self):
        """Return the broadcast batch shape carried by the time object."""
        return self.jd1.shape

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("shape", format_shape(self.shape)),
                ("jd", format_float_array(self.jd, precision=9, scientific=False, signed=False)),
                ("iso", format_string_array(self.iso_string)),
            ],
        )


class TTView(TimeView):
    """``TT`` view of a :class:`Time` object.

    ``TT`` is the proper time of an Earth-based observer and is directly related to the atomic timescale ``TAI``. In DiffOrb it is the bridge between Earth-rotation timescales such as ``UTC`` and ``UT1``, and the dynamical timescale ``TDB``.
    """
    jd1: Float[Array, "..."]
    jd2: Float[Array, "..."]
    time: Time
    _DEFAULT_STRING_TEMPLATE: ClassVar[str] = "{YYYY}-{MM}-{DD} {hh}:{mm}:{ss.3}"


class TDBView(TimeView):
    """``TDB`` view of a :class:`Time` object.

    ``TDB`` is the barycentric relativistic timescale used for solar-system dynamics and ephemeris access.
    """
    jd1: Float[Array, "..."]
    jd2: Float[Array, "..."]
    time: Time
    _DEFAULT_STRING_TEMPLATE: ClassVar[str] = "{YYYY}-{MM}-{DD} {hh}:{mm}:{ss.3}"


class TAIView(TimeView):
    """``TAI`` view of a :class:`Time` object.

    ``TAI`` is a continuous atomic timescale. In DiffOrb it mainly serves as the intermediate timescale between ``UTC`` and ``TT``.
    """
    jd1: Float[Array, "..."]
    jd2: Float[Array, "..."]
    time: Time
    _DEFAULT_STRING_TEMPLATE: ClassVar[str] = "{YYYY}-{MM}-{DD} {hh}:{mm}:{ss.3}"


class UTView(TimeView):
    """Mixed ``UT`` view of a :class:`Time` object.

    Mixed ``UT`` represents ``UT1`` before 1962-01-01 and ``UTC`` on and after 1962-01-01.
    """
    jd1: Float[Array, "..."]
    jd2: Float[Array, "..."]
    time: Time
    _DEFAULT_STRING_TEMPLATE: ClassVar[str] = "{YYYY}-{MM}-{DD} {hh}:{mm}:{ss.3}"


class UT1View(TimeView):
    """``UT1`` view of a :class:`Time` object.

    ``UT1`` is the Earth-rotation timescale that follows the actual rotation angle of the Earth.
    """
    jd1: Float[Array, "..."]
    jd2: Float[Array, "..."]
    time: Time
    _DEFAULT_STRING_TEMPLATE: ClassVar[str] = "{YYYY}-{MM}-{DD} {hh}:{mm}:{ss.3}"


class UTCView(TimeView):
    """``UTC`` view of a :class:`Time` object.

    DiffOrb only defines ``UTC`` for epochs on and after 1962-01-01. Earlier epochs should use ``UT1`` instead. Mixed batches that cross the 1962 boundary should use :class:`UTView`.
    """
    jd1: Float[Array, "..."]
    jd2: Float[Array, "..."]
    time: Time
    _DEFAULT_STRING_TEMPLATE: ClassVar[str] = "{YYYY}-{MM}-{DD} {hh}:{mm}:{ss.3}"
