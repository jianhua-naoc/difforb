"""Uniform time-interval container for ``time``.

This module defines :class:`TimeDelta`, a split-Julian-date interval type used by the ``time`` package. A ``TimeDelta`` stores a uniform duration as ``jd1 + jd2`` in day units, with ``1 day = 86400 SI seconds``.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float

from difforb.core.batch import BatchableObject
from difforb.core.constants import DAY_S
from difforb.core.time.utils import renormalize_split_jd

jax.config.update("jax_enable_x64", True)


class TimeDelta(BatchableObject):
    """Split-Julian-date container for uniform time intervals.

    ``TimeDelta`` represents a uniform duration rather than a civil calendar offset. The stored value is ``jd1 + jd2`` in day units, where ``1 day`` is exactly ``86400`` SI seconds.

    Parameters
    ----------
    jd1 : Float[ArrayLike, "..."]
        Large component of the split day interval.
    jd2 : Float[ArrayLike, "..."], default=0.0
        Small remainder component of the split day interval.
    """

    _jd1: Float[Array, "..."]
    _jd2: Float[Array, "..."]

    def __init__(self, jd1: Float[ArrayLike, "..."], jd2: Float[ArrayLike, "..."] = 0.0):
        """Initialize a time-interval container from a split day interval."""
        jd1 = jnp.asarray(jd1, dtype=jnp.float64)
        jd2 = jnp.asarray(jd2, dtype=jnp.float64)
        jd1, jd2 = jnp.broadcast_arrays(jd1, jd2)
        self._jd1, self._jd2 = renormalize_split_jd(jd1, jd2)

    @classmethod
    def from_days(cls, days: Float[ArrayLike, "..."]) -> "TimeDelta":
        """Build a time interval from durations in day units.

        Parameters
        ----------
        days : Float[ArrayLike, "..."]
            Interval length in day units.

        Returns
        -------
        TimeDelta
            Time interval that represents the requested duration.
        """
        return cls(0.0, days)

    @classmethod
    def from_seconds(cls, seconds: Float[ArrayLike, "..."]) -> "TimeDelta":
        """Build a time interval from durations in SI seconds.

        Parameters
        ----------
        seconds : Float[ArrayLike, "..."]
            Interval length in SI seconds.

        Returns
        -------
        TimeDelta
            Time interval that represents the requested duration.
        """
        seconds = jnp.asarray(seconds, dtype=jnp.float64)
        return cls(0.0, seconds / DAY_S)

    @property
    def shape(self):
        """Return the broadcast batch shape carried by the interval."""
        return self._jd1.shape

    @property
    def jd1(self) -> Float[Array, "..."]:
        """Return the large component of the split day interval."""
        return self._jd1

    @property
    def jd2(self) -> Float[Array, "..."]:
        """Return the small remainder component of the split day interval."""
        return self._jd2

    @property
    def jd(self) -> Float[Array, "..."]:
        """Return the total interval in day units."""
        return self._jd1 + self._jd2

    @property
    def days(self) -> Float[Array, "..."]:
        """Return the total interval in day units."""
        return self.jd

    @property
    def seconds(self) -> Float[Array, "..."]:
        """Return the total interval in SI seconds."""
        return self.jd * DAY_S

    def __add__(self, other: "TimeDelta") -> "TimeDelta":
        """Add two time intervals."""
        if not isinstance(other, TimeDelta):
            return NotImplemented
        return TimeDelta(self._jd1 + other._jd1, self._jd2 + other._jd2)

    def __sub__(self, other: "TimeDelta") -> "TimeDelta":
        """Subtract one time interval from another."""
        if not isinstance(other, TimeDelta):
            return NotImplemented
        return TimeDelta(self._jd1 - other._jd1, self._jd2 - other._jd2)

    def __neg__(self) -> "TimeDelta":
        """Negate the time interval."""
        return TimeDelta(-self._jd1, -self._jd2)
