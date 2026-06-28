"""Origin definitions and state providers for ``state``.

This module defines the origin enum used by the new state-vector system and registers provider functions that return each supported origin state with respect to the Solar System Barycenter (``SSB``) in ``ICRS``.

The current built-in origins are ``SSB``, ``SUN``, and ``EARTH``. The nontrivial origin states are obtained from :class:`difforb.body.ephbody.EphemerisBody`, so callers provide the required ephemeris bodies explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

if TYPE_CHECKING:
    from difforb.body.ephbody import EphemerisBody
    from difforb.core.time.timescale import TDBView

OriginProviderFn: TypeAlias = Callable[
    ["TDBView", "EphemerisBody | None", "EphemerisBody | None"],
    tuple[Float[Array, "... 3"], Float[Array, "... 3"]],
]


class Origin(str, Enum):
    """Supported Cartesian origins for state vectors.

    Attributes
    ----------
    SSB
        Solar System Barycenter.
    SUN
        Sun center of mass.
    EARTH
        Earth center of mass.
    """

    SSB = "SSB"
    SUN = "SUN"
    EARTH = "EARTH"


def _ssb_origin_in_ssb_icrs(
        tdb: "TDBView",
        _sun: "EphemerisBody | None",
        _earth: "EphemerisBody | None",
) -> tuple[Float[Array, "... 3"], Float[Array, "... 3"]]:
    """Return the ``SSB`` origin state in ``SSB`` and ``ICRS``.

    Parameters
    ----------
    tdb : TDBView
        Epoch in ``TDB``.
    _sun, _earth : EphemerisBody, optional
        Unused ephemeris-body arguments.

    Returns
    -------
    tuple[Float[Array, "... 3"], Float[Array, "... 3"]]
        Zero position and velocity with shape ``tdb.shape + (3,)``.
    """

    shape = tdb.shape + (3,)
    zeros = jnp.zeros(shape, dtype=float)
    return zeros, zeros


def _sun_origin_in_ssb_icrs(
        tdb: "TDBView",
        sun: "EphemerisBody | None",
        _earth: "EphemerisBody | None",
) -> tuple[Float[Array, "... 3"], Float[Array, "... 3"]]:
    """Return the Sun origin state in ``SSB`` and ``ICRS``.

    Parameters
    ----------
    tdb : TDBView
        Epoch in ``TDB``.
    sun : EphemerisBody, optional
        Sun ephemeris body. It must be provided.
    _earth : EphemerisBody, optional
        Unused Earth ephemeris body.

    Returns
    -------
    tuple[Float[Array, "... 3"], Float[Array, "... 3"]]
        Sun barycentric position and velocity in ``au`` and ``au / day``.

    Raises
    ------
    ValueError
        If ``sun`` is not provided.
    """

    if sun is None:
        raise ValueError("Origin ``SUN`` requires the ``sun`` ephemeris body.")
    return sun._bcrs_pv_jd(tdb.jd1, tdb.jd2)


def _earth_origin_in_ssb_icrs(
        tdb: "TDBView",
        _sun: "EphemerisBody | None",
        earth: "EphemerisBody | None",
) -> tuple[Float[Array, "... 3"], Float[Array, "... 3"]]:
    """Return the Earth origin state in ``SSB`` and ``ICRS``.

    Parameters
    ----------
    tdb : TDBView
        Epoch in ``TDB``.
    _sun : EphemerisBody, optional
        Unused Sun ephemeris body.
    earth : EphemerisBody, optional
        Earth ephemeris body. It must be provided.

    Returns
    -------
    tuple[Float[Array, "... 3"], Float[Array, "... 3"]]
        Earth barycentric position and velocity in ``au`` and ``au / day``.

    Raises
    ------
    ValueError
        If ``earth`` is not provided.
    """

    if earth is None:
        raise ValueError("Origin ``EARTH`` requires the ``earth`` ephemeris body.")
    return earth._bcrs_pv_jd(tdb.jd1, tdb.jd2)


ORIGIN_IN_SSB_ICRS: dict[Origin, OriginProviderFn] = {
    Origin.SSB: _ssb_origin_in_ssb_icrs,
    Origin.SUN: _sun_origin_in_ssb_icrs,
    Origin.EARTH: _earth_origin_in_ssb_icrs,
}


def origin_in_ssb_icrs(
        origin: Origin,
        tdb: "TDBView",
        *,
        sun: "EphemerisBody | None" = None,
        earth: "EphemerisBody | None" = None,
) -> tuple[Float[Array, "... 3"], Float[Array, "... 3"]]:
    """Return one origin state in ``SSB`` and ``ICRS``.

    Parameters
    ----------
    origin : Origin
        Requested origin.
    tdb : TDBView
        Epoch in ``TDB``.
    sun : EphemerisBody, optional
        Sun ephemeris body used when the origin is ``SUN``.
    earth : EphemerisBody, optional
        Earth ephemeris body used when the origin is ``EARTH``.

    Returns
    -------
    tuple[Float[Array, "... 3"], Float[Array, "... 3"]]
        Origin position and velocity with respect to ``SSB`` in ``ICRS``.

    Raises
    ------
    KeyError
        If ``origin`` does not have a registered provider.
    """

    try:
        provider = ORIGIN_IN_SSB_ICRS[origin]
    except KeyError as exc:
        raise KeyError(f"No provider in ``SSB`` and ``ICRS`` is registered for origin {origin.value!r}.") from exc
    return provider(tdb, sun, earth)
