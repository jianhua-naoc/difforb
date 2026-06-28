"""Axis definitions and fixed rotations for ``state``.

This module defines the axis-family enum used by the new state-vector system and registers fixed rotation matrices from each supported axis family to ``ICRS``.

The registered axes follow the same conventions and numerical constants as :mod:`difforb.core.state`. In particular, ``ECLIP_J2000`` matches the JPL Horizons ecliptic-of-J2000 frame, and ``J2000`` uses the fixed bias rotation between the mean-equator J2000 frame and ``ICRS``.
"""

from __future__ import annotations

from enum import Enum

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from difforb.utils import R1_single, R2_single, R3_single, arcsec_to_rad


class Axes(str, Enum):
    """Supported Cartesian axis families for state vectors.

    The enum values represent only the orientation of the coordinate axes. Origin semantics are handled separately by the origin layer of ``state``.

    Attributes
    ----------
    ``ICRS``
        International Celestial Reference System axes.
    J2000
        J2000 mean-equator axes with the fixed bias relative to ``ICRS``.
    ECLIP_J2000
        JPL Horizons ecliptic-of-J2000 axes. It is produced by rotating around the ``ICRS`` x-axis by a standard fixed obliquity angle of 84381.448 arcseconds (IAU 76 precession-nutation model). It is not the same as the mean J2000 ecliptic reference system.
    """

    ICRS = "ICRS"
    J2000 = "J2000"
    ECLIP_J2000 = "ECLIP_J2000"


ICRS_TO_ICRS: Float[Array, "3 3"] = jnp.eye(3)
ECLIP_J2000_TO_ICRS: Float[Array, "3 3"] = R1_single(jnp.deg2rad(84381.448 / 3600.0))
J2000_TO_ICRS: Float[Array, "3 3"] = (
        R3_single(-arcsec_to_rad(-0.01460))
        @ R2_single(-arcsec_to_rad(-0.0166170))
        @ R1_single(arcsec_to_rad(-0.0068192))
)

AXES_TO_ICRS_ROT: dict[Axes, Float[Array, "3 3"]] = {
    Axes.ICRS: ICRS_TO_ICRS,
    Axes.J2000: J2000_TO_ICRS,
    Axes.ECLIP_J2000: ECLIP_J2000_TO_ICRS,
}


def axes_to_icrs_rotation(axes: Axes) -> Float[Array, "3 3"]:
    """Return the registered fixed rotation from ``axes`` to ``ICRS``.

    Parameters
    ----------
    axes : Axes
        Source axis family.

    Returns
    -------
    Float[Array, "3 3"]
        Rotation matrix from ``axes`` to ``ICRS``.

    Raises
    ------
    KeyError
        If ``axes`` does not have a registered rotation.
    """

    try:
        return AXES_TO_ICRS_ROT[axes]
    except KeyError as exc:
        raise KeyError(f"No rotation to ``ICRS`` is registered for axes {axes.value!r}.") from exc


def icrs_to_axes_rotation(axes: Axes) -> Float[Array, "3 3"]:
    """Return the fixed rotation from ``ICRS`` to ``axes``.

    Parameters
    ----------
    axes : Axes
        Target axis family.

    Returns
    -------
    Float[Array, "3 3"]
        Rotation matrix from ``ICRS`` to ``axes``.

    Notes
    -----
    For the currently supported fixed axis families, the inverse rotation is the transpose of the forward rotation.
    """

    return axes_to_icrs_rotation(axes).T
