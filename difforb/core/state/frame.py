"""Frame definitions for ``state``.

This module combines axis families and origins into one immutable frame object that can be attached to a Cartesian state vector. A frame in ``state`` is the engineering-level semantic contract for one state representation: it specifies both the axis orientation and the origin used by the stored position and velocity.

The predefined frame constants cover the common combinations already used by the current DiffOrb state-vector API, such as ``BCRS``, ``GCRS``, and heliocentric J2000-like frames.
"""

from __future__ import annotations

from dataclasses import dataclass

from difforb.report.text import build_repr

from .axes import Axes
from .origins import Origin


@dataclass(frozen=True)
class Frame:
    """Cartesian frame defined by one axis family and one origin.

    Parameters
    ----------
    axes : Axes
        Axis family used by the state vector.
    origin : Origin
        Origin used by the state vector.
    name : str, optional
        Human-facing short label for the frame. If omitted, the repr falls back to the explicit ``origin`` and ``axes`` pair.
    """

    axes: Axes
    origin: Origin
    name: str | None = None

    def __repr__(self) -> str:
        fields = [("name", self.name)]
        if self.name is None:
            fields.extend(
                [
                    ("origin", self.origin.value),
                    ("axes", self.axes.value),
                ]
            )
        return build_repr(self.__class__.__name__, fields)


BCRS = Frame(axes=Axes.ICRS, origin=Origin.SSB, name="BCRS")
GCRS = Frame(axes=Axes.ICRS, origin=Origin.EARTH, name="GCRS")
HELIO_ICRS = Frame(axes=Axes.ICRS, origin=Origin.SUN, name="HELIO_ICRS")
HELIO_J2000 = Frame(axes=Axes.J2000, origin=Origin.SUN, name="HELIO_J2000")
HELIO_ECLIP_J2000 = Frame(axes=Axes.ECLIP_J2000, origin=Origin.SUN, name="HELIO_ECLIP_J2000")
