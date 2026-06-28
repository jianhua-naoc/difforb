"""Pytest fixtures for ``core.state`` tests.

This module provides minimal fake ephemeris bodies that satisfy the
``_bcrs_pv_jd`` protocol consumed by :mod:`difforb.core.state.origins` and
:mod:`difforb.core.state.state`. The fixtures keep the origin translation tests
local to ``tests/state`` and avoid any dependency on external SPK
ephemerides.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import pytest
from jax import Array
from jaxtyping import Float


@dataclass(frozen=True)
class FakeEphemerisBody:
    """Minimal fake ephemeris body for ``core.state`` tests.

    Parameters
    ----------
    _pos0_values : tuple[float, float, float]
        Barycentric position in ``au`` at ``jd_ref``.
    _vel0_values : tuple[float, float, float]
        Barycentric velocity in ``au / day``. The fake body follows a constant
        velocity model.
    jd_ref : float, default=2451545.0
        Reference Julian date in ``TDB`` used by the linear state model.

    Notes
    -----
    The returned state is

    ``pos(jd) = pos0 + (jd - jd_ref) * vel0``

    and

    ``vel(jd) = vel0``.

    This is not intended to emulate physical ephemerides. It only provides a
    deterministic and shape-aware origin state for ``core.state`` tests.
    """

    _pos0_values: tuple[float, float, float]
    _vel0_values: tuple[float, float, float]
    jd_ref: float = 2451545.0

    @property
    def pos0(self) -> Float[Array, "3"]:
        """Return the reference barycentric position in ``au``."""

        return jnp.asarray(self._pos0_values, dtype=float)

    @property
    def vel0(self) -> Float[Array, "3"]:
        """Return the reference barycentric velocity in ``au / day``."""

        return jnp.asarray(self._vel0_values, dtype=float)

    def _bcrs_pv_jd(
            self,
            jd1: Float[Array, "..."],
            jd2: Float[Array, "..."],
    ) -> tuple[Float[Array, "... 3"], Float[Array, "... 3"]]:
        """Return the fake barycentric state at one ``TDB`` epoch.

        Parameters
        ----------
        jd1 : Float[Array, "..."]
            Large split-Julian-date component in ``TDB``.
        jd2 : Float[Array, "..."]
            Small split-Julian-date component in ``TDB``.

        Returns
        -------
        tuple[Float[Array, "... 3"], Float[Array, "... 3"]]
            Fake barycentric position and velocity in ``au`` and ``au / day``.
        """

        jd = jnp.asarray(jd1, dtype=float) + jnp.asarray(jd2, dtype=float)
        dt = jd - float(self.jd_ref)
        pos = jnp.asarray(self.pos0, dtype=float) + dt[..., None] * jnp.asarray(self.vel0, dtype=float)
        vel = jnp.broadcast_to(jnp.asarray(self.vel0, dtype=float), pos.shape)
        return pos, vel


@pytest.fixture
def fake_sun() -> FakeEphemerisBody:
    """Build one deterministic fake Sun ephemeris body for ``core.state`` tests."""

    return FakeEphemerisBody(
        _pos0_values=(0.1245, -0.2875, 0.03125),
        _vel0_values=(0.0021, -0.0017, 0.0008),
        jd_ref=2451545.0,
    )


@pytest.fixture
def fake_earth() -> FakeEphemerisBody:
    """Build one deterministic fake Earth ephemeris body for ``core.state`` tests."""

    return FakeEphemerisBody(
        _pos0_values=(-0.742, 0.615, 0.102),
        _vel0_values=(0.0112, -0.0134, 0.0046),
        jd_ref=2451545.0,
    )
