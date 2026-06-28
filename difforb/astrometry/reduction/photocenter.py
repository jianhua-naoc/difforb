"""Photocenter corrections for comet optical astrometry.

This module contains small empirical corrections that move the optical
measurement point away from the dynamical center of mass before right ascension
and declination are formed. The correction is applied to solved optical
``LightPath`` objects in ``BCRS`` and does not alter the propagated center-of-mass
trajectory.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Bool, Float

from difforb.astrometry.reduction.lt import LightPath
from difforb.body.ephbody import EphemerisBody
from difforb.core.constants import AU_KM, C
from difforb.core.state.state import State

jax.config.update("jax_enable_x64", True)


def _apply_s0(
        sun: EphemerisBody,
        light_path: LightPath,
        s0: Float[ArrayLike, ""],
        mask: Bool[ArrayLike, "..."] | None = None,
) -> LightPath:
    """Apply a scalar center-of-light offset to one optical light path.

    Parameters
    ----------
    sun : EphemerisBody
        Solar ephemeris body used to build the heliocentric comet direction.
    light_path : LightPath
        Solved optical path whose start state is the comet center of mass in
        ``BCRS``.
    s0 : Float[ArrayLike, ""]
        Center-of-light to center-of-mass offset at ``1 au``, in ``km``. A
        positive value shifts the optical point away from the Sun along the
        Sun-comet direction. At heliocentric distance ``r_h`` in ``au``, the
        applied distance is ``s0 / r_h**2``.
    mask : Bool[ArrayLike, "..."] or None, optional
        Per-observation mask selecting which path rows receive the correction.
        If omitted, every row is corrected.

    Returns
    -------
    LightPath
        Optical path whose start position and path vector point to the
        corrected center of light.

    Notes
    -----
    This is an observation-model correction. It is applied after the
    center-of-mass light-time solution and before angular observables are
    formed. The correction changes the path geometry but does not re-integrate
    the light-time equation.
    """
    sun_pos = sun._bcrs_pos_jd(light_path.start.tdb.jd1, light_path.start.tdb.jd2)
    sun_to_com = light_path.start.pos - sun_pos
    rh = jnp.linalg.norm(sun_to_com, axis=-1)
    offset = (jnp.asarray(s0, dtype=light_path.start.pos.dtype) / AU_KM) * sun_to_com / rh[..., None] ** 3
    if mask is not None:
        offset = jnp.where(jnp.asarray(mask, dtype=bool)[..., None], offset, 0.0)

    photocenter_pos = light_path.start.pos + offset
    path_pos = photocenter_pos - light_path.end.pos
    path_dist = jnp.linalg.norm(path_pos, axis=-1)
    start = State(
        tdb=light_path.start.tdb,
        pos=photocenter_pos,
        vel=light_path.start.vel,
        frame=light_path.start.frame,
    )
    return LightPath(
        pos=path_pos,
        vel=light_path.vel,
        dist=path_dist,
        lt=path_dist / C,
        start=start,
        end=light_path.end,
    )


class PhotocenterCorrection(eqx.Module):
    """Scalar comet optical photocenter correction.

    Parameters
    ----------
    s0 : Float[ArrayLike, ""], default=0
        Center-of-light offset at ``1 au``, in ``km``. A positive value moves
        the optical point away from the Sun along the Sun-comet direction.
    estimate : bool, default=False
        If ``True``, include ``s0`` as one least-squares measurement-model
        parameter.

    Notes
    -----
    The measurement model applies this correction to optical observations and
    skips zero-aperture optical astrometry marked by note code ``e``. These
    data-selection rules are fixed by the observation model, not by this
    parameter object.
    """

    s0: Float[Array, ""]
    estimate: bool = eqx.field(static=True)

    def __init__(
            self,
            s0: Float[ArrayLike, ""] = 0.0,
            *,
            estimate: bool = False,
    ) -> None:
        """Initialize the correction configuration."""
        self.s0 = jnp.asarray(s0, dtype=jnp.float64)
        self.estimate = bool(estimate)

    @property
    def n_estimated_params(self) -> int:
        """Return the number of estimated photocenter parameters."""
        return 1 if self.estimate else 0

    def get_estimated_params(self) -> Float[Array, "N_estimated"]:
        """Return estimated photocenter parameters in ``km``."""
        if self.estimate:
            return jnp.atleast_1d(self.s0)
        return jnp.array([], dtype=self.s0.dtype)

    def get_estimated_param_scales(self) -> Float[Array, "N_estimated"]:
        """Return characteristic scales for estimated photocenter parameters."""
        if self.estimate:
            return jnp.ones((1,), dtype=self.s0.dtype) * 1000.0
        return jnp.array([], dtype=self.s0.dtype)

    def get_estimated_param_names(self) -> list[str]:
        """Return estimated photocenter parameter names."""
        return ["S0"] if self.estimate else []

    def update_estimated_params(self, new_params: Float[Array, "N_estimated"]) -> "PhotocenterCorrection":
        """Return a copy with updated estimated photocenter parameters.

        Parameters
        ----------
        new_params : Float[Array, "N_estimated"]
            New parameter values. For the global model this contains ``S0`` in
            ``km`` when ``estimate=True``.

        Returns
        -------
        PhotocenterCorrection
            Correction configuration with the updated ``S0`` value.
        """
        if not self.estimate:
            return self
        return eqx.tree_at(lambda correction: correction.s0, self, jnp.asarray(new_params, dtype=self.s0.dtype)[0])

    def apply(self, sun: EphemerisBody, light_path: LightPath, mask: Bool[ArrayLike, "..."] | None = None) -> LightPath:
        """Apply this correction to an optical light path."""
        return _apply_s0(sun, light_path, self.s0, mask)
