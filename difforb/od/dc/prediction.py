from functools import partial
from typing import NamedTuple, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from difforb.astrometry.data import ObservationData
from difforb.astrometry.debias import DebiasResult
from difforb.astrometry.reduction.lt import LightTimeContext
from difforb.astrometry.reduction.optical import compute_astrometric_vector, correct_light_bending
from difforb.astrometry.reduction.photocenter import PhotocenterCorrection
from difforb.astrometry.reduction.radar import compute_radar_obs
from difforb.body.ephbody import EphemerisBody
from difforb.body.smallbody import SmallBody
from difforb.body.site import Site
from difforb.core.batch import BatchableObject
from difforb.core.state.frame import BCRS, Frame
from difforb.core.state.state import State
from difforb.core.time.timescale import TDBView, Time
from difforb.dynamics.force_model import ForceModel
from difforb.integrator.integrator import NumericalIntegrator
from difforb.utils import car2sph

jax.config.update("jax_enable_x64", True)


def _sky_plane_rates(
        pos: Float[Array, "... 3"],
        vel: Float[Array, "... 3"],
) -> Float[Array, "... 2"]:
    """Return tangent-plane angular rates from relative Cartesian state.

    Parameters
    ----------
    pos : Array, shape (..., 3)
        Relative observer-to-target position in ``au``.
    vel : Array, shape (..., 3)
        Relative observer-to-target velocity in ``au / day``.

    Returns
    -------
    Array, shape (..., 2)
        Angular rates in radians per day, ordered as
        ``(ra_dot_cos_dec, dec_dot)``.
    """
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    vx, vy, vz = vel[..., 0], vel[..., 1], vel[..., 2]
    rho2 = x * x + y * y
    rho = jnp.sqrt(rho2)
    r2 = rho2 + z * z
    r = jnp.sqrt(r2)
    ra_dot = (x * vy - y * vx) / rho2
    ra_dot_cos_dec = ra_dot * rho / r
    rho_dot = (x * vx + y * vy) / rho
    dec_dot = (rho * vz - z * rho_dot) / r2
    return jnp.stack([ra_dot_cos_dec, dec_dot], axis=-1)


class PrecomputedSite(BatchableObject):
    bcrs_state: State

    def state(self, t_obs, frame: Frame = BCRS, *, sun=None, earth=None, grid=False):
        return self.bcrs_state

    @property
    def shape(self):
        return self.bcrs_state.shape


class AstrometryMeasurementModel(NamedTuple):
    epoch_tdb: TDBView
    t_start: TDBView
    t_end: TDBView
    optical_t: Time
    optical_rx: PrecomputedSite
    optical_values: Float[Array, "N_optical_obs 2"]
    optical_photocenter_mask: Bool[Array, "N_optical_obs"]
    radar_t: Time
    radar_rx: Site
    radar_tx: Site
    radar_tx_freq: Float[Array, "N_radar_obs"]
    radar_values: Float[Array, "N_radar_obs"]
    radar_types: Int[Array, "N_radar_obs"]
    sun: EphemerisBody
    earth: EphemerisBody
    shapiro_bodies: Tuple
    photocenter_correction: PhotocenterCorrection

    @classmethod
    def build(cls, data: ObservationData, epoch_tdb: TDBView, sun: EphemerisBody, earth: EphemerisBody,
              debias_result: DebiasResult, photocenter_correction: PhotocenterCorrection | None = None):
        if photocenter_correction is None:
            photocenter_correction = PhotocenterCorrection()
        t_start = (data.t_start - 50.0).tdb()
        t_end = (data.t_end + 50.0).tdb()

        # 1. Optical
        optical_data = data.optical
        optical_rx = PrecomputedSite(
            Site.from_code(optical_data.rx_codes).state(optical_data.t, frame=BCRS, earth=earth),
        )
        optical_photocenter_mask = np.array(["e" not in str(code) for code in optical_data.note_codes], dtype=bool)

        # 2. Radar
        radar_data = data.radar
        radar_delay_mask = radar_data.is_delay
        radar_types = jnp.where(radar_delay_mask, 0, 1)

        return cls(
            epoch_tdb=epoch_tdb,
            t_start=t_start,
            t_end=t_end,
            optical_t=optical_data.t,
            optical_rx=optical_rx,
            optical_values=jnp.asarray(optical_data.values - debias_result.optical_bias),
            optical_photocenter_mask=jnp.asarray(optical_photocenter_mask),
            radar_t=radar_data.t,
            radar_rx=Site.from_code(radar_data.rx_codes).require_ground(),
            radar_tx=Site.from_code(radar_data.tx_codes).require_ground(),
            radar_tx_freq=jnp.asarray(radar_data.tx_freq),
            radar_values=jnp.asarray(radar_data.values),
            radar_types=radar_types,
            sun=sun,
            earth=earth,
            shapiro_bodies=(sun,),
            photocenter_correction=photocenter_correction,
        )

    def compute_residuals_core(self, params: Float[Array, "N_param"], force_model: ForceModel, integrator:
    NumericalIntegrator) -> tuple[Float[Array, "N_flat_obs"], Float[Array, "N_flat_obs"]]:
        sun, earth = self.sun, self.earth
        optical_context = LightTimeContext(sun=sun, earth=earth, shapiro_bodies=self.shapiro_bodies)
        radar_context = LightTimeContext(sun=sun, earth=earth, atmos_cor_enable=True, corona_cor_enable=True,
                                         shapiro_bodies=self.shapiro_bodies)
        target, photocenter_correction = self._target_and_photocenter(params, force_model, integrator)

        astro_path = compute_astrometric_vector(
            self.optical_t, self.optical_rx, target, optical_context
        )
        astro_path = photocenter_correction.apply(sun, astro_path, self.optical_photocenter_mask)
        bent_pos = correct_light_bending(sun, astro_path)
        pred_ra, pred_dec = car2sph(bent_pos)
        opt_pred = jnp.stack([pred_ra, pred_dec], axis=1)

        radar_obs = compute_radar_obs(
            self.radar_t, target, self.radar_rx, self.radar_tx,
            self.radar_tx_freq, radar_context
        )
        candidates = [
            radar_obs.delay, radar_obs.doppler_shift,
        ]
        flat_rad_pred = jnp.where(self.radar_types == 0, candidates[0], candidates[1])

        raw_opt_residuals = self.optical_values - opt_pred
        raw_ra_residuals = raw_opt_residuals[..., 0]
        ra_residuals = jnp.mod(raw_ra_residuals + jnp.pi, 2. * jnp.pi) - jnp.pi
        obs_dec = self.optical_values[..., 1]
        ra_residuals = ra_residuals * jnp.cos(obs_dec)
        dec_residuals = raw_opt_residuals[..., 1]
        flat_opt_residuals = jnp.stack([ra_residuals, dec_residuals], axis=1).ravel()
        flat_rad_residuals = self.radar_values - flat_rad_pred
        flat_residuals = jnp.concatenate([flat_opt_residuals, flat_rad_residuals])

        return flat_residuals, flat_residuals

    def _target_and_photocenter(
            self,
            params: Float[Array, "N_param"],
            force_model: ForceModel,
            integrator: NumericalIntegrator,
    ) -> tuple[SmallBody, PhotocenterCorrection]:
        """Build the propagated target and current photocenter correction."""
        orbit_params = params[..., :6]
        n_force_params = force_model.get_all_estimated_params().shape[-1]
        force_params = params[..., 6:6 + n_force_params]
        photocenter_params = params[..., 6 + n_force_params:]

        new_force_model = force_model.update_estimated_params(force_params)
        photocenter_correction = self.photocenter_correction.update_estimated_params(photocenter_params)
        target_orbit = State.from_array(self.epoch_tdb, orbit_params, BCRS)
        target = SmallBody.create(target_orbit)
        target = target.propagate(self.t_start, self.t_end, new_force_model, integrator, grid=False)
        return target, photocenter_correction

    def compute_optical_rates_core(
            self,
            params: Float[Array, "N_param"],
            force_model: ForceModel,
            integrator: NumericalIntegrator,
    ) -> Float[Array, "N_optical_obs 2"]:
        """Compute modeled optical tangent-plane rates for time weighting."""
        if self.optical_values.shape[0] == 0:
            return jnp.empty((0, 2), dtype=params.dtype)

        sun, earth = self.sun, self.earth
        optical_context = LightTimeContext(sun=sun, earth=earth, shapiro_bodies=self.shapiro_bodies)
        target, photocenter_correction = self._target_and_photocenter(params, force_model, integrator)
        astro_path = compute_astrometric_vector(
            self.optical_t, self.optical_rx, target, optical_context
        )
        astro_path = photocenter_correction.apply(sun, astro_path, self.optical_photocenter_mask)
        bent_pos = correct_light_bending(sun, astro_path)
        return _sky_plane_rates(bent_pos, astro_path.vel)

    @eqx.filter_jit
    def compute_residuals(self, params: Float[Array, "N_param"], force_model: ForceModel, integrator: NumericalIntegrator) -> \
            Float[Array, "N_flat_obs"]:
        wrapped_residual_func = partial(self.compute_residuals_core, integrator=integrator, force_model=force_model)
        flat_residuals, _ = wrapped_residual_func(params)
        return flat_residuals

    @eqx.filter_jit
    def compute_jacobian_with_residuals(self, params: Float[Array, "N_param"], force_model: ForceModel, integrator:
    NumericalIntegrator) -> tuple[Float[Array, "N_flat_obs N_param"], Float[Array, "N_flat_obs"]]:
        wrapped_residual_func = partial(self.compute_residuals_core, integrator=integrator, force_model=force_model)
        jac, flat_residuals = jax.jacfwd(wrapped_residual_func, has_aux=True)(params)
        return jac, flat_residuals

    @eqx.filter_jit
    def compute_optical_rates(
            self,
            params: Float[Array, "N_param"],
            force_model: ForceModel,
            integrator: NumericalIntegrator,
    ) -> Float[Array, "N_optical_obs 2"]:
        """Return modeled optical tangent-plane rates in radians per day."""
        return self.compute_optical_rates_core(params, force_model, integrator)
