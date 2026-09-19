from typing import NamedTuple, Type

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from difforb.astrometry.data import ObservationLayout
from difforb.astrometry.reduction.photocenter import PhotocenterCorrection
from difforb.body.smallbody import Orbit
from difforb.body.ephbody import EphemerisBody
from difforb.core.element import KepElement
from difforb.core.state.frame import Frame
from difforb.core.state.state import State
from difforb.dynamics.force_model import ForceModel
from difforb.od.dc.lsq import LSQTermination, RobustResult
from difforb.od.dc.lsq.core import compute_unweighted_rms, whiten_residuals
from difforb.report.display_units import orbit_element_specs, repr_fields_from_specs, STATE_REPR_SPECS
from difforb.report.text import build_repr, format_float_array

jax.config.update("jax_enable_x64", True)


@jax.jit
def compute_weighted_rms(
        residuals: Float[Array, "N_obs"],
        weights: Float[Array, "N_obs"],
        inlier_mask: Bool[Array, "N_obs"],
) -> Float[Array, ""]:
    """Compute a report-only RMS from marginal flattened weights."""
    used_weights = jnp.where(inlier_mask, weights, 0.0)
    return jnp.sqrt(jnp.sum(residuals * residuals * used_weights) / jnp.sum(used_weights))


class DCEstimate(NamedTuple):
    """Final parameter estimate from a differential-correction solve.

    Parameters
    ----------
    orbit : Orbit
        Estimated orbit. The first six covariance parameters use ``orbit.array.squeeze()`` in the native order of the concrete orbit representation.
    model_params : Float[Array, "N_model"]
        Estimated non-orbit model parameters appended after the six orbit parameters.
    model_param_names : list[str]
        Names corresponding to ``model_params``. This list is empty when the solve estimated only the six orbit parameters.
    cov_mat_post : Float[Array, "N_param N_param"]
        Posterior covariance matrix for ``[orbit.array.squeeze(), model_params]``, where ``N_param = 6 + N_model``.
    """

    orbit: Orbit
    model_params: Float[Array, "N_model"]
    model_param_names: list[str]
    cov_mat_post: Float[Array, "N_param N_param"]

    @property
    def uncertainties(self) -> Float[Array, "N_param"]:
        return jnp.sqrt(jnp.diagonal(self.cov_mat_post))


class OpticalResult(NamedTuple):
    residuals: Float[Array, "N 2"]
    normalized_residuals: Float[Array, "N 2"]
    weighted_rms: float
    unweighted_rms: float
    inlier_masks: Bool[Array, "N"]
    metrics: Float[Array, "N"]

    @property
    def n_obs(self) -> int:
        return len(self.inlier_masks)

    @property
    def n_inliers(self) -> int:
        return int(jnp.sum(self.inlier_masks))

    @property
    def n_outliers(self) -> int:
        return self.n_obs - self.n_inliers


class RadarResult(NamedTuple):
    residuals: Float[Array, "N"]
    normalized_residuals: Float[Array, "N"]
    inlier_masks: Bool[Array, "N"]
    metrics: Float[Array, "N"]

    delay_weighted_rms: float
    delay_unweighted_rms: float
    doppler_weighted_rms: float
    doppler_unweighted_rms: float

    @property
    def n_obs(self) -> int:
        return len(self.inlier_masks)

    @property
    def n_inliers(self) -> int:
        return int(jnp.sum(self.inlier_masks))

    @property
    def n_outliers(self) -> int:
        return self.n_obs - self.n_inliers


class LSQDiagnostics(NamedTuple):
    """
    Diagnostics from the final robust differential-correction solve.

    ``lsq_iterations`` counts accepted Levenberg-Marquardt steps accumulated
    across all inlier-mask solves in the robust loop. ``outlier_iterations``
    counts outer outlier-rejection iterations evaluated with usable residual,
    Jacobian, and covariance diagnostics from the inner solves.
    ``termination_reason`` describes why the final least-squares solve stopped
    at the returned estimate. ``optical_weight_matrices`` and
    ``radar_weights`` are the canonical weights used by the solver;
    ``flat_weights`` is only a marginal display view.
    """
    flat_jacobian: Float[Array, "N_flat N_param"]
    flat_weights: Float[Array, "N_flat"]
    optical_weight_matrices: Float[Array, "N_optical 2 2"]
    radar_weights: Float[Array, "N_radar"]
    cov_mat_prior: Float[Array, "N_param N_param"]
    cov_rank: Int[Array, ""]
    cov_condition: Float[Array, ""]
    cov_valid: Bool[Array, ""]
    converged: bool
    termination_reason: str
    lsq_iterations: int
    outlier_iterations: int


class DCResult(NamedTuple):
    """Differential-correction result with the final estimate and residual diagnostics.

    Parameters
    ----------
    estimate : DCEstimate
        Final orbit, estimated model parameters, and posterior covariance. Use ``estimate.cov_mat_post`` for the covariance matrix in the parameter order documented by :class:`DCEstimate`.
    optical : OpticalResult
        Residual diagnostics for optical observations.
    radar : RadarResult
        Residual diagnostics for radar observations.
    lsq_diagnostics : LSQDiagnostics
        Final least-squares diagnostics, including covariance validity, rank, condition, and termination state.
    normalized_residual_rms : float
        Root mean square of normalized residuals for the final solution. Dimensionless.
    """
    estimate: DCEstimate
    optical: OpticalResult
    radar: RadarResult
    lsq_diagnostics: LSQDiagnostics
    normalized_residual_rms: float

    def transform(self, target: Frame | Type[KepElement]) -> "DCResult":
        """Convert the solved orbit and covariance; pass ``KepElement`` from ``difforb.core`` for Keplerian-element covariance, returned in ``estimate.cov_mat_post``.

        Parameters
        ----------
        target : Frame or type[KepElement]
            Target state frame or the :class:`KepElement` class.

        Returns
        -------
        DCResult
            Result with ``estimate.orbit`` and ``estimate.cov_mat_post`` transformed to the target representation.

        Notes
        -----
        Model parameters remain appended after the six orbit parameters.
        """
        if target is KepElement and isinstance(self.estimate.orbit, KepElement):
            return self
        if isinstance(target, Frame) and isinstance(self.estimate.orbit, State) and self.estimate.orbit.frame == target:
            return self

        sun = EphemerisBody("sun")
        earth = EphemerisBody("earth")
        epoch_tdb = self.estimate.orbit.tdb

        def conversion_func(params: Float[Array, "6+N"]) -> tuple[
            Float[Array, "6+N"], Float[Array, "6+N"]]:
            orbit_params = params[:6]
            model_params = params[6:]
            if isinstance(self.estimate.orbit, State):
                cur_orbit = State.from_array(
                    self.estimate.orbit.tdb,
                    orbit_params,
                    self.estimate.orbit.frame,
                )
            elif isinstance(self.estimate.orbit, KepElement):
                cur_orbit = KepElement.from_array(
                    self.estimate.orbit.tdb,
                    orbit_params,
                )
            else:
                raise TypeError(
                    f"Unsupported orbit type: {type(self.estimate.orbit)}"
                )

            if target is KepElement:
                new_orbit = (
                    cur_orbit
                    if isinstance(cur_orbit, KepElement)
                    else KepElement.from_state(cur_orbit, sun=sun, earth=earth)
                )
            else:
                state = cur_orbit.state() if isinstance(cur_orbit, KepElement) else cur_orbit
                new_orbit = state.to(target, sun=sun, earth=earth)
            new_orbit_params = new_orbit.array.squeeze()
            new_params = jnp.concatenate((new_orbit_params, model_params))
            return new_params, new_params

        params = jnp.concatenate([self.estimate.orbit.array.squeeze(), self.estimate.model_params])
        j, new_params = jax.jacfwd(conversion_func, has_aux=True)(params)
        orbit_params = new_params[:6]
        if target is KepElement:
            new_orbit = KepElement.from_array(epoch_tdb, orbit_params)
        else:
            new_orbit = State.from_array(epoch_tdb, orbit_params, target)
        new_cov_mat = j @ self.estimate.cov_mat_post @ j.T

        new_estimate = DCEstimate(
            orbit=new_orbit,
            model_params=self.estimate.model_params,
            model_param_names=self.estimate.model_param_names,
            cov_mat_post=new_cov_mat
        )
        return DCResult(new_estimate, self.optical, self.radar, self.lsq_diagnostics,
                        self.normalized_residual_rms)

    @property
    def quality_code(self) -> Float[Array, ""]:
        """
        IAU MPC Uncertainty Parameter U.

        Range: 0 (good) to 9 (poor).

        Ref: https://www.minorplanetcenter.net/iau/info/UValue.html
        """
        kep_result = self.transform(KepElement)
        kep_orbit = kep_result.estimate.orbit
        kep_cov_mat = kep_result.estimate.cov_mat_post[:6, :6]

        def compute_derived_p_tp(
                params: Float[Array, "6"],
        ) -> Float[Array, "2"]:
            ele = KepElement.from_array(kep_orbit.tdb, params)
            p = ele.period.squeeze()
            tp_jd = ele.perit_jd.squeeze()
            return jnp.stack([p, tp_jd])

        jac = jax.jacfwd(compute_derived_p_tp)(kep_orbit.array.squeeze())
        derived_cov = jac @ kep_cov_mat @ jac.T
        sigma_p = jnp.sqrt(derived_cov[0, 0])
        sigma_tp = jnp.sqrt(derived_cov[1, 1])

        e = kep_orbit.e
        period = kep_orbit.period / 365.25
        ko = 180 / jnp.pi * 0.01720209895
        runoff = (sigma_tp * e + 10 / period * sigma_p) * ko / period * 3600 * 3
        cons = jnp.log(648000) / 9.
        u = jnp.floor(jnp.log(runoff) / cons) + 1
        u = u.squeeze().astype(int)
        return u

    def __repr__(self) -> str:
        orbit = self.estimate.orbit
        if isinstance(orbit, KepElement):
            orbit_type = orbit.__class__.__name__
        else:
            orbit_type = orbit.frame.name or f"{orbit.frame.origin.value}+{orbit.frame.axes.value}"
        orbit_fields = [("orbit_type", orbit_type)]
        if isinstance(orbit, KepElement):
            orbit_fields.extend(repr_fields_from_specs(orbit, orbit_element_specs(orbit)))
        else:
            orbit_fields.extend(repr_fields_from_specs(orbit, STATE_REPR_SPECS))

        try:
            quality_code = str(int(np.asarray(self.quality_code).item()))
        except Exception:
            quality_code = "N/A"

        m_params = "{" + ", ".join(f"{n}={format_float_array(v, precision=3)}" for n, v in zip(self.estimate.model_param_names,
                                                                                               self.estimate.model_params)) + "}" if self.estimate.model_param_names else None

        return build_repr(self.__class__.__name__, [
            ("epoch_jd", format_float_array(orbit.tdb.jd, precision=9, scientific=False, signed=False)),
            *orbit_fields,
            ("normalized_residual_rms", format_float_array(self.normalized_residual_rms)),
            ("u", quality_code),
            ("optical", f"{self.optical.n_inliers}/{self.optical.n_obs}"),
            ("radar", f"{self.radar.n_inliers}/{self.radar.n_obs}"),
            ("iters", f"{self.lsq_diagnostics.lsq_iterations}+{self.lsq_diagnostics.outlier_iterations}"),
            ("params", m_params),
        ])


def build_dc_result(
        robust_result: RobustResult,
        layout: ObservationLayout,
        initial_orbit: State,
        force_model: ForceModel,
        photocenter_correction: PhotocenterCorrection,
) -> DCResult:
    """Convert one array-only robust result to the public result model."""
    lsq_result = robust_result.lsq_result
    rejection = robust_result.rej_result
    optical_weights = jnp.asarray(lsq_result.optical_weight_matrices)
    radar_weights = jnp.asarray(lsq_result.radar_weights)
    flat_weights = layout.concat_to_flat_array(
        jnp.diagonal(optical_weights, axis1=1, axis2=2), radar_weights,
    )
    flat_optical_weights, flat_radar_weights = layout.split_flat_array(flat_weights)
    optical_residuals, radar_residuals = layout.split_flat_array_to_array(lsq_result.residuals)
    normalized = whiten_residuals(
        lsq_result.residuals,
        optical_weights,
        radar_weights,
        jnp.ones_like(lsq_result.residuals, dtype=bool),
    )
    optical_normalized, radar_normalized = layout.split_flat_array_to_array(normalized)

    inlier_masks = layout.flat_mask_to_mask(rejection.flat_inlier_mask)
    flat_optical_inliers, flat_radar_inliers = layout.split_flat_array(rejection.flat_inlier_mask)
    optical_inliers, radar_inliers = layout.split_array(inlier_masks)
    optical_metrics, radar_metrics = layout.split_array(rejection.metric)
    delay_inliers = flat_radar_inliers & layout.data.radar.is_delay
    doppler_inliers = flat_radar_inliers & layout.data.radar.is_doppler

    estimate = DCEstimate(
        orbit=State.from_array(initial_orbit.tdb, lsq_result.params[:6], initial_orbit.frame),
        model_params=lsq_result.params[6:],
        model_param_names=(
            force_model.get_all_estimated_param_names()
            + photocenter_correction.get_estimated_param_names()
        ),
        cov_mat_post=lsq_result.cov_mat_post,
    )
    optical = OpticalResult(
        residuals=optical_residuals,
        normalized_residuals=optical_normalized,
        inlier_masks=optical_inliers,
        metrics=optical_metrics,
        weighted_rms=float(compute_weighted_rms(
            optical_residuals.ravel(), flat_optical_weights, flat_optical_inliers,
        )),
        unweighted_rms=float(compute_unweighted_rms(
            optical_residuals.ravel(), flat_optical_inliers,
        )),
    )
    radar = RadarResult(
        residuals=radar_residuals,
        normalized_residuals=radar_normalized,
        inlier_masks=radar_inliers,
        metrics=radar_metrics,
        delay_weighted_rms=float(compute_weighted_rms(
            radar_residuals, flat_radar_weights, delay_inliers,
        )),
        delay_unweighted_rms=float(compute_unweighted_rms(radar_residuals, delay_inliers)),
        doppler_weighted_rms=float(compute_weighted_rms(
            radar_residuals, flat_radar_weights, doppler_inliers,
        )),
        doppler_unweighted_rms=float(compute_unweighted_rms(radar_residuals, doppler_inliers)),
    )
    diagnostics = LSQDiagnostics(
        flat_jacobian=lsq_result.jacobian,
        flat_weights=flat_weights,
        optical_weight_matrices=optical_weights,
        radar_weights=radar_weights,
        cov_mat_prior=lsq_result.cov_mat_prior,
        cov_rank=lsq_result.cov_rank,
        cov_condition=lsq_result.cov_condition,
        cov_valid=lsq_result.cov_valid,
        converged=bool(lsq_result.converged),
        termination_reason=LSQTermination(int(lsq_result.termination_code)).name,
        lsq_iterations=int(robust_result.lsq_iter_num),
        outlier_iterations=int(robust_result.outlier_iter_num),
    )
    return DCResult(
        estimate, optical, radar, diagnostics,
        float(lsq_result.normalized_residual_rms),
    )
