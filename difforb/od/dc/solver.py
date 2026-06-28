"""Differential-correction solver entry points."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from difforb.astrometry.data import ObservationData, ObservationLayout
from difforb.astrometry.debias import DebiasPolicy
from difforb.astrometry.reduction.photocenter import PhotocenterCorrection
from difforb.astrometry.weight import WeightPolicy
from difforb.body.ephbody import EphemerisBody
from difforb.body.smallbody import Orbit
from difforb.core.element import KepElement
from difforb.core.state.frame import BCRS
from difforb.core.state.state import State
from difforb.dynamics.force_model import ForceModel
from difforb.integrator.integrator import NumericalIntegrator
from difforb.od.dc.bucket import DCBucketPolicy, crop_dc_result, pack_dc_observations
from difforb.od.dc.prediction import AstrometryMeasurementModel
from difforb.od.events import SolverEventHandler, SolverEventLogger, SolverLogDetail
from difforb.od.lsq import (
    LeastSquares,
    RobustLeastSquares,
    RobustResult,
    build_time_inflated_optical_weight_matrices,
    compute_unweighted_rms,
    whiten_residuals,
)
from difforb.report.text import build_repr, format_float_array

from difforb.od.dc.result import DCResult, DCEstimate, OpticalResult, RadarResult, LSQDiagnostics
from difforb.od.outlier.policy import InteractiveOutlierPolicy

jax.config.update("jax_enable_x64", True)


@jax.jit
def _compute_weighted_rms(residuals: Float[Array, "N"], weights: Float[Array, "N"],
                          inlier_mask: Bool[Array, "N"]) -> Float[Array, ""]:
    """Compute a report-only RMS from marginal flattened weights."""
    used_weights = jnp.where(inlier_mask, weights, 0.)
    return jnp.sqrt(jnp.sum(residuals * residuals * used_weights) / jnp.sum(used_weights))


def _canonicalize_initial_orbit(initial_orbit: Orbit, sun: EphemerisBody, earth: EphemerisBody) -> State:
    if isinstance(initial_orbit, KepElement):
        state = initial_orbit.state()
    elif isinstance(initial_orbit, State):
        state = initial_orbit
    else:
        raise TypeError(f"Unsupported orbit type: {type(initial_orbit)}.")
    if state.frame != BCRS:
        state = state.to(BCRS, sun=sun, earth=earth)
    return state


class DCSolver:
    """
    Differential-correction solver built on Levenberg-Marquardt least squares.

    The solver always runs through :class:`RobustLeastSquares` so that chi-square
    diagnostics, inlier masks, and rejection statistics are produced
    consistently. Disabling outlier rejection only skips the rejection step; it
    does not disable robust diagnostics. When optical time uncertainties are
    available, the solver refreshes the time-inflated optical weight matrices at
    each least-squares linearization point from the modeled sky-plane rates.
    """

    def __init__(self,
                 lsq_tol: float = 1e-11,
                 lsq_max_iters: int = 20,
                 *,
                 sun: EphemerisBody | None = None,
                 earth: EphemerisBody | None = None,
                 bucket_policy: DCBucketPolicy | None = None):
        """
        Create a differential-correction solver.

        Parameters
        ----------
        lsq_tol : float, default=1e-11
            Base convergence threshold for the inner least-squares solve. This
            value controls the relative scaled step norm and the relative loss
            reduction; the scaled gradient threshold defaults to its square root.
        lsq_max_iters : int, default=20
            Maximum number of iterations allowed in each least-squares solve.
        sun : EphemerisBody or None, optional
            Ephemeris-backed Sun body used by the light-time and residual
            models. If omitted, the solver resolves ``EphemerisBody("sun")``
            during construction.
        earth : EphemerisBody or None, optional
            Ephemeris-backed Earth body used by the site and light-time models.
            If omitted, the solver resolves ``EphemerisBody("earth")`` during
            construction.
        bucket_policy : DCBucketPolicy or None, optional
            Optional shape-bucket policy for padding non-empty observation
            tables before residual and Jacobian evaluation. Padded residuals
            are structurally masked and are cropped from the returned result.

        Raises
        ------
        ValueError
            Raised when a numeric option falls outside its valid range.
        """
        self.lsq_tol = lsq_tol
        self.lsq_max_iter = lsq_max_iters
        self.bucket_policy = bucket_policy

        self._sun = sun if sun is not None else EphemerisBody('sun')
        self._earth = earth if earth is not None else EphemerisBody('earth')

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("lsq_tol", format_float_array(self.lsq_tol)),
                ("lsq_max_iter", str(self.lsq_max_iter)),
                ("bucket_policy", self.bucket_policy.__class__.__name__ if self.bucket_policy is not None else None),
            ],
        )

    def _build_result(self, robust_result: RobustResult, layout: ObservationLayout, initial_orbit: State,
                      force_model: ForceModel, photocenter_correction: PhotocenterCorrection) -> DCResult:
        lsq_result = robust_result.lsq_result
        rej_result = robust_result.rej_result

        params = lsq_result.params
        model_param_names = force_model.get_all_estimated_param_names() + photocenter_correction.get_estimated_param_names()
        estimate = DCEstimate(
            orbit=State.from_array(initial_orbit.tdb, params[:6], BCRS),
            model_params=params[6:],
            model_param_names=model_param_names,
            cov_mat_post=lsq_result.cov_mat_post,
        )

        optical_weight_matrices = jnp.asarray(lsq_result.optical_weight_matrices)
        radar_weights = jnp.asarray(lsq_result.radar_weights)
        flat_weights = layout.concat_to_flat_array(
            jnp.diagonal(optical_weight_matrices, axis1=1, axis2=2),
            radar_weights,
        )
        flat_optical_weights, flat_radar_weights = layout.split_flat_array(flat_weights)

        optical_residuals, radar_residuals = layout.split_flat_array_to_array(lsq_result.residuals)
        normalized_flat_residuals = whiten_residuals(
            lsq_result.residuals,
            optical_weight_matrices,
            radar_weights,
            jnp.ones_like(lsq_result.residuals, dtype=bool),
        )
        optical_normalized_residuals, radar_normalized_residuals = layout.split_flat_array_to_array(normalized_flat_residuals)

        inlier_masks = layout.flat_mask_to_mask(rej_result.flat_inlier_mask)
        flat_optical_inliers, flat_radar_inliers = layout.split_flat_array(rej_result.flat_inlier_mask)
        optical_inliers, radar_inliers = layout.split_array(inlier_masks)
        optical_metrics, radar_metrics = layout.split_array(rej_result.metric)

        optical_weighted_rms = float(
            _compute_weighted_rms(optical_residuals.ravel(), flat_optical_weights, flat_optical_inliers))
        is_delay_mask = layout.data.radar.is_delay
        is_doppler_mask = layout.data.radar.is_doppler
        radar_delay_weighted_rms = float(
            _compute_weighted_rms(radar_residuals, flat_radar_weights, flat_radar_inliers & is_delay_mask
                                  ))
        radar_doppler_weighted_rms = float(
            _compute_weighted_rms(radar_residuals, flat_radar_weights, flat_radar_inliers & is_doppler_mask))

        optical_unweighted_rms = float(compute_unweighted_rms(optical_residuals.ravel(), flat_optical_inliers))
        radar_delay_unweighted_rms = float(compute_unweighted_rms(radar_residuals, flat_radar_inliers & is_delay_mask))
        radar_doppler_unweighted_rms = float(compute_unweighted_rms(radar_residuals, flat_radar_inliers & is_doppler_mask))

        optical_result = OpticalResult(residuals=optical_residuals,
                                       normalized_residuals=optical_normalized_residuals,
                                       inlier_masks=optical_inliers,
                                       metrics=optical_metrics,
                                       weighted_rms=optical_weighted_rms,
                                       unweighted_rms=optical_unweighted_rms)
        radar_result = RadarResult(residuals=radar_residuals, normalized_residuals=radar_normalized_residuals,
                                   inlier_masks=radar_inliers, metrics=radar_metrics, delay_weighted_rms=radar_delay_weighted_rms,
                                   delay_unweighted_rms=radar_delay_unweighted_rms,
                                   doppler_weighted_rms=radar_doppler_weighted_rms,
                                   doppler_unweighted_rms=radar_doppler_unweighted_rms)

        lsq_diagnostics = LSQDiagnostics(flat_jacobian=lsq_result.jacobian,
                                         flat_weights=flat_weights,
                                         optical_weight_matrices=optical_weight_matrices,
                                         radar_weights=radar_weights,
                                         cov_mat_prior=lsq_result.cov_mat_prior,
                                         cov_rank=lsq_result.cov_rank,
                                         cov_condition=lsq_result.cov_condition,
                                         cov_valid=lsq_result.cov_valid,
                                         converged=lsq_result.converged,
                                         termination_reason=lsq_result.termination_reason,
                                         lsq_iterations=robust_result.lsq_iter_num,
                                         outlier_iterations=robust_result.outlier_iter_num)

        return DCResult(estimate=estimate, optical=optical_result, radar=radar_result,
                        lsq_diagnostics=lsq_diagnostics, normalized_residual_rms=float(lsq_result.normalized_residual_rms))

    def solve(self, data: ObservationData, initial_orbit: Orbit, force_model: ForceModel,
              integrator: NumericalIntegrator, weight_policy: WeightPolicy, debias_policy: DebiasPolicy,
              outlier_policy: InteractiveOutlierPolicy, *,
              photocenter_correction: PhotocenterCorrection | None = None,
              event_handler: SolverEventHandler | None = None,
              log_detail: SolverLogDetail = "iter",
              event_logger: SolverEventLogger | None = None) -> DCResult:
        """
        Run differential correction for a specific orbit-estimation problem.

        Parameters
        ----------
        data : ObservationData
            Observations used for the fit.
        initial_orbit : Orbit
            Initial orbital state used as the least-squares starting point.
        force_model : ForceModel
            Dynamical model when propagating the orbit, whose estimable parameters
            are solved jointly with the orbit, if applicable.
        integrator : NumericalIntegrator
            Integrator used to propagate the orbit.
        weight_policy : WeightPolicy
            Policy for per-observation sigmas and inverse-variance weights.
        debias_policy : DebiasPolicy
            Policy for optical astrometric debias corrections.
        outlier_policy : InteractiveOutlierPolicy
            Policy for initial, manual, and statistical inlier masks.
        photocenter_correction : PhotocenterCorrection or None, optional
            Optional optical center-of-light correction. Estimated photocenter
            parameters are appended after the dynamical model parameters in the
            least-squares vector.
        event_handler : SolverEventHandler or None, optional
            Optional callback that receives least-squares and outlier-rejection
            progress events.
        log_detail : {"quiet", "summary", "iter", "trial"}, default="iter"
            Minimum solver-log detail emitted to ``event_handler``.
        event_logger : SolverEventLogger or None, optional
            Context-aware structured event logger shared across staged
            differential correction.
        Returns
        -------
        DCResult
            Final differential-correction result together with residual,
            chi-square, inlier-mask, and iteration diagnostics.
        """
        packed_data = None
        solve_data = data
        if self.bucket_policy is not None:
            packed_data = pack_dc_observations(data, self.bucket_policy)
            solve_data = packed_data.data

        layout = ObservationLayout(solve_data)

        weight_results = weight_policy.weights(solve_data)
        debias_result = debias_policy.bias(solve_data)
        compiled_outlier_policy = outlier_policy.compiled(
            layout,
            flat_valid_mask=packed_data.flat_valid_mask if packed_data is not None else None,
        )

        init_state = _canonicalize_initial_orbit(initial_orbit, self._sun, self._earth)
        if photocenter_correction is None:
            photocenter_correction = PhotocenterCorrection()
        measure_model = AstrometryMeasurementModel.build(
            solve_data,
            init_state.tdb,
            self._sun,
            self._earth,
            debias_result,
            photocenter_correction,
        )
        res_func = partial(measure_model.compute_residuals, force_model=force_model, integrator=integrator)
        res_jac_func = partial(measure_model.compute_jacobian_with_residuals, force_model=force_model,
                               integrator=integrator)

        init_state_params = init_state.array.squeeze()
        init_model_params = force_model.get_all_estimated_params()
        init_photocenter_params = photocenter_correction.get_estimated_params()
        init_params = jnp.concatenate([init_state_params, init_model_params, init_photocenter_params])
        weight_array_func = None
        has_optical_time_uncertainty = np.any(
            np.isfinite(weight_results.optical_time_uncertainties)
            & (weight_results.optical_time_uncertainties != 0.0)
        )
        if has_optical_time_uncertainty:
            base_optical_covariances = jnp.asarray(weight_results.optical_covariances)
            optical_time_uncertainties = jnp.asarray(weight_results.optical_time_uncertainties)
            radar_weights = jnp.asarray(weight_results.radar_weights)

            def weight_array_func(params):
                optical_rates = measure_model.compute_optical_rates(params, force_model, integrator)
                optical_weight_matrices = build_time_inflated_optical_weight_matrices(
                    base_optical_covariances,
                    optical_time_uncertainties,
                    optical_rates,
                )
                return optical_weight_matrices, radar_weights

        model_param_scale = jnp.asarray(force_model.get_all_estimated_param_scales(), dtype=init_state_params.dtype)
        photocenter_param_scale = jnp.asarray(photocenter_correction.get_estimated_param_scales(), dtype=init_state_params.dtype)
        param_scale = jnp.concatenate([jnp.ones_like(init_state_params), model_param_scale, photocenter_param_scale])

        lsq_solver = LeastSquares(tol=self.lsq_tol, max_iter=self.lsq_max_iter)
        robust_solver = RobustLeastSquares(lsq_solver)
        robust_lsq_result = robust_solver.solve(
            init_params,
            weight_results,
            compiled_outlier_policy,
            res_func,
            res_jac_func,
            param_scale=param_scale,
            weight_array_func=weight_array_func,
            event_handler=event_handler,
            log_detail=log_detail,
            event_logger=event_logger,
        )
        result = self._build_result(robust_lsq_result, layout, initial_orbit, force_model, photocenter_correction)
        if packed_data is not None:
            result = crop_dc_result(result, packed_data)
        return result
