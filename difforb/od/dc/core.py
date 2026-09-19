"""Numerical kernels shared by scalar and batched differential correction."""

import equinox as eqx
from jax import Array
from jaxtyping import Float

from difforb.dynamics.force_model import ForceModel
from difforb.integrator.integrator import NumericalIntegrator
from difforb.od.dc.lsq.core import (
    LinearizationFunction,
    LMOptions,
    RobustResult,
    build_time_inflated_optical_weight_matrices,
    solve_robust,
)
from difforb.od.dc.prediction import AstrometryMeasurementModel
from difforb.od.outlier.policy import CompiledOutlierPolicy


def linearize_observations(
        params: Float[Array, "N_param"],
        *,
        measure_model: AstrometryMeasurementModel,
        force_model: ForceModel,
        integrator: NumericalIntegrator,
        base_optical_covariances: Float[Array, "N_optical 2 2"],
        optical_time_uncertainties: Float[Array, "N_optical"],
        radar_weights: Float[Array, "N_radar"],
) -> tuple[
    Float[Array, "N_flat_obs N_param"],
    Float[Array, "N_flat_obs"],
    Float[Array, "N_optical 2 2"],
    Float[Array, "N_radar"],
]:
    """Linearize one strategy and refresh its optical weights."""
    jacobian, residuals, optical_rates = (
        measure_model.compute_jacobian_with_residuals_and_optical_rates(
            params, force_model, integrator,
        )
    )
    optical_weight_matrices = build_time_inflated_optical_weight_matrices(
        base_optical_covariances,
        optical_time_uncertainties,
        optical_rates,
    )
    return jacobian, residuals, optical_weight_matrices, radar_weights


def bind_linearization(
        measure_model: AstrometryMeasurementModel,
        force_model: ForceModel,
        integrator: NumericalIntegrator,
        base_optical_covariances: Float[Array, "N_optical 2 2"],
        optical_time_uncertainties: Float[Array, "N_optical"],
        radar_weights: Float[Array, "N_radar"],
) -> LinearizationFunction:
    """Bind one strategy's measurement linearization."""
    return eqx.Partial(
        linearize_observations,
        measure_model=measure_model,
        force_model=force_model,
        integrator=integrator,
        base_optical_covariances=base_optical_covariances,
        optical_time_uncertainties=optical_time_uncertainties,
        radar_weights=radar_weights,
    )


def solve_differential_correction_single(
        base_optical_covariances: Float[Array, "N_optical 2 2"],
        optical_time_uncertainties: Float[Array, "N_optical"],
        radar_weights: Float[Array, "N_radar"],
        outlier_policy: CompiledOutlierPolicy,
        init_params: Float[Array, "N_param"],
        force_model: ForceModel,
        *,
        measure_model: AstrometryMeasurementModel,
        integrator: NumericalIntegrator,
        solver_options: LMOptions,
) -> RobustResult:
    """Fit one prepared strategy with compiled robust control flow."""
    linearize = bind_linearization(
        measure_model,
        force_model,
        integrator,
        base_optical_covariances,
        optical_time_uncertainties,
        radar_weights,
    )
    return solve_robust(
        init_params, outlier_policy, linearize,
        solver_options,
    )


@eqx.filter_jit
def solve_differential_correction_batch(
        base_optical_covariances: Float[Array, "B N_optical 2 2"],
        optical_time_uncertainties: Float[Array, "B N_optical"],
        radar_weights: Float[Array, "B N_radar"],
        outlier_policies: CompiledOutlierPolicy,
        init_params: Float[Array, "B N_param"],
        force_models: ForceModel,
        force_model_axes: object,
        measure_model: AstrometryMeasurementModel,
        integrator: NumericalIntegrator,
        solver_options: LMOptions,
) -> RobustResult:
    """Vectorize the single-strategy kernel over one compatible batch."""
    fit_one = eqx.Partial(
        solve_differential_correction_single,
        measure_model=measure_model,
        integrator=integrator,
        solver_options=solver_options,
    )
    axes = (0, 0, 0, 0, 0, force_model_axes)
    return eqx.filter_vmap(fit_one, in_axes=axes)(
        base_optical_covariances,
        optical_time_uncertainties,
        radar_weights,
        outlier_policies,
        init_params,
        force_models,
    )
