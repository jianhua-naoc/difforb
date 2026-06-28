import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from difforb.astrometry.weight import WeightResult

from difforb.od.lsq import (
    LeastSquares,
    RobustLeastSquares,
    compute_normalized_residual_rms,
    compute_prior_covariance,
    compute_unweighted_rms,
    compute_weighted_lsq_loss,
    solve_normal_equation,
)
from difforb.od.outlier.chi2 import Chi2OutlierRejecter
from difforb.od.outlier.policy import CompiledOutlierPolicy
from tests.assertions import assert_allclose, assert_array_equal


def scalar_weight_result(weights):
    weights_np = np.asarray(weights, dtype=float)
    return WeightResult(
        optical_uncertainties=np.empty((0, 2), dtype=float),
        radar_uncertainties=1.0 / np.sqrt(weights_np),
        optical_sources=np.asarray([], dtype=object),
        radar_sources=np.asarray(["TEST"] * len(weights_np), dtype=object),
        optical_correlations=np.asarray([], dtype=float),
        optical_time_uncertainties=np.asarray([], dtype=float),
    )


def scalar_weight_arrays(weights):
    weights_array = jnp.asarray(weights)
    return jnp.zeros((0, 2, 2), dtype=weights_array.dtype), weights_array


def test_solve_normal_equation_matches_weighted_closed_form():
    jacobian = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ]
    )
    rhs = jnp.asarray([1.0, -2.0, 0.5, 3.0])
    weights = jnp.asarray([1.0, 4.0, 0.25, 2.0])
    inlier_mask = jnp.asarray([True, True, False, True])
    param_scale = jnp.ones(2)
    damping_diag = jnp.sum(jacobian * jacobian * weights[:, None], axis=0)
    optical_weight_matrices, radar_weights = scalar_weight_arrays(weights)

    actual = solve_normal_equation(
        jacobian,
        rhs,
        optical_weight_matrices,
        radar_weights,
        inlier_mask,
        damping=0.0,
        damping_diag=damping_diag,
        param_scale=param_scale,
    )

    sqrt_weights = jnp.sqrt(jnp.where(inlier_mask, weights, 0.0))
    expected, *_ = jnp.linalg.lstsq(jacobian * sqrt_weights[:, None], rhs * sqrt_weights, rcond=1.0e-15)

    print(
        "[od.lsq.normal_equation] "
        f"param_max_abs_diff={float(jnp.max(jnp.abs(actual - expected))):.12e}"
    )

    assert_allclose(actual, expected, atol=1.0e-13, rtol=0.0)


def test_least_squares_solves_linear_model_against_closed_form():
    design = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [-1.0, 2.0],
        ]
    )
    observed = jnp.asarray([1.2, -0.4, 0.9, 3.1, -2.3])
    weights = jnp.asarray([1.0, 2.0, 1.5, 0.75, 3.0])
    inlier_mask = jnp.ones(observed.shape, dtype=bool)
    init_params = jnp.asarray([-2.0, 2.5])

    def residuals(params):
        return design @ params - observed

    def jacobian_with_residuals(params):
        return design, residuals(params)

    result = LeastSquares(tol=1.0e-13, max_iter=50).solve(
        init_params,
        scalar_weight_result(weights),
        inlier_mask,
        residuals,
        jacobian_with_residuals,
    )

    sqrt_weights = jnp.sqrt(weights)
    expected, *_ = jnp.linalg.lstsq(design * sqrt_weights[:, None], observed * sqrt_weights, rcond=1.0e-15)

    print(
        "[od.lsq.linear] "
        f"iter={result.iter_num} "
        f"reason={result.termination_reason} "
        f"param_max_abs_diff={float(jnp.max(jnp.abs(result.params - expected))):.12e} "
        f"normalized_rms={float(result.normalized_residual_rms):.12e}"
    )

    assert result.converged
    assert result.cov_valid
    assert result.cov_rank == 2
    assert_allclose(result.params, expected, atol=1.0e-9, rtol=0.0)
    assert_allclose(result.residuals, design @ expected - observed, atol=1.0e-9, rtol=0.0)


def test_least_squares_refreshes_dynamic_weights_at_final_parameters():
    init_params = jnp.asarray([0.0])
    inlier_mask = jnp.asarray([True])

    def residuals(params):
        return jnp.asarray([params[0] - 1.0])

    def jacobian_with_residuals(params):
        return jnp.asarray([[1.0]]), residuals(params)

    def weight_array_func(params):
        return jnp.empty((0, 2, 2), dtype=params.dtype), jnp.asarray([1.0 + params[0] * params[0]])

    result = LeastSquares(tol=1.0e-13, max_iter=20).solve(
        init_params,
        scalar_weight_result(jnp.asarray([1.0])),
        inlier_mask,
        residuals,
        jacobian_with_residuals,
        weight_array_func=weight_array_func,
    )

    assert result.converged
    assert_allclose(result.params, jnp.asarray([1.0]), atol=1.0e-9, rtol=0.0)
    assert_allclose(result.radar_weights, jnp.asarray([2.0]), atol=1.0e-9, rtol=0.0)


def test_least_squares_uses_optical_correlation_blocks():
    design = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    observed = jnp.asarray([1.0, -2.0, 0.5])
    covariance = jnp.asarray([[4.0, 1.2], [1.2, 9.0]])
    optical_weight_matrices = jnp.linalg.inv(covariance)[None, :, :]
    radar_weight = jnp.asarray([0.25])
    weights = WeightResult(
        optical_uncertainties=np.asarray([[2.0, 3.0]], dtype=float),
        radar_uncertainties=np.asarray([2.0], dtype=float),
        optical_sources=np.asarray(["TEST"], dtype=object),
        radar_sources=np.asarray(["TEST"], dtype=object),
        optical_correlations=np.asarray([0.2], dtype=float),
        optical_time_uncertainties=np.asarray([np.nan], dtype=float),
    )
    inlier_mask = jnp.ones(3, dtype=bool)
    init_params = jnp.asarray([0.0, 0.0])

    def residuals(params):
        return design @ params - observed

    def jacobian_with_residuals(params):
        return design, residuals(params)

    result = LeastSquares(tol=1.0e-13, max_iter=50).solve(
        init_params,
        weights,
        inlier_mask,
        residuals,
        jacobian_with_residuals,
    )

    full_weight_matrix = jnp.asarray(
        [
            [optical_weight_matrices[0, 0, 0], optical_weight_matrices[0, 0, 1], 0.0],
            [optical_weight_matrices[0, 1, 0], optical_weight_matrices[0, 1, 1], 0.0],
            [0.0, 0.0, radar_weight[0]],
        ]
    )
    expected = jnp.linalg.solve(design.T @ full_weight_matrix @ design, design.T @ full_weight_matrix @ observed)

    assert result.converged
    assert_allclose(result.params, expected, atol=1.0e-9, rtol=0.0)


def test_least_squares_ignores_masked_outlier_rows():
    design = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ]
    )
    observed = jnp.asarray([2.0, -1.0, 1.0, 1.0e6])
    weights = jnp.ones(observed.shape)
    inlier_mask = jnp.asarray([True, True, True, False])
    init_params = jnp.asarray([0.0, 0.0])

    def residuals(params):
        return design @ params - observed

    def jacobian_with_residuals(params):
        return design, residuals(params)

    result = LeastSquares(tol=1.0e-13, max_iter=50).solve(
        init_params,
        scalar_weight_result(weights),
        inlier_mask,
        residuals,
        jacobian_with_residuals,
    )

    sqrt_weights = jnp.sqrt(jnp.where(inlier_mask, weights, 0.0))
    expected, *_ = jnp.linalg.lstsq(design * sqrt_weights[:, None], observed * sqrt_weights, rcond=1.0e-15)

    print(
        "[od.lsq.inlier_mask] "
        f"param_max_abs_diff={float(jnp.max(jnp.abs(result.params - expected))):.12e} "
        f"masked_residual={float(result.residuals[-1]):+.12e}"
    )

    assert result.converged
    assert_allclose(result.params, expected, atol=1.0e-9, rtol=0.0)
    assert abs(float(result.residuals[-1])) > 1.0e5


def test_lsq_metrics_ignore_outliers():
    residuals = jnp.asarray([1.0, -2.0, 100.0, 4.0])
    weights = jnp.asarray([1.0, 4.0, 9.0, 0.25])
    inlier_mask = jnp.asarray([True, True, False, True])
    optical_weight_matrices, radar_weights = scalar_weight_arrays(weights)

    expected_loss = 0.5 * (1.0 * 1.0**2 + 4.0 * (-2.0) ** 2 + 0.25 * 4.0**2)
    expected_unweighted_rms = jnp.sqrt((1.0**2 + (-2.0) ** 2 + 4.0**2) / 3.0)
    expected_normalized_rms = jnp.sqrt((1.0 * 1.0**2 + 4.0 * (-2.0) ** 2 + 0.25 * 4.0**2) / 3.0)

    assert_allclose(compute_weighted_lsq_loss(residuals, optical_weight_matrices, radar_weights, inlier_mask),
                    expected_loss, atol=1.0e-15, rtol=0.0)
    assert_allclose(compute_unweighted_rms(residuals, inlier_mask), expected_unweighted_rms, atol=1.0e-15, rtol=0.0)
    assert_allclose(compute_normalized_residual_rms(residuals, optical_weight_matrices, radar_weights, inlier_mask),
                    expected_normalized_rms, atol=1.0e-15, rtol=0.0)


def test_robust_lsq_events_ignore_structural_padding():
    design = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [100.0, 100.0],
        ]
    )
    observed = jnp.asarray([2.0, -1.0, 1.0, 1.0e6])
    weights = jnp.ones(observed.shape)
    valid_mask = jnp.asarray([True, True, True, False])
    init_params = jnp.asarray([0.0, 0.0])
    events = []

    def residuals(params):
        return design @ params - observed

    def jacobian_with_residuals(params):
        return design, residuals(params)

    policy = CompiledOutlierPolicy(
        auto_rejecter=None,
        enable_auto_rejection=False,
        max_iters=1,
        n_2d=0,
        n_1d=4,
        flat_manual_outlier_mask=jnp.zeros(4, dtype=bool),
        flat_manual_inlier_mask=jnp.zeros(4, dtype=bool),
        flat_valid_mask=valid_mask,
        observation_valid_mask=valid_mask,
    )

    RobustLeastSquares(LeastSquares(tol=1.0e-13, max_iter=50)).solve(
        init_params,
        scalar_weight_result(weights),
        policy,
        residuals,
        jacobian_with_residuals,
        event_handler=events.append,
        log_detail="iter",
    )

    outlier_events = {event.event: event for event in events if event.event in {"outlier_iteration_start", "outlier_disabled"}}

    assert outlier_events["outlier_iteration_start"].data["observation_count"] == 3
    assert outlier_events["outlier_iteration_start"].data["inlier_count"] == 3
    assert outlier_events["outlier_iteration_start"].data["outlier_count"] == 0
    assert outlier_events["outlier_disabled"].data["observation_count"] == 3


def test_robust_lsq_refits_final_mask_when_max_outlier_iterations_reached():
    design = jnp.ones((4, 1))
    observed = jnp.asarray([0.0, 0.0, 0.0, 100.0])
    weights = jnp.ones(observed.shape)
    init_params = jnp.asarray([10.0])

    def residuals(params):
        return design @ params - observed

    def jacobian_with_residuals(params):
        return design, residuals(params)

    policy = CompiledOutlierPolicy(
        auto_rejecter=Chi2OutlierRejecter().with_observation_structure(n_2d=0, n_1d=4),
        enable_auto_rejection=True,
        max_iters=1,
        n_2d=0,
        n_1d=4,
        flat_manual_outlier_mask=jnp.zeros(4, dtype=bool),
        flat_manual_inlier_mask=jnp.zeros(4, dtype=bool),
        flat_valid_mask=jnp.ones(4, dtype=bool),
        observation_valid_mask=jnp.ones(4, dtype=bool),
    )

    result = RobustLeastSquares(LeastSquares(tol=1.0e-13, max_iter=50)).solve(
        init_params,
        scalar_weight_result(weights),
        policy,
        residuals,
        jacobian_with_residuals,
    )

    assert result.outlier_iter_num == 1
    assert_array_equal(result.rej_result.flat_inlier_mask, jnp.asarray([True, True, True, False]))
    assert_allclose(result.lsq_result.params, jnp.asarray([0.0]), atol=1.0e-9, rtol=0.0)
    assert_allclose(result.lsq_result.normalized_residual_rms, 0.0, atol=1.0e-9, rtol=0.0)


def test_prior_covariance_reports_rank_deficiency():
    jacobian = jnp.asarray(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [-1.0, -2.0],
        ]
    )
    weights = jnp.ones(3)
    inlier_mask = jnp.ones(3, dtype=bool)
    optical_weight_matrices, radar_weights = scalar_weight_arrays(weights)

    result = compute_prior_covariance(jacobian, optical_weight_matrices, radar_weights, inlier_mask)

    assert result.rank == 1
    assert not bool(result.valid)
    assert jnp.isinf(result.condition)
    assert jnp.all(jnp.isfinite(result.cov_mat))
