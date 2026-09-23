import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from difforb.astrometry.weight import WeightResult

from difforb.od.dc.lsq import LMOptions, LeastSquares, RobustLeastSquares
from difforb.od.dc.lsq.core import (
    compute_normalized_residual_rms,
    compute_prior_covariance,
    compute_unweighted_rms,
    evaluate_lm_trial,
    initialize_lsq,
    linearize_at,
)
from difforb.od.outlier.chi2 import Chi2OutlierRejecter
from difforb.od.outlier.policy import CompiledOutlierPolicy
from tests.assertions import assert_allclose, assert_array_equal


def scalar_weight_arrays(weights):
    weights_array = jnp.asarray(weights)
    return jnp.zeros((0, 2, 2), dtype=weights_array.dtype), weights_array


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

    optical_weights, radar_weights = scalar_weight_arrays(weights)

    def linearize(params):
        return design, residuals(params), optical_weights, radar_weights

    result = LeastSquares(max_iter=50).solve(
        init_params,
        inlier_mask,
        linearize,
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

    def linearize(params):
        return (
            jnp.asarray([[1.0]]),
            residuals(params),
            jnp.empty((0, 2, 2), dtype=params.dtype),
            jnp.asarray([1.0 + params[0] * params[0]]),
        )

    result = LeastSquares(max_iter=20).solve(
        init_params,
        inlier_mask,
        linearize,
    )

    assert result.converged
    assert_allclose(result.params, jnp.asarray([1.0]), atol=1.0e-9, rtol=0.0)
    assert_allclose(result.radar_weights, jnp.asarray([2.0]), atol=1.0e-9, rtol=0.0)
    assert_allclose(result.radar_weights, 1.0 + result.params**2, atol=0.0, rtol=0.0)


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

    fixed_optical_weights = jnp.asarray(weights.optical_weight_matrices)
    fixed_radar_weights = jnp.asarray(weights.radar_weights)

    def linearize(params):
        return design, residuals(params), fixed_optical_weights, fixed_radar_weights

    result = LeastSquares(max_iter=50).solve(
        init_params,
        inlier_mask,
        linearize,
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

    optical_weights, radar_weights = scalar_weight_arrays(weights)

    def linearize(params):
        return design, residuals(params), optical_weights, radar_weights

    result = LeastSquares(max_iter=50).solve(
        init_params,
        inlier_mask,
        linearize,
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

    expected_unweighted_rms = jnp.sqrt((1.0**2 + (-2.0) ** 2 + 4.0**2) / 3.0)
    expected_normalized_rms = jnp.sqrt((1.0 * 1.0**2 + 4.0 * (-2.0) ** 2 + 0.25 * 4.0**2) / 3.0)

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

    optical_weights, radar_weights = scalar_weight_arrays(weights)

    def linearize(params):
        return design, residuals(params), optical_weights, radar_weights

    policy = CompiledOutlierPolicy(
        auto_rejecter=None,
        enable_auto_rejection=True,
        max_iters=1,
        n_2d=0,
        n_1d=4,
        flat_manual_outlier_mask=jnp.zeros(4, dtype=bool),
        flat_manual_inlier_mask=jnp.zeros(4, dtype=bool),
        flat_valid_mask=valid_mask,
        observation_valid_mask=valid_mask,
    )

    RobustLeastSquares(LeastSquares(max_iter=50)).solve(
        init_params,
        policy,
        linearize,
        verbose=lambda event, **data: events.append((event, data)),
    )

    assert [event for event, _ in events if event == "outlier_iteration"] == [
        "outlier_iteration",
    ]
    outlier_data = next(data for event, data in events if event == "outlier_iteration")
    assert outlier_data["observation_count"] == 3
    assert outlier_data["inlier_count"] == 3
    assert outlier_data["outlier_count"] == 0


def test_robust_lsq_refits_final_mask_when_max_outlier_iterations_reached():
    design = jnp.ones((4, 1))
    observed = jnp.asarray([0.0, 0.0, 0.0, 100.0])
    weights = jnp.ones(observed.shape)
    init_params = jnp.asarray([10.0])

    def residuals(params):
        return design @ params - observed

    optical_weights, radar_weights = scalar_weight_arrays(weights)

    def linearize(params):
        return design, residuals(params), optical_weights, radar_weights

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

    result = RobustLeastSquares(LeastSquares(max_iter=50)).solve(
        init_params,
        policy,
        linearize,
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


def test_least_squares_keeps_linearization_weights_during_trials():
    def linearize(params):
        radar_weights = jnp.asarray([2.0 + params[0] ** 2])
        return (
            jnp.asarray([[2.0 * params[0]]]),
            jnp.asarray([params[0] ** 2 - 1.0]),
            jnp.empty((0, 2, 2)),
            radar_weights,
        )

    def residuals(params):
        return jnp.asarray([params[0] ** 2 - 1.0])

    solver = LeastSquares(max_iter=50)
    x0 = jnp.asarray([0.1])
    mask = jnp.asarray([True])
    state = initialize_lsq(
        x0,
        linearize_at(x0, mask, linearize),
        LMOptions(50, solver.max_damping_iter, solver.damping_init),
    )
    trial, candidate_model = evaluate_lm_trial(state, mask, linearize)
    assert not trial.accepted
    assert_allclose(candidate_model.residuals, residuals(trial.candidate), atol=0.0, rtol=0.0)

    result = solver.solve(
        x0, mask, linearize,
    )
    assert result.converged
    # The first scalar LM correction is known analytically. Its rejected RMS
    # must use W(0.1), even though the candidate has a very different weight.
    initial = 0.1
    candidate = initial + (1.0 - initial**2) / (2.0*initial*(1.0 + solver.damping_init))
    expected_rms = abs(candidate**2 - 1.0) * np.sqrt(2.0 + initial**2)
    assert_allclose(trial.rms, expected_rms, rtol=1e-12, atol=0.)
    # For this scalar root, the normal-matrix correction norm equals the whitened residual.
    assert result.params[0] > 0.
    assert result.normalized_residual_rms < 1e-3
    assert_allclose(result.radar_weights, 2.0 + result.params**2, atol=0.0, rtol=0.0)
