import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from difforb.od.outlier.chi2 import Chi2OutlierRejecter, compute_individual_chi2, update_inlier_mask
from difforb.od.outlier.policy import CompiledOutlierPolicy
from tests.assertions import assert_allclose, assert_array_equal


def _static_array_warning_messages(caught) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if "A JAX array is being set as static" in str(w.message)
    ]


def test_individual_chi2_matches_diagonal_closed_form():
    flat_residuals = jnp.asarray([1.0, -2.0, 3.0, -4.0, 5.0])
    flat_measure_var = jnp.asarray([4.0, 9.0, 16.0, 25.0, 36.0])
    jacobian = jnp.zeros((5, 2))
    cov_mat = jnp.zeros((2, 2))
    inlier_mask = jnp.asarray([True, True, True])

    actual = compute_individual_chi2(
        flat_residuals,
        flat_measure_var,
        jacobian,
        cov_mat,
        inlier_mask,
        n_2d=2,
        n_1d=1,
    )
    expected = jnp.asarray([
        1.0**2 / 4.0 + (-2.0) ** 2 / 9.0,
        3.0**2 / 16.0 + (-4.0) ** 2 / 25.0,
        5.0**2 / 36.0,
    ])

    print(
        "[od.outlier.chi2] "
        f"metric_max_abs_diff={float(jnp.max(jnp.abs(actual - expected))):.12e}"
    )

    assert_allclose(actual, expected, atol=1.0e-15, rtol=0.0)


def test_chi2_rejecter_uses_optical_covariance_blocks():
    rejecter = Chi2OutlierRejecter().with_observation_structure(n_2d=1, n_1d=0)
    residuals = jnp.asarray([1.0, 2.0])
    covariance = jnp.asarray([[4.0, 1.2], [1.2, 9.0]])
    weight_matrix = jnp.linalg.inv(covariance)

    result = rejecter.reject(
        flat_residuals=residuals,
        optical_weight_matrices=weight_matrix[None, :, :],
        radar_weights=jnp.asarray([], dtype=residuals.dtype),
        jac=jnp.zeros((2, 1)),
        cov_mat=jnp.zeros((1, 1)),
        flat_inlier_mask=jnp.ones(2, dtype=bool),
    )

    expected = residuals @ weight_matrix @ residuals
    assert_allclose(result.metric, jnp.asarray([expected]), atol=1.0e-15, rtol=0.0)


def test_update_inlier_mask_rejects_and_recovers():
    current_inlier_mask = jnp.asarray([True, True, False, False])
    chi2 = jnp.asarray([1.0, 1000.0, 2.0, 20.0])

    new_mask, reject_threshold, chi2_max = update_inlier_mask(
        current_inlier_mask,
        chi2,
        chi2_rej=8.0,
        chi2_rec=5.0,
        progressive_alpha=0.25,
    )

    expected_reject_threshold = 8.0 + 400.0 / (1.2**2)

    assert_array_equal(new_mask, jnp.asarray([True, False, True, False]))
    assert_allclose(chi2_max, 1000.0, atol=0.0, rtol=0.0)
    assert_allclose(reject_threshold, expected_reject_threshold, atol=1.0e-15, rtol=0.0)


def test_chi2_rejecter_updates_2d_and_1d_masks():
    rejecter = Chi2OutlierRejecter().with_observation_structure(n_2d=2, n_1d=2)
    flat_residuals = jnp.asarray([
        0.0,
        0.0,
        30.0,
        0.0,
        0.0,
        30.0,
    ])
    optical_weight_matrices = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
    radar_weights = jnp.ones(2)
    jacobian = jnp.zeros((6, 2))
    cov_mat = jnp.zeros((2, 2))
    flat_inlier_mask = jnp.ones(6, dtype=bool)

    result = rejecter.reject(flat_residuals, optical_weight_matrices, radar_weights, jacobian, cov_mat, flat_inlier_mask)

    print(
        "[od.outlier.reject] "
        f"metric={jnp.asarray(result.metric)} "
        f"flat_inliers={jnp.asarray(result.flat_inlier_mask)}"
    )

    assert_array_equal(result.flat_inlier_mask, jnp.asarray([True, True, False, False, True, False]))
    assert_allclose(result.metric, jnp.asarray([0.0, 900.0, 0.0, 900.0]), atol=1.0e-12, rtol=0.0)


def test_compiled_policy_manual_masks_override_current_mask():
    policy = CompiledOutlierPolicy(
        auto_rejecter=None,
        enable_auto_rejection=True,
        max_iters=1,
        n_2d=2,
        n_1d=1,
        flat_manual_outlier_mask=jnp.asarray([False, False, True, True, False]),
        flat_manual_inlier_mask=jnp.asarray([False, False, False, False, True]),
        flat_valid_mask=jnp.ones(5, dtype=bool),
        observation_valid_mask=jnp.ones(3, dtype=bool),
    )
    current_mask = jnp.asarray([True, True, True, True, False])

    init_mask = policy.get_init_mask()
    result = policy.apply(
        flat_residuals=jnp.zeros(5),
        optical_weight_matrices=jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
        radar_weights=jnp.ones(1),
        jac=jnp.zeros((5, 2)),
        cov_mat=jnp.eye(2),
        cur_flat_inlier_mask=current_mask,
    )

    assert_array_equal(init_mask, jnp.asarray([True, True, False, False, True]))
    assert_array_equal(result.flat_inlier_mask, jnp.asarray([True, True, False, False, True]))
    assert_allclose(result.metric, jnp.full((3,), jnp.nan), equal_nan=True)


def test_outlier_policy_normalizes_jax_scalar_static_options():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rejecter = Chi2OutlierRejecter(
            n_2d=jnp.asarray(2, dtype=jnp.int32),
            n_1d=jnp.asarray(1, dtype=jnp.int32),
        )
        policy = CompiledOutlierPolicy(
            auto_rejecter=None,
            enable_auto_rejection=jnp.asarray(True),
            max_iters=jnp.asarray(1, dtype=jnp.int32),
            n_2d=jnp.asarray(2, dtype=jnp.int32),
            n_1d=jnp.asarray(1, dtype=jnp.int32),
            flat_manual_outlier_mask=jnp.zeros(5, dtype=bool),
            flat_manual_inlier_mask=jnp.zeros(5, dtype=bool),
            flat_valid_mask=jnp.ones(5, dtype=bool),
            observation_valid_mask=jnp.ones(3, dtype=bool),
        )

    assert _static_array_warning_messages(caught) == []
    assert isinstance(rejecter.n_2d, int)
    assert isinstance(rejecter.n_1d, int)
    assert policy.enable_auto_rejection is True
    assert isinstance(policy.max_iters, int)
    assert isinstance(policy.n_2d, int)
    assert isinstance(policy.n_1d, int)


def test_compiled_policy_structural_mask_blocks_manual_inlier():
    policy = CompiledOutlierPolicy(
        auto_rejecter=None,
        enable_auto_rejection=True,
        max_iters=1,
        n_2d=2,
        n_1d=1,
        flat_manual_outlier_mask=jnp.zeros(5, dtype=bool),
        flat_manual_inlier_mask=jnp.asarray([False, False, False, False, True]),
        flat_valid_mask=jnp.asarray([True, True, True, True, False]),
        observation_valid_mask=jnp.asarray([True, True, False]),
    )
    current_mask = jnp.asarray([True, True, True, True, False])

    init_mask = policy.get_init_mask()
    result = policy.apply(
        flat_residuals=jnp.zeros(5),
        optical_weight_matrices=jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
        radar_weights=jnp.ones(1),
        jac=jnp.zeros((5, 2)),
        cov_mat=jnp.eye(2),
        cur_flat_inlier_mask=current_mask,
    )

    assert_array_equal(init_mask, jnp.asarray([True, True, True, True, False]))
    assert_array_equal(result.flat_inlier_mask, jnp.asarray([True, True, True, True, False]))
