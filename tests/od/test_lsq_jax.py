"""Compiled-control-flow and numerical contracts of the JAX LM backend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difforb.od.dc.lsq import LMOptions, LSQTermination, LeastSquares, RobustLeastSquares
from difforb.od.dc.lsq.core import (
    LMTrial,
    advance_lsq,
    compute_correction_norm,
    initialize_lsq,
    compute_prior_covariance,
    update_lm_state,
)
from difforb.od.outlier.chi2 import Chi2OutlierRejecter
from difforb.od.outlier.policy import CompiledOutlierPolicy

jax.config.update("jax_enable_x64", True)


def nonlinear_problem():
    def residual(x):
        return jnp.asarray([x[0]**2 - 1.])

    def linearize(x):
        return jnp.asarray([[2.*x[0]]]), residual(x), jnp.empty((0, 2, 2)), jnp.asarray([2.+x[0]**2])

    return residual, linearize


def outlier_policy(max_iters=1, *, enabled=True, valid=None):
    valid = jnp.ones(4, bool) if valid is None else valid
    return CompiledOutlierPolicy(
        Chi2OutlierRejecter().with_observation_structure(n_2d=0, n_1d=4), enabled, max_iters, 0, 4,
        jnp.zeros(4, bool), jnp.zeros(4, bool), valid, valid,
    )


def assert_compiled_control_flow(compiled, *args):
    ir = str(compiled.lower(*args).compiler_ir(dialect="stablehlo"))
    assert "stablehlo.while" in ir
    assert "stablehlo.case" in ir
    assert "callback" not in ir.lower()


@pytest.mark.parametrize("max_iter", [0, 1])
def test_lm_reports_accepted_step_limits(max_iter):
    residual, linearize = nonlinear_problem()
    result = LeastSquares(max_iter=max_iter).solve(
        jnp.asarray([2.]), jnp.ones(1, bool), linearize,
    )
    assert result.termination_code == LSQTermination.max_iter_reached
    assert result.iter_num == max_iter


def test_lm_stops_after_damping_failure():
    residual = lambda x: jnp.where(x > 0., jnp.nan, x-1.)
    linearize = lambda x: (jnp.ones((1, 1)), residual(x), jnp.empty((0, 2, 2)), jnp.ones(1))
    solver = LeastSquares()
    solver.max_damping_iter = 3
    result = solver.solve(
        jnp.zeros(1), jnp.ones(1, bool), linearize,
    )
    assert result.termination_code == LSQTermination.damping_failed
    assert result.iter_num == 0
    np.testing.assert_array_equal(result.params, [0.])


def test_correction_norm_matches_full_normal_matrix_and_parameter_units():
    rng = np.random.default_rng(17)
    design = rng.normal(size=(12, 7))
    design[:, 6] += 2. * design[:, 0]
    residuals = rng.normal(size=12)
    delta = np.linalg.lstsq(design, -residuals, rcond=None)[0]
    expected = np.sqrt(delta @ (design.T @ design) @ delta / 7)
    np.testing.assert_allclose(compute_correction_norm(jnp.asarray(design), jnp.asarray(residuals)), expected, rtol=1e-12)
    # Changing physical units must not change the statistical correction size.
    scaled = design / np.asarray([1., 2., 3., .5, 4., 1., 1e-6])
    np.testing.assert_allclose(compute_correction_norm(jnp.asarray(scaled), jnp.asarray(residuals)), expected, rtol=1e-8)


def test_correction_norm_includes_additional_fitted_parameters():
    residual = jnp.asarray([0., 0., 0., 0., 0., 0., 7.])
    np.testing.assert_allclose(compute_correction_norm(jnp.eye(7), residual), np.sqrt(7.))


def test_statistically_small_correction_can_have_large_physical_change():
    design = jnp.diag(jnp.asarray([1., 1., 1., 1., 1., 1., 1e-9]))
    target = jnp.asarray([0., 0., 0., 0., 0., 0., 1.])
    residual = lambda x: design @ (x-target)
    linearize = lambda x: (design, residual(x), jnp.empty((0, 2, 2)), jnp.ones(7))
    result = LeastSquares(max_iter=1).solve(jnp.zeros(7), jnp.ones(7, bool), linearize)
    assert result.params[-1] > .5
    assert result.converged
    assert result.termination_code == LSQTermination.correction_converged


def test_large_damping_cannot_satisfy_correction_criterion():
    residual = lambda x: x-1.
    linearize = lambda x: (jnp.ones((1, 1)), residual(x), jnp.empty((0, 2, 2)), jnp.ones(1))
    solver = LeastSquares(max_iter=1)
    solver.damping_init = 1e12
    result = solver.solve(jnp.zeros(1), jnp.ones(1, bool), linearize)
    assert 0. < result.params[0] < 1e-10
    assert not result.converged
    assert result.termination_code == LSQTermination.max_iter_reached


def test_accepted_candidate_correction_converges_at_iteration_limit():
    residual = lambda x: x-1.
    linearize = lambda x: (jnp.ones((1, 1)), residual(x), jnp.empty((0, 2, 2)), jnp.ones(1))
    result = LeastSquares(max_iter=1).solve(
        jnp.zeros(1), jnp.ones(1, bool), linearize,
    )
    assert result.converged
    assert result.termination_code == LSQTermination.correction_converged
    assert result.iter_num == 1
    assert result.params[0] > .99


def plateau_problem():
    residual = lambda x: jnp.asarray([x[0], 1000.])
    linearize = lambda x: (jnp.asarray([[1.], [0.]]), residual(x), jnp.empty((0, 2, 2)), jnp.ones(2))
    return residual, linearize


def test_rms_stagnation_is_an_independent_success_reason_with_logging():
    residual, linearize = plateau_problem()
    solver = LeastSquares(max_iter=20)
    solver.damping_init = 1e6
    events = []
    result = solver.solve(jnp.ones(1), jnp.ones(2, bool), linearize,
                          verbose=lambda event, **data: events.append((event, data)))
    assert result.converged
    assert result.termination_reason == "rms_stagnated"
    assert result.iter_num == 10
    assert result.params[0] > .8  # delnor remains far above its threshold.
    assert [event for event, _ in events] == ["least_squares_step"] * 10
    np.testing.assert_allclose(
        events[-1][1]["normalized_residual_rms"],
        result.normalized_residual_rms,
    )


@pytest.mark.parametrize("reject", [False, True])
def test_stagnation_counter_resets_on_improvement_and_ignores_rejected_trials(reject):
    residual = lambda x: jnp.where(reject & (x > 0.), jnp.nan, x-1.)
    linearize = lambda x: (jnp.ones((1, 1)), residual(x), jnp.empty((0, 2, 2)), jnp.ones(1))
    x, mask = jnp.zeros(1), jnp.ones(1, bool)
    options = LMOptions(max_iter=20, max_damping_iter=10, damping_init=1e-3)
    state = initialize_lsq(x, mask, linearize, options)._replace(stagnant_steps=jnp.asarray(9))
    state = advance_lsq(state, mask, linearize, options)
    assert state.stagnant_steps == (9 if reject else 0)
    assert state.termination_code == LSQTermination.running


def test_half_tenth_percent_rms_improvement_resets_stagnation():
    residual, linearize = plateau_problem()
    options = LMOptions(max_iter=20, max_damping_iter=10, damping_init=1e-3)
    state = initialize_lsq(
        jnp.ones(1), jnp.ones(2, bool), linearize, options,
    )._replace(stagnant_steps=jnp.asarray(9))
    next_model = state.model._replace(rms=state.model.rms * (1.0 - 5e-4))
    trial = LMTrial(
        state.params, next_model.residuals, jnp.zeros(1), next_model.rms,
        jnp.asarray(1.0), jnp.asarray(True), state.damping, jnp.asarray(True),
    )
    updated = update_lm_state(
        state, trial, next_model, options.max_iter, options.max_damping_iter,
    )
    assert updated.stagnant_steps == 0
    assert updated.termination_code == LSQTermination.running


@pytest.mark.parametrize("exponent,reason", [(2., "rms_stagnated"), (1000., "rms_increasing")])
def test_rms_increase_uses_refreshed_weights_and_distinguishes_failure(exponent, reason):
    residual = lambda x: x-2.
    linearize = lambda x: (jnp.ones((1, 1)), residual(x), jnp.empty((0, 2, 2)), jnp.exp(exponent*x))
    solver = LeastSquares(max_iter=20)
    solver.damping_init = 1e6
    events = []
    result = solver.solve(jnp.zeros(1), jnp.ones(1, bool), linearize,
                          verbose=lambda event, **data: events.append((event, data)))
    assert result.iter_num == 10
    assert result.converged == (reason == "rms_stagnated")
    assert result.termination_reason == reason
    assert result.normalized_residual_rms > 2.
    assert [event for event, _ in events] == ["least_squares_step"] * 10
    np.testing.assert_allclose(
        events[-1][1]["normalized_residual_rms"],
        result.normalized_residual_rms,
    )


def test_inner_solve_jit_and_vmap_have_no_host_callbacks():
    residual, linearize = nonlinear_problem()
    solver = LeastSquares(max_iter=50, solver_jit=True)
    solve = jax.jit(lambda x: solver.solve(x, jnp.ones(1, bool), linearize))
    starts = jnp.asarray([[.1], [.5], [2.], [1.]])
    assert_compiled_control_flow(solve, starts[0])
    results = jax.jit(jax.vmap(solve))(starts)
    assert np.all(results.params > 0.)
    assert np.all(results.normalized_residual_rms < 1e-3)
    np.testing.assert_allclose(results.radar_weights, 2.+results.params**2, atol=1e-14)
    assert np.all(results.converged)
    assert int(results.iter_num[-1]) == 1
    assert np.all(np.asarray(results.iter_num[:-1]) > 0)


@pytest.mark.parametrize("max_iters", [1, 5])
def test_robust_loop_is_compiled_and_refits_changed_mask(max_iters):
    solver = RobustLeastSquares(LeastSquares(max_iter=50, solver_jit=True))
    policy = outlier_policy(max_iters)
    def residual(x):
        return x[0] - jnp.asarray([0., 0., 0., 100.])

    def linearize(x):
        return jnp.ones((4, 1)), residual(x), jnp.empty((0, 2, 2)), jnp.ones(4)

    solve = jax.jit(lambda x: solver.solve(x, policy, linearize))
    x0 = jnp.asarray([10.])
    assert_compiled_control_flow(solve, x0)
    results = jax.jit(jax.vmap(solve))(jnp.asarray([[10.], [30.]]))
    assert np.all(results.lsq_result.converged)
    np.testing.assert_array_equal(results.rej_result.flat_inlier_mask, [[True, True, True, False]] * 2)
    np.testing.assert_allclose(results.lsq_result.params, 0., atol=1e-9, rtol=0.)
    expected = compute_prior_covariance(jnp.ones((4, 1)), jnp.empty((0, 2, 2)), jnp.ones(4),
                                         jnp.asarray([True, True, True, False]))
    np.testing.assert_allclose(results.lsq_result.cov_mat_prior, np.broadcast_to(expected.cov_mat, (2, 1, 1)))
    assert np.all(results.lsq_iter_num > results.lsq_result.iter_num)
    assert np.all(results.outlier_iter_num <= max_iters)


def test_dynamic_weights_nonzero_residual_fixed_point():
    solver = LeastSquares(max_iter=50)
    def residual(x):
        return jnp.asarray([x[0], x[0]-2.])

    def linearize(x):
        return jnp.ones((2, 1)), residual(x), jnp.empty((0, 2, 2)), jnp.asarray([1., 1.+.05*x[0]**2])

    result = solver.solve(jnp.zeros(1), jnp.ones(2, bool), linearize)
    # J.T W(x) r(x)=0 gives this cubic, not the derivative of r.T W(x) r.
    roots = np.roots([.05, -.1, 2., -2.])
    expected = roots[np.abs(roots.imag) < 1e-12].real
    assert result.converged
    # Check the analytic fixed point at the fixed statistical accuracy, not a tunable physical tolerance.
    statistical_scale = np.sqrt(np.sum(result.radar_weights))
    np.testing.assert_allclose(result.params, expected, atol=1e-3/statistical_scale, rtol=0.)
    correction_norm = abs(np.sum(result.radar_weights * result.residuals)) / statistical_scale
    assert correction_norm < 1e-3


def test_automatic_scaling_preserves_small_model_parameters():
    design = jnp.concatenate([jnp.eye(7), jnp.ones((1, 7))])
    scales = jnp.asarray([1., 1., 1., 1., 1., 1., 1e-12])
    jac = design / scales
    expected = jnp.arange(1., 8.) * scales
    residual = lambda x: jac @ (x-expected)
    linearize = lambda x: (jac, residual(x), jnp.empty((0, 2, 2)), jnp.ones(8))
    result = LeastSquares(max_iter=50).solve(jnp.zeros(7), jnp.ones(8, bool), linearize)
    assert result.converged
    np.testing.assert_allclose(result.params/scales, expected/scales, atol=1e-8, rtol=0.)


def test_parameter_units_do_not_change_the_fit():
    design = jnp.asarray([
        [1.0, 0.0, 1.0e-15],
        [0.0, 1.0, -2.0e-15],
        [1.0, 1.0, 3.0e-15],
        [2.0, -1.0, 1.0e-15],
    ])
    expected = jnp.asarray([2.0, -3.0, 4.0e12])
    residual = lambda x: design @ (x - expected)
    linearize = lambda x: (
        design, residual(x), jnp.empty((0, 2, 2)), jnp.ones(design.shape[0]),
    )
    units = jnp.asarray([1.0e-3, 1.0e3, 1.0e12])
    scaled_design = design * units
    scaled_expected = expected / units
    scaled_residual = lambda x: scaled_design @ (x - scaled_expected)
    scaled_linearize = lambda x: (
        scaled_design,
        scaled_residual(x),
        jnp.empty((0, 2, 2)),
        jnp.ones(scaled_design.shape[0]),
    )
    solver = LeastSquares(max_iter=50)
    unit = solver.solve(jnp.zeros(3), jnp.ones(design.shape[0], bool), linearize)
    reparameterized = solver.solve(
        jnp.zeros(3), jnp.ones(scaled_design.shape[0], bool), scaled_linearize,
    )
    np.testing.assert_allclose(reparameterized.params * units, unit.params, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        reparameterized.normalized_residual_rms,
        unit.normalized_residual_rms,
        rtol=1e-12,
        atol=1e-12,
    )
    assert reparameterized.termination_code == unit.termination_code
    assert reparameterized.iter_num == unit.iter_num


@pytest.mark.parametrize("solver_jit", [False, True])
def test_initial_convergence_limits_and_nonfinite_models(solver_jit):
    residual, linearize = nonlinear_problem()
    solver = LeastSquares(max_iter=0, solver_jit=solver_jit)
    exact = solver.solve(jnp.ones(1), jnp.ones(1, bool), linearize)
    limited = solver.solve(jnp.asarray([.1]), jnp.ones(1, bool), linearize)
    invalid = solver.solve(jnp.asarray([jnp.nan]), jnp.ones(1, bool), linearize)
    assert exact.termination_code == LSQTermination.max_iter_reached
    assert not exact.converged
    assert limited.termination_code == LSQTermination.max_iter_reached
    assert not limited.converged
    assert invalid.termination_code == LSQTermination.nonfinite_model
    assert not invalid.cov_valid
    assert exact.iter_num == limited.iter_num == invalid.iter_num == 0


@pytest.mark.parametrize("solver_jit", [False, True])
def test_nonfinite_trial_is_rejected_without_changing_parameters(solver_jit):
    residual = lambda x: jnp.where(x > 0., jnp.nan, x-1.)
    linearize = lambda x: (jnp.ones((1, 1)), residual(x), jnp.empty((0, 2, 2)), jnp.ones(1))
    result = LeastSquares(solver_jit=solver_jit).solve(jnp.zeros(1), jnp.ones(1, bool), linearize)
    assert result.termination_code == LSQTermination.damping_failed
    assert not result.converged
    assert result.iter_num == 0
    np.testing.assert_array_equal(result.params, [0.])


@pytest.mark.parametrize("solver_jit", [False, True])
def test_singular_fit_skips_rejection_with_unavailable_metrics(solver_jit):
    residual = lambda x: jnp.ones(4)*(x.sum()-2.)
    linearize = lambda x: (jnp.ones((4, 2)), residual(x), jnp.empty((0, 2, 2)), jnp.ones(4))
    result = RobustLeastSquares(LeastSquares(solver_jit=solver_jit)).solve(jnp.zeros(2), outlier_policy(), linearize)
    assert not result.lsq_result.cov_valid
    assert result.outlier_iter_num == 0
    assert np.all(np.isnan(result.rej_result.metric))
    np.testing.assert_array_equal(result.rej_result.flat_inlier_mask, jnp.ones(4, bool))


def test_logging_preserves_numerical_results_and_final_refit():
    residual = lambda x: x[0]-jnp.asarray([0., 0., 0., 100.])
    linearize = lambda x: (jnp.ones((4, 1)), residual(x), jnp.empty((0, 2, 2)), jnp.ones(4))
    solver = RobustLeastSquares(LeastSquares(max_iter=50))
    quiet = solver.solve(jnp.asarray([10.]), outlier_policy(), linearize)
    events = []
    logged = solver.solve(jnp.asarray([10.]), outlier_policy(), linearize,
                           verbose=lambda event, **data: events.append((event, data)))
    np.testing.assert_allclose(logged.lsq_result.params, quiet.lsq_result.params, atol=1e-12)
    np.testing.assert_array_equal(logged.rej_result.flat_inlier_mask, quiet.rej_result.flat_inlier_mask)
    assert len([event for event, _ in events if event == "least_squares_step"]) == logged.lsq_iter_num
    outlier_events = [data for event, data in events if event == "outlier_iteration"]
    assert len(outlier_events) == 1
    assert outlier_events[0]["mask_changed"]
    assert outlier_events[0]["inlier_count"] == 3


def test_batched_stagnation_and_correction_stop_independently():
    residual, linearize = plateau_problem()
    solver = LeastSquares(max_iter=10, solver_jit=True)
    solver.damping_init = 1e6
    solve = jax.jit(jax.vmap(lambda x: solver.solve(x, jnp.ones(2, bool), linearize)))
    result = solve(jnp.asarray([[0.], [1.]]))
    np.testing.assert_array_equal(result.termination_code,
                                  [LSQTermination.correction_converged, LSQTermination.rms_stagnated])
    np.testing.assert_array_equal(result.iter_num, [1, 10])


def test_robust_refits_use_identical_convergence_settings():
    residual = lambda x: x[0]-jnp.asarray([0., 0., 0., 100.])
    linearize = lambda x: (jnp.ones((4, 1)), residual(x), jnp.empty((0, 2, 2)), jnp.ones(4))
    solver = LeastSquares(max_iter=20)
    x0 = jnp.asarray([10.])
    first = solver.solve(x0, jnp.ones(4, bool), linearize)
    final_mask = jnp.asarray([True, True, True, False])
    second = solver.solve(first.params, final_mask, linearize)
    robust = RobustLeastSquares(solver).solve(x0, outlier_policy(), linearize)
    assert first.converged and second.converged and robust.lsq_result.converged
    np.testing.assert_array_equal(robust.rej_result.flat_inlier_mask, final_mask)
    np.testing.assert_allclose(robust.lsq_result.params, second.params, atol=1e-15)
    assert robust.lsq_iter_num == first.iter_num + second.iter_num


@pytest.mark.parametrize("robust", [False, True])
def test_solver_jit_modes_preserve_fit(robust):
    if robust:
        residual = lambda x: x[0]-jnp.asarray([0., 0., 0., 100.])
        linearize = lambda x: (jnp.ones((4, 1)), residual(x), jnp.empty((0, 2, 2)), jnp.ones(4))
        x0, selection = jnp.asarray([10.]), outlier_policy()
    else:
        residual, linearize = nonlinear_problem()
        x0, selection = jnp.asarray([.1]), jnp.ones(1, bool)
    results, histories = [], []
    for compiled in [False, True]:
        inner = LeastSquares(max_iter=50, solver_jit=compiled)
        solver = RobustLeastSquares(inner) if robust else inner
        events = []
        result = solver.solve(
            x0,
            selection,
            linearize,
            verbose=lambda event, **data: events.append((event, data)),
        )
        results.append(result)
        histories.append([event for event, _ in events])
    for eager, compiled in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1])):
        if isinstance(eager, str):
            assert eager == compiled
        else:
            np.testing.assert_allclose(eager, compiled, rtol=1e-10, atol=1e-12, equal_nan=True)
    assert "least_squares_step" in histories[0]
    assert histories[0] == histories[1]


def test_python_solver_calls_compiled_steps_without_changing_jit_context(monkeypatch):
    from difforb.od.dc.lsq import core

    traces = []

    @jax.jit
    def residual(x):
        traces.append("residual")
        return x-1.

    @jax.jit
    def linearize(x):
        traces.append("linearize")
        return jnp.ones((1, 1)), residual(x), jnp.empty((0, 2, 2)), jnp.ones(1)

    x0 = jnp.zeros(1)
    jax.block_until_ready((residual(x0), linearize(x0)))
    initial_traces = list(traces)
    step_contexts = []
    original_step = core.evaluate_lm_trial

    def step(*args):
        step_contexts.append((jax.config.jax_disable_jit, isinstance(args[0], jax.core.Tracer)))
        return original_step(*args)

    monkeypatch.setattr(core, "evaluate_lm_trial", step)
    result = LeastSquares().solve(x0, jnp.ones(1, bool), linearize)
    assert result.converged
    assert len(step_contexts) >= int(result.iter_num)
    assert all(not disabled and not traced for disabled, traced in step_contexts)
    assert traces == initial_traces  # Repeated model calls execute already compiled kernels.
    assert not jax.config.jax_disable_jit


@pytest.mark.parametrize("solver_jit", [False, True])
def test_solver_jit_respects_enclosing_disable_context(solver_jit):
    contexts = []

    def residual(x):
        contexts.append(jax.config.jax_disable_jit)
        return x-1.

    linearize = lambda x: (jnp.ones((1, 1)), residual(x), jnp.empty((0, 2, 2)), jnp.ones(1))
    with jax.disable_jit():
        result = LeastSquares(solver_jit=solver_jit).solve(
            jnp.zeros(1), jnp.ones(1, bool), linearize,
        )
        assert result.converged
        assert jax.config.jax_disable_jit
    assert contexts and all(contexts)
    assert not jax.config.jax_disable_jit


@pytest.mark.parametrize("robust", [False, True])
def test_python_loops_bypass_complete_compiled_driver(monkeypatch, robust):
    from difforb.od.dc.lsq import core

    calls = dict(initialize=0, evaluate_lm_trial=0, finish=0)

    def observe(name, compiled):
        def invoke(*args, **kwargs):
            assert not isinstance(args[0], jax.core.Tracer)
            calls[name] += 1
            return compiled(*args, **kwargs)
        return invoke

    attrs = {
        "initialize": "initialize_lsq",
        "evaluate_lm_trial": "evaluate_lm_trial",
        "finish": "finish_lsq",
    }
    for name, attr in attrs.items():
        monkeypatch.setattr(core, attr, observe(name, getattr(core, attr)))

    def forbid(*args, **kwargs):
        raise AssertionError("The Python driver must not call the full compiled driver")

    monkeypatch.setattr(core, "solve_lsq", forbid)
    residual = lambda x: x[0]-jnp.asarray([0., 0., 0., 100.])
    linearize = lambda x: (jnp.ones((4, 1)), residual(x), jnp.empty((0, 2, 2)), jnp.ones(4))
    inner = LeastSquares()
    if robust:
        result = RobustLeastSquares(inner).solve(jnp.asarray([10.]), outlier_policy(), linearize)
        expected_fits = 2
        fit = result.lsq_result
        np.testing.assert_array_equal(result.rej_result.flat_inlier_mask, [True, True, True, False])
    else:
        fit = inner.solve(jnp.asarray([10.]), jnp.ones(4, bool), linearize)
        expected_fits = 1
    assert fit.converged
    assert calls["initialize"] == calls["finish"] == expected_fits
    assert calls["evaluate_lm_trial"] >= int(fit.iter_num)
