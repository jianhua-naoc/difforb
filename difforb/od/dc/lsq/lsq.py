"""Least-squares configuration and result objects for differential correction."""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from difforb.core.validate import coerce_scalar_bool, coerce_scalar_int
from difforb.od.dc.lsq.core import (
    LSQTermination,
    LeastSquaresResult,
    LinearizationFunction,
    PriorCovarianceResult,
    RobustResult,
    flat_inlier_mask_to_observation_mask,
    solve_lsq,
    solve_lsq_python,
    solve_robust,
    valid_chi2,
)
from difforb.od.events import (
    SolverEventHandler,
    SolverEventLogger,
    SolverLogDetail,
    make_solver_event_logger,
)
from difforb.od.outlier.outlier import RejResult
from difforb.od.outlier.policy import CompiledOutlierPolicy


def _emit_lsq_done(logger, result, n_params):
    reason = LSQTermination(int(result.termination_code)).name
    converged = reason in {"correction_converged", "rms_stagnated"}
    if reason in {"damping_failed", "nonfinite_model", "linear_solve_failed", "rms_increasing"}:
        logger.bind(lsq_step=int(result.iter_num) + 1).emit(
            "least_squares", "lsq_failed", "warning", reason=reason,
            normalized_residual_rms=float(result.normalized_residual_rms),
        )
    logger.emit(
        "least_squares", "lsq_done", "info" if converged else "warning",
        converged=converged, reason=reason, steps=int(result.iter_num),
        normalized_residual_rms=float(result.normalized_residual_rms),
        cov_rank=int(result.cov_rank), cov_condition=float(result.cov_condition),
        cov_valid=bool(result.cov_valid), n_params=n_params,
    )


class LeastSquares:
    """Dynamically weighted Levenberg--Marquardt least squares.

    The default host driver keeps Python control flow around small JIT-compiled
    numerical kernels. The compiled driver places the same method in a JAX
    ``while_loop`` for transformation with ``jit`` and ``vmap``.
    """

    _DAMPING_INIT = 1e-3
    _MAX_DAMPING_ITER = 10

    def __init__(self, max_iter: int = 20, *, solver_jit: bool = False):
        """Initialize one least-squares solver."""
        self.solver_jit = coerce_scalar_bool("solver_jit", solver_jit)
        self.max_iter = coerce_scalar_int("max_iter", max_iter)
        if self.max_iter < 0:
            raise ValueError("max_iter must be nonnegative.")
        self.damping_init = self._DAMPING_INIT
        self.max_damping_iter = self._MAX_DAMPING_ITER

    def _solver_options(self):
        return {
            "max_iter": self.max_iter,
            "max_damping_iter": self.max_damping_iter,
            "damping_init": self.damping_init,
        }

    def solve(self, init_params: Float[Array, "N_param"],
              inlier_mask: Bool[Array, "N_flat_obs"],
              res_func: Callable, linearize_func: LinearizationFunction, *,
              param_scale: Float[Array, "N_param"] | None = None,
              event_handler: SolverEventHandler | None = None,
              log_detail: SolverLogDetail = "iter",
              event_logger: SolverEventLogger | None = None) -> LeastSquaresResult:
        """Fit one fixed inlier set and return a JAX-compatible result."""
        logger = event_logger if event_logger is not None else make_solver_event_logger(event_handler, log_detail)
        scale = jnp.ones_like(init_params) if param_scale is None else jnp.asarray(param_scale)
        options = self._solver_options()

        if self.solver_jit:
            result = solve_lsq(
                init_params, inlier_mask, res_func, linearize_func, scale, options,
            )
        else:
            def initial_callback(model):
                logger.emit(
                    "least_squares", "lsq_start",
                    normalized_residual_rms=float(model.rms),
                    damping=self.damping_init,
                    n_inlier_residuals=int(jnp.sum(inlier_mask)),
                    n_params=init_params.size,
                )

            def trial_callback(state, trial, next_model):
                step = int(state.steps) + 1
                trial_index = int(state.damping_trials) + 1
                bound = logger.bind(lsq_step=step, trial=trial_index)
                bound.emit("least_squares", "lm_trial_start", damping=float(state.damping))
                if bool(trial.accepted):
                    logger.bind(lsq_step=step).emit(
                        "least_squares", "lsq_step_accepted",
                        normalized_residual_rms_before=float(state.model.rms),
                        normalized_residual_rms_after=float(next_model.rms),
                        rho=float(trial.rho), damping=float(state.damping),
                        next_damping=float(trial.next_damping),
                        damping_trials=trial_index,
                    )
                else:
                    bound.emit(
                        "least_squares", "lm_trial_rejected",
                        normalized_residual_rms=float(trial.rms),
                        rho=float(trial.rho), damping=float(state.damping),
                    )

            result = solve_lsq_python(
                init_params, inlier_mask, res_func, linearize_func, scale, options,
                initial_callback=initial_callback,
                trial_callback=trial_callback,
            )

        if not isinstance(result.termination_code, jax.core.Tracer):
            _emit_lsq_done(logger, result, init_params.size)
        return result


class RobustLeastSquares:
    """Alternate least-squares fits and fixed-shape outlier rejection."""

    def __init__(self, solver: LeastSquares) -> None:
        """Initialize robust fitting with one configured inner solver."""
        self.solver = solver

    def solve(self, init_param: Float[Array, "N_param"],
              compiled_outlier_policy: CompiledOutlierPolicy,
              res_func: Callable, linearize_func: LinearizationFunction, *,
              param_scale: Float[Array, "N_param"] | None = None,
              event_handler: SolverEventHandler | None = None,
              log_detail: SolverLogDetail = "iter",
              event_logger: SolverEventLogger | None = None) -> RobustResult:
        """Run robust fitting and optionally emit host-side progress."""
        if compiled_outlier_policy.max_iters < 1:
            raise ValueError("The outlier iteration budget must be positive.")
        logger = event_logger if event_logger is not None else make_solver_event_logger(event_handler, log_detail)

        if self.solver.solver_jit:
            scale = jnp.ones_like(init_param) if param_scale is None else jnp.asarray(param_scale)
            result = solve_robust(
                init_param, compiled_outlier_policy, res_func, linearize_func,
                scale, self.solver._solver_options(),
            )
            if not isinstance(result.lsq_result.termination_code, jax.core.Tracer):
                _emit_lsq_done(logger, result.lsq_result, init_param.size)
                self._emit_robust_done(logger, result, compiled_outlier_policy)
            return result

        return self._solve_host(
            init_param, compiled_outlier_policy, res_func, linearize_func,
            param_scale, logger,
        )

    @staticmethod
    def _emit_robust_done(logger, result, policy):
        observation_count = int(np.asarray(policy.observation_valid_mask).sum())
        observation_mask = flat_inlier_mask_to_observation_mask(
            result.rej_result.flat_inlier_mask, policy.n_2d,
        )
        inlier_count = int(np.asarray(observation_mask & policy.observation_valid_mask).sum())
        bound = logger.bind(outlier_iteration=max(1, int(result.outlier_iter_num)))
        if not bool(valid_chi2(result.lsq_result)):
            bound.emit(
                "robust_least_squares", "outlier_skipped", "warning",
                stop_reason="chi2_unavailable",
                lsq_termination_reason=LSQTermination(int(result.lsq_result.termination_code)).name,
                cov_valid=bool(result.lsq_result.cov_valid),
                cov_rank=int(result.lsq_result.cov_rank),
                total_accepted_step_count=int(result.lsq_iter_num),
            )
        elif not policy.enable_auto_rejection:
            bound.emit(
                "robust_least_squares", "outlier_disabled",
                observation_count=observation_count, inlier_count=inlier_count,
                outlier_count=observation_count - inlier_count,
                normalized_residual_rms=float(result.lsq_result.normalized_residual_rms),
            )
        else:
            bound.emit(
                "robust_least_squares", "outlier_done",
                stop_reason="mask_unchanged",
                outlier_iteration_count=int(result.outlier_iter_num),
                observation_count=observation_count, inlier_count=inlier_count,
                outlier_count=observation_count - inlier_count,
            )

    def _solve_host(self, init_param, policy, residual, linearize, param_scale, logger):
        params = init_param
        mask = policy.get_init_mask()
        total_steps = 0
        outlier_iterations = 0
        result = None
        rejected = None
        mask_changed = False
        observation_count = int(np.asarray(policy.observation_valid_mask).sum())

        def unavailable(current_mask, fit_result):
            dtype = jnp.result_type(init_param, fit_result.residuals)
            return RejResult(current_mask, jnp.full((policy.n_2d + policy.n_1d,), jnp.nan, dtype))

        for index in range(policy.max_iters):
            observation_mask = flat_inlier_mask_to_observation_mask(mask, policy.n_2d)
            inlier_count = int(np.asarray(observation_mask & policy.observation_valid_mask).sum())
            iteration_logger = logger.bind(outlier_iteration=index + 1, lsq_solve=index + 1)
            iteration_logger.emit(
                "robust_least_squares", "outlier_iteration_start",
                observation_count=observation_count, inlier_count=inlier_count,
                outlier_count=observation_count - inlier_count,
            )
            result = self.solver.solve(
                params, mask, residual, linearize, param_scale=param_scale,
                event_logger=iteration_logger,
            )
            total_steps += int(result.iter_num)
            mask_changed = False

            if not bool(valid_chi2(result)):
                rejected = unavailable(mask, result)
                iteration_logger.emit(
                    "robust_least_squares", "outlier_skipped", "warning",
                    stop_reason="chi2_unavailable",
                    lsq_termination_reason=result.termination_reason,
                    cov_valid=bool(result.cov_valid), cov_rank=int(result.cov_rank),
                    total_accepted_step_count=total_steps,
                )
                break

            rejected = policy.apply(
                result.residuals, result.optical_weight_matrices,
                result.radar_weights, result.jacobian,
                result.cov_mat_prior, mask,
            )
            if not policy.enable_auto_rejection:
                iteration_logger.emit(
                    "robust_least_squares", "outlier_disabled",
                    observation_count=observation_count, inlier_count=inlier_count,
                    outlier_count=observation_count - inlier_count,
                    normalized_residual_rms=float(result.normalized_residual_rms),
                )
                break

            outlier_iterations += 1
            new_mask = rejected.flat_inlier_mask
            old_observations = flat_inlier_mask_to_observation_mask(mask, policy.n_2d)
            new_observations = flat_inlier_mask_to_observation_mask(new_mask, policy.n_2d)
            valid = policy.observation_valid_mask
            new_inlier_count = int(np.asarray(new_observations & valid).sum())
            iteration_logger.emit(
                "robust_least_squares", "outlier_update",
                normalized_residual_rms=float(result.normalized_residual_rms),
                changed_to_outlier_count=int(np.asarray(old_observations & ~new_observations & valid).sum()),
                changed_to_inlier_count=int(np.asarray(~old_observations & new_observations & valid).sum()),
                observation_count=observation_count, inlier_count=new_inlier_count,
                outlier_count=observation_count - new_inlier_count,
            )
            if bool(jnp.array_equal(new_mask, mask)):
                iteration_logger.emit(
                    "robust_least_squares", "outlier_done",
                    stop_reason="mask_unchanged",
                    outlier_iteration_count=outlier_iterations,
                    observation_count=observation_count,
                    inlier_count=new_inlier_count,
                    outlier_count=observation_count - new_inlier_count,
                )
                break

            params = result.params
            mask = new_mask
            mask_changed = True
        else:
            if mask_changed:
                result = self.solver.solve(
                    params, mask, residual, linearize, param_scale=param_scale,
                    event_logger=logger.bind(
                        outlier_iteration=policy.max_iters,
                        lsq_solve=policy.max_iters + 1,
                    ),
                )
                total_steps += int(result.iter_num)
                if bool(valid_chi2(result)):
                    metric = policy.apply(
                        result.residuals, result.optical_weight_matrices,
                        result.radar_weights, result.jacobian,
                        result.cov_mat_prior, mask,
                    ).metric
                    rejected = RejResult(mask, metric)
                else:
                    rejected = unavailable(mask, result)

            observation_mask = flat_inlier_mask_to_observation_mask(mask, policy.n_2d)
            inlier_count = int(np.asarray(observation_mask & policy.observation_valid_mask).sum())
            logger.bind(outlier_iteration=policy.max_iters).emit(
                "robust_least_squares", "outlier_done", "warning",
                stop_reason="max_iterations_reached",
                max_iterations=policy.max_iters,
                outlier_iteration_count=outlier_iterations,
                observation_count=observation_count, inlier_count=inlier_count,
                outlier_count=observation_count - inlier_count,
            )

        assert result is not None and rejected is not None
        return RobustResult(
            result,
            rejected,
            jnp.asarray(outlier_iterations, dtype=jnp.int32),
            jnp.asarray(total_steps, dtype=jnp.int32),
        )


__all__ = [
    "LSQTermination", "LeastSquares", "LeastSquaresResult",
    "LinearizationFunction", "PriorCovarianceResult", "RobustLeastSquares",
    "RobustResult",
]
