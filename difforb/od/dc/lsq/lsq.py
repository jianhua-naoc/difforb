"""Least-squares configuration and result objects for differential correction."""

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from difforb.core.validate import coerce_scalar_bool, coerce_scalar_int
from difforb.od.dc.lsq.core import (
    LSQTermination,
    LMOptions,
    LeastSquaresResult,
    LinearizationFunction,
    PriorCovarianceResult,
    RobustResult,
    advance_robust,
    continue_robust,
    evaluate_rejection,
    finish_robust,
    flat_inlier_mask_to_observation_mask,
    initialize_robust,
    solve_lsq,
    solve_lsq_python,
    solve_robust,
)
from difforb.od.outlier.policy import CompiledOutlierPolicy
from difforb.od.progress import SolverReporter, solver_progress_reporter


class LeastSquares:
    """Dynamically weighted Levenberg--Marquardt least squares.

    The default host driver keeps Python control flow around small JIT-compiled
    numerical kernels. The compiled driver places the same method in a JAX
    ``while_loop`` for transformation with ``jit`` and ``vmap``.
    """

    _DAMPING_INIT = 1e-3
    _MAX_DAMPING_ITER = 10

    def __init__(self, max_iter: int = 20, *, solver_jit: bool = False) -> None:
        """Initialize one least-squares solver."""
        self.solver_jit = coerce_scalar_bool("solver_jit", solver_jit)
        self.max_iter = coerce_scalar_int("max_iter", max_iter)
        if self.max_iter < 0:
            raise ValueError("max_iter must be nonnegative.")
        self.damping_init = self._DAMPING_INIT
        self.max_damping_iter = self._MAX_DAMPING_ITER

    @property
    def options(self) -> LMOptions:
        """Numerical options consumed by the array-only solver core."""
        return LMOptions(
            self.max_iter,
            self.max_damping_iter,
            self.damping_init,
        )

    def solve(self, init_params: Float[Array, "N_param"],
              inlier_mask: Bool[Array, "N_flat_obs"],
              linearize_func: LinearizationFunction, *,
              verbose: bool | SolverReporter = False) -> LeastSquaresResult:
        """Fit one fixed inlier set and return a JAX-compatible result."""
        reporter = solver_progress_reporter(verbose)
        options = self.options

        if self.solver_jit and reporter is None:
            return solve_lsq(init_params, inlier_mask, linearize_func, options)
        return solve_lsq_python(
            init_params, inlier_mask, linearize_func, options,
            step_callback=reporter,
        )


class RobustLeastSquares:
    """Alternate least-squares fits and fixed-shape outlier rejection."""

    def __init__(self, solver: LeastSquares) -> None:
        """Initialize robust fitting with one configured inner solver."""
        self.solver = solver

    def solve(self, init_param: Float[Array, "N_param"],
              compiled_outlier_policy: CompiledOutlierPolicy,
              linearize_func: LinearizationFunction, *,
              verbose: bool | SolverReporter = False) -> RobustResult:
        """Run robust fitting and optionally report host-side progress."""
        if compiled_outlier_policy.max_iters < 1:
            raise ValueError("The outlier iteration budget must be positive.")
        reporter = solver_progress_reporter(verbose)

        if self.solver.solver_jit and reporter is None:
            return solve_robust(
                init_param, compiled_outlier_policy, linearize_func,
                self.solver.options,
            )

        mask = compiled_outlier_policy.get_init_mask()
        result = self.solver.solve(
            init_param, mask, linearize_func,
            verbose=False if reporter is None else reporter,
        )
        state = initialize_robust(result, mask)

        while bool(continue_robust(state, compiled_outlier_policy)):
            rejection = evaluate_rejection(state, compiled_outlier_policy)
            changed = bool(jnp.any(rejection.flat_inlier_mask != state.mask))
            if changed:
                next_result = self.solver.solve(
                    state.result.params,
                    rejection.flat_inlier_mask,
                    linearize_func,
                    verbose=False if reporter is None else reporter,
                )
            else:
                next_result = state.result
            state = advance_robust(state, rejection, next_result)
            if reporter is not None:
                observation_mask = flat_inlier_mask_to_observation_mask(
                    state.mask,
                    compiled_outlier_policy.n_2d,
                )
                valid = compiled_outlier_policy.observation_valid_mask
                observation_count = int(np.asarray(valid).sum())
                inlier_count = int(np.asarray(observation_mask & valid).sum())
                reporter(
                    "outlier_iteration",
                    iteration=int(state.iterations),
                    observation_count=observation_count,
                    inlier_count=inlier_count,
                    outlier_count=observation_count - inlier_count,
                    normalized_residual_rms=float(
                        state.result.normalized_residual_rms,
                    ),
                    mask_changed=changed,
                )

        return finish_robust(
            state, compiled_outlier_policy, init_param.dtype,
        )


__all__ = [
    "LMOptions", "LSQTermination", "LeastSquares", "LeastSquaresResult",
    "LinearizationFunction", "PriorCovarianceResult", "RobustLeastSquares",
    "RobustResult",
]
