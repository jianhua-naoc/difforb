"""JAX numerical kernels for differential-correction least squares.

The host and compiled drivers implement the same dynamically weighted
Levenberg--Marquardt method. Observation weights are refreshed only at
accepted parameters and remain fixed while damping trials are evaluated.
"""

from enum import IntEnum
from typing import Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array
from jax.typing import DTypeLike
from jaxtyping import Bool, Float, Int

from difforb.core.constants import DAY_S
from difforb.od.outlier.outlier import RejResult
from difforb.od.outlier.policy import CompiledOutlierPolicy

jax.config.update("jax_enable_x64", True)

CORRECTION_TOL = 1e-3
MIN_RMS_DECREASE = 1e-4
STAGNATION_STEPS = 10
COVARIANCE_RCOND = 1e-12
JACOBIAN_COLUMN_SCALE_RCOND = 1e-12
TRUST_REGION_LOW_CUTOFF = 0.01
TRUST_REGION_HIGH_CUTOFF = 0.99
TRUST_REGION_SHRINK = 4.0
TRUST_REGION_GROW = 3.5

LinearizationFunction = Callable[
    [Float[Array, "N_param"]],
    tuple[
        Float[Array, "N_flat_obs N_param"],
        Float[Array, "N_flat_obs"],
        Float[Array, "N_optical 2 2"],
        Float[Array, "N_radar"],
    ],
]


class LMOptions(NamedTuple):
    """Static configuration for Levenberg--Marquardt control flow."""

    max_iter: int
    max_damping_iter: int
    damping_init: float


class LSQTermination(IntEnum):
    """Numeric termination codes returned by the array interface."""

    running = 0
    correction_converged = 1
    rms_stagnated = 2
    max_iter_reached = 3
    damping_failed = 4
    nonfinite_model = 5
    linear_solve_failed = 6
    rms_increasing = 7


class PriorCovarianceResult(NamedTuple):
    """Unscaled covariance matrix and rank diagnostics."""

    cov_mat: Float[Array, "N_param N_param"]
    rank: Int[Array, ""]
    condition: Float[Array, ""]
    valid: Bool[Array, ""]


class LeastSquaresResult(NamedTuple):
    """JAX-compatible result from one fixed-mask least-squares fit."""

    params: Float[Array, "N_param"]
    jacobian: Float[Array, "N_obs N_param"]
    residuals: Float[Array, "N_obs"]
    normalized_residual_rms: Float[Array, ""]
    optical_weight_matrices: Float[Array, "N_optical 2 2"]
    radar_weights: Float[Array, "N_radar"]
    cov_mat_prior: Float[Array, "N_param N_param"]
    cov_mat_post: Float[Array, "N_param N_param"]
    cov_rank: Int[Array, ""]
    cov_condition: Float[Array, ""]
    cov_valid: Bool[Array, ""]
    converged: Bool[Array, ""]
    termination_code: Int[Array, ""]
    iter_num: Int[Array, ""]

    @property
    def termination_reason(self) -> str:
        """Return the host-readable termination reason."""
        return LSQTermination(int(self.termination_code)).name


class RobustResult(NamedTuple):
    """JAX-compatible result from robust fitting and rejection."""

    lsq_result: LeastSquaresResult
    rej_result: RejResult
    outlier_iter_num: Int[Array, ""]
    lsq_iter_num: Int[Array, ""]


class Linearization(NamedTuple):
    """Weighted model at one accepted parameter vector."""

    jacobian: Float[Array, "N_obs N_param"]
    residuals: Float[Array, "N_obs"]
    optical_weights: Float[Array, "N_optical 2 2"]
    radar_weights: Float[Array, "N_radar"]
    design: Float[Array, "N_obs N_param"]
    whitened_residuals: Float[Array, "N_obs"]
    step_scale: Float[Array, "N_param"]
    correction_norm: Float[Array, ""]
    rms: Float[Array, ""]
    finite: Bool[Array, ""]


class LMState(NamedTuple):
    """State carried between LM damping trials."""

    params: Float[Array, "N_param"]
    model: Linearization
    damping: Float[Array, ""]
    steps: Int[Array, ""]
    damping_trials: Int[Array, ""]
    stagnant_steps: Int[Array, ""]
    termination_code: Int[Array, ""]


class LMTrial(NamedTuple):
    """Numerical outcome of one damping trial."""

    candidate: Float[Array, "N_param"]
    residuals: Float[Array, "N_obs"]
    scaled_delta: Float[Array, "N_param"]
    rms: Float[Array, ""]
    rho: Float[Array, ""]
    accepted: Bool[Array, ""]
    next_damping: Float[Array, ""]
    linear_ok: Bool[Array, ""]


class RobustState(NamedTuple):
    """State carried between compiled rejection passes."""

    result: LeastSquaresResult
    mask: Bool[Array, "N_obs"]
    valid: Bool[Array, ""]
    done: Bool[Array, ""]
    iterations: Int[Array, ""]
    solves: Int[Array, ""]
    total_steps: Int[Array, ""]


def evaluate_linearization(
        params: Float[Array, "N_param"],
        linearize_func: LinearizationFunction,
) -> tuple[
    Float[Array, "N_obs N_param"],
    Float[Array, "N_obs"],
    Float[Array, "N_optical 2 2"],
    Float[Array, "N_radar"],
]:
    """Evaluate residuals, their Jacobian, and weights at one parameter vector."""
    jacobian, residuals, optical_weights, radar_weights = linearize_func(params)
    return (
        jacobian,
        residuals,
        jax.lax.stop_gradient(jnp.asarray(optical_weights)),
        jax.lax.stop_gradient(jnp.asarray(radar_weights)),
    )


@jax.jit
def build_time_inflated_optical_weight_matrices(
        base_optical_covariances: Float[Array, "N_optical 2 2"],
        optical_time_uncertainties: Float[Array, "N_optical"],
        optical_rates: Float[Array, "N_optical 2"],
) -> Float[Array, "N_optical 2 2"]:
    """Return inverse covariance blocks with optical time uncertainty included."""
    finite_mask = jnp.isfinite(optical_time_uncertainties) & (optical_time_uncertainties != 0.0)
    time_days = jnp.where(
        finite_mask,
        optical_time_uncertainties / jnp.asarray(DAY_S, dtype=optical_rates.dtype),
        0.0,
    )
    covariances = base_optical_covariances + jnp.einsum(
        "ni,nj,n->nij", optical_rates, optical_rates, time_days * time_days,
    )
    return jnp.linalg.inv(covariances)


def flat_inlier_mask_to_observation_mask(
        flat_inlier_mask: Bool[Array, "N_flat_obs"],
        n_2d: int,
) -> Bool[Array, "N_obs"]:
    """Convert a flat residual mask to one mask value per observation."""
    optical = flat_inlier_mask[:2 * n_2d].reshape((n_2d, 2)).all(axis=1)
    return jnp.concatenate([optical, flat_inlier_mask[2 * n_2d:]])


def _optical_observation_mask(
        inlier_mask: Bool[Array, "N_flat_obs"],
        n_2d: int,
) -> Bool[Array, "N_optical"]:
    return jnp.all(inlier_mask[:2 * n_2d].reshape((n_2d, 2)), axis=1)


def _weight_cholesky_factor(
        optical_weight_matrices: Float[Array, "N_optical 2 2"],
) -> Float[Array, "N_optical 2 2"]:
    return jnp.swapaxes(jnp.linalg.cholesky(optical_weight_matrices), -1, -2)


@jax.jit
def whiten_residuals(
        residuals: Float[Array, "N_flat_obs"],
        optical_weight_matrices: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
) -> Float[Array, "N_flat_obs"]:
    """Apply block square-root weights and the fixed inlier mask."""
    n_2d = optical_weight_matrices.shape[0]
    n_flat_2d = 2 * n_2d
    optical_residuals = residuals[:n_flat_2d].reshape((n_2d, 2))
    optical_factor = _weight_cholesky_factor(optical_weight_matrices)
    optical_factor *= _optical_observation_mask(inlier_mask, n_2d)[:, None, None]
    optical = jnp.einsum("nij,nj->ni", optical_factor, optical_residuals).reshape(-1)
    radar_sqrt_weights = jnp.sqrt(jnp.where(inlier_mask[n_flat_2d:], radar_weights, 0.0))
    radar = residuals[n_flat_2d:] * radar_sqrt_weights
    return jnp.concatenate([optical, radar])


@jax.jit
def whiten_design_matrix(
        design: Float[Array, "N_flat_obs N_param"],
        optical_weight_matrices: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
) -> Float[Array, "N_flat_obs N_param"]:
    """Apply block square-root weights and the fixed inlier mask to a Jacobian."""
    n_2d = optical_weight_matrices.shape[0]
    n_flat_2d = 2 * n_2d
    optical_design = design[:n_flat_2d].reshape((n_2d, 2, design.shape[-1]))
    optical_factor = _weight_cholesky_factor(optical_weight_matrices)
    optical_factor *= _optical_observation_mask(inlier_mask, n_2d)[:, None, None]
    optical = jnp.einsum("nij,njk->nik", optical_factor, optical_design).reshape((-1, design.shape[-1]))
    radar_sqrt_weights = jnp.sqrt(jnp.where(inlier_mask[n_flat_2d:], radar_weights, 0.0))
    radar = design[n_flat_2d:] * radar_sqrt_weights[:, None]
    return jnp.concatenate([optical, radar])


def _jacobian_column_scale(
        design: Float[Array, "N_flat_obs N_param"],
) -> Float[Array, "N_param"]:
    """Build column-normalization factors from the weighted Jacobian."""
    col_norms = jnp.sqrt(jnp.sum(design * design, axis=0))
    finite_positive = jnp.where(jnp.isfinite(col_norms) & (col_norms > 0.0), col_norms, 0.0)
    max_norm = jnp.max(finite_positive)
    floor = max_norm * jnp.maximum(
        jnp.asarray(JACOBIAN_COLUMN_SCALE_RCOND, design.dtype), jnp.finfo(design.dtype).eps,
    )
    return jnp.where(
        max_norm > 0.0,
        1.0 / jnp.maximum(finite_positive, floor),
        jnp.ones_like(col_norms),
    )


@jax.jit
def compute_jacobian_column_scale(
        jacobian: Float[Array, "N_flat_obs N_param"],
        optical_weight_matrices: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
) -> Float[Array, "N_param"]:
    """Build automatic column-norm scales from a weighted Jacobian."""
    design = whiten_design_matrix(jacobian, optical_weight_matrices, radar_weights, inlier_mask)
    return _jacobian_column_scale(design)


@jax.jit
def compute_correction_norm(
        design: Float[Array, "N_flat_obs N_param"],
        residuals: Float[Array, "N_flat_obs"],
) -> Float[Array, ""]:
    """Measure the undamped Gauss--Newton correction in the normal metric."""
    solution = lx.linear_solve(
        lx.MatrixLinearOperator(design), -residuals,
        solver=lx.SVD(rcond=COVARIANCE_RCOND), throw=False,
    )
    norm = jnp.linalg.norm(design @ solution.value) / jnp.sqrt(design.shape[1])
    return jnp.where(solution.result == lx.RESULTS.successful, norm, jnp.inf)


@jax.jit
def compute_unweighted_rms(
        residuals: Float[Array, "N_flat_obs"],
        inlier_mask: Bool[Array, "N_flat_obs"],
) -> Float[Array, ""]:
    """Compute RMS of selected residual components in their native units."""
    used = jnp.where(inlier_mask, residuals, 0.0)
    return jnp.sqrt(jnp.sum(used * used) / jnp.sum(inlier_mask))


@jax.jit
def compute_normalized_residual_rms(
        residuals: Float[Array, "N_flat_obs"],
        optical_weights: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
) -> Float[Array, ""]:
    """Compute RMS of block-whitened residuals."""
    whitened = whiten_residuals(residuals, optical_weights, radar_weights, inlier_mask)
    return jnp.sqrt(jnp.sum(whitened * whitened) / jnp.sum(inlier_mask))


@jax.jit
def compute_prior_covariance(
        jacobian: Float[Array, "N_flat_obs N_param"],
        optical_weights: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
) -> PriorCovarianceResult:
    """Calculate the unscaled covariance matrix and rank diagnostics."""
    design = whiten_design_matrix(jacobian, optical_weights, radar_weights, inlier_mask)
    _, singular_values, vt = jnp.linalg.svd(design, full_matrices=False)
    zero = jnp.asarray(0.0, design.dtype)
    inf = jnp.asarray(jnp.inf, design.dtype)
    max_singular = jnp.max(jnp.concatenate([singular_values, zero[None]]))
    threshold_scale = jnp.maximum(
        jnp.asarray(COVARIANCE_RCOND, design.dtype),
        jnp.finfo(design.dtype).eps * max(design.shape),
    )
    valid_singular = singular_values > threshold_scale * max_singular
    rank = jnp.sum(valid_singular)
    safe_singular = jnp.where(valid_singular, singular_values, 1.0)
    inverse_square = jnp.where(valid_singular, 1.0 / (safe_singular * safe_singular), 0.0)
    covariance = (vt.T * inverse_square) @ vt
    min_valid = jnp.min(jnp.concatenate([jnp.where(valid_singular, singular_values, inf), inf[None]]))
    full_rank = rank == jacobian.shape[1]
    condition = jnp.where(full_rank & (rank > 0), max_singular / min_valid, inf)
    return PriorCovarianceResult(covariance, rank, condition, full_rank)


@jax.jit
def compute_post_cov_mat(
        cov_prior: Float[Array, "N_param N_param"],
        residuals: Float[Array, "N_flat_obs"],
        optical_weights: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
) -> Float[Array, "N_param N_param"]:
    """Scale the normal-matrix inverse by the posterior variance factor."""
    whitened = whiten_residuals(residuals, optical_weights, radar_weights, inlier_mask)
    dof = jnp.sum(inlier_mask) - cov_prior.shape[0]
    sigma0_sq = jnp.where(
        dof > 0,
        jnp.maximum(jnp.sum(whitened * whitened) / jnp.maximum(dof, 1), 1.0),
        1.0,
    )
    return cov_prior * sigma0_sq


@jax.jit
def prepare_linearization(
        params: Float[Array, "N_param"],
        jacobian: Float[Array, "N_flat_obs N_param"],
        residuals: Float[Array, "N_flat_obs"],
        optical_weights: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        mask: Bool[Array, "N_flat_obs"],
) -> Linearization:
    """Build the weighted and scaled model at one accepted point."""
    whitened = whiten_residuals(residuals, optical_weights, radar_weights, mask)
    physical_design = whiten_design_matrix(jacobian, optical_weights, radar_weights, mask)
    column_scale = _jacobian_column_scale(physical_design)
    diagonal = jnp.sum(physical_design * physical_design, axis=0) * column_scale * column_scale
    floor = jnp.max(diagonal) * jnp.finfo(params.dtype).eps
    safe_diagonal = jnp.maximum(diagonal, floor)
    safe_diagonal = jnp.where(safe_diagonal > 0.0, safe_diagonal, 1.0)
    step_scale = column_scale / jnp.sqrt(safe_diagonal)
    design = physical_design * step_scale
    rms = jnp.sqrt(jnp.sum(whitened * whitened) / jnp.sum(mask))
    finite = (
        jnp.all(jnp.isfinite(params))
        & jnp.all(jnp.isfinite(design))
        & jnp.all(jnp.isfinite(whitened))
        & jnp.all(jnp.isfinite(step_scale))
        & jnp.isfinite(rms)
    )
    correction_norm = jax.lax.cond(
        finite,
        lambda: compute_correction_norm(design, whitened),
        lambda: jnp.asarray(jnp.inf, params.dtype),
    )
    return Linearization(
        jacobian, residuals, optical_weights, radar_weights, design,
        whitened, step_scale, correction_norm, rms, finite,
    )


@eqx.filter_jit
def linearize_at(
        params: Float[Array, "N_param"],
        mask: Bool[Array, "N_flat_obs"],
        linearize: LinearizationFunction,
) -> Linearization:
    """Evaluate and prepare one accepted linearization point."""
    jacobian, residuals, optical, radar = evaluate_linearization(params, linearize)
    return prepare_linearization(params, jacobian, residuals, optical, radar, mask)


@jax.jit
def solve_damped_step(
        model: Linearization,
        damping: Float[Array, ""],
) -> tuple[Float[Array, "N_param"], Bool[Array, ""]]:
    """Solve one LM step by QR on the augmented least-squares system."""
    design = lx.MatrixLinearOperator(model.design)
    structure = design.in_structure()
    augmented_design = lx.FunctionLinearOperator(
        lambda value: (design.mv(value), jnp.sqrt(damping) * value),
        structure,
    )
    solution = lx.linear_solve(
        augmented_design,
        (model.whitened_residuals, jnp.zeros_like(model.step_scale)),
        solver=lx.QR(), throw=False,
    )
    scaled_delta = -solution.value
    linear_ok = (
        (solution.result == lx.RESULTS.successful)
        & jnp.all(jnp.isfinite(scaled_delta))
    )
    return scaled_delta, linear_ok


@jax.jit
def evaluate_trial(
        state: LMState,
        scaled_delta: Float[Array, "N_param"],
        linear_ok: Bool[Array, ""],
        trial_residuals: Float[Array, "N_flat_obs"],
        mask: Bool[Array, "N_flat_obs"],
) -> LMTrial:
    """Evaluate one candidate using weights frozen at the current point."""
    model = state.model
    candidate = state.params + model.step_scale * scaled_delta
    trial_whitened = whiten_residuals(
        trial_residuals, model.optical_weights, model.radar_weights, mask,
    )
    current_loss = 0.5 * jnp.sum(model.whitened_residuals * model.whitened_residuals)
    trial_loss = 0.5 * jnp.sum(trial_whitened * trial_whitened)
    linearized = model.whitened_residuals + model.design @ scaled_delta
    predicted_loss = 0.5 * jnp.sum(linearized * linearized)
    actual_change = trial_loss - current_loss
    predicted_change = predicted_loss - current_loss
    accepted = actual_change <= TRUST_REGION_LOW_CUTOFF * predicted_change
    good = (predicted_change < 0.0) & (actual_change < TRUST_REGION_HIGH_CUTOFF * predicted_change)
    finite = jnp.all(jnp.isfinite(candidate)) & jnp.all(jnp.isfinite(trial_whitened))
    accepted &= linear_ok & finite
    next_damping = jnp.where(
        good & accepted,
        state.damping / TRUST_REGION_GROW,
        jnp.where(accepted, state.damping, state.damping * TRUST_REGION_SHRINK),
    )
    rho = jnp.where(predicted_change < 0.0, actual_change / predicted_change, -jnp.inf)
    rms = jnp.sqrt(jnp.sum(trial_whitened * trial_whitened) / jnp.sum(mask))
    return LMTrial(candidate, trial_residuals, scaled_delta, rms, rho, accepted, next_damping, linear_ok)


def evaluate_lm_trial(
        state: LMState,
        mask: Bool[Array, "N_flat_obs"],
        linearize: LinearizationFunction,
) -> tuple[LMTrial, Linearization]:
    """Evaluate a trial and retain its candidate linearization for reuse."""
    scaled_delta, linear_ok = solve_damped_step(state.model, state.damping)
    candidate = state.params + state.model.step_scale * scaled_delta
    candidate_model = linearize_at(candidate, mask, linearize)
    trial = evaluate_trial(
        state, scaled_delta, linear_ok, candidate_model.residuals, mask,
    )
    return trial, candidate_model


@jax.jit
def update_lm_state(
        state: LMState,
        trial: LMTrial,
        next_model: Linearization,
        max_iter: int,
        max_damping_iter: int,
) -> LMState:
    """Apply a trial outcome and the established DiffOrb stop rules."""
    accepted = trial.accepted
    steps = state.steps + accepted.astype(jnp.int32)
    damping_trials = jnp.where(accepted, 0, state.damping_trials + 1)
    stalled = next_model.rms > state.model.rms * (1.0 - MIN_RMS_DECREASE)
    stagnant_steps = jnp.where(
        accepted,
        jnp.where(stalled, state.stagnant_steps + 1, 0),
        state.stagnant_steps,
    )
    rms_stop = accepted & (stagnant_steps >= STAGNATION_STEPS)
    rms_code = jnp.where(
        next_model.rms > 1.1 * state.model.rms,
        LSQTermination.rms_increasing,
        LSQTermination.rms_stagnated,
    )
    code = jnp.where(steps >= max_iter, LSQTermination.max_iter_reached, LSQTermination.running)
    code = jnp.where(damping_trials >= max_damping_iter, LSQTermination.damping_failed, code)
    code = jnp.where(rms_stop, rms_code, code)
    correction_converged = state.model.correction_norm < CORRECTION_TOL
    correction_converged |= (
        (steps >= max_iter)
        & (next_model.correction_norm < CORRECTION_TOL)
    )
    code = jnp.where(
        accepted & correction_converged,
        LSQTermination.correction_converged,
        code,
    )
    code = jnp.where(next_model.finite, code, LSQTermination.nonfinite_model)
    code = jnp.where(trial.linear_ok, code, LSQTermination.linear_solve_failed).astype(jnp.int32)
    params = jnp.where(accepted, trial.candidate, state.params)
    return LMState(
        params, next_model, trial.next_damping, steps, damping_trials,
        stagnant_steps, code,
    )


def initialize_lsq(
        params: Float[Array, "N_param"],
        model: Linearization,
        options: LMOptions,
) -> LMState:
    """Reset damping, counters, and termination for a prepared linearization."""
    code = jnp.where(model.finite, LSQTermination.running, LSQTermination.nonfinite_model)
    code = jnp.where(
        model.finite & (options.max_iter == 0),
        LSQTermination.max_iter_reached,
        code,
    ).astype(jnp.int32)
    zero = jnp.asarray(0, jnp.int32)
    return LMState(
        params, model, jnp.asarray(options.damping_init, params.dtype),
        zero, zero, zero, code,
    )


@jax.jit
def initialize_refit(
        result: LeastSquaresResult,
        mask: Bool[Array, "N_flat_obs"],
        options: LMOptions,
) -> LMState:
    """Initialize a new mask at the previous fit's final parameters.

    Notes
    -----
    Only the inlier mask may change. The residuals, Jacobian, and dynamic weights belong to the unchanged parameter vector and measurement model. Weighted quantities and all iteration controls are rebuilt for the new fit.
    """
    model = prepare_linearization(
        result.params, result.jacobian, result.residuals,
        result.optical_weight_matrices, result.radar_weights, mask,
    )
    return initialize_lsq(result.params, model, options)


def advance_lsq(
        state: LMState,
        mask: Bool[Array, "N_flat_obs"],
        linearize: LinearizationFunction,
        options: LMOptions,
) -> LMState:
    """Evaluate one damping trial for the compiled driver."""
    trial, candidate_model = evaluate_lm_trial(state, mask, linearize)
    next_model = jax.lax.cond(
        trial.accepted,
        lambda: candidate_model,
        lambda: state.model,
    )
    return update_lm_state(
        state, trial, next_model, options.max_iter, options.max_damping_iter,
    )


@jax.jit
def finish_lsq(
        state: LMState,
        mask: Bool[Array, "N_flat_obs"],
) -> LeastSquaresResult:
    """Build covariance diagnostics and the array result."""
    code = jnp.where(
        state.termination_code == LSQTermination.running,
        LSQTermination.max_iter_reached,
        state.termination_code,
    ).astype(jnp.int32)

    def covariance() -> tuple[Array, Array, Array, Array, Array]:
        prior = compute_prior_covariance(
            state.model.jacobian, state.model.optical_weights,
            state.model.radar_weights, mask,
        )
        post = compute_post_cov_mat(
            prior.cov_mat, state.model.residuals, state.model.optical_weights,
            state.model.radar_weights, mask,
        )
        return prior.cov_mat, post, prior.rank, prior.condition, prior.valid

    def unavailable() -> tuple[Array, Array, Array, Array, Array]:
        matrix = jnp.full((state.params.size, state.params.size), jnp.nan, state.params.dtype)
        return matrix, matrix, jnp.asarray(0, jnp.int64), jnp.asarray(jnp.inf, state.params.dtype), jnp.asarray(False)

    prior, post, rank, condition, valid = jax.lax.cond(state.model.finite, covariance, unavailable)
    converged = (code == LSQTermination.correction_converged) | (code == LSQTermination.rms_stagnated)
    return LeastSquaresResult(
        state.params, state.model.jacobian, state.model.residuals, state.model.rms,
        state.model.optical_weights, state.model.radar_weights, prior, post,
        rank, condition, valid, converged, code, state.steps,
    )


@eqx.filter_jit
def solve_lsq(
        initial: LMState,
        mask: Bool[Array, "N_flat_obs"],
        linearize: LinearizationFunction,
        options: LMOptions,
) -> LeastSquaresResult:
    """Run the compiled LM loop from an initialized state."""
    final = jax.lax.while_loop(
        lambda state: state.termination_code == LSQTermination.running,
        lambda state: advance_lsq(state, mask, linearize, options),
        initial,
    )
    return finish_lsq(final, mask)


def solve_lsq_python(
        state: LMState,
        mask: Bool[Array, "N_flat_obs"],
        linearize: LinearizationFunction,
        options: LMOptions,
        *,
        step_callback: Callable[..., None] | None = None,
) -> LeastSquaresResult:
    """Run the host LM loop from an initialized state."""
    while int(state.termination_code) == LSQTermination.running:
        previous = state
        trial, candidate_model = evaluate_lm_trial(state, mask, linearize)
        next_model = (
            candidate_model
            if bool(trial.accepted) else state.model
        )
        state = update_lm_state(
            state, trial, next_model, options.max_iter, options.max_damping_iter,
        )
        if step_callback is not None and bool(trial.accepted):
            step_callback(
                "least_squares_step",
                step=int(previous.steps) + 1,
                normalized_residual_rms=float(next_model.rms),
                damping=float(previous.damping),
                next_damping=float(trial.next_damping),
            )
    return finish_lsq(state, mask)


def valid_chi2(result: LeastSquaresResult) -> Bool[Array, ""]:
    """Return whether a fit can support covariance-based rejection."""
    return (
        result.cov_valid
        & jnp.isfinite(result.normalized_residual_rms)
        & jnp.all(jnp.isfinite(result.residuals))
        & jnp.all(jnp.isfinite(result.jacobian))
        & jnp.all(jnp.isfinite(result.cov_mat_prior))
        & jnp.isfinite(result.cov_condition)
    )


def initialize_robust(
        result: LeastSquaresResult,
        mask: Bool[Array, "N_flat_obs"],
) -> RobustState:
    """Initialize robust rejection from the first fixed-mask fit."""
    return RobustState(
        result,
        mask,
        valid_chi2(result),
        jnp.asarray(False),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(1, jnp.int32),
        result.iter_num,
    )


def evaluate_rejection(
        state: RobustState,
        policy: CompiledOutlierPolicy,
) -> RejResult:
    """Evaluate one outlier policy at the current robust state."""
    return policy.apply(
        state.result.residuals,
        state.result.optical_weight_matrices,
        state.result.radar_weights,
        state.result.jacobian,
        state.result.cov_mat_prior,
        state.mask,
    )


@jax.jit
def advance_robust(
        state: RobustState,
        rejection: RejResult,
        next_result: LeastSquaresResult,
) -> RobustState:
    """Advance robust state after one rejection evaluation and optional refit."""
    new_mask = rejection.flat_inlier_mask
    changed = jnp.any(new_mask != state.mask)
    return RobustState(
        next_result,
        new_mask,
        valid_chi2(next_result),
        ~changed,
        state.iterations + 1,
        state.solves + changed.astype(jnp.int32),
        state.total_steps + jnp.where(changed, next_result.iter_num, 0),
    )


def continue_robust(
        state: RobustState,
        policy: CompiledOutlierPolicy,
) -> Bool[Array, ""]:
    """Return whether another rejection pass is required."""
    return (
        policy.enable_auto_rejection
        & state.valid
        & ~state.done
        & (state.iterations < policy.max_iters)
    )


@eqx.filter_jit
def finish_robust(
        state: RobustState,
        policy: CompiledOutlierPolicy,
        dtype: DTypeLike,
) -> RobustResult:
    """Build the final array-only robust result."""
    metric = jax.lax.cond(
        state.valid,
        lambda: evaluate_rejection(state, policy).metric,
        lambda: jnp.full((policy.n_2d + policy.n_1d,), jnp.nan, dtype),
    )
    return RobustResult(
        state.result,
        RejResult(state.mask, metric),
        state.iterations,
        state.total_steps,
    )


@eqx.filter_jit
def solve_robust(
        params: Float[Array, "N_param"],
        policy: CompiledOutlierPolicy,
        linearize: LinearizationFunction,
        options: LMOptions,
) -> RobustResult:
    """Drive shared robust state transitions with JAX control flow."""
    mask = policy.get_init_mask()
    model = linearize_at(params, mask, linearize)
    result = solve_lsq(initialize_lsq(params, model, options), mask, linearize, options)
    initial = initialize_robust(result, mask)

    def step(state: RobustState) -> RobustState:
        rejection = evaluate_rejection(state, policy)
        changed = jnp.any(rejection.flat_inlier_mask != state.mask)
        next_result = jax.lax.cond(
            changed,
            lambda: solve_lsq(
                initialize_refit(state.result, rejection.flat_inlier_mask, options),
                rejection.flat_inlier_mask,
                linearize,
                options,
            ),
            lambda: state.result,
        )
        return advance_robust(state, rejection, next_result)

    final = jax.lax.while_loop(
        lambda state: continue_robust(state, policy),
        step,
        initial,
    )
    return finish_robust(final, policy, params.dtype)
