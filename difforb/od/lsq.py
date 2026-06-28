"""Least-squares solvers for orbit determination and robust outlier rejection."""

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Bool, Int
from typing import Any, Callable, NamedTuple

from difforb.astrometry.weight import WeightResult
from difforb.core.constants import DAY_S
from difforb.od.events import SolverEventHandler, SolverEventLogger, SolverLogDetail, make_solver_event_logger
from difforb.od.outlier.outlier import RejResult
from difforb.od.outlier.policy import CompiledOutlierPolicy

jax.config.update("jax_enable_x64", True)

COVARIANCE_RCOND = 1e-12
STATE_PARAM_COUNT = 6
STATE_PARAM_SCALE_RCOND = 1e-12
WeightArrayFunction = Callable[[Float[Array, "N_param"]], tuple[Float[Array, "N_optical 2 2"], Float[Array, "N_radar"]]]


def _scalar_float(value: Any) -> float:
    """Return one Python float from a scalar-like value."""
    return float(jnp.asarray(value).reshape(-1)[0])


def _scalar_int(value: Any) -> int:
    """Return one Python integer from a scalar-like value."""
    return int(jnp.asarray(value).reshape(-1)[0])


def _scalar_bool(value: Any) -> bool:
    """Return one Python boolean from a scalar-like value."""
    return bool(jnp.asarray(value).reshape(-1)[0])


def _finite_array(value: Any) -> bool:
    """Return whether every entry in an array-like value is finite."""
    return _scalar_bool(jnp.all(jnp.isfinite(value)))


def _weight_result_arrays(weight_result: WeightResult) -> tuple[Float[Array, "N_optical 2 2"], Float[Array, "N_radar"]]:
    """Return JAX weight arrays from one resolved weight result."""
    return (
        jnp.asarray(weight_result.optical_weight_matrices),
        jnp.asarray(weight_result.radar_weights),
    )


def _resolve_weight_arrays(
        weight_result: WeightResult,
        weight_array_func: WeightArrayFunction | None,
        params: Float[Array, "N_param"],
) -> tuple[Float[Array, "N_optical 2 2"], Float[Array, "N_radar"]]:
    """Return fixed or parameter-refreshed weight arrays for one linearization point."""
    if weight_array_func is None:
        optical_weight_matrices, radar_weights = _weight_result_arrays(weight_result)
    else:
        optical_weight_matrices, radar_weights = weight_array_func(params)
    return jax.lax.stop_gradient(jnp.asarray(optical_weight_matrices)), jax.lax.stop_gradient(jnp.asarray(radar_weights))


@jax.jit
def build_time_inflated_optical_weight_matrices(
        base_optical_covariances: Float[Array, "N_optical 2 2"],
        optical_time_uncertainties: Float[Array, "N_optical"],
        optical_rates: Float[Array, "N_optical 2"],
) -> Float[Array, "N_optical 2 2"]:
    """Return inverse covariance blocks with optical time uncertainty folded in.

    Parameters
    ----------
    base_optical_covariances : Array, shape (N_optical, 2, 2)
        Optical covariance matrices before time-uncertainty inflation, in
        radians squared.
    optical_time_uncertainties : Array, shape (N_optical,)
        Observation-time uncertainties in seconds. Non-finite and zero entries
        leave the corresponding covariance unchanged.
    optical_rates : Array, shape (N_optical, 2)
        Tangent-plane angular rates in radians per day, ordered as
        ``(ra_dot_cos_dec, dec_dot)``.

    Returns
    -------
    Array, shape (N_optical, 2, 2)
        Inverse covariance blocks for the current optical rates.
    """
    finite_mask = jnp.isfinite(optical_time_uncertainties) & (optical_time_uncertainties != 0.0)
    time_days = jnp.where(finite_mask, optical_time_uncertainties / jnp.asarray(DAY_S, dtype=optical_rates.dtype), 0.0)
    covariances = base_optical_covariances + jnp.einsum("ni,nj,n->nij", optical_rates, optical_rates, time_days * time_days)
    return jnp.linalg.inv(covariances)


def _lsq_result_has_valid_chi2_inputs(result: "LeastSquaresResult") -> bool:
    """Return whether covariance-based Chi2 values can be evaluated."""
    return (
            _scalar_bool(result.cov_valid)
            and _finite_array(result.normalized_residual_rms)
            and _finite_array(result.residuals)
            and _finite_array(result.jacobian)
            and _finite_array(result.cov_mat_prior)
            and _finite_array(result.cov_condition)
    )


def _flat_inlier_mask_to_observation_mask(flat_inlier_mask: Bool[Array, "N_obs"], n_2d: int) -> Bool[Array, "N_obs"]:
    """Convert a flat residual mask to one mask value per observation."""
    n_2d = int(n_2d)
    optical_flat = flat_inlier_mask[:2 * n_2d]
    optical_mask = optical_flat.reshape((n_2d, 2)).all(axis=1)
    scalar_mask = flat_inlier_mask[2 * n_2d:]
    return jnp.concatenate([optical_mask, scalar_mask])


@jax.jit
def compute_relative_scaled_step(delta_param: Float[Array, "N_param"], cur_param: Float[Array, "N_param"],
                                 param_scale: Float[Array, "N_param"]) -> Float[Array, ""]:
    """
    Calculate the relative step norm in scaled parameter space.

    Parameters
    ----------
    delta_param : Float[Array, "N_param"]
        Accepted physical parameter increment.
    cur_param : Float[Array, "N_param"]
        Parameter vector before applying the accepted increment.
    param_scale : Float[Array, "N_param"]
        Multiplicative parameter scales used by the current linearized solve.

    Returns
    -------
    Float[Array, ""]
        Dimensionless relative step norm.
    """
    safe_scale = jnp.where(jnp.isfinite(param_scale) & (param_scale > 0.), param_scale, 1.)
    scaled_delta_norm = jnp.linalg.norm(delta_param / safe_scale)
    scaled_param_norm = jnp.linalg.norm(cur_param / safe_scale)
    return scaled_delta_norm / (1. + scaled_param_norm)


def _optical_observation_mask(inlier_mask: Bool[Array, "N_flat_obs"], n_2d: int) -> Bool[Array, "N_optical"]:
    """Return one mask value per optical residual pair."""
    n_flat_2d = 2 * n_2d
    return jnp.all(inlier_mask[:n_flat_2d].reshape((n_2d, 2)), axis=1)


@jax.jit
def _weight_cholesky_factor(optical_weight_matrices: Float[Array, "N_optical 2 2"]) -> Float[Array, "N_optical 2 2"]:
    """Return factors ``S`` such that ``S.T @ S`` applies each weight matrix."""
    return jnp.swapaxes(jnp.linalg.cholesky(optical_weight_matrices), -1, -2)


@jax.jit
def whiten_residuals(
        residuals: Float[Array, "N_flat_obs"],
        optical_weight_matrices: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
) -> Float[Array, "N_flat_obs"]:
    """Apply the square-root optical weight matrices to one residual vector."""
    n_2d = optical_weight_matrices.shape[0]
    n_flat_2d = 2 * n_2d
    optical_residuals = residuals[:n_flat_2d].reshape((n_2d, 2))
    radar_residuals = residuals[n_flat_2d:]

    optical_mask = _optical_observation_mask(inlier_mask, n_2d)
    optical_factor = _weight_cholesky_factor(optical_weight_matrices)
    optical_factor = optical_factor * optical_mask[:, None, None]
    optical_whitened = jnp.einsum("nij,nj->ni", optical_factor, optical_residuals).reshape(-1)

    radar_mask = inlier_mask[n_flat_2d:]
    radar_sqrt_weights = jnp.sqrt(jnp.where(radar_mask, radar_weights, 0.))
    radar_whitened = radar_residuals * radar_sqrt_weights
    return jnp.concatenate([optical_whitened, radar_whitened], axis=0)


@jax.jit
def whiten_design_matrix(
        A: Float[Array, "N_flat_obs N_param"],
        optical_weight_matrices: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
) -> Float[Array, "N_flat_obs N_param"]:
    """Apply the square-root optical weight matrices to one Jacobian matrix."""
    n_2d = optical_weight_matrices.shape[0]
    n_flat_2d = 2 * n_2d
    optical_jacobian = A[:n_flat_2d].reshape((n_2d, 2, A.shape[-1]))
    radar_jacobian = A[n_flat_2d:]

    optical_mask = _optical_observation_mask(inlier_mask, n_2d)
    optical_factor = _weight_cholesky_factor(optical_weight_matrices)
    optical_factor = optical_factor * optical_mask[:, None, None]
    optical_whitened = jnp.einsum("nij,njk->nik", optical_factor, optical_jacobian).reshape((-1, A.shape[-1]))

    radar_mask = inlier_mask[n_flat_2d:]
    radar_sqrt_weights = jnp.sqrt(jnp.where(radar_mask, radar_weights, 0.))
    radar_whitened = radar_jacobian * radar_sqrt_weights[:, None]
    return jnp.concatenate([optical_whitened, radar_whitened], axis=0)


@jax.jit
def apply_weight_matrix_to_vector(
        residuals: Float[Array, "N_flat_obs"],
        optical_weight_matrices: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
) -> Float[Array, "N_flat_obs"]:
    """Apply the optical weight matrices to one residual-like vector."""
    n_2d = optical_weight_matrices.shape[0]
    n_flat_2d = 2 * n_2d
    optical_residuals = residuals[:n_flat_2d].reshape((n_2d, 2))
    radar_residuals = residuals[n_flat_2d:]

    optical_mask = _optical_observation_mask(inlier_mask, n_2d)
    masked_optical_weight_matrices = optical_weight_matrices * optical_mask[:, None, None]
    optical_weighted = jnp.einsum("nij,nj->ni", masked_optical_weight_matrices, optical_residuals).reshape(-1)

    radar_mask = inlier_mask[n_flat_2d:]
    radar_weighted = radar_residuals * jnp.where(radar_mask, radar_weights, 0.)
    return jnp.concatenate([optical_weighted, radar_weighted], axis=0)


@jax.jit
def compute_scaled_gradient_inf(
        A: Float[Array, "N_flat_obs N_param"],
        residuals: Float[Array, "N_flat_obs"],
        optical_weight_matrices: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
        param_scale: Float[Array, "N_param"],
) -> Float[Array, ""]:
    """Calculate the scaled weighted-gradient norm for block weights."""
    weighted_residuals = apply_weight_matrix_to_vector(residuals, optical_weight_matrices, radar_weights, inlier_mask)
    gradient = A.T @ weighted_residuals
    return jnp.max(jnp.abs(param_scale * gradient))


@jax.jit
def compute_state_param_scale(
        A: Float[Array, "N_flat_obs N_param"],
        optical_weight_matrices: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
) -> Float[Array, "N_param"]:
    """Build automatic column-norm scales from the whitened Jacobian."""
    whitened_A = whiten_design_matrix(A, optical_weight_matrices, radar_weights, inlier_mask)
    col_norms = jnp.sqrt(jnp.sum(whitened_A * whitened_A, axis=0))
    param_scale = jnp.ones_like(col_norms)

    state_count = min(STATE_PARAM_COUNT, A.shape[1])
    state_norms = col_norms[:state_count]
    finite_positive_norms = jnp.where(jnp.isfinite(state_norms) & (state_norms > 0.), state_norms, 0.)
    max_state_norm = jnp.max(finite_positive_norms)
    relative_floor = max_state_norm * jnp.maximum(
        jnp.asarray(STATE_PARAM_SCALE_RCOND, dtype=A.dtype),
        jnp.finfo(A.dtype).eps,
    )
    safe_state_norms = jnp.maximum(finite_positive_norms, relative_floor)
    state_scale = jnp.where(max_state_norm > 0., 1. / safe_state_norms, jnp.ones_like(state_norms))
    return param_scale.at[:state_count].set(state_scale)


@jax.jit
def compute_lsq_param_scale(
        A: Float[Array, "N_flat_obs N_param"],
        optical_weight_matrices: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
        base_param_scale: Float[Array, "N_param"],
) -> Float[Array, "N_param"]:
    """Combine state column-norm scales with model-provided scales."""
    safe_base_scale = jnp.where(jnp.isfinite(base_param_scale) & (base_param_scale > 0.), base_param_scale, 1.)
    state_param_scale = compute_state_param_scale(A, optical_weight_matrices, radar_weights, inlier_mask)
    state_count = min(STATE_PARAM_COUNT, A.shape[1])
    return safe_base_scale.at[:state_count].set(state_param_scale[:state_count])


@jax.jit
def solve_normal_equation(
        A: Float[Array, "N_flat_obs N_param"],
        b: Float[Array, "N_flat_obs"],
        optical_weight_matrices: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
        damping: Float[Array, ""],
        damping_diag: Float[Array, "N_param"],
        param_scale: Float[Array, "N_param"],
) -> Float[Array, "N_param"]:
    """Solve one damped least-squares step with block weight matrices."""
    scaled_A = A * param_scale[None, :]
    A_tilde = whiten_design_matrix(scaled_A, optical_weight_matrices, radar_weights, inlier_mask)
    b_tilde = whiten_residuals(b, optical_weight_matrices, radar_weights, inlier_mask)
    scaled_damping_diag = damping_diag * param_scale * param_scale
    finite_positive_diag = jnp.where(
        jnp.isfinite(scaled_damping_diag) & (scaled_damping_diag > 0.),
        scaled_damping_diag,
        0.,
    )
    max_damping_diag = jnp.max(finite_positive_diag)
    relative_floor = max_damping_diag * jnp.finfo(scaled_damping_diag.dtype).eps
    safe_damping_diag = jnp.where(finite_positive_diag > 0., finite_positive_diag, relative_floor)
    safe_damping_diag = jnp.where(max_damping_diag > 0., jnp.maximum(safe_damping_diag, relative_floor),
                                  jnp.zeros_like(scaled_damping_diag))
    damped_diag = jnp.sqrt(damping * safe_damping_diag)
    A_tilde = jnp.concatenate([A_tilde, jnp.diag(damped_diag)], axis=0)
    b_tilde = jnp.concatenate([b_tilde, jnp.zeros_like(damped_diag)], axis=0)
    x, residuals, rank, s = jnp.linalg.lstsq(A_tilde, b_tilde, rcond=1e-15)
    return x * param_scale


@jax.jit
def compute_weighted_lsq_loss(
        residuals: Float[Array, "N_flat_obs"],
        optical_weight_matrices: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
) -> Float[Array, ""]:
    """Calculate the weighted least-squares loss for block weights."""
    whitened_residuals = whiten_residuals(residuals, optical_weight_matrices, radar_weights, inlier_mask)
    return 0.5 * jnp.sum(whitened_residuals * whitened_residuals)


@jax.jit
def compute_unweighted_rms(residuals: Float[Array, "N"],
                           inlier_mask: Bool[Array, "N"]) -> Float[Array, ""]:
    used_residuals = jnp.where(inlier_mask, residuals, 0.)
    rms = jnp.sqrt(jnp.sum(used_residuals * used_residuals) / jnp.sum(
        inlier_mask))
    return rms


@jax.jit
def compute_normalized_residual_rms(
        residuals: Float[Array, "N_flat_obs"],
        optical_weight_matrices: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
) -> Float[Array, ""]:
    """Compute the RMS of block-whitened residuals."""
    normalized_residuals = whiten_residuals(residuals, optical_weight_matrices, radar_weights, inlier_mask)
    rms = jnp.sqrt(jnp.sum(normalized_residuals * normalized_residuals) / jnp.sum(inlier_mask))
    return rms


class PriorCovarianceResult(NamedTuple):
    """Unscaled covariance matrix and rank diagnostics for one weighted Jacobian."""
    cov_mat: Float[Array, "N_param N_param"]
    rank: Int[Array, ""]
    condition: Float[Array, ""]
    valid: Bool[Array, ""]


@jax.jit
def compute_prior_covariance(
        A: Float[Array, "N_flat_obs N_param"],
        optical_weight_matrices: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
) -> PriorCovarianceResult:
    """Calculate the unscaled covariance matrix for block weight matrices."""
    A_tilde = whiten_design_matrix(A, optical_weight_matrices, radar_weights, inlier_mask)

    _, singular_values, vt = jnp.linalg.svd(A_tilde, full_matrices=False)
    zero = jnp.asarray(0., dtype=A_tilde.dtype)
    inf = jnp.asarray(jnp.inf, dtype=A_tilde.dtype)
    max_singular = jnp.max(jnp.concatenate([singular_values, zero[None]]))
    threshold_scale = jnp.maximum(jnp.asarray(COVARIANCE_RCOND, dtype=A_tilde.dtype),
                                  jnp.finfo(A_tilde.dtype).eps * max(A_tilde.shape))
    cutoff = threshold_scale * max_singular
    valid_singular = singular_values > cutoff
    rank = jnp.sum(valid_singular)

    safe_singular = jnp.where(valid_singular, singular_values, 1.)
    inv_singular_sq = jnp.where(valid_singular, 1. / (safe_singular * safe_singular), 0.)
    cov_mat = (vt.T * inv_singular_sq) @ vt

    min_valid_singular = jnp.min(jnp.concatenate([jnp.where(valid_singular, singular_values, inf), inf[None]]))
    full_rank = rank == A.shape[1]
    condition = jnp.where(full_rank & (rank > 0), max_singular / min_valid_singular, inf)
    return PriorCovarianceResult(cov_mat=cov_mat, rank=rank, condition=condition, valid=full_rank)


@jax.jit
def compute_post_cov_mat(
        cov_prior: Float[Array, "N_param N_param"],
        residuals: Float[Array, "N_flat_obs"],
        optical_weight_matrices: Float[Array, "N_optical 2 2"],
        radar_weights: Float[Array, "N_radar"],
        inlier_mask: Bool[Array, "N_flat_obs"],
) -> Float[Array, "N_param N_param"]:
    """Scale the normal-matrix inverse by the posterior variance factor."""
    n_params = cov_prior.shape[0]
    normalized_residuals = whiten_residuals(residuals, optical_weight_matrices, radar_weights, inlier_mask)
    chi2 = jnp.sum(normalized_residuals * normalized_residuals)
    n_obs = jnp.sum(inlier_mask)
    dof = n_obs - n_params
    safe_dof = jnp.where(dof > 0, dof, 1)
    sigma0_sq_raw = chi2 / safe_dof
    sigma0_sq = jnp.where(dof > 0, jnp.maximum(sigma0_sq_raw, 1.0), 1.0)
    return cov_prior * sigma0_sq


class LeastSquaresResult(NamedTuple):
    """
    Intermediate states in the least squares iterative solution.

    ``normalized_residual_rms`` is the RMS of normalized residuals. It is
    dimensionless.
    """
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
    converged: bool
    termination_reason: str
    iter_num: int


class LeastSquares:
    """
    Levenberg-Marquardt least-squares solver for orbit determination.

    The solver linearizes the residual model at the current parameter vector,
    solves a damped weighted least-squares step, and accepts trial steps with
    a positive trust-region ratio for the current inlier set. The damping
    factor is updated with a Nielsen-style rule driven by that ratio.
    Convergence is tested in scaled parameter space so mixed state and
    force-model parameters do not share one implicit physical unit.

    If a dynamic weight function is supplied, weights are refreshed once at
    each linearization point and held fixed during that damping trial search.
    The refreshed weights are treated as constants through
    :func:`jax.lax.stop_gradient`, so the solver follows an iteratively
    reweighted least-squares convention rather than differentiating the weight
    model with respect to the estimated parameters.

    References
    ----------
    Oliver Montenbruck and Eberhard Gill, *Satellite Orbits: Models, Methods
    and Applications*, Section 8.1.
    """

    _DAMPING_INIT = 1e-3
    _NU_INIT = 2.0
    _NU_MAX = 8.0
    _MAX_DAMPING_ITER = 10

    def __init__(self, tol: float = 1e-11, max_iter: int = 20):
        """
        Initialize the least-squares solver.

        Parameters
        ----------
        tol : float, default=1e-11
            Base convergence threshold. The solver uses it directly for the
            relative scaled step test and uses its square root for the scaled
            weighted-gradient test.
        max_iter : int, default=20
            Maximum number of accepted outer iterations.

        Notes
        -----
        Levenberg-Marquardt damping controls and the derived step and gradient
        thresholds are fixed internal defaults. They are intentionally not part
        of the public solver interface.
        """
        self.tol = tol
        self.max_iter = max_iter
        self.damping_init = self._DAMPING_INIT
        self.nu_init = self._NU_INIT
        self.nu_max = self._NU_MAX
        self.max_damping_iter = self._MAX_DAMPING_ITER
        self.step_tol = tol
        self.gradient_tol = tol ** 0.5

    def solve(self, init_params: Float[Array, "N_param"],
              weights: WeightResult, inlier_mask: Bool[Array, "N_obs"],
              res_func: Callable, res_jac_func: Callable,
              param_scale: Float[Array, "N_param"] | None = None,
              weight_array_func: WeightArrayFunction | None = None,
              event_handler: SolverEventHandler | None = None,
              log_detail: SolverLogDetail = "iter",
              event_logger: SolverEventLogger | None = None) -> 'LeastSquaresResult':
        """
        Solve the nonlinear weighted least-squares problem.

        Parameters
        ----------
        init_params : Float[Array, "N_param"]
            Initial parameter vector used as the first linearization point.
        weights : WeightResult
            Resolved observation weights. Optical rows are represented by full
            2-by-2 weight matrices and radar rows by scalar inverse
            variances.
        inlier_mask : Bool[Array, "N_obs"]
            Flat inlier mask held fixed during this inner solve.
        res_func : Callable
            Callable returning residuals for one parameter vector.
        res_jac_func : Callable
            Callable returning ``(jacobian, residuals)`` for one parameter
            vector.
        param_scale : Float[Array, "N_param"] or None, optional
            Characteristic scales for all parameters. If omitted, all
            non-state parameters use unit scale; the first six state parameters
            still use automatic weighted Jacobian column-norm scaling.
        weight_array_func : callable or None, optional
            Optional function that returns ``(optical_weight_matrices,
            radar_weights)`` for the current parameter vector. Returned weights
            are refreshed once per linearization point and held fixed through
            the damping trial search.
        event_handler : SolverEventHandler or None, optional
            Optional callback that receives structured progress events. The
            solver is quiet when this is omitted.
        log_detail : {"quiet", "summary", "iter", "trial"}, default="iter"
            Minimum solver-log detail emitted to ``event_handler``.
        event_logger : SolverEventLogger or None, optional
            Context-aware structured event logger. When supplied, it owns
            filtering and context.

        Returns
        -------
        LeastSquaresResult
            Final parameter vector, residuals, Jacobian, covariance matrices,
            and iteration count at the accepted solution.
        """
        logger = event_logger if event_logger is not None else make_solver_event_logger(event_handler, log_detail)
        cur_param = init_params
        base_param_scale = jnp.ones_like(init_params) if param_scale is None else jnp.asarray(param_scale)
        cur_damping = self.damping_init
        cur_nu = self.nu_init
        iter_num = 0
        termination_reason = "max_iter_reached"
        last_relative_step_norm = jnp.asarray(jnp.inf, dtype=init_params.dtype)
        for i in range(self.max_iter):
            # 1. Calculate residuals and Jacobian matrix
            jac, residuals = res_jac_func(cur_param)
            optical_weight_matrices, radar_weights = _resolve_weight_arrays(weights, weight_array_func, cur_param)
            cur_rms = compute_normalized_residual_rms(residuals, optical_weight_matrices, radar_weights, inlier_mask)
            if i == 0 and logger.enabled("lsq_start"):
                n_inlier_residuals = _scalar_int(jnp.sum(inlier_mask))
                n_params = int(init_params.shape[0])
                logger.emit(
                    "least_squares",
                    "lsq_start",
                    "info",
                    normalized_residual_rms=_scalar_float(cur_rms),
                    damping=_scalar_float(cur_damping),
                    n_inlier_residuals=n_inlier_residuals,
                    n_params=n_params,
                )
            whitened_jac = whiten_design_matrix(jac, optical_weight_matrices, radar_weights, inlier_mask)
            damping_diag = jnp.sum(whitened_jac * whitened_jac, axis=0)
            step_param_scale = compute_lsq_param_scale(jac, optical_weight_matrices, radar_weights, inlier_mask, base_param_scale)
            cur_loss = compute_weighted_lsq_loss(residuals, optical_weight_matrices, radar_weights, inlier_mask)
            scaled_gradient_inf = compute_scaled_gradient_inf(jac, residuals, optical_weight_matrices, radar_weights, inlier_mask,
                                                              step_param_scale)
            if scaled_gradient_inf <= self.gradient_tol:
                termination_reason = "gradient_converged"
                break

            # 2. Search one accepted damped step for the current linearization.
            accepted = False
            trial_damping = cur_damping
            trial_nu = cur_nu
            for damping_iter in range(self.max_damping_iter):
                trial_index = damping_iter + 1
                trial_logger = logger.bind(lsq_step=i + 1, trial=trial_index)
                if trial_logger.enabled("lm_trial_start"):
                    trial_logger.emit(
                        "least_squares",
                        "lm_trial_start",
                        "info",
                        damping=_scalar_float(trial_damping),
                    )
                delta_param = solve_normal_equation(jac, -residuals, optical_weight_matrices, radar_weights, inlier_mask, trial_damping,
                                                    damping_diag, step_param_scale)
                new_param = cur_param + delta_param
                new_residuals = res_func(new_param)
                new_rms = compute_normalized_residual_rms(new_residuals, optical_weight_matrices, radar_weights, inlier_mask)
                new_loss = compute_weighted_lsq_loss(new_residuals, optical_weight_matrices, radar_weights, inlier_mask)
                jac_delta = jac @ delta_param
                weighted_residuals = apply_weight_matrix_to_vector(residuals, optical_weight_matrices, radar_weights, inlier_mask)
                weighted_jac_delta = apply_weight_matrix_to_vector(jac_delta, optical_weight_matrices, radar_weights, inlier_mask)
                predicted_reduction = -jnp.dot(weighted_residuals, jac_delta) - 0.5 * jnp.dot(
                    weighted_jac_delta, jac_delta)
                actual_reduction = cur_loss - new_loss
                safe_predicted_reduction = jnp.where(predicted_reduction > 0., predicted_reduction, 1.0)
                rho = jnp.where(predicted_reduction > 0., actual_reduction / safe_predicted_reduction, -jnp.inf)
                if rho > 0.:
                    relative_step_norm = compute_relative_scaled_step(delta_param, cur_param, step_param_scale)
                    cur_param = new_param
                    damping_scale = jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                    accepted_damping = trial_damping
                    cur_damping = jnp.maximum(trial_damping * damping_scale, 1e-9)
                    cur_nu = self.nu_init
                    last_relative_step_norm = relative_step_norm
                    accepted = True
                    next_iter_num = i + 1
                    step_logger = logger.bind(lsq_step=next_iter_num)
                    if step_logger.enabled("lsq_step_accepted"):
                        step_logger.emit(
                            "least_squares",
                            "lsq_step_accepted",
                            "info",
                            normalized_residual_rms_before=_scalar_float(cur_rms),
                            normalized_residual_rms_after=_scalar_float(new_rms),
                            rho=_scalar_float(rho),
                            damping=_scalar_float(accepted_damping),
                            next_damping=_scalar_float(cur_damping),
                            damping_trials=trial_index,
                        )
                    break
                if trial_logger.enabled("lm_trial_rejected"):
                    trial_logger.emit(
                        "least_squares",
                        "lm_trial_rejected",
                        "info",
                        normalized_residual_rms=_scalar_float(new_rms),
                        rho=_scalar_float(rho),
                        damping=_scalar_float(trial_damping),
                    )
                trial_damping = trial_damping * trial_nu
                trial_nu = jnp.minimum(2.0 * trial_nu, self.nu_max)

            if not accepted:
                termination_reason = "damping_failed"
                failed_logger = logger.bind(lsq_step=i + 1)
                if failed_logger.enabled("lsq_failed"):
                    failed_logger.emit(
                        "least_squares",
                        "lsq_failed",
                        "warning",
                        reason=termination_reason,
                        normalized_residual_rms=_scalar_float(cur_rms),
                        damping_trials=self.max_damping_iter,
                    )
                break

            iter_num = i + 1
            if last_relative_step_norm <= self.step_tol:
                termination_reason = "step_converged"
                break
        else:
            iter_num = self.max_iter

        jac, residuals = res_jac_func(cur_param)
        optical_weight_matrices, radar_weights = _resolve_weight_arrays(weights, weight_array_func, cur_param)
        rms = compute_normalized_residual_rms(residuals, optical_weight_matrices, radar_weights, inlier_mask)
        converged = termination_reason in ("gradient_converged", "step_converged")
        cov_result = compute_prior_covariance(jac, optical_weight_matrices, radar_weights, inlier_mask)
        cov_mat_prior = cov_result.cov_mat
        cov_mat_post = compute_post_cov_mat(cov_mat_prior, residuals, optical_weight_matrices, radar_weights, inlier_mask)

        if logger.enabled("lsq_done"):
            cov_rank = _scalar_int(cov_result.rank)
            n_params = int(init_params.shape[0])
            logger.emit(
                "least_squares",
                "lsq_done",
                "info" if converged else "warning",
                converged=bool(converged),
                reason=termination_reason,
                steps=int(iter_num),
                normalized_residual_rms=_scalar_float(rms),
                cov_rank=cov_rank,
                cov_condition=_scalar_float(cov_result.condition),
                cov_valid=_scalar_bool(cov_result.valid),
                n_params=n_params,
            )

        return LeastSquaresResult(params=cur_param, jacobian=jac, residuals=residuals, normalized_residual_rms=rms,
                                  optical_weight_matrices=optical_weight_matrices,
                                  radar_weights=radar_weights,
                                  cov_mat_prior=cov_mat_prior,
                                  cov_mat_post=cov_mat_post,
                                  cov_rank=cov_result.rank,
                                  cov_condition=cov_result.condition,
                                  cov_valid=cov_result.valid,
                                  converged=converged,
                                  termination_reason=termination_reason,
                                  iter_num=iter_num)


class RobustResult(NamedTuple):
    """
    Result of the robust least-squares loop.

    ``lsq_iter_num`` is the accumulated number of accepted LM steps across all
    inlier-mask solves. It is intentionally distinct from
    ``lsq_result.iter_num``, which only describes the final inner solve.
    ``outlier_iter_num`` counts outer outlier-rejection iterations that were actually
    evaluated with finite residuals, a finite Jacobian, and valid covariance
    diagnostics from the inner least-squares solve.
    """
    lsq_result: LeastSquaresResult
    rej_result: RejResult
    outlier_iter_num: int
    lsq_iter_num: int


class RobustLeastSquares:
    """
    Combines a LeastSquares solver with an OutlierRejecter to perform
    iterative robust estimation.
    """

    def __init__(self, solver: LeastSquares) -> None:
        self.solver = solver

    def solve(self, init_param: Float[Array, "N_param"],
              weights: WeightResult, compiled_outlier_policy: CompiledOutlierPolicy,
              res_func: Callable, res_jac_func: Callable,
              param_scale: Float[Array, "N_param"] | None = None,
              weight_array_func: WeightArrayFunction | None = None,
              event_handler: SolverEventHandler | None = None,
              log_detail: SolverLogDetail = "iter",
              event_logger: SolverEventLogger | None = None) -> RobustResult:
        """Solve with fixed-point alternation between least squares and rejection."""
        logger = event_logger if event_logger is not None else make_solver_event_logger(event_handler, log_detail)
        optical_weight_matrices, radar_weights = _resolve_weight_arrays(weights, weight_array_func, init_param)
        metric_dtype = jnp.result_type(init_param, optical_weight_matrices, radar_weights)
        cur_param = init_param
        cur_flat_inlier_mask = compiled_outlier_policy.get_init_mask()
        total_lsq_iter_num = 0
        outlier_iteration_count = 0
        mask_changed_after_last_lsq = False

        def unavailable_rej_result(flat_inlier_mask):
            n_obs = compiled_outlier_policy.n_2d + compiled_outlier_policy.n_1d
            return RejResult(flat_inlier_mask, jnp.full((n_obs,), jnp.nan, dtype=metric_dtype))

        def observation_mask_counts(flat_inlier_mask):
            obs_mask = _flat_inlier_mask_to_observation_mask(flat_inlier_mask, compiled_outlier_policy.n_2d)
            valid_obs_mask = compiled_outlier_policy.observation_valid_mask
            observation_count = _scalar_int(jnp.sum(valid_obs_mask))
            inlier_count = _scalar_int(jnp.sum(obs_mask & valid_obs_mask))
            outlier_count = observation_count - inlier_count
            return obs_mask, observation_count, inlier_count, outlier_count

        lsq_result = None
        final_rej_result = None
        for i in range(compiled_outlier_policy.max_iters):
            iteration_logger = logger.bind(outlier_iteration=i + 1, lsq_solve=i + 1)
            if iteration_logger.enabled("outlier_iteration_start"):
                _, observation_count, inlier_count, outlier_count = observation_mask_counts(cur_flat_inlier_mask)
                iteration_logger.emit(
                    "robust_least_squares",
                    "outlier_iteration_start",
                    "info",
                    inlier_count=inlier_count,
                    outlier_count=outlier_count,
                    observation_count=observation_count,
                )
            # 1. Run lsq solver
            lsq_result = self.solver.solve(cur_param, weights, cur_flat_inlier_mask, res_func, res_jac_func,
                                           param_scale=param_scale,
                                           weight_array_func=weight_array_func,
                                           event_logger=iteration_logger.bind(lsq_solve=i + 1))
            optical_weight_matrices = lsq_result.optical_weight_matrices
            radar_weights = lsq_result.radar_weights
            total_lsq_iter_num += lsq_result.iter_num
            mask_changed_after_last_lsq = False
            # 2. Stop before mask updates if the current fit cannot support reliable Chi2 values.
            if not _lsq_result_has_valid_chi2_inputs(lsq_result):
                final_rej_result = unavailable_rej_result(cur_flat_inlier_mask)
                if iteration_logger.enabled("outlier_skipped"):
                    iteration_logger.emit(
                        "robust_least_squares",
                        "outlier_skipped",
                        "warning",
                        stop_reason="chi2_unavailable",
                        lsq_termination_reason=lsq_result.termination_reason,
                        cov_valid=_scalar_bool(lsq_result.cov_valid),
                        cov_rank=_scalar_int(lsq_result.cov_rank),
                        total_accepted_step_count=total_lsq_iter_num,
                    )
                break
            # 3. break if disable rejection
            if not compiled_outlier_policy.enable_auto_rejection:
                final_rej_result = compiled_outlier_policy.apply(lsq_result.residuals, optical_weight_matrices, radar_weights,
                                                                 lsq_result.jacobian, lsq_result.cov_mat_prior,
                                                                 cur_flat_inlier_mask)
                if iteration_logger.enabled("outlier_disabled"):
                    _, observation_count, inlier_count, outlier_count = observation_mask_counts(final_rej_result.flat_inlier_mask)
                    iteration_logger.emit(
                        "robust_least_squares",
                        "outlier_disabled",
                        "info",
                        inlier_count=inlier_count,
                        outlier_count=outlier_count,
                        observation_count=observation_count,
                        normalized_residual_rms=_scalar_float(lsq_result.normalized_residual_rms),
                    )
                break
            # 4. Update the inlier mask with one outer outlier-rejection iteration.
            rej_result = compiled_outlier_policy.apply(lsq_result.residuals, optical_weight_matrices, radar_weights,
                                                       lsq_result.jacobian, lsq_result.cov_mat_prior,
                                                       cur_flat_inlier_mask)
            outlier_iteration_count += 1
            if iteration_logger.enabled("outlier_update"):
                old_obs_mask = _flat_inlier_mask_to_observation_mask(cur_flat_inlier_mask,
                                                                     compiled_outlier_policy.n_2d)
                new_obs_mask = _flat_inlier_mask_to_observation_mask(rej_result.flat_inlier_mask,
                                                                     compiled_outlier_policy.n_2d)
                valid_obs_mask = compiled_outlier_policy.observation_valid_mask
                changed_to_outlier_count = _scalar_int(jnp.sum(valid_obs_mask & old_obs_mask & ~new_obs_mask))
                changed_to_inlier_count = _scalar_int(jnp.sum(valid_obs_mask & ~old_obs_mask & new_obs_mask))
                observation_count = _scalar_int(jnp.sum(valid_obs_mask))
                inlier_count = _scalar_int(jnp.sum(new_obs_mask & valid_obs_mask))
                outlier_count = observation_count - inlier_count
                iteration_logger.emit(
                    "robust_least_squares",
                    "outlier_update",
                    "info",
                    normalized_residual_rms=_scalar_float(lsq_result.normalized_residual_rms),
                    changed_to_outlier_count=changed_to_outlier_count,
                    changed_to_inlier_count=changed_to_inlier_count,
                    inlier_count=inlier_count,
                    outlier_count=outlier_count,
                    observation_count=observation_count,
                )
            new_flat_inlier_mask = rej_result.flat_inlier_mask
            # 5. Check convergence
            if _scalar_bool(jnp.array_equal(new_flat_inlier_mask, cur_flat_inlier_mask)):
                final_rej_result = rej_result
                if iteration_logger.enabled("outlier_done"):
                    _, observation_count, inlier_count, outlier_count = observation_mask_counts(new_flat_inlier_mask)
                    iteration_logger.emit(
                        "robust_least_squares",
                        "outlier_done",
                        "info",
                        stop_reason="mask_unchanged",
                        outlier_iteration_count=outlier_iteration_count,
                        inlier_count=inlier_count,
                        outlier_count=outlier_count,
                        observation_count=observation_count,
                    )
                break

            cur_param = lsq_result.params
            cur_flat_inlier_mask = new_flat_inlier_mask
            final_rej_result = rej_result
            mask_changed_after_last_lsq = True
        else:
            final_rej_result = final_rej_result or unavailable_rej_result(cur_flat_inlier_mask)
            if mask_changed_after_last_lsq:
                # The last allowed rejection pass changed the mask. Refit once with
                # that final mask so returned parameters, residuals, and metrics are
                # aligned even though no further rejection pass is permitted.
                lsq_result = self.solver.solve(cur_param, weights, cur_flat_inlier_mask, res_func, res_jac_func,
                                               param_scale=param_scale,
                                               weight_array_func=weight_array_func,
                                               event_logger=logger.bind(lsq_solve=compiled_outlier_policy.max_iters + 1))
                optical_weight_matrices = lsq_result.optical_weight_matrices
                radar_weights = lsq_result.radar_weights
                total_lsq_iter_num += lsq_result.iter_num
                if _lsq_result_has_valid_chi2_inputs(lsq_result):
                    metric_result = compiled_outlier_policy.apply(lsq_result.residuals, optical_weight_matrices, radar_weights,
                                                                  lsq_result.jacobian, lsq_result.cov_mat_prior,
                                                                  cur_flat_inlier_mask)
                    final_rej_result = RejResult(cur_flat_inlier_mask, metric_result.metric)
                else:
                    final_rej_result = unavailable_rej_result(cur_flat_inlier_mask)
            done_logger = logger.bind(outlier_iteration=compiled_outlier_policy.max_iters, lsq_solve=compiled_outlier_policy.max_iters)
            if done_logger.enabled("outlier_done"):
                _, observation_count, inlier_count, outlier_count = observation_mask_counts(final_rej_result.flat_inlier_mask)
                done_logger.emit(
                    "robust_least_squares",
                    "outlier_done",
                    "warning",
                    stop_reason="max_iterations_reached",
                    outlier_iteration_count=outlier_iteration_count,
                    max_iterations=compiled_outlier_policy.max_iters,
                    inlier_count=inlier_count,
                    outlier_count=outlier_count,
                    observation_count=observation_count,
                )

        return RobustResult(lsq_result=lsq_result, rej_result=final_rej_result,
                            outlier_iter_num=outlier_iteration_count, lsq_iter_num=total_lsq_iter_num)
