import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Bool

from difforb.core.validate import coerce_scalar_int
from difforb.od.outlier.outlier import OutlierRejecter, RejResult

jax.config.update("jax_enable_x64", True)


def compute_individual_chi2_2d_single(residual: Float[Array, "2"], jac: Float[Array, "2 N_param"],
                                      cov_mat: Float[Array, "N_param N_param"],
                                      measure_var: Float[Array, "2"],
                                      is_inlier: bool):
    # 1. Measurement covariance matrix
    measure_cov_mat = jnp.diag(measure_var)
    # 2. Covariance matrix propagated from least-squares fitting
    modified_term = jac @ cov_mat @ jac.T
    # 3. Covariance matrix of residuals
    signs = jnp.where(is_inlier, -1.0, 1.0)
    residual_cov_mat = measure_cov_mat + signs * modified_term
    # 4. Calculate Chi2
    inv_residual_cov_mat = jnp.linalg.solve(residual_cov_mat, residual)
    chi2 = jnp.dot(residual, inv_residual_cov_mat)
    return chi2


def compute_individual_chi2_2d_cov_single(residual: Float[Array, "2"], jac: Float[Array, "2 N_param"],
                                          cov_mat: Float[Array, "N_param N_param"],
                                          measure_cov: Float[Array, "2 2"],
                                          is_inlier: bool):
    # 1. Covariance matrix propagated from least-squares fitting
    modified_term = jac @ cov_mat @ jac.T
    # 2. Covariance matrix of residuals
    signs = jnp.where(is_inlier, -1.0, 1.0)
    residual_cov_mat = measure_cov + signs * modified_term
    # 3. Calculate Chi2
    inv_residual_cov_mat = jnp.linalg.solve(residual_cov_mat, residual)
    chi2 = jnp.dot(residual, inv_residual_cov_mat)
    return chi2


def compute_individual_chi2_1d_single(residual: Float[Array, "1"], jac_row: Float[Array, "N_param"],
                                      cov_mat: Float[Array, "N_param N_param"], measure_var: Float[Array, "1"],
                                      is_inlier: bool):
    # 1. Variance propagated from least-squares fitting
    modified_term = jnp.dot(jac_row, jnp.dot(cov_mat, jac_row))
    # 2. Variance of residuals
    signs = jnp.where(is_inlier, -1.0, 1.0)
    residual_var = measure_var + signs * modified_term
    # 3. Calculate Chi2
    chi2 = (residual ** 2) / residual_var
    return chi2


compute_individual_chi2_2d = jax.jit(jax.vmap(compute_individual_chi2_2d_single, in_axes=(0, 0, None, 0, 0)))

compute_individual_chi2_2d_cov = jax.jit(jax.vmap(compute_individual_chi2_2d_cov_single, in_axes=(0, 0, None, 0, 0)))

compute_individual_chi2_1d = jax.jit(jax.vmap(compute_individual_chi2_1d_single, in_axes=(0, 0, None, 0, 0)))


@eqx.filter_jit
def compute_individual_chi2(flat_residuals: Float[Array, "N_flat_obs"],
                            flat_measure_var: Float[Array, "N_flat_obs"],
                            jac: Float[Array, "N_flat_obs N_param"],
                            cov_mat: Float[Array, "N_param N_param"],
                            inlier_mask: Bool[Array, "N_obs"], n_2d: int, n_1d: int) -> Float[
    Array, "N_obs"]:
    n_obs = n_2d + n_1d
    n_flat_2d = 2 * n_2d
    chi2 = jnp.zeros(n_obs)
    # 2D Observations
    residuals_2d = flat_residuals[:n_flat_2d].reshape(-1, 2)  # [N_2d, 2]
    jac_2d = jac[:n_flat_2d].reshape(-1, 2, jac.shape[-1])  # [N_2d, 2, N_param]
    var_2d = flat_measure_var[:n_flat_2d].reshape(-1, 2)
    inlier_mask_2d = inlier_mask[:n_2d]  # [N_2d]
    chi2_2d = compute_individual_chi2_2d(residuals_2d, jac_2d, cov_mat, var_2d, inlier_mask_2d)
    chi2 = chi2.at[:n_2d].set(chi2_2d)
    # 1D Observation
    residuals_1d = flat_residuals[n_flat_2d:]  # [N_1d]
    jac_1d = jac[n_flat_2d:]  # [N_1d, N_param]
    var_1d = flat_measure_var[n_flat_2d:]
    inlier_mask_1d = inlier_mask[n_2d:]
    chi2_1d = compute_individual_chi2_1d(residuals_1d, jac_1d, cov_mat, var_1d, inlier_mask_1d)
    chi2 = chi2.at[n_2d:].set(chi2_1d)
    return chi2


@eqx.filter_jit
def compute_individual_chi2_weighted(flat_residuals: Float[Array, "N_flat_obs"],
                                     optical_weight_matrices: Float[Array, "N_optical 2 2"],
                                     radar_weights: Float[Array, "N_radar"],
                                     jac: Float[Array, "N_flat_obs N_param"],
                                     cov_mat: Float[Array, "N_param N_param"],
                                     inlier_mask: Bool[Array, "N_obs"], n_2d: int, n_1d: int) -> Float[
    Array, "N_obs"]:
    n_obs = n_2d + n_1d
    n_flat_2d = 2 * n_2d
    chi2 = jnp.zeros(n_obs)
    # 2D Observations
    residuals_2d = flat_residuals[:n_flat_2d].reshape(-1, 2)
    jac_2d = jac[:n_flat_2d].reshape(-1, 2, jac.shape[-1])
    cov_2d = jnp.linalg.inv(optical_weight_matrices)
    inlier_mask_2d = inlier_mask[:n_2d]
    chi2_2d = compute_individual_chi2_2d_cov(residuals_2d, jac_2d, cov_mat, cov_2d, inlier_mask_2d)
    chi2 = chi2.at[:n_2d].set(chi2_2d)
    # 1D Observation
    residuals_1d = flat_residuals[n_flat_2d:]
    jac_1d = jac[n_flat_2d:]
    var_1d = jnp.reciprocal(radar_weights)
    inlier_mask_1d = inlier_mask[n_2d:]
    chi2_1d = compute_individual_chi2_1d(residuals_1d, jac_1d, cov_mat, var_1d, inlier_mask_1d)
    chi2 = chi2.at[n_2d:].set(chi2_1d)
    return chi2


def fudge_term(n_sel: Float[ArrayLike, ""]) -> Float[Array, ""]:
    return 400. * jnp.reciprocal(1.2 ** n_sel)


def update_inlier_mask(current_inlier_mask: Bool[Array, "N_obs"],
                       chi2: Float[Array, "N_obs"],
                       chi2_rej: Float[ArrayLike, ""],
                       chi2_rec: Float[ArrayLike, ""],
                       progressive_alpha: Float[ArrayLike, ""]):
    """
    Update an inlier mask with explicit rejection and recovery branches.

    Parameters
    ----------
    current_inlier_mask : Bool[Array, "N_obs"]
        Current observation-level inlier mask.
    chi2 : Float[Array, "N_obs"]
        Individual chi-square values for the same observations.
    chi2_rej : Float[ArrayLike, ""]
        Base rejection threshold.
    chi2_rec : Float[ArrayLike, ""]
        Recovery threshold for current outliers.
    progressive_alpha : Float[ArrayLike, ""]
        Progressive rejection factor proposed by Carpino et al. (2003).

    Returns
    -------
    tuple[Bool[Array, "N_obs"], Float[Array, ""], Float[Array, ""]]
        Updated inlier mask, effective rejection threshold, and maximum chi-square among current inliers.
    """
    n_sel = jnp.sum(current_inlier_mask)
    rej_threshold = chi2_rej + fudge_term(n_sel)
    chi2_max = jnp.max(jnp.where(current_inlier_mask, chi2, -jnp.inf), initial=0.)
    reject_threshold = jnp.maximum(rej_threshold, progressive_alpha * chi2_max)
    reject_mask = current_inlier_mask & (chi2 >= reject_threshold)
    recover_mask = (~current_inlier_mask) & (chi2 < chi2_rec)
    new_inlier_mask = (current_inlier_mask & (~reject_mask)) | recover_mask
    return new_inlier_mask, reject_threshold, chi2_max


class Chi2OutlierRejecter(OutlierRejecter):
    """
    Outlier rejecter based on Chi2 test proposed by Carpino et al. (2003). Ref: Carpino M, Milani A, Chesley S R. Error statistics of asteroid optical astrometric observations[J]. Icarus, 2003, 166(2): 248-270.
    """
    chi2_rej_2d: Float[Array, ""]
    chi2_rec_2d: Float[Array, ""]
    chi2_rej_1d: Float[Array, ""]
    chi2_rec_1d: Float[Array, ""]
    progressive_alpha: Float[Array, ""]
    n_2d: int = eqx.field(static=True)
    n_1d: int = eqx.field(static=True)

    def __init__(
            self,
            chi2_rej_2d: Float[ArrayLike, ""] = 8.0,
            chi2_rec_2d: Float[ArrayLike, ""] = 7.0,
            chi2_rej_1d: Float[ArrayLike, ""] = 6.0,
            chi2_rec_1d: Float[ArrayLike, ""] = 5.0,
            progressive_alpha: Float[ArrayLike, ""] = 0.25,
            n_2d: int = 0,
            n_1d: int = 0,
    ):
        self.chi2_rej_2d = jnp.array(chi2_rej_2d, dtype=jnp.float64)
        self.chi2_rec_2d = jnp.array(chi2_rec_2d, dtype=jnp.float64)
        self.chi2_rej_1d = jnp.array(chi2_rej_1d, dtype=jnp.float64)
        self.chi2_rec_1d = jnp.array(chi2_rec_1d, dtype=jnp.float64)
        self.progressive_alpha = jnp.array(progressive_alpha, dtype=jnp.float64)
        self.n_2d = coerce_scalar_int("n_2d", n_2d)
        self.n_1d = coerce_scalar_int("n_1d", n_1d)

    @staticmethod
    def _fudge_term(n_sel):
        return fudge_term(n_sel)

    def with_observation_structure(self, n_2d: int, n_1d: int) -> 'Chi2OutlierRejecter':
        return Chi2OutlierRejecter(
            chi2_rej_2d=self.chi2_rej_2d,
            chi2_rec_2d=self.chi2_rec_2d,
            chi2_rej_1d=self.chi2_rej_1d,
            chi2_rec_1d=self.chi2_rec_1d,
            progressive_alpha=self.progressive_alpha,
            n_2d=n_2d,
            n_1d=n_1d,
        )

    @eqx.filter_jit
    def reject(self, flat_residuals: Float[Array, "N_flat_obs"],
               optical_weight_matrices: Float[Array, "N_optical 2 2"],
               radar_weights: Float[Array, "N_radar"],
               jac: Float[Array, "N_flat_obs N_param"], cov_mat: Float[Array, "N_param N_param"],
               flat_inlier_mask: Bool[Array, "N_flat_obs"]) -> RejResult:
        """
        Perform one hysteretic rejection/recovery update.

        Current inliers are tested only against the rejection threshold,
        while current outliers are tested only against the recovery
        threshold. The individual Chi2 values are computed with the
        covariance sign convention of Carpino et al. (2003), so inliers
        and outliers use different residual covariance models.
        """
        n_flat_2d = 2 * self.n_2d
        inlier_mask_2d = flat_inlier_mask[:n_flat_2d][::2]
        inlier_mask_1d = flat_inlier_mask[n_flat_2d:]
        inlier_mask = jnp.concatenate([inlier_mask_2d, inlier_mask_1d])
        new_flat_inlier_mask = jnp.ones(n_flat_2d + self.n_1d, dtype=bool)

        # 1. Calculate individual Chi2
        chi2 = compute_individual_chi2_weighted(
            flat_residuals,
            optical_weight_matrices,
            radar_weights,
            jac,
            cov_mat,
            inlier_mask,
            self.n_2d,
            self.n_1d,
        )

        # 2. Update mask for 2D observations with explicit rejection/recovery.
        chi2_2d = chi2[:self.n_2d]
        new_inlier_mask_2d, _, _ = update_inlier_mask(
            inlier_mask_2d,
            chi2_2d,
            self.chi2_rej_2d,
            self.chi2_rec_2d,
            self.progressive_alpha
        )
        new_flat_inlier_mask_2d = jnp.repeat(new_inlier_mask_2d, 2)
        new_flat_inlier_mask = new_flat_inlier_mask.at[:n_flat_2d].set(new_flat_inlier_mask_2d)

        # 3. Update mask for 1D observations with the scalar analogue.
        chi2_1d = chi2[self.n_2d:]
        new_inlier_mask_1d, _, _ = update_inlier_mask(
            inlier_mask_1d,
            chi2_1d,
            self.chi2_rej_1d,
            self.chi2_rec_1d,
            self.progressive_alpha
        )
        new_flat_inlier_mask = new_flat_inlier_mask.at[n_flat_2d:].set(new_inlier_mask_1d)

        result = RejResult(new_flat_inlier_mask, chi2)
        return result
