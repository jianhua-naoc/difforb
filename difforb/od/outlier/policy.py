from typing import Iterable
import jax.numpy as jnp
import equinox as eqx
from jax import Array
from jaxtyping import Float, Bool, ArrayLike

from difforb.astrometry.data import ObservationLayout
from difforb.core.validate import coerce_scalar_bool, coerce_scalar_int
from difforb.od.outlier.outlier import OutlierRejecter, RejResult


class CompiledOutlierPolicy(eqx.Module):
    auto_rejecter: OutlierRejecter | None
    enable_auto_rejection: bool = eqx.field(static=True)
    max_iters: int = eqx.field(static=True)
    n_2d: int = eqx.field(static=True)
    n_1d: int = eqx.field(static=True)
    flat_manual_outlier_mask: Bool[Array, "N_flat_obs"]
    flat_manual_inlier_mask: Bool[Array, "N_flat_obs"]
    flat_valid_mask: Bool[Array, "N_flat_obs"]
    observation_valid_mask: Bool[Array, "N_obs"]

    def __init__(
            self,
            auto_rejecter: OutlierRejecter | None,
            enable_auto_rejection: bool,
            max_iters: int,
            n_2d: int,
            n_1d: int,
            flat_manual_outlier_mask: Bool[ArrayLike, "N_flat_obs"],
            flat_manual_inlier_mask: Bool[ArrayLike, "N_flat_obs"],
            flat_valid_mask: Bool[ArrayLike, "N_flat_obs"],
            observation_valid_mask: Bool[ArrayLike, "N_obs"],
    ):
        self.auto_rejecter = auto_rejecter
        self.enable_auto_rejection = coerce_scalar_bool("enable_auto_rejection", enable_auto_rejection)
        self.max_iters = coerce_scalar_int("max_iters", max_iters)
        self.n_2d = coerce_scalar_int("n_2d", n_2d)
        self.n_1d = coerce_scalar_int("n_1d", n_1d)
        self.flat_manual_outlier_mask = jnp.asarray(flat_manual_outlier_mask, dtype=bool)
        self.flat_manual_inlier_mask = jnp.asarray(flat_manual_inlier_mask, dtype=bool)
        self.flat_valid_mask = jnp.asarray(flat_valid_mask, dtype=bool)
        self.observation_valid_mask = jnp.asarray(observation_valid_mask, dtype=bool)

    def get_init_mask(self) -> Bool[Array, "N_flat_obs"]:
        init_inlier_mask = jnp.ones_like(self.flat_manual_outlier_mask)
        return (init_inlier_mask | self.flat_manual_inlier_mask) & (~self.flat_manual_outlier_mask) & self.flat_valid_mask

    @eqx.filter_jit
    def apply(self, flat_residuals: Float[Array, "N_flat_obs"],
              optical_weight_matrices: Float[Array, "N_optical 2 2"],
              radar_weights: Float[Array, "N_radar"],
              jac: Float[Array, "N_flat_obs N_param"], cov_mat: Float[Array, "N_param N_param"],
              cur_flat_inlier_mask: Bool[Array, "N_flat_obs"]) -> RejResult:
        if self.auto_rejecter is None:
            n_obs = self.n_2d + self.n_1d
            metric = jnp.full((n_obs,), jnp.nan, dtype=flat_residuals.dtype)
            auto_inlier_mask = cur_flat_inlier_mask
        else:
            auto_res = self.auto_rejecter.reject(flat_residuals, optical_weight_matrices, radar_weights, jac, cov_mat, cur_flat_inlier_mask)
            metric = auto_res.metric
            auto_inlier_mask = auto_res.flat_inlier_mask

        if not self.enable_auto_rejection:
            auto_inlier_mask = cur_flat_inlier_mask

        final_inlier_mask = (auto_inlier_mask | self.flat_manual_inlier_mask) & (~self.flat_manual_outlier_mask) & self.flat_valid_mask
        return RejResult(flat_inlier_mask=final_inlier_mask, metric=metric)


class InteractiveOutlierPolicy:
    def __init__(self, auto_rejecter: OutlierRejecter | None, enable_auto_rejecter: bool = True,
                 max_iters: int = 10):
        self.auto_rejecter = auto_rejecter
        self.enable_auto_rejecter = enable_auto_rejecter
        self.max_iters = max_iters
        # Manual overrides
        self._manual_outlier_index: set[int] = set()
        self._manual_inlier_index: set[int] = set()

    @staticmethod
    def _normalize_indices(input_indices: int | Iterable[int] | Float[ArrayLike, "..."]) -> list[int]:
        """Convert scalar or array-like index input to a Python integer list."""
        if isinstance(input_indices, int):
            return [int(input_indices)]

        return [int(i.item()) if hasattr(i, 'item') else int(i) for i in input_indices]

    def force_outlier(self, input_indices: int | Iterable[int] | Float[ArrayLike, "..."]):
        for idx in self._normalize_indices(input_indices):
            self._manual_outlier_index.add(idx)
            self._manual_inlier_index.discard(idx)

    def force_inlier(self, input_indices: int | Iterable[int] | Float[ArrayLike, "..."]):
        for idx in self._normalize_indices(input_indices):
            self._manual_inlier_index.add(idx)
            self._manual_outlier_index.discard(idx)

    def restore_manual(self, input_indices: int | Iterable[int] | Float[ArrayLike, "..."] = None):
        if input_indices is not None:
            for idx in self._normalize_indices(input_indices):
                self._manual_outlier_index.discard(idx)
                self._manual_inlier_index.discard(idx)
        else:
            self._manual_outlier_index.clear()
            self._manual_inlier_index.clear()

    def compiled(self, layout: ObservationLayout, flat_valid_mask: Bool[Array, "N_flat_obs"] | None = None) -> CompiledOutlierPolicy:
        flat_manual_outlier_mask = layout.input_indices_to_flat_mask(self._manual_outlier_index)
        flat_manual_inlier_mask = layout.input_indices_to_flat_mask(self._manual_inlier_index)
        if flat_valid_mask is None:
            flat_valid_mask = jnp.ones_like(flat_manual_outlier_mask)
        else:
            flat_valid_mask = jnp.asarray(flat_valid_mask, dtype=bool)
        observation_valid_mask = layout.flat_mask_to_mask(flat_valid_mask)

        auto_rejecter = None
        if self.auto_rejecter is not None:
            auto_rejecter = self.auto_rejecter.with_observation_structure(
                n_2d=layout.n_2d,
                n_1d=layout.n_1d
            )

        return CompiledOutlierPolicy(auto_rejecter=auto_rejecter, enable_auto_rejection=self.enable_auto_rejecter,
                                     max_iters=self.max_iters, n_2d=layout.n_2d, n_1d=layout.n_1d,
                                     flat_manual_outlier_mask=flat_manual_outlier_mask,
                                     flat_manual_inlier_mask=flat_manual_inlier_mask,
                                     flat_valid_mask=flat_valid_mask,
                                     observation_valid_mask=observation_valid_mask)
