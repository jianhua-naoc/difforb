from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx
from jax import Array
from jaxtyping import Float, Bool


class RejResult(NamedTuple):
    flat_inlier_mask: Bool[Array, "N_flat_obs"]
    metric: Float[Array, "N_obs"]


class OutlierRejecter(eqx.Module):
    @abstractmethod
    def reject(self, flat_residuals: Float[Array, "N_flat_obs"],
               optical_weight_matrices: Float[Array, "N_optical 2 2"],
               radar_weights: Float[Array, "N_radar"],
               jac: Float[Array, "N_flat_obs N_param"], cov_mat: Float[Array, "N_param N_param"],
               flat_inlier_mask: Bool[Array, "N_flat_obs"]) -> RejResult:
        pass

    @abstractmethod
    def with_observation_structure(self, n_2d: int, n_1d: int) -> 'OutlierRejecter':
        pass
