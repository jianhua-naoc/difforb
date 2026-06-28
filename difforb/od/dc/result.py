from typing import NamedTuple, Type

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from difforb.body.smallbody import Orbit
from difforb.body.ephbody import EphemerisBody
from difforb.core.element import KepElement
from difforb.core.state.frame import Frame
from difforb.core.state.state import State
from difforb.report.display_units import orbit_element_specs, repr_fields_from_specs, STATE_REPR_SPECS
from difforb.report.text import build_repr, format_float_array

jax.config.update("jax_enable_x64", True)


class DCEstimate(NamedTuple):
    """Final parameter estimate from a differential-correction solve.

    Parameters
    ----------
    orbit : Orbit
        Estimated orbit. The first six covariance parameters use ``orbit.array.squeeze()`` in the native order of the concrete orbit representation.
    model_params : Float[Array, "N_model"]
        Estimated non-orbit model parameters appended after the six orbit parameters.
    model_param_names : list[str]
        Names corresponding to ``model_params``. This list is empty when the solve estimated only the six orbit parameters.
    cov_mat_post : Float[Array, "N_param N_param"]
        Posterior covariance matrix for ``[orbit.array.squeeze(), model_params]``, where ``N_param = 6 + N_model``.
    """

    orbit: Orbit
    model_params: Float[Array, "N_model"]
    model_param_names: list[str]
    cov_mat_post: Float[Array, "N_param N_param"]

    @property
    def uncertainties(self) -> Float[Array, "N_param"]:
        return jnp.sqrt(jnp.diagonal(self.cov_mat_post))


class OpticalResult(NamedTuple):
    residuals: Float[Array, "N 2"]
    normalized_residuals: Float[Array, "N 2"]
    weighted_rms: float
    unweighted_rms: float
    inlier_masks: Bool[Array, "N"]
    metrics: Float[Array, "N"]

    @property
    def n_obs(self) -> int:
        return len(self.inlier_masks)

    @property
    def n_inliers(self) -> int:
        return int(jnp.sum(self.inlier_masks))

    @property
    def n_outliers(self) -> int:
        return self.n_obs - self.n_inliers


class RadarResult(NamedTuple):
    residuals: Float[Array, "N"]
    normalized_residuals: Float[Array, "N"]
    inlier_masks: Bool[Array, "N"]
    metrics: Float[Array, "N"]

    delay_weighted_rms: float
    delay_unweighted_rms: float
    doppler_weighted_rms: float
    doppler_unweighted_rms: float

    @property
    def n_obs(self) -> int:
        return len(self.inlier_masks)

    @property
    def n_inliers(self) -> int:
        return int(jnp.sum(self.inlier_masks))

    @property
    def n_outliers(self) -> int:
        return self.n_obs - self.n_inliers


class LSQDiagnostics(NamedTuple):
    """
    Diagnostics from the final robust differential-correction solve.

    ``lsq_iterations`` counts accepted Levenberg-Marquardt steps accumulated
    across all inlier-mask solves in the robust loop. ``outlier_iterations``
    counts outer outlier-rejection iterations evaluated with usable residual,
    Jacobian, and covariance diagnostics from the inner solves.
    ``termination_reason`` describes why the final least-squares solve stopped
    at the returned estimate. ``optical_weight_matrices`` and
    ``radar_weights`` are the canonical weights used by the solver;
    ``flat_weights`` is only a marginal display view.
    """
    flat_jacobian: Float[Array, "N_flat N_param"]
    flat_weights: Float[Array, "N_flat"]
    optical_weight_matrices: Float[Array, "N_optical 2 2"]
    radar_weights: Float[Array, "N_radar"]
    cov_mat_prior: Float[Array, "N_param N_param"]
    cov_rank: Int[Array, ""]
    cov_condition: Float[Array, ""]
    cov_valid: Bool[Array, ""]
    converged: bool
    termination_reason: str
    lsq_iterations: int
    outlier_iterations: int


class DCResult(NamedTuple):
    """Differential-correction result with the final estimate and residual diagnostics.

    Parameters
    ----------
    estimate : DCEstimate
        Final orbit, estimated model parameters, and posterior covariance. Use ``estimate.cov_mat_post`` for the covariance matrix in the parameter order documented by :class:`DCEstimate`.
    optical : OpticalResult
        Residual diagnostics for optical observations.
    radar : RadarResult
        Residual diagnostics for radar observations.
    lsq_diagnostics : LSQDiagnostics
        Final least-squares diagnostics, including covariance validity, rank, condition, and termination state.
    normalized_residual_rms : float
        Root mean square of normalized residuals for the final solution. Dimensionless.
    """
    estimate: DCEstimate
    optical: OpticalResult
    radar: RadarResult
    lsq_diagnostics: LSQDiagnostics
    normalized_residual_rms: float

    @staticmethod
    def _rebuild_orbit(orbit: Orbit, orbit_params: Float[Array, "6"]) -> Orbit:
        if isinstance(orbit, State):
            return State.from_array(orbit.tdb, orbit_params, orbit.frame)
        if isinstance(orbit, KepElement):
            return KepElement.from_array(orbit.tdb, orbit_params)
        raise TypeError(f"Unsupported orbit type: {type(orbit)}")

    @staticmethod
    def _convert_orbit(orbit: Orbit, target: Frame | Type[KepElement], sun: EphemerisBody,
                       earth: EphemerisBody) -> Orbit:
        if target is KepElement:
            if isinstance(orbit, KepElement):
                return orbit
            return KepElement.from_state(orbit, sun=sun, earth=earth)

        if isinstance(orbit, KepElement):
            orbit = orbit.state()
        return orbit.to(target, sun=sun, earth=earth)

    def transform(self, target: Frame | Type[KepElement]) -> "DCResult":
        """Convert the solved orbit and covariance; pass ``KepElement`` from ``difforb.core`` for Keplerian-element covariance, returned in ``estimate.cov_mat_post``.

        Parameters
        ----------
        target : Frame or type[KepElement]
            Target state frame or the :class:`KepElement` class.

        Returns
        -------
        DCResult
            Result with ``estimate.orbit`` and ``estimate.cov_mat_post`` transformed to the target representation.

        Notes
        -----
        Model parameters remain appended after the six orbit parameters.
        """
        if target is KepElement and isinstance(self.estimate.orbit, KepElement):
            return self
        if isinstance(target, Frame) and isinstance(self.estimate.orbit, State) and self.estimate.orbit.frame == target:
            return self

        sun = EphemerisBody("sun")
        earth = EphemerisBody("earth")
        epoch_tdb = self.estimate.orbit.tdb

        def conversion_func(params: Float[Array, "6+N"]) -> tuple[
            Float[Array, "6+N"], Float[Array, "6+N"]]:
            orbit_params = params[:6]
            model_params = params[6:]
            cur_orbit = self._rebuild_orbit(self.estimate.orbit, orbit_params)
            new_orbit = self._convert_orbit(cur_orbit, target, sun, earth)
            new_orbit_params = new_orbit.array.squeeze()
            new_params = jnp.concatenate((new_orbit_params, model_params))
            return new_params, new_params

        params = jnp.concatenate([self.estimate.orbit.array.squeeze(), self.estimate.model_params])
        j, new_params = jax.jacfwd(conversion_func, has_aux=True)(params)
        orbit_params = new_params[:6]
        if target is KepElement:
            new_orbit = KepElement.from_array(epoch_tdb, orbit_params)
        else:
            new_orbit = State.from_array(epoch_tdb, orbit_params, target)
        new_cov_mat = j @ self.estimate.cov_mat_post @ j.T

        new_estimate = DCEstimate(
            orbit=new_orbit,
            model_params=self.estimate.model_params,
            model_param_names=self.estimate.model_param_names,
            cov_mat_post=new_cov_mat
        )
        return DCResult(new_estimate, self.optical, self.radar, self.lsq_diagnostics,
                        self.normalized_residual_rms)

    @property
    def quality_code(self) -> Float[Array, ""]:
        """
        IAU MPC Uncertainty Parameter U.

        Range: 0 (good) to 9 (poor).

        Ref: https://www.minorplanetcenter.net/iau/info/UValue.html
        """
        kep_result = self.transform(KepElement)
        kep_orbit = kep_result.estimate.orbit
        kep_cov_mat = kep_result.estimate.cov_mat_post[:6, :6]

        def compute_derived_p_tp(params):
            ele = KepElement.from_array(kep_orbit.tdb, params)
            p = ele.period.squeeze()
            tp_jd = ele.perit_jd.squeeze()
            return jnp.stack([p, tp_jd])

        jac = jax.jacfwd(compute_derived_p_tp)(kep_orbit.array.squeeze())
        derived_cov = jac @ kep_cov_mat @ jac.T
        sigma_p = jnp.sqrt(derived_cov[0, 0])
        sigma_tp = jnp.sqrt(derived_cov[1, 1])

        e = kep_orbit.e
        period = kep_orbit.period / 365.25
        ko = 180 / jnp.pi * 0.01720209895
        runoff = (sigma_tp * e + 10 / period * sigma_p) * ko / period * 3600 * 3
        cons = jnp.log(648000) / 9.
        u = jnp.floor(jnp.log(runoff) / cons) + 1
        u = u.squeeze().astype(int)
        return u

    def __repr__(self) -> str:
        orbit = self.estimate.orbit
        if isinstance(orbit, KepElement):
            orbit_type = orbit.__class__.__name__
        else:
            orbit_type = orbit.frame.name or f"{orbit.frame.origin.value}+{orbit.frame.axes.value}"
        orbit_fields = [("orbit_type", orbit_type)]
        if isinstance(orbit, KepElement):
            orbit_fields.extend(repr_fields_from_specs(orbit, orbit_element_specs(orbit)))
        else:
            orbit_fields.extend(repr_fields_from_specs(orbit, STATE_REPR_SPECS))

        try:
            quality_code = str(int(np.asarray(self.quality_code).item()))
        except Exception:
            quality_code = "N/A"

        m_params = "{" + ", ".join(f"{n}={format_float_array(v, precision=3)}" for n, v in zip(self.estimate.model_param_names,
                                                                                               self.estimate.model_params)) + "}" if self.estimate.model_param_names else None

        return build_repr(self.__class__.__name__, [
            ("epoch_jd", format_float_array(orbit.tdb.jd, precision=9, scientific=False, signed=False)),
            *orbit_fields,
            ("normalized_residual_rms", format_float_array(self.normalized_residual_rms)),
            ("u", quality_code),
            ("optical", f"{self.optical.n_inliers}/{self.optical.n_obs}"),
            ("radar", f"{self.radar.n_inliers}/{self.radar.n_obs}"),
            ("iters", f"{self.lsq_diagnostics.lsq_iterations}+{self.lsq_diagnostics.outlier_iterations}"),
            ("params", m_params),
        ])
