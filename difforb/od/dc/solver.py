"""Differential-correction solver entry point."""

import jax
import numpy as np

from difforb.astrometry.data import ObservationData, ObservationLayout
from difforb.astrometry.debias import DebiasPolicy
from difforb.astrometry.reduction.photocenter import PhotocenterCorrection
from difforb.astrometry.weight import WeightPolicy
from difforb.body.ephbody import EphemerisBody
from difforb.body.smallbody import Orbit, SmallBody
from difforb.core.device import put_arrays
from difforb.core.validate import coerce_scalar_bool, coerce_scalar_int
from difforb.dynamics.force_model import ForceModel
from difforb.integrator.integrator import NumericalIntegrator
from difforb.od.dc.strategy import (
    dispatch_dc_strategies,
    expand_dc_strategies,
    prepare_dc_strategy,
)
from difforb.od.dc.lsq import LeastSquares
from difforb.od.dc.prediction import AstrometryMeasurementModel
from difforb.od.dc.result import DCResult, build_dc_result
from difforb.od.outlier.policy import InteractiveOutlierPolicy
from difforb.od.progress import SolverReporter, solver_progress_reporter
from difforb.report.text import build_repr

jax.config.update("jax_enable_x64", True)


class DCSolver:
    """Differential correction with optional compiled solver control flow."""

    def __init__(self,
                 lsq_max_iters: int = 20,
                 *,
                 solver_jit: bool = False,
                 sun: EphemerisBody | None = None,
                 earth: EphemerisBody | None = None) -> None:
        """Create a differential-correction solver.

        Parameters
        ----------
        lsq_max_iters : int, default=20
            Maximum accepted steps in each least-squares solve.
        solver_jit : bool, default=False
            Compile the complete least-squares and outlier-rejection control
            flow. Residual, Jacobian, and propagation kernels retain their own
            JIT settings in either mode.
        sun, earth : EphemerisBody or None, optional
            Ephemeris bodies used by frame conversion and measurement models.
        """
        self.lsq_max_iter = coerce_scalar_int("lsq_max_iters", lsq_max_iters)
        if self.lsq_max_iter < 0:
            raise ValueError("`lsq_max_iters` must be nonnegative.")
        self.solver_jit = coerce_scalar_bool("solver_jit", solver_jit)
        self.sun = sun if sun is not None else EphemerisBody("sun")
        self.earth = earth if earth is not None else EphemerisBody("earth")

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("lsq_max_iter", str(self.lsq_max_iter)),
                ("solver_jit", str(self.solver_jit)),
            ],
        )

    def solve(self, data: ObservationData, initial_orbit: Orbit,
              force_model: ForceModel | list[ForceModel] | tuple[ForceModel, ...],
              integrator: NumericalIntegrator,
              weight_policy: WeightPolicy | list[WeightPolicy] | tuple[WeightPolicy, ...],
              debias_policy: DebiasPolicy,
              outlier_policy: InteractiveOutlierPolicy | list[InteractiveOutlierPolicy] | tuple[InteractiveOutlierPolicy, ...], *,
              photocenter_correction: PhotocenterCorrection | None = None,
              verbose: bool | SolverReporter = False,
              device: jax.Device | None = None,
              grid: bool = False,
              batch_size: int | None = None) -> DCResult | np.ndarray:
        """Run differential correction for one orbit-estimation problem.

        Scalar strategy arguments return :class:`DCResult`. Sequence-valued
        force, weight, or outlier arguments return an object array of
        :class:`DCResult` using point-wise broadcasting, or their Cartesian
        product when ``grid=True``.
        Compatible strategies use ``vmap`` when ``solver_jit=True`` and
        progress reporting is disabled.

        Parameters
        ----------
        data : ObservationData
            Observations used for every strategy.
        initial_orbit : Orbit
            Common least-squares starting orbit.
        force_model : ForceModel or sequence of ForceModel
            Dynamical model or models.
        integrator : NumericalIntegrator
            Numerical orbit integrator.
        weight_policy : WeightPolicy or sequence of WeightPolicy
            Observation-weight policy or policies.
        debias_policy : DebiasPolicy
            Common optical-debias policy.
        outlier_policy : InteractiveOutlierPolicy or sequence
            Outlier policy or policies.
        photocenter_correction : PhotocenterCorrection or None, optional
            Optical center-of-light model.
        verbose : bool or callable, default=False
            Print solver progress, or pass a callback that accepts an event
            name and keyword data.
        device : jax.Device or None, optional
            Device receiving numerical arrays.
        grid : bool, default=False
            Use Cartesian-product strategy semantics instead of point-wise
            broadcasting.
        batch_size : int or None, optional
            Maximum compatible strategies in one mapped solve.

        Returns
        -------
        DCResult or numpy.ndarray
            A scalar result, or an object array of results with the broadcast
            or Cartesian-product strategy shape.
        """
        reporter = solver_progress_reporter(verbose)
        if device is not None and not isinstance(device, jax.Device):
            raise TypeError("`device` must be a jax.Device or None.")
        grid = coerce_scalar_bool("grid", grid)
        if batch_size is not None:
            batch_size = coerce_scalar_int("batch_size", batch_size)
            if batch_size < 1:
                raise ValueError("`batch_size` must be positive.")

        strategies, result_shape, is_batched = expand_dc_strategies(
            force_model, weight_policy, outlier_policy, grid,
        )
        layout = ObservationLayout(data)
        initial_state = SmallBody.create(
            initial_orbit, sun=self.sun, earth=self.earth,
        ).orbit0
        photocenter = (
            PhotocenterCorrection()
            if photocenter_correction is None
            else photocenter_correction
        )
        measure_model = AstrometryMeasurementModel.build(
            data,
            initial_state.tdb,
            self.sun,
            self.earth,
            debias_policy.bias(data),
            photocenter,
        )
        initial_state_params = initial_state.array.squeeze()
        if device is not None:
            measure_model, integrator = put_arrays((measure_model, integrator), device)
        prepared_strategies = tuple(
            prepare_dc_strategy(
                strategy, data, layout, initial_state_params, photocenter,
            )
            for strategy in strategies
        )
        result_initial_state = initial_state
        if device is not None:
            prepared_strategies = put_arrays(prepared_strategies, device)
            result_initial_state = put_arrays(initial_state, device)
        least_squares = LeastSquares(
            max_iter=self.lsq_max_iter,
            solver_jit=self.solver_jit and reporter is None,
        )

        fitted = dispatch_dc_strategies(
            prepared_strategies,
            is_batched,
            measure_model,
            integrator,
            least_squares,
            batch_size,
            reporter,
        )
        results = tuple(
            build_dc_result(
                result, layout, result_initial_state, model, photocenter,
            )
            for result, model in fitted
        )

        if reporter is not None:
            for index, result in enumerate(results):
                progress = {
                    "termination_reason": result.lsq_diagnostics.termination_reason,
                    "lsq_iterations": result.lsq_diagnostics.lsq_iterations,
                    "outlier_iterations": result.lsq_diagnostics.outlier_iterations,
                    "normalized_residual_rms": float(result.normalized_residual_rms),
                    "inlier_count": result.optical.n_inliers + result.radar.n_inliers,
                }
                if is_batched:
                    progress["index"] = index
                reporter("result", **progress)

        if not is_batched:
            return results[0]

        batch_results = np.empty(result_shape, dtype=object)
        for index, result in enumerate(results):
            batch_results.flat[index] = result
        return batch_results
