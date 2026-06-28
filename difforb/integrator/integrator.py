"""Numerical orbit integrator and dense interpolator.

This module is the wrapper based on ``diffrax`` and the custom ``IAS15`` implementation.

The integrated state uses the project six-component Cartesian convention, with position first and velocity second. This module only manages numerical propagation and interpolation. The physical frame, origin, and units are defined by the supplied ``ForceModel`` and initial state.
"""

from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import Array
from jaxtyping import Float, Bool, ArrayLike
import diffrax

from typing import Literal, Optional, Tuple
from difforb.dynamics.force_model import ForceModel
from difforb.integrator.ias15 import IAS15Solver, IAS15StepSizeController, IAS15Term
from difforb.core.batch import safe_dispatch, safe_cartesian_dispatch, BatchableObject
from difforb.core.validate import coerce_optional_scalar_float, coerce_scalar_float, coerce_scalar_int
from difforb.report.text import build_repr, format_float_array

jax.config.update("jax_enable_x64", True)

SUPPORTED_INTEGRATOR_METHODS = ("IAS15", "DOPRI8", "DOPRI5")
IAS15_ADAPTIVE_MODES = (1, 2)


def is_covered_single(interp: 'BiDirectionalInterpolator', jd1: Float[Array, ""], jd2: Float[Array, ""]) -> Bool[Array, ""]:
    tau = (jd1 - interp.t0_jd1) + (jd2 - interp.t0_jd2)
    tau_min = (interp.t_min_jd1 - interp.t0_jd1) + (interp.t_min_jd2 - interp.t0_jd2)
    tau_max = (interp.t_max_jd1 - interp.t0_jd1) + (interp.t_max_jd2 - interp.t0_jd2)
    in_range = (tau >= tau_min) & (tau <= tau_max)
    return in_range


def evaluate_single(
        interp: 'BiDirectionalInterpolator',
        jd1: Float[Array, ""],
        jd2: Float[Array, ""],
) -> tuple[Float[Array, "3"], Float[Array, "3"]]:
    tau = (jd1 - interp.t0_jd1) + (jd2 - interp.t0_jd2)
    pos, vel = jax.lax.cond(
        tau >= 0.,
        lambda _: interp.sol_fwd.evaluate(tau),
        lambda _: interp.sol_bwd.evaluate(tau),
        operand=None
    )
    return pos, vel


class BiDirectionalInterpolator(BatchableObject):
    """Dense interpolator with forward and backward coverage from one reference epoch.

    Parameters
    ----------
    t0_jd1, t0_jd2 : Float[Array, "..."]
        Reference epoch in split Julian Date form.
    t_min_jd1, t_min_jd2 : Float[Array, "..."]
        Lower bound of the covered epoch range, in split Julian Date form.
    t_max_jd1, t_max_jd2 : Float[Array, "..."]
        Upper bound of the covered epoch range, in split Julian Date form.
    sol_fwd : diffrax.Solution
        Dense forward solution from the reference epoch toward later times.
    sol_bwd : diffrax.Solution
        Dense backward solution from the reference epoch toward earlier times.

    Notes
    -----
    The interpolator stores two dense solutions. Evaluation picks ``sol_fwd`` for epochs at or after the reference epoch, and ``sol_bwd`` for earlier epochs.
    """
    t0_jd1: Float[Array, "..."]
    t0_jd2: Float[Array, "..."]
    t_min_jd1: Float[Array, "..."]
    t_min_jd2: Float[Array, "..."]
    t_max_jd1: Float[Array, "..."]
    t_max_jd2: Float[Array, "..."]
    sol_fwd: diffrax.Solution
    sol_bwd: diffrax.Solution

    @eqx.filter_jit
    def is_covered(self, jd1: Float[Array, "..."], jd2: Float[Array, "..."], grid: bool = False) -> Bool[
        Array, ""]:
        """Check whether epochs are inside the covered range.

        Parameters
        ----------
        jd1, jd2 : Float[Array, "..."]
            Query epochs in split Julian Date form.
        grid : bool, default=False
            If ``True``, use the Cartesian product of the interpolator batch and the time batch. If ``False``, use point-wise
            broadcasting.

        Returns
        -------
        Bool[Array, "..."]
            Coverage mask for the queried epochs.

        Notes
        -----
        Vectorize :func:`difforb.integrator.integrator._is_covered_scalar`.
        """
        if not grid:
            return safe_dispatch(is_covered_single, (0, 0, 0), self, jd1, jd2)
        else:
            return safe_cartesian_dispatch(is_covered_single, ((0,), (self,)), ((0, 0), (jd1, jd2)))

    @eqx.filter_jit
    def evaluate(
            self,
            jd1: Float[Array, "..."],
            jd2: Float[Array, "..."],
            grid: bool = False,
    ) -> tuple[Float[Array, "... 3"], Float[Array, "... 3"]]:
        """Evaluate the interpolated state at the requested epochs.

        Parameters
        ----------
        jd1, jd2 : Float[Array, "..."]
            Query epochs in split Julian Date form.
        grid : bool, default=False
            If ``True``, use the Cartesian product of the interpolator batch and the time batch. If ``False``, use point-wise
            broadcasting.

        Returns
        -------
        tuple[Float[Array, "... 3"], Float[Array, "... 3"]]
            Interpolated Cartesian position and velocity in ``au`` and ``au / day``.

        Notes
        -----
        Vectorize :func:`difforb.integrator.integrator._evaluate_scalar`.
        """
        if not grid:
            return safe_dispatch(evaluate_single, (0, 0, 0), self, jd1, jd2)
        else:
            return safe_cartesian_dispatch(evaluate_single, ((0,), (self,)), ((0, 0), (jd1, jd2)))

    def __call__(
            self,
            jd1: Float[Array, "..."],
            jd2: Float[Array, "..."],
            grid: bool = False,
    ) -> tuple[Float[Array, "... 3"], Float[Array, "... 3"]]:
        """Evaluate the interpolated state at the requested epochs.

        Parameters
        ----------
        jd1, jd2 : Float[Array, "..."]
            Query epochs in split Julian Date form.
        grid : bool, default=False
            If ``True``, use the Cartesian product of the interpolator batch and the time batch. If ``False``, use point-wise
            broadcasting.

        Returns
        -------
        tuple[Float[Array, "... 3"], Float[Array, "... 3"]]
            Interpolated Cartesian position and velocity in ``au`` and ``au / day``.

        Notes
        -----
        This is a thin alias of :`evaluate`.
        """
        return self.evaluate(jd1, jd2, grid)

    @property
    def shape(self):
        """Batch shape of the interpolator."""
        return self.t0_jd1.shape


def _ode_vector_field(t, state, args, force_model):
    """Return the first-order state derivative for the Cartesian ODE."""
    pos, vel = state
    acc = force_model(t, state, args)
    return vel, acc


class NumericalIntegrator(eqx.Module):
    """Numerical propagator for orbit integration.

    This wrapper provides a compact user-facing API around the underlying ``diffrax`` solvers and the custom ``IAS15``
    implementation. It builds the solver object, the step-size controller, and return the bidirectional dense-output integrator
    :class:`BiDirectionalInterpolator`.

    Notes
    -----
    The current implementation always enables dense output and uses forward-mode differentiation internally. These behaviors are part of the runtime contract of :meth:`__call__` and are therefore not exposed as constructor options.
    """
    solver: diffrax.AbstractSolver
    step_controller: diffrax.AbstractStepSizeController
    method: str = eqx.field(static=True)
    tol: Optional[float] = eqx.field(static=True)
    rtol: Optional[float] = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    initial_step: float = eqx.field(static=True)
    ias15_adaptive_mode: int = eqx.field(static=True)
    ias15_safety_factor: float = eqx.field(static=True)
    ias15_min_step: float = eqx.field(static=True)

    def __init__(self,
                 method: Literal["IAS15", "DOPRI8", "DOPRI5"] = 'IAS15',
                 tol: Optional[float] = 1e-9,
                 max_steps: int = 4096,
                 *,
                 rtol: Optional[float] = None,
                 atol: Optional[float] = None,
                 initial_step: float = 1e-6,
                 ias15_adaptive_mode: int = 2,
                 ias15_safety_factor: float = 0.25,
                 ias15_min_step: float = 0.0):
        """
        Create a numerical integrator with method-specific adaptive controls.

        Parameters
        ----------
        method : {"IAS15", "DOPRI8", "DOPRI5"}, default="IAS15"
            Integration method name. The value is case-insensitive and is normalized to upper case internally.

            - ``"IAS15"``: the complete implementation of IAS15 under the ``diffrax`` framework.
            - ``"DOPRI8"``: 8th-order Dormand-Prince solver from ``diffrax``.
            - ``"DOPRI5"``: 5th-order Dormand-Prince solver from ``diffrax``.
        tol : float or None, default=1e-9
            Compatibility shorthand tolerance.

            For ``"DOPRI8"`` and ``"DOPRI5"``, this acts as the fallback value used to fill missing ``rtol`` or ``atol``:

            - if only ``tol`` is given, then ``rtol = atol = tol``;
            - if ``tol`` and ``rtol`` are given, then ``atol = tol``;
            - if ``tol`` and ``atol`` are given, then ``rtol = tol``.

            For ``"IAS15"``, only the absolute tolerance is meaningful, so ``tol`` is interpreted as the fallback value for ``atol``.
        max_steps : int, default=4096
            Maximum number of adaptive steps allowed in each forward or backward solve.
        rtol : float or None, optional
            Relative tolerance for ``"DOPRI8"`` and ``"DOPRI5"``.

            If both ``rtol`` and ``atol`` are provided, they are used directly. If only ``rtol`` is provided, then ``tol`` must also be provided so that ``atol`` can be inferred. This option is invalid for ``"IAS15"`` and will raise ``ValueError``.
        atol : float or None, optional
            Absolute tolerance used by the adaptive step controller.

            For ``"DOPRI8"`` and ``"DOPRI5"``, this is the absolute component of ``diffrax.PIDController``.

            For ``"IAS15"``, this is the only effective error threshold and is passed to :class:`IAS15StepSizeController`. If omitted, the value is inferred from ``tol``.
        initial_step : float, default=1e-6
            Absolute value of the initial trial step size, in days.

            The propagator always starts the forward solve with ``+initial_step`` and the backward solve with ``-initial_step``. This parameter only controls the first trial step; subsequent step sizes are chosen adaptively by the selected controller.
        ias15_adaptive_mode : {1, 2}, default=2
            Adaptive step-size formula used by ``"IAS15"``.

            - ``1`` selects the Rein & Spiegel (2015)-style error ratio.
            - ``2`` selects the PRS23-based timescale heuristic currently used as the project default.

            This option is only valid when ``method="IAS15"``.
        ias15_safety_factor : float, default=0.25
            Acceptance threshold and growth limiter for the IAS15 adaptive controller. Larger values make the controller more conservative. This option is only valid when ``method="IAS15"``.
        ias15_min_step : float, default=0.0
            Lower bound for the next proposed IAS15 step size, in days. Use ``0.0`` to keep the current unrestricted behavior. This option is only valid when ``method="IAS15"``.

        Raises
        ------
        ValueError
            Raised when the integration method is unknown, when tolerance combinations are incomplete or incompatible with the selected method, or when a numeric option violates its valid range.

        Notes
        -----
        Method-specific options are validated eagerly during construction. In particular, IAS15-only options are rejected for non-IAS15 methods instead of being silently ignored.
        """
        method = str(method).upper()
        tol = coerce_optional_scalar_float("tol", tol)
        rtol = coerce_optional_scalar_float("rtol", rtol)
        atol = coerce_optional_scalar_float("atol", atol)
        max_steps = coerce_scalar_int("max_steps", max_steps)
        initial_step = coerce_scalar_float("initial_step", initial_step)
        ias15_adaptive_mode = coerce_scalar_int("ias15_adaptive_mode", ias15_adaptive_mode)
        ias15_safety_factor = coerce_scalar_float("ias15_safety_factor", ias15_safety_factor)
        ias15_min_step = coerce_scalar_float("ias15_min_step", ias15_min_step)

        self.method = method
        self.tol = tol
        self._validate_method(self.method)
        self._validate_scalar_option("max_steps", max_steps, allow_zero=False, integer_only=True)
        self._validate_scalar_option("initial_step", initial_step, allow_zero=False)
        self._validate_scalar_option("ias15_safety_factor", ias15_safety_factor, allow_zero=False)
        self._validate_scalar_option("ias15_min_step", ias15_min_step, allow_zero=True)
        self._validate_optional_positive("tol", tol)
        self._validate_optional_positive("rtol", rtol)
        self._validate_optional_positive("atol", atol)
        self._validate_ias15_mode(ias15_adaptive_mode)

        self.rtol, self.atol = self._resolve_tolerances(self.method, tol, rtol, atol)
        self.max_steps = max_steps
        self.initial_step = initial_step
        self.ias15_adaptive_mode = ias15_adaptive_mode
        self.ias15_safety_factor = ias15_safety_factor
        self.ias15_min_step = ias15_min_step
        self._validate_method_specific_options()
        self.solver, self.step_controller = self._setup_solver()

    def __repr__(self) -> str:
        repr_fields = [
            ("method", self.method),
            ("rtol", format_float_array(self.rtol) if self.rtol is not None else "none"),
            ("atol", format_float_array(self.atol)),
            ("max_steps", str(self.max_steps)),
            ("initial_step", format_float_array(self.initial_step)),
        ]
        if self.method == 'IAS15':
            repr_fields.extend([
                ("adaptive_mode", str(self.ias15_adaptive_mode)),
                ("safety_factor", format_float_array(self.ias15_safety_factor)),
                ("min_step", format_float_array(self.ias15_min_step)),
            ])
        repr_fields.extend([
            ("solver", self.solver.__class__.__name__),
            ("step_controller", self.step_controller.__class__.__name__),
        ])
        return build_repr(self.__class__.__name__, repr_fields)

    @staticmethod
    def _validate_method(method: str) -> None:
        if method not in SUPPORTED_INTEGRATOR_METHODS:
            supported = "', '".join(SUPPORTED_INTEGRATOR_METHODS)
            raise ValueError(
                f"Unknown method: {method}. Currently supported methods: '{supported}'"
            )

    @staticmethod
    def _validate_scalar_option(name: str, value, allow_zero: bool, integer_only: bool = False) -> None:
        if integer_only and not isinstance(value, int):
            raise ValueError(f"{name} must be an integer.")
        if allow_zero:
            is_valid = value >= 0
            relation = ">= 0"
        else:
            is_valid = value > 0
            relation = "> 0"
        if not is_valid:
            raise ValueError(f"{name} must be {relation}.")

    @classmethod
    def _validate_optional_positive(cls, name: str, value: Optional[float]) -> None:
        if value is None:
            return
        cls._validate_scalar_option(name, value, allow_zero=False)

    @staticmethod
    def _validate_ias15_mode(adaptive_mode: int) -> None:
        if adaptive_mode not in IAS15_ADAPTIVE_MODES:
            raise ValueError(
                f"ias15_adaptive_mode must be one of {IAS15_ADAPTIVE_MODES}."
            )

    @staticmethod
    def _resolve_tolerances(method: str,
                            tol: Optional[float],
                            rtol: Optional[float],
                            atol: Optional[float]) -> Tuple[Optional[float], float]:
        if method == 'IAS15':
            if rtol is not None:
                raise ValueError(f"{method} does not support rtol; use tol or atol.")
            resolved_atol = atol if atol is not None else tol
            if resolved_atol is None:
                raise ValueError(f"{method} requires tol or atol.")
            return None, resolved_atol

        if rtol is None and atol is None:
            if tol is None:
                raise ValueError("DOPRI methods require tol or both rtol and atol.")
            return tol, tol
        if rtol is not None and atol is not None:
            return rtol, atol
        if tol is None:
            raise ValueError(
                "If only one of rtol or atol is provided, tol must also be provided."
            )
        if rtol is not None:
            return rtol, tol
        return tol, atol

    def _validate_method_specific_options(self) -> None:
        if self.method == 'IAS15':
            return
        if self.ias15_adaptive_mode != 2:
            raise ValueError("ias15_adaptive_mode is only valid when method='IAS15'.")
        if self.ias15_safety_factor != 0.25:
            raise ValueError("ias15_safety_factor is only valid when method='IAS15'.")
        if self.ias15_min_step != 0.0:
            raise ValueError("ias15_min_step is only valid when method='IAS15'.")

    def _setup_solver(self):
        """Build the solver and adaptive step controller for the selected method."""
        if self.method == 'IAS15':
            solver = IAS15Solver()
            step_controller = IAS15StepSizeController(
                atol=self.atol,
                adaptive_mode=self.ias15_adaptive_mode,
                safety_factor=self.ias15_safety_factor,
                min_dt=self.ias15_min_step,
            )
        else:
            if self.method == 'DOPRI8':
                solver = diffrax.Dopri8()
            elif self.method == 'DOPRI5':
                solver = diffrax.Dopri5()
            step_controller = diffrax.PIDController(rtol=self.rtol, atol=self.atol)
        return solver, step_controller

    def _get_term(self, force_model: ForceModel) -> diffrax.AbstractTerm:
        """Build the ``diffrax`` term object for the selected solver."""
        if self.method == 'IAS15':
            return IAS15Term(force_model)
        vector_field = partial(_ode_vector_field, force_model=force_model)
        return diffrax.ODETerm(vector_field)

    def _solve_core(self, force_model: ForceModel, y0: Float[Array, "6"], t0_jd1: Float[Array, ""],
                    t0_jd2: Float[Array, ""], tau_fwd: Float[Array, ""],
                    tau_bwd: Float[Array, ""]) -> Tuple[diffrax.Solution, diffrax.Solution]:
        """Solve the forward and backward dense trajectories around one reference epoch.

        Parameters
        ----------
        force_model : ForceModel
            Force model used by the ODE right-hand side.
        y0 : Float[Array, "6"]
            Initial Cartesian state at the reference epoch, with position first and velocity second.
        t0_jd1, t0_jd2 : Float[Array, ""]
            Reference epoch in split Julian Date form.
        tau_fwd : Float[Array, ""]
            End time of the forward solve, measured from the reference epoch in days.
        tau_bwd : Float[Array, ""]
            End time of the backward solve, measured from the reference epoch in days. This value is non-positive.

        Returns
        -------
        tuple[diffrax.Solution, diffrax.Solution]
            Dense forward and backward solutions.

        Notes
        -----
        This method always enables dense output and uses ``diffrax.ForwardMode`` for differentiation.
        """
        term = self._get_term(force_model)
        args = (t0_jd1, t0_jd2)
        y0 = (y0[:3], y0[3:6])
        adjoint = diffrax.ForwardMode()
        saveat = diffrax.SaveAt(dense=True)
        sol_fwd = diffrax.diffeqsolve(
            term, self.solver, 0., tau_fwd, dt0=self.initial_step, y0=y0, args=args,
            stepsize_controller=self.step_controller, max_steps=self.max_steps,
            saveat=saveat, adjoint=adjoint
        )
        sol_bwd = diffrax.diffeqsolve(
            term, self.solver, 0., tau_bwd, dt0=-self.initial_step, y0=y0, args=args,
            stepsize_controller=self.step_controller, max_steps=self.max_steps,
            saveat=saveat, adjoint=adjoint
        )

        return sol_fwd, sol_bwd

    def _solve_single(self, force_model: ForceModel, y0: Float[Array, "6"], t0_jd1: Float[Array, ""],
                      t0_jd2: Float[Array, ""], t_start_jd1: Float[Array, ""], t_start_jd2: Float[Array, ""],
                      t_end_jd1: Float[Array, ""], t_end_jd2: Float[Array, ""]) -> BiDirectionalInterpolator:
        """Build one bidirectional interpolator over the requested time span.

        Parameters
        ----------
        force_model : ForceModel
            Force model used by the ODE right-hand side.
        y0 : Float[Array, "6"]
            Initial Cartesian state at the reference epoch, with position first and velocity second.
        t0_jd1, t0_jd2 : Float[Array, ""]
            Reference epoch in split Julian Date form.
        t_start_jd1, t_start_jd2 : Float[Array, ""]
            One end of the covered interval, in split Julian Date form.
        t_end_jd1, t_end_jd2 : Float[Array, ""]
            The other end of the covered interval, in split Julian Date form.

        Returns
        -------
        BiDirectionalInterpolator
            Interpolator that covers the closed interval between the two endpoint epochs.
        """
        tau_start = (t_start_jd1 - t0_jd1) + (t_start_jd2 - t0_jd2)
        tau_end = (t_end_jd1 - t0_jd1) + (t_end_jd2 - t0_jd2)
        tau_min = jnp.minimum(tau_start, tau_end)
        tau_max = jnp.maximum(tau_start, tau_end)
        tau_fwd = jnp.maximum(tau_max, 0.)
        tau_bwd = jnp.minimum(tau_min, 0.)
        sol_fwd, sol_bwd = self._solve_core(force_model, y0, t0_jd1, t0_jd2, tau_fwd, tau_bwd)
        is_start_lt_end = tau_start < tau_end
        t_min_jd1 = jnp.where(is_start_lt_end, t_start_jd1, t_end_jd1)
        t_min_jd2 = jnp.where(is_start_lt_end, t_start_jd2, t_end_jd2)
        t_max_jd1 = jnp.where(is_start_lt_end, t_end_jd1, t_start_jd1)
        t_max_jd2 = jnp.where(is_start_lt_end, t_end_jd2, t_start_jd2)
        return BiDirectionalInterpolator(t0_jd1=t0_jd1, t0_jd2=t0_jd2, t_min_jd1=t_min_jd1, t_min_jd2=t_min_jd2,
                                         t_max_jd1=t_max_jd1, t_max_jd2=t_max_jd2, sol_fwd=sol_fwd, sol_bwd=sol_bwd)

    @eqx.filter_jit
    def __call__(self, force_model: ForceModel, y0: Float[ArrayLike, "... 6"], t0_jd1: Float[ArrayLike, "..."],
                 t0_jd2: Float[ArrayLike, "..."], t_start_jd1: Float[ArrayLike, "..."],
                 t_start_jd2: Float[ArrayLike, "..."], t_end_jd1: Float[ArrayLike, "..."],
                 t_end_jd2: Float[ArrayLike, "..."], grid: bool = False) -> BiDirectionalInterpolator:
        """Integrate the orbit and build dense interpolators.

        Parameters
        ----------
        force_model : ForceModel
            Force model or batch of force models used by the ODE right-hand side.
        y0 : Float[ArrayLike, "... 6"]
            Initial Cartesian state or batch of states at the reference epoch, with position first and velocity second.
        t0_jd1, t0_jd2 : Float[ArrayLike, "..."]
            Reference epoch or batch of reference epochs in split Julian Date form.
        t_start_jd1, t_start_jd2 : Float[ArrayLike, "..."]
            One end of the covered interval, in split Julian Date form.
        t_end_jd1, t_end_jd2 : Float[ArrayLike, "..."]
            The other end of the covered interval, in split Julian Date form.
        grid : bool, default=False
            If ``False``, use point-wise broadcasting. If ``True``, use the Cartesian product of the input batches.

        Returns
        -------
        BiDirectionalInterpolator
            Dense interpolator or batch of interpolators for the requested propagation spans.

        Notes
        -----
        The returned interpolator covers the full closed interval between ``t_start`` and ``t_end``, even when the reference epoch lies inside that interval. Forward propagation covers epochs at or after ``t0``. Backward propagation covers epochs before ``t0``.
        """
        if not isinstance(force_model, ForceModel):
            raise TypeError("difforb.integrator.NumericalIntegrator requires a difforb.dynamics.ForceModel.")
        if not grid:
            return safe_dispatch(self._solve_single, (0, 1, 0, 0, 0, 0, 0, 0), force_model, y0,
                                 t0_jd1, t0_jd2,
                                 t_start_jd1,
                                 t_start_jd2, t_end_jd1, t_end_jd2)
        else:
            return safe_cartesian_dispatch(self._solve_single, ((0, 1), (force_model, y0)),
                                           ((0, 0, 0, 0, 0, 0), (t0_jd1, t0_jd2,
                                                                 t_start_jd1,
                                                                 t_start_jd2, t_end_jd1,
                                                                 t_end_jd2)))
