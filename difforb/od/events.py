"""Structured event model for orbit-determination progress logs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

SolverLogDetail = Literal["quiet", "summary", "iter", "trial"]

SOLVER_LOG_DETAILS = ("quiet", "summary", "iter", "trial")
_SOLVER_LOG_DETAIL_ORDER = {name: index for index, name in enumerate(SOLVER_LOG_DETAILS)}
_SOLVER_EVENT_MIN_DETAIL = {
    "lsq_done": "summary",
    "lsq_failed": "summary",
    "outlier_done": "summary",
    "outlier_skipped": "summary",
    "outlier_disabled": "summary",
    "outlier_iteration_start": "iter",
    "outlier_update": "iter",
    "lsq_start": "iter",
    "lsq_step_accepted": "iter",
    "lm_trial_start": "trial",
    "lm_trial_rejected": "trial",
}


@dataclass(frozen=True)
class SolverEvent:
    """
    Structured progress event emitted by orbit-determination solvers.

    Parameters
    ----------
    component : str
        Solver component that produced the event.
    event : str
        Stable machine-readable event name.
    level : str
        Severity label such as ``"info"`` or ``"warning"``.
    context : Mapping[str, Any]
        Hierarchical solve context, such as stage, outlier iteration, LSQ solve,
        LSQ step, and damping trial.
    data : Mapping[str, Any]
        Event-specific scalar payload.
    """
    component: str
    event: str
    level: str = "info"
    context: Mapping[str, Any] = field(default_factory=dict)
    data: Mapping[str, Any] = field(default_factory=dict)


SolverEventHandler = Callable[[SolverEvent], None]


def normalize_solver_log_detail(log_detail: str) -> SolverLogDetail:
    """Normalize and validate one solver log-detail option."""
    detail = str(log_detail).strip().lower()
    if detail not in _SOLVER_LOG_DETAIL_ORDER:
        valid = "', '".join(SOLVER_LOG_DETAILS)
        raise ValueError(f"Unknown solver log detail: {log_detail!r}. Expected one of '{valid}'.")
    return detail


def solver_log_event_enabled(log_detail: str, event_name: str) -> bool:
    """Return whether one solver event should be emitted at the requested detail."""
    detail = normalize_solver_log_detail(log_detail)
    if detail == "quiet":
        return False
    min_detail = _SOLVER_EVENT_MIN_DETAIL.get(event_name, "iter")
    return _SOLVER_LOG_DETAIL_ORDER[detail] >= _SOLVER_LOG_DETAIL_ORDER[min_detail]


class SolverEventLogger:
    """
    Context-aware emitter for structured solver events.

    The logger owns filtering and hierarchical context. Numerical code can bind
    the context it knows, emit scalar event payloads, and leave text rendering
    to an event handler.
    """

    def __init__(self, handler: SolverEventHandler | None = None, log_detail: SolverLogDetail = "iter",
                 context: Mapping[str, Any] | None = None) -> None:
        """Initialize a solver event logger."""
        self.handler = handler
        self.log_detail = normalize_solver_log_detail(log_detail)
        self.context = dict(context or {})

    def bind(self, **context: Any) -> "SolverEventLogger":
        """Return a new logger with additional hierarchical context."""
        next_context = dict(self.context)
        next_context.update({key: value for key, value in context.items() if value is not None})
        return SolverEventLogger(self.handler, self.log_detail, next_context)

    def enabled(self, event_name: str) -> bool:
        """Return whether one event would be delivered to the handler."""
        return self.handler is not None and solver_log_event_enabled(self.log_detail, event_name)

    def emit(self, component: str, event: str, level: str = "info", **data: Any) -> None:
        """Emit one structured solver event if it passes the current filter."""
        if not self.enabled(event):
            return
        assert self.handler is not None
        self.handler(SolverEvent(component=component, event=event, level=level,
                                 context=dict(self.context), data=dict(data)))


def make_solver_event_logger(handler: SolverEventHandler | None,
                             log_detail: SolverLogDetail = "iter") -> SolverEventLogger:
    """Build a solver event logger from one optional event handler."""
    return SolverEventLogger(handler, log_detail)


def _format_sci(value: Any) -> str:
    """Format one numeric scalar for human-readable solver logs."""
    return f"{float(value):.3e}"


def _indent(depth: int) -> str:
    """Return the standard text indentation for one event depth."""
    return "  " * max(int(depth), 0)


def _lsq_depth(context: Mapping[str, Any]) -> int:
    """Return the LSQ text depth implied by one event context."""
    return 2 if "outlier_iteration" in context else 1


def render_solver_event_text(event: SolverEvent) -> str:
    """Render one solver event as a plain-text progress line."""
    ctx = event.context
    data = event.data
    name = event.event

    if name == "outlier_iteration_start":
        depth = 1
        return (f"{_indent(depth)}Outlier iteration {ctx.get('outlier_iteration', '?')} start: "
                f"inliers={data.get('inlier_count', '?')}/{data.get('observation_count', '?')}")
    if name == "outlier_update":
        depth = 1
        return (f"{_indent(depth)}Outlier update {ctx.get('outlier_iteration', '?')}: "
                f"normalized residual RMS={_format_sci(data['normalized_residual_rms'])}, "
                f"to_outlier={data.get('changed_to_outlier_count', '?')}, "
                f"to_inlier={data.get('changed_to_inlier_count', '?')}, "
                f"inliers={data.get('inlier_count', '?')}/{data.get('observation_count', '?')}")
    if name == "outlier_done":
        if data.get("stop_reason") == "max_iterations_reached":
            max_iterations = data.get("max_iterations", data.get("outlier_iteration_count", "?"))
            return f"{_indent(1)}Outlier rejection stopped: maximum iterations reached ({max_iterations})."
        return f"{_indent(1)}Outlier rejection done: mask unchanged after {data.get('outlier_iteration_count', '?')} iteration(s)."
    if name == "outlier_skipped":
        return (f"{_indent(1)}Outlier rejection skipped: Chi2 value unavailable "
                f"(LSQ reason: {data.get('lsq_termination_reason', data.get('stop_reason', 'unknown'))}).")
    if name == "outlier_disabled":
        return (f"{_indent(1)}Outlier rejection disabled: "
                f"inliers={data.get('inlier_count', '?')}/{data.get('observation_count', '?')}, "
                f"normalized residual RMS={_format_sci(data['normalized_residual_rms'])}")
    lsq_depth = _lsq_depth(ctx)
    if name == "lsq_start":
        return (f"{_indent(lsq_depth)}LSQ start: normalized residual RMS={_format_sci(data['normalized_residual_rms'])}, "
                f"damping={_format_sci(data['damping'])}, "
                f"inlier residuals={data.get('n_inlier_residuals', '?')}, params={data.get('n_params', '?')}")
    if name == "lsq_step_accepted":
        return (f"{_indent(lsq_depth)}LSQ step {ctx.get('lsq_step', data.get('step', '?'))} accepted: "
                f"normalized residual RMS {_format_sci(data['normalized_residual_rms_before'])} -> "
                f"{_format_sci(data['normalized_residual_rms_after'])}, "
                f"rho={_format_sci(data['rho'])}, next damping={_format_sci(data['next_damping'])}, "
                f"trials={data.get('damping_trials', '?')}")
    if name == "lsq_done":
        status = "converged" if data.get("converged", False) else "not converged"
        return (f"{_indent(lsq_depth)}LSQ done: {status}, reason={data.get('reason', 'unknown')}, "
                f"steps={data.get('steps', '?')}, "
                f"normalized residual RMS={_format_sci(data['normalized_residual_rms'])}, "
                f"covariance rank={data.get('cov_rank', '?')}/{data.get('n_params', '?')}")
    if name == "lsq_failed":
        return (f"{_indent(lsq_depth)}LSQ stopped: damping failed at step "
                f"{ctx.get('lsq_step', data.get('step', '?'))}, "
                f"normalized residual RMS={_format_sci(data['normalized_residual_rms'])}")

    trial_depth = lsq_depth + 1
    if name == "lm_trial_start":
        return (f"{_indent(trial_depth)}Trial {ctx.get('lsq_step', '?')}.{ctx.get('trial', '?')} start: "
                f"damping={_format_sci(data['damping'])}")
    if name == "lm_trial_rejected":
        return (f"{_indent(trial_depth)}Trial {ctx.get('lsq_step', '?')}.{ctx.get('trial', '?')} rejected: "
                f"normalized residual RMS={_format_sci(data['normalized_residual_rms'])}, rho={_format_sci(data['rho'])}")

    return f"{event.component}.{event.event}: {dict(data)}"


class CompositeSolverEventHandler:
    """Forward each solver event to multiple handlers."""

    def __init__(self, *handlers: SolverEventHandler | None) -> None:
        """Initialize the handler list, skipping missing handlers."""
        self.handlers = tuple(handler for handler in handlers if handler is not None)

    def __call__(self, event: SolverEvent) -> None:
        """Deliver one event to every configured handler."""
        for handler in self.handlers:
            handler(event)


class RunLogHandler:
    """
    Render solver events as plain text and pass them to a writer callback.

    This handler is intended for console output and human-readable run logs. It
    does not persist structured events.
    """

    def __init__(self, writer: Callable[[str], None]) -> None:
        """Initialize the text handler with one line-oriented writer."""
        self.writer = writer

    def __call__(self, event: SolverEvent) -> None:
        """Render and write one solver event."""
        self.writer(render_solver_event_text(event))


def print_solver_event(event: SolverEvent) -> None:
    """Print one solver event with the default text renderer."""
    print(render_solver_event_text(event))
