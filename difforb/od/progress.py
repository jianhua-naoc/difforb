"""Plain-language progress reporting for orbit determination."""

from collections.abc import Callable
from typing import Any


SolverReporter = Callable[..., None]


def print_solver_progress(event: str, **data: Any) -> None:
    """Print one supported solver progress update."""
    if event == "least_squares_step":
        print(
            f"Least-squares step {data['step']} accepted: "
            f"normalized residual RMS={data['normalized_residual_rms']:.6g}, "
            f"damping={data['damping']:.3g} -> {data['next_damping']:.3g}"
        )
        return
    if event == "outlier_iteration":
        changed = "changed" if data["mask_changed"] else "unchanged"
        print(
            f"Outlier iteration {data['iteration']}: "
            f"inliers={data['inlier_count']}/{data['observation_count']}, "
            f"outliers={data['outlier_count']}, mask={changed}, "
            f"normalized residual RMS={data['normalized_residual_rms']:.6g}"
        )
        return
    if event == "result":
        prefix = "Result" if "index" not in data else f"Result {data['index']}"
        print(
            f"{prefix}: reason={data['termination_reason']}, "
            f"steps={data['lsq_iterations']}/{data['outlier_iterations']}, "
            f"normalized residual RMS={data['normalized_residual_rms']:.6g}, "
            f"inliers={data['inlier_count']}"
        )
        return
    raise ValueError(f"Unknown solver progress event: {event!r}.")


def solver_progress_reporter(
        verbose: bool | SolverReporter,
) -> SolverReporter | None:
    """Normalize a progress option to a reporter callback or ``None``."""
    if verbose is False:
        return None
    if verbose is True:
        return print_solver_progress
    if callable(verbose):
        return verbose
    raise TypeError("`verbose` must be a bool or callable.")


__all__ = ["SolverReporter", "print_solver_progress", "solver_progress_reporter"]
