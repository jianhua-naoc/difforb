import pytest

from difforb.od.progress import print_solver_progress, solver_progress_reporter


def test_solver_progress_reporter_normalizes_supported_values():
    callback = lambda event, **data: None

    assert solver_progress_reporter(False) is None
    assert solver_progress_reporter(True) is print_solver_progress
    assert solver_progress_reporter(callback) is callback


@pytest.mark.parametrize("verbose", [None, 0, 1, "yes", object()])
def test_solver_progress_reporter_rejects_other_values(verbose):
    with pytest.raises(TypeError):
        solver_progress_reporter(verbose)


def test_default_progress_reporter_prints_supported_events(capsys):
    print_solver_progress(
        "least_squares_step",
        step=2,
        normalized_residual_rms=1.25,
        damping=0.01,
        next_damping=0.002,
    )
    print_solver_progress(
        "outlier_iteration",
        iteration=1,
        observation_count=8,
        inlier_count=7,
        outlier_count=1,
        normalized_residual_rms=0.75,
        mask_changed=True,
    )
    print_solver_progress(
        "result",
        index=3,
        termination_reason="correction_converged",
        lsq_iterations=4,
        outlier_iterations=1,
        normalized_residual_rms=0.5,
        inlier_count=7,
    )

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Least-squares step 2 accepted:")
    assert lines[1].startswith("Outlier iteration 1:")
    assert lines[2].startswith("Result 3:")


def test_default_progress_reporter_rejects_unknown_events():
    with pytest.raises(ValueError):
        print_solver_progress("unknown")
