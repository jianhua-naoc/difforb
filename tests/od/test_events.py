import pytest

from difforb.od.events import (
    CompositeSolverEventHandler,
    RunLogHandler,
    SolverEvent,
    SolverEventLogger,
    make_solver_event_logger,
    normalize_solver_log_detail,
    render_solver_event_text,
    solver_log_event_enabled,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("quiet", "quiet"),
        (" Summary ", "summary"),
        ("ITER", "iter"),
        ("trial", "trial"),
    ],
)
def test_log_detail_normalization(raw, expected):
    assert normalize_solver_log_detail(raw) == expected


def test_log_detail_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown solver log detail"):
        normalize_solver_log_detail("verbose")


@pytest.mark.parametrize(
    ("detail", "event_name", "expected"),
    [
        ("quiet", "lsq_done", False),
        ("quiet", "lm_trial_start", False),
        ("summary", "lsq_done", True),
        ("summary", "outlier_disabled", True),
        ("summary", "lsq_start", False),
        ("summary", "unknown_event", False),
        ("iter", "lsq_done", True),
        ("iter", "lsq_start", True),
        ("iter", "outlier_update", True),
        ("iter", "lm_trial_start", False),
        ("iter", "unknown_event", True),
        ("trial", "lm_trial_start", True),
        ("trial", "lm_trial_rejected", True),
        ("trial", "unknown_event", True),
    ],
)
def test_event_filter_by_detail(detail, event_name, expected):
    assert solver_log_event_enabled(detail, event_name) is expected


def test_logger_bind_and_emit():
    events = []
    logger = SolverEventLogger(events.append, log_detail="iter", context={"stage": 2})
    bound = logger.bind(outlier_iteration=3, ignored=None)

    bound.emit("lsq", "lsq_start", normalized_residual_rms=1.25, damping=0.01, n_inlier_residuals=8, n_params=6)
    bound.emit("lsq", "lm_trial_start", damping=0.02)

    assert logger.context == {"stage": 2}
    assert bound.context == {"stage": 2, "outlier_iteration": 3}
    assert len(events) == 1
    assert events[0] == SolverEvent(
        component="lsq",
        event="lsq_start",
        level="info",
        context={"stage": 2, "outlier_iteration": 3},
        data={
            "normalized_residual_rms": 1.25,
            "damping": 0.01,
            "n_inlier_residuals": 8,
            "n_params": 6,
        },
    )


def test_logger_without_handler_is_disabled():
    logger = make_solver_event_logger(None, log_detail="trial")

    assert not logger.enabled("lsq_done")
    logger.emit("lsq", "lsq_done", normalized_residual_rms=0.0)


def test_composite_handler_forwards_events():
    first = []
    second = []
    event = SolverEvent("lsq", "lsq_done")
    handler = CompositeSolverEventHandler(first.append, None, second.append)

    handler(event)

    assert first == [event]
    assert second == [event]


def test_run_log_handler_renders_events():
    lines = []
    handler = RunLogHandler(lines.append)

    handler(
        SolverEvent(
            "lsq",
            "lsq_done",
            context={"stage": 1},
            data={
                "converged": True,
                "reason": "gradient_converged",
                "steps": 2,
                "normalized_residual_rms": 3.0e-4,
                "cov_rank": 6,
                "n_params": 6,
            },
        )
    )

    assert lines == [
        "  LSQ done: converged, reason=gradient_converged, steps=2, "
        "normalized residual RMS=3.000e-04, covariance rank=6/6"
    ]


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            SolverEvent(
                "outlier",
                "outlier_update",
                context={"outlier_iteration": 2},
                data={
                    "normalized_residual_rms": 0.0125,
                    "changed_to_outlier_count": 1,
                    "changed_to_inlier_count": 0,
                    "inlier_count": 7,
                    "observation_count": 8,
                },
            ),
            "  Outlier update 2: normalized residual RMS=1.250e-02, "
            "to_outlier=1, to_inlier=0, inliers=7/8",
        ),
        (
            SolverEvent(
                "lsq",
                "lsq_start",
                context={"stage": 1},
                data={
                    "normalized_residual_rms": 2.5,
                    "damping": 0.001,
                    "n_inlier_residuals": 12,
                    "n_params": 6,
                },
            ),
            "  LSQ start: normalized residual RMS=2.500e+00, damping=1.000e-03, "
            "inlier residuals=12, params=6",
        ),
        (
            SolverEvent(
                "lsq",
                "lsq_step_accepted",
                context={"outlier_iteration": 1, "lsq_step": 4},
                data={
                    "normalized_residual_rms_before": 1.0,
                    "normalized_residual_rms_after": 0.5,
                    "rho": 0.75,
                    "next_damping": 0.0002,
                    "damping_trials": 3,
                },
            ),
            "    LSQ step 4 accepted: normalized residual RMS 1.000e+00 -> 5.000e-01, "
            "rho=7.500e-01, next damping=2.000e-04, trials=3",
        ),
        (
            SolverEvent(
                "lsq",
                "lm_trial_rejected",
                context={"outlier_iteration": 1, "lsq_step": 4, "trial": 2},
                data={"normalized_residual_rms": 1.5, "rho": -0.1},
            ),
            "      Trial 4.2 rejected: normalized residual RMS=1.500e+00, rho=-1.000e-01",
        ),
        (
            SolverEvent("custom", "unknown_event", data={"value": 3}),
            "custom.unknown_event: {'value': 3}",
        ),
    ],
)
def test_render_event_text(event, expected):
    assert render_solver_event_text(event) == expected
