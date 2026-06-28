import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from difforb.core.constants import GM_SUN
from difforb.core.element import KepElement
from difforb.core.time.timescale import Time
from difforb.dynamics.force_model import Force, ForceModel
from difforb.dynamics.two_body import kepler_propagate
from difforb.integrator.ias15 import IAS15StepSizeController
from difforb.integrator.integrator import BiDirectionalInterpolator, NumericalIntegrator
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)


def _static_array_warning_messages(caught) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if "A JAX array is being set as static" in str(w.message)
    ]


class CentralGravityForce(Force):
    mu: float = eqx.field(static=True)

    def __init__(self, mu):
        self.mu = float(mu)

    def __call__(self, tdb_jd1, tdb_jd2, state, args):
        pos, _ = state
        r = jnp.linalg.norm(pos)
        return -self.mu * pos / (r * r * r)

    @property
    def shape(self):
        return ()


class ConstantAccelerationForce(Force):
    acceleration: jax.Array

    def __init__(self, acceleration):
        self.acceleration = jnp.asarray(acceleration, dtype=float)

    def __call__(self, tdb_jd1, tdb_jd2, state, args):
        return self.acceleration

    @property
    def shape(self):
        return self.acceleration.shape[:-1]


@pytest.mark.parametrize(
    ("method", "kwargs", "pos_atol", "vel_atol"),
    [
        ("IAS15", {"tol": 1.0e-12, "initial_step": 0.05, "max_steps": 4096}, 3.0e-11, 3.0e-12),
        ("DOPRI8", {"tol": 1.0e-12, "initial_step": 0.05, "max_steps": 4096}, 3.0e-11, 3.0e-12),
        ("DOPRI5", {"tol": 1.0e-11, "initial_step": 0.05, "max_steps": 4096}, 3.0e-10, 3.0e-11),
    ],
)
def test_integrator_two_body_against_kepler_propagate(method, kwargs, pos_atol, vel_atol):
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    element = KepElement.from_classical(
        tdb=tdb,
        a=2.1,
        e=0.25,
        inc=12.0,
        node=75.0,
        peri=140.0,
        m=35.0,
    )
    state = element.state()
    y0 = jnp.concatenate([state.pos, state.vel])
    t0_jd1 = jnp.asarray(2460741.0, dtype=float)
    t0_jd2 = jnp.asarray(0.5, dtype=float)
    force_model = ForceModel([CentralGravityForce(GM_SUN)])
    integrator = NumericalIntegrator(method=method, **kwargs)

    trajectory = integrator(
        force_model,
        y0,
        t0_jd1,
        t0_jd2,
        t0_jd1,
        t0_jd2 - 12.0,
        t0_jd1,
        t0_jd2 + 18.0,
    )
    offsets = jnp.asarray([-10.0, 0.0, 15.0], dtype=float)
    actual_pos, actual_vel = trajectory.evaluate(t0_jd1, t0_jd2 + offsets)
    expected_pos, expected_vel = kepler_propagate(
        jnp.broadcast_to(state.pos, (offsets.shape[0], 3)),
        jnp.broadcast_to(state.vel, (offsets.shape[0], 3)),
        offsets,
        GM_SUN,
    )

    print(
        "[integrator.two_body] "
        f"method={method:<6} "
        f"pos_max_abs_diff={float(jnp.max(jnp.abs(actual_pos - expected_pos))):+.12e} au "
        f"vel_max_abs_diff={float(jnp.max(jnp.abs(actual_vel - expected_vel))):+.12e} au/day"
    )

    assert isinstance(trajectory, BiDirectionalInterpolator)
    assert trajectory.shape == ()
    assert_allclose(actual_pos, expected_pos, atol=pos_atol, rtol=0.0)
    assert_allclose(actual_vel, expected_vel, atol=vel_atol, rtol=0.0)


def test_integrator_constant_acceleration_closed_form():
    acceleration = jnp.asarray([2.0e-6, -3.0e-6, 4.0e-6], dtype=float)
    force_model = ForceModel([ConstantAccelerationForce(acceleration)])
    integrator = NumericalIntegrator(method="DOPRI8", tol=1.0e-13, initial_step=0.05, max_steps=4096)
    pos0 = jnp.asarray([1.0, -2.0, 0.5], dtype=float)
    vel0 = jnp.asarray([0.01, 0.02, -0.03], dtype=float)
    y0 = jnp.concatenate([pos0, vel0])
    t0_jd1 = jnp.asarray(2460741.0, dtype=float)
    t0_jd2 = jnp.asarray(0.5, dtype=float)

    trajectory = integrator(
        force_model,
        y0,
        t0_jd1,
        t0_jd2,
        t0_jd1,
        t0_jd2 - 2.0,
        t0_jd1,
        t0_jd2 + 3.0,
    )
    offsets = jnp.asarray([-1.5, 0.0, 2.5], dtype=float)
    actual_pos, actual_vel = trajectory.evaluate(t0_jd1, t0_jd2 + offsets)
    expected_pos = pos0 + offsets[:, None] * vel0 + 0.5 * offsets[:, None] * offsets[:, None] * acceleration
    expected_vel = vel0 + offsets[:, None] * acceleration

    print(
        "[integrator.constant_acc] "
        f"pos_max_abs_diff={float(jnp.max(jnp.abs(actual_pos - expected_pos))):+.12e} au "
        f"vel_max_abs_diff={float(jnp.max(jnp.abs(actual_vel - expected_vel))):+.12e} au/day"
    )

    assert_allclose(actual_pos, expected_pos, atol=1.0e-13, rtol=0.0)
    assert_allclose(actual_vel, expected_vel, atol=1.0e-13, rtol=0.0)


def test_bidirectional_interpolator_coverage_and_reversed_span():
    acceleration = jnp.asarray([1.0e-6, 0.0, -2.0e-6], dtype=float)
    force_model = ForceModel([ConstantAccelerationForce(acceleration)])
    integrator = NumericalIntegrator(method="DOPRI8", tol=1.0e-13, initial_step=0.05, max_steps=4096)
    pos0 = jnp.asarray([0.2, -0.3, 0.4], dtype=float)
    vel0 = jnp.asarray([0.005, -0.006, 0.007], dtype=float)
    y0 = jnp.concatenate([pos0, vel0])
    t0_jd1 = jnp.asarray(2460741.0, dtype=float)
    t0_jd2 = jnp.asarray(0.5, dtype=float)

    trajectory = integrator(
        force_model,
        y0,
        t0_jd1,
        t0_jd2,
        t0_jd1,
        t0_jd2 + 4.0,
        t0_jd1,
        t0_jd2 - 3.0,
    )
    coverage_offsets = jnp.asarray([-3.0, 0.0, 4.0, -3.01, 4.01], dtype=float)
    actual_covered = trajectory.is_covered(t0_jd1, t0_jd2 + coverage_offsets)
    expected_covered = jnp.asarray([True, True, True, False, False])

    sample_offsets = jnp.asarray([-2.5, 0.0, 3.5], dtype=float)
    actual_pos, actual_vel = trajectory.evaluate(t0_jd1, t0_jd2 + sample_offsets)
    expected_pos = pos0 + sample_offsets[:, None] * vel0 + 0.5 * sample_offsets[:, None] * sample_offsets[:, None] * acceleration
    expected_vel = vel0 + sample_offsets[:, None] * acceleration

    print(
        "[integrator.coverage] "
        f"covered={actual_covered.tolist()} "
        f"pos_max_abs_diff={float(jnp.max(jnp.abs(actual_pos - expected_pos))):+.12e} au"
    )

    assert_allclose(actual_covered, expected_covered, atol=0.0, rtol=0.0)
    assert_allclose(actual_pos, expected_pos, atol=1.0e-13, rtol=0.0)
    assert_allclose(actual_vel, expected_vel, atol=1.0e-13, rtol=0.0)


def test_integrator_pointwise_batch_shape():
    acceleration = jnp.asarray([0.0, 0.0, 0.0], dtype=float)
    force_model = ForceModel([ConstantAccelerationForce(acceleration)])
    integrator = NumericalIntegrator(method="DOPRI8", tol=1.0e-12, initial_step=0.05, max_steps=4096)
    pos0 = jnp.asarray([[1.0, 0.0, 0.0], [2.0, -1.0, 0.5]], dtype=float)
    vel0 = jnp.asarray([[0.01, 0.02, 0.03], [-0.02, 0.01, 0.04]], dtype=float)
    y0 = jnp.concatenate([pos0, vel0], axis=-1)
    t0_jd1 = jnp.asarray([2460741.0, 2460741.0], dtype=float)
    t0_jd2 = jnp.asarray([0.5, 0.5], dtype=float)

    trajectory = integrator(
        force_model,
        y0,
        t0_jd1,
        t0_jd2,
        t0_jd1,
        t0_jd2 - jnp.asarray([1.0, 2.0], dtype=float),
        t0_jd1,
        t0_jd2 + jnp.asarray([2.0, 3.0], dtype=float),
    )
    offsets = jnp.asarray([1.5, -1.0], dtype=float)
    actual_pos, actual_vel = trajectory.evaluate(t0_jd1, t0_jd2 + offsets)
    expected_pos = pos0 + offsets[:, None] * vel0

    print(
        "[integrator.batch] "
        f"trajectory_shape={trajectory.shape} "
        f"pos_shape={actual_pos.shape} "
        f"vel_shape={actual_vel.shape}"
    )

    assert trajectory.shape == (2,)
    assert actual_pos.shape == (2, 3)
    assert actual_vel.shape == (2, 3)
    assert_allclose(actual_pos, expected_pos, atol=1.0e-13, rtol=0.0)
    assert_allclose(actual_vel, vel0, atol=1.0e-13, rtol=0.0)


def test_integrator_grid_shape():
    force_model = ForceModel([ConstantAccelerationForce([0.0, 0.0, 0.0])])
    integrator = NumericalIntegrator(method="DOPRI5", tol=1.0e-10, initial_step=0.05, max_steps=4096)
    pos0 = jnp.asarray([[1.0, 0.0, 0.0], [2.0, -1.0, 0.5]], dtype=float)
    vel0 = jnp.asarray([[0.01, 0.02, 0.03], [-0.02, 0.01, 0.04]], dtype=float)
    y0 = jnp.concatenate([pos0, vel0], axis=-1)
    t0_jd1 = jnp.asarray([2460741.0, 2460741.0], dtype=float)
    t0_jd2 = jnp.asarray([0.5, 0.5], dtype=float)

    trajectory = integrator(
        force_model,
        y0,
        t0_jd1,
        t0_jd2,
        t0_jd1,
        t0_jd2 - jnp.asarray([1.0, 2.0], dtype=float),
        t0_jd1,
        t0_jd2 + jnp.asarray([2.0, 3.0], dtype=float),
        grid=True,
    )
    query_offsets = jnp.asarray([-0.5, 0.0, 1.5], dtype=float)
    actual_pos, actual_vel = trajectory.evaluate(2460741.0, 0.5 + query_offsets, grid=True)

    print(
        "[integrator.grid] "
        f"trajectory_shape={trajectory.shape} "
        f"pos_shape={actual_pos.shape} "
        f"vel_shape={actual_vel.shape}"
    )

    assert trajectory.shape == (2, 2)
    assert actual_pos.shape == (2, 2, 3, 3)
    assert actual_vel.shape == (2, 2, 3, 3)


@pytest.mark.parametrize(
    ("method", "kwargs", "expected_method", "expected_rtol", "expected_atol", "solver_name", "controller_name"),
    [
        ("ias15", {"tol": 1.0e-10}, "IAS15", None, 1.0e-10, "IAS15Solver", "IAS15StepSizeController"),
        ("DOPRI8", {"tol": 3.0e-10}, "DOPRI8", 3.0e-10, 3.0e-10, "Dopri8", "PIDController"),
        ("DOPRI5", {"tol": None, "rtol": 4.0e-10, "atol": 5.0e-11}, "DOPRI5", 4.0e-10, 5.0e-11, "Dopri5", "PIDController"),
    ],
)
def test_numerical_integrator_resolves_options(
        method,
        kwargs,
        expected_method,
        expected_rtol,
        expected_atol,
        solver_name,
        controller_name,
):
    integrator = NumericalIntegrator(method=method, **kwargs)
    text = repr(integrator)

    print(
        "[integrator.options] "
        f"method={integrator.method:<6} "
        f"solver={integrator.solver.__class__.__name__} "
        f"controller={integrator.step_controller.__class__.__name__}"
    )

    assert integrator.method == expected_method
    assert integrator.rtol == expected_rtol
    assert integrator.atol == expected_atol
    assert integrator.solver.__class__.__name__ == solver_name
    assert integrator.step_controller.__class__.__name__ == controller_name
    assert expected_method in text
    assert solver_name in text


def test_integrator_normalizes_jax_scalar_static_options():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        integrator = NumericalIntegrator(
            method="IAS15",
            tol=jnp.asarray(1.0e-10),
            max_steps=jnp.asarray(128, dtype=jnp.int32),
            initial_step=jnp.asarray(1.0e-6),
            ias15_adaptive_mode=jnp.asarray(2, dtype=jnp.int32),
            ias15_safety_factor=jnp.asarray(0.25),
            ias15_min_step=jnp.asarray(0.0),
        )
        controller = IAS15StepSizeController(adaptive_mode=jnp.asarray(2, dtype=jnp.int32))

    assert _static_array_warning_messages(caught) == []
    assert isinstance(integrator.tol, float)
    assert isinstance(integrator.max_steps, int)
    assert isinstance(integrator.initial_step, float)
    assert isinstance(integrator.ias15_adaptive_mode, int)
    assert isinstance(controller.adaptive_mode, int)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "RK4"}, "Unknown method"),
        ({"max_steps": 0}, "max_steps must be"),
        ({"initial_step": 0.0}, "initial_step must be"),
        ({"tol": None}, "IAS15 requires tol or atol"),
        ({"method": "IAS15", "rtol": 1.0e-10}, "IAS15 does not support rtol"),
        ({"method": "DOPRI8", "tol": None, "rtol": 1.0e-10}, "If only one of rtol or atol"),
        ({"method": "DOPRI8", "ias15_adaptive_mode": 1}, "ias15_adaptive_mode is only valid"),
        ({"method": "DOPRI8", "ias15_min_step": 1.0e-6}, "ias15_min_step is only valid"),
    ],
)
def test_numerical_integrator_rejects_invalid_options(kwargs, match):
    print(f"[integrator.invalid_options] kwargs={kwargs}")

    with pytest.raises(ValueError, match=match):
        NumericalIntegrator(**kwargs)
