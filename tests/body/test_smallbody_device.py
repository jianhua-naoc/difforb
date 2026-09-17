"""Propagation placement and differentiation on CPU and optional CUDA devices.

Run with ``JAX_NUM_CPU_DEVICES=2`` to exercise transfers between CPU devices.
"""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difforb.body.ephbody import EphemerisBody
from difforb.body.smallbody import SmallBody
from difforb.core.state.frame import BCRS
from difforb.core.state.state import State
from difforb.core.time.timescale import Time
from difforb.dynamics.force_model import (
    EarthJ2Perturbation, Force, ForceModel, NewtonianGravity, PPNGravity,
    RTNDistanceLawNonGravEffect, SolarJ2Perturbation,
)
from difforb.integrator import NumericalIntegrator
from difforb.spk.spk import Ephemeris


class ConstantAcceleration(Force):
    acceleration: jax.Array

    def __call__(self, jd1, jd2, state, args):
        return self.acceleration

    @property
    def shape(self):
        return self.acceleration.shape[:-1]


def place(tree, device):
    arrays, static = eqx.partition(tree, eqx.is_array)
    return eqx.combine(jax.device_put(arrays, device), static)


def assert_device(tree, device):
    arrays = [x for x in jax.tree_util.tree_leaves(tree) if eqx.is_array(x)]
    assert arrays
    assert all(x.devices() == {device} for x in arrays)
    assert all(x.dtype == jnp.float64 for x in arrays if jnp.issubdtype(x.dtype, jnp.floating))


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cpu":
        return jax.devices("cpu")[-1]
    try:
        return jax.devices("cuda")[0]
    except RuntimeError:
        pytest.skip("CUDA is not available")


@pytest.fixture
def inputs():
    with jax.default_device(jax.devices("cpu")[0]):
        epoch = Time.from_tdb_jd(2460741.5, 0.0).tdb()
        start = Time.from_tdb_jd(2460741.5, -1.0).tdb()
        end = Time.from_tdb_jd(2460741.5, 1.0).tdb()
        query = Time.from_tdb_jd(2460741.5, 0.5).tdb()
        body = SmallBody(State(tdb=epoch, pos=jnp.array([1.0, 0.2, -0.1]),
                               vel=jnp.array([0.0, 0.01, 0.002]), frame=BCRS))
        force = ForceModel([ConstantAcceleration(jnp.array([0.001, 0.002, -0.001]))])
    return place((body, force, start, end, query), jax.devices("cpu")[0])


@pytest.mark.parametrize("method", ["IAS15", "DOPRI5", "DOPRI8"])
def test_placement_and_forward_differentiation(inputs, device, method):
    body, force, start, end, query = inputs
    integrator = NumericalIntegrator(method=method, tol=1e-11, initial_step=0.1, max_steps=64)
    result = body.propagate(start, end, force, integrator, device=device)
    assert_device(result, device)
    assert_device(body, jax.devices("cpu")[0])
    assert_device(force, jax.devices("cpu")[0])
    assert body.trajectory is None

    reference = body.propagate(start, end, force, integrator).state(query)
    actual = result.state(place(query, device))
    np.testing.assert_allclose(actual.array, reference.array, atol=2e-13, rtol=0)

    def forward(params, offset):
        target = eqx.tree_at(lambda b: (b.orbit0.pos, b.orbit0.vel), body, (params[:3], params[3:6]))
        model = ForceModel([ConstantAcceleration(params[6:])])
        propagated = target.propagate(start, end, model, integrator, device=device)
        offset = jax.device_put(offset, device)
        p, v = propagated._bcrs_pv_jd(
            propagated.trajectory.t0_jd1, propagated.trajectory.t0_jd2 + offset,
        )
        return jnp.concatenate((p, v))

    params = jnp.concatenate((body.orbit0.array, force.forces[0].acceleration))
    target_params, offset = place((params, jnp.asarray(0.5)), device)
    expected = np.concatenate((
        np.block([[np.eye(3), 0.5 * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]]),
        np.concatenate((0.125 * np.eye(3), 0.5 * np.eye(3))),
    ), axis=1)
    for derivative in (jax.jit(jax.jacfwd(forward)), jax.jacfwd(jax.jit(forward))):
        jac = derivative(target_params, offset)
        assert_device(jac, device)
        np.testing.assert_allclose(jac, expected, atol=3e-13, rtol=0)

    # The differentiated function itself transfers the original CPU inputs.
    jac = jax.jacfwd(forward)(params, jnp.asarray(0.5))
    np.testing.assert_allclose(jac, expected, atol=3e-13, rtol=0)
    time_jac = jax.jit(jax.jacfwd(forward, argnums=1))(target_params, offset)
    expected_time = np.concatenate((np.asarray(params[3:6] + 0.5 * params[6:]), np.asarray(params[6:])))
    np.testing.assert_allclose(time_jac, expected_time, atol=3e-13, rtol=0)
    _, tangent = jax.jvp(jax.jit(forward), (target_params, offset),
                         (jnp.ones_like(target_params), jnp.zeros_like(offset)))
    np.testing.assert_allclose(tangent, expected.sum(axis=1), atol=3e-13, rtol=0)


@pytest.mark.parametrize("grid", [False, True])
def test_batch_and_repropagation(inputs, device, grid):
    body, _, start, end, query = inputs
    with jax.default_device(jax.devices("cpu")[0]):
        epoch = Time.from_tdb_jd(jnp.full(2, 2460741.5), jnp.zeros(2)).tdb()
        body = SmallBody(State(tdb=epoch, pos=jnp.stack((body.orbit0.pos, body.orbit0.pos * 2)),
                               vel=jnp.stack((body.orbit0.vel, body.orbit0.vel * 2)), frame=BCRS))
        force = ForceModel([ConstantAcceleration(jnp.array([[0.001, 0., 0.], [0., 0.002, 0.]]))])
        query = Time.from_tdb_jd(jnp.full(2, 2460741.5), jnp.array([-0.5, 0.5])).tdb()
    integrator = NumericalIntegrator(tol=1e-11, initial_step=0.1, max_steps=64)
    original = body.propagate(start, end, force, integrator)
    result = original.propagate(query, end, force, integrator, grid=grid, device=device)
    assert_device(result, device)
    assert_device(original, jax.devices("cpu")[0])
    expected = body.propagate(query, end, force, integrator, grid=grid).state(end)
    actual = result.state(place(end, device))
    assert actual.shape == ((2, 2) if grid else (2,))
    np.testing.assert_allclose(actual.array, expected.array, atol=2e-13, rtol=0)


@pytest.mark.parametrize("gravity_type", [NewtonianGravity, PPNGravity])
def test_spk_forces_and_parameter_derivatives(inputs, device, gravity_type):
    path = Path(__file__).resolve().parents[1] / "data/spk/de441_2017_2025_excerpt.bsp"
    if not path.exists():
        pytest.skip("local DE441 SPK excerpt is not installed")
    body, _, start, end, query = inputs
    with jax.default_device(jax.devices("cpu")[0]):
        ephemeris = Ephemeris(str(path))
        sun = EphemerisBody("sun", eph=ephemeris)
        earth = EphemerisBody("earth", eph=ephemeris)
        earth_state = earth.state(body.orbit0.tdb)
        body = SmallBody(State(tdb=body.orbit0.tdb, pos=earth_state.pos + jnp.array([0.01, 0.002, -0.001]),
                               vel=earth_state.vel + jnp.array([0., 0.001, 0.]), frame=BCRS))
        model = ForceModel([
            gravity_type([sun, earth]),
            SolarJ2Perturbation(sun),
            EarthJ2Perturbation(earth, pole_unit_vec=jnp.array([0.002, 0.001, 1.])),
            RTNDistanceLawNonGravEffect(sun, A1=1e-10, A2=-2e-10, A3=3e-10),
        ])
    body, model = place((body, model), jax.devices("cpu")[0])
    assert_device(model, jax.devices("cpu")[0])
    params = jnp.concatenate((body.orbit0.array, model.get_all_estimated_params() / 1e-10))
    params = jax.device_put(params, jax.devices("cpu")[0])
    integrator = NumericalIntegrator(tol=1e-12, initial_step=0.05, max_steps=128)

    def forward(params, selected_device):
        target = eqx.tree_at(lambda b: (b.orbit0.pos, b.orbit0.vel), body, (params[:3], params[3:6]))
        forces = model.update_estimated_params(params[6:] * 1e-10)
        target = target.propagate(start, end, forces, integrator, device=selected_device)
        times = query if selected_device is None else place(query, selected_device)
        return target.state(times).array

    reference = lambda p: forward(p, None)
    on_device = lambda p: forward(p, device)
    placed_params = jax.device_put(params, device)
    actual = jax.jit(on_device)(placed_params)
    assert_device(actual, device)
    np.testing.assert_allclose(actual, jax.jit(reference)(params), atol=3e-12, rtol=0)
    jac = jax.jit(jax.jacfwd(on_device))(placed_params)
    ref_jac = jax.jit(jax.jacfwd(reference))(params)
    np.testing.assert_allclose(jac, ref_jac, atol=3e-11, rtol=1e-10)
    assert np.all(np.isfinite(jac))
    assert np.linalg.norm(np.asarray(jac[:, 6:])) > 0
    direction = jax.device_put(jnp.array([1., -0.5, 0.2, 0.1, -0.2, 0.3, 0., 0., 0.]), device)
    step = 1e-6
    finite_difference = (jax.jit(on_device)(placed_params + step * direction)
                         - jax.jit(on_device)(placed_params - step * direction)) / (2 * step)
    np.testing.assert_allclose(np.asarray(jac) @ np.asarray(direction), finite_difference, atol=3e-8, rtol=0)
    assert_device(model, jax.devices("cpu")[0])


def test_invalid_device(inputs):
    body, force, start, end, _ = inputs
    with pytest.raises(TypeError, match="jax.Device"):
        body.propagate(start, end, force, NumericalIntegrator(), device="cpu")
