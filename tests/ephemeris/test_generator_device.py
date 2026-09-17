"""End-to-end FP64 ephemerides on CPU and optional CUDA devices."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difforb.spk as spk
from difforb.body.ephbody import EphemerisBody
from difforb.body.site import Site
from difforb.body.smallbody import SmallBody
from difforb.core.element import KepElement
from difforb.core.time.timescale import Time
from difforb.dynamics.force_model import ForceModel, PPNGravity
from difforb.ephemeris.generator import EphemerisGenerator
from difforb.integrator import NumericalIntegrator
from difforb.spk.spk import Ephemeris
from tests.body.test_smallbody_device import assert_device, place


@pytest.fixture(params=["cpu", "cuda"])
def selected_device(request):
    if request.param == "cpu":
        return jax.devices("cpu")[-1]
    try:
        return jax.devices("cuda")[0]
    except RuntimeError:
        pytest.skip("CUDA is not available")


@pytest.fixture(scope="module")
def scene():
    path = Path(__file__).resolve().parents[1] / "data/spk/de441_2017_2025_excerpt.bsp"
    if not path.exists():
        pytest.skip("local DE441 SPK excerpt is not installed")
    with jax.default_device(jax.devices("cpu")[0]):
        ephemeris = Ephemeris(str(path))
        spk.set_default_ephemeris(ephemeris)
        sun = EphemerisBody("sun")
        earth = EphemerisBody("earth")
        epoch = Time.from_tdb_jd(2460690.5, 0.).tdb()
        body = SmallBody.create(KepElement.from_classical(epoch, 2.15, .22, 10., 75., 140., 35.), sun=sun)
        model = ForceModel([PPNGravity([sun, earth])])
        start = Time.from_tdb_jd(2460690.5, -1.).tdb()
        end = Time.from_tdb_jd(2460690.5, 1.).tdb()
        times = Time.from_tt_jd(jnp.full(2, 2460690.5), jnp.array([.2, .4]))
        sites = Site.from_code(["-14", "-13"]).require_ground()
    yield place((body, model, start, end, times, sites), jax.devices("cpu")[0])
    spk.clear_default_ephemeris()


def table_for(generator, kind, times, sites, grid=False):
    if kind == "elements":
        return generator.elements_table(times.tdb(), grid=grid)
    if kind.startswith("radar"):
        epoch_at = "transmit" if kind == "radar_transmit" else "receive"
        return generator.radar_table(times, sites, tx_freq=8.56e9, epoch_at=epoch_at, grid=grid)
    return getattr(generator, f"{kind}_table")(times, sites, grid=grid)


def observables(table, kind):
    if kind == "elements":
        return table.array
    if kind == "vector":
        return jnp.concatenate((table.geometric.pos, table.astrometric.pos, table.apparent.pos,
                                table.apparent.vel, table.light_time[..., None]), axis=-1)
    if kind == "optical":
        fields = ("astrometric_ra", "astrometric_dec", "apparent_ra", "apparent_dec",
                  "azimuth", "elevation", "delta", "r", "phase_angle", "elongation")
    else:
        fields = ("radar_range", "radar_rate", "radar_doppler", "tx_azimuth",
                  "tx_elevation", "rx_azimuth", "rx_elevation")
    return jnp.stack([getattr(table, name) for name in fields], axis=-1)


@pytest.mark.parametrize("kind", ["vector", "optical", "radar", "radar_transmit", "elements"])
def test_table_placement_and_grid(scene, selected_device, kind):
    body, model, start, end, times, sites = scene
    integrator = NumericalIntegrator(tol=1e-12, initial_step=.05, max_steps=64)
    reference = EphemerisGenerator(body.propagate(start, end, model, integrator))
    actual = EphemerisGenerator(body.propagate(start, end, model, integrator, device=selected_device))
    actual, placed_times, placed_sites = place((actual, times, sites), selected_device)
    assert_device(actual, selected_device)
    for grid in (False, True):
        expected = table_for(reference, kind, times, sites, grid)
        result = table_for(actual, kind, placed_times, placed_sites, grid)
        assert result.shape == ((2, 2) if grid and kind != "elements" else (2,))
        assert_device(result, selected_device)
        for x, y in zip(jax.tree_util.tree_leaves(result), jax.tree_util.tree_leaves(expected)):
            if eqx.is_array(x):
                np.testing.assert_allclose(x, y, atol=2e-9, rtol=2e-12, equal_nan=True)
    assert_device(body, jax.devices("cpu")[0])


@pytest.mark.parametrize("kind", ["vector", "optical", "radar", "radar_transmit", "elements"])
def test_propagation_to_table_forward_ad(scene, selected_device, kind):
    body, model, start, end, times, sites = scene
    integrator = NumericalIntegrator(tol=1e-12, initial_step=.05, max_steps=64)
    template = EphemerisGenerator(body)

    def forward(params, device):
        target = eqx.tree_at(lambda b: (b.orbit0.pos, b.orbit0.vel), body, (params[:3], params[3:6]))
        target = target.propagate(start, end, model, integrator, device=device)
        generator = eqx.tree_at(lambda g: g.target, place(template, device), target)
        query, observers = place((times, sites), device)
        query = query + params[6]
        return observables(table_for(generator, kind, query, observers), kind)

    cpu_params = jax.device_put(jnp.concatenate((body.orbit0.array, jnp.zeros(1))), jax.devices("cpu")[0])
    params = jax.device_put(cpu_params, selected_device)
    evaluate = jax.jit(lambda p: forward(p, selected_device))
    result = evaluate(params)
    jacobian = jax.jit(jax.jacfwd(evaluate))(params)
    reference = jax.jit(jax.jacfwd(lambda p: forward(p, jax.devices("cpu")[0])))(cpu_params)
    assert_device((result, jacobian), selected_device)
    assert np.isfinite(np.asarray(jacobian)).all()
    np.testing.assert_allclose(jacobian, reference, atol=2e-8, rtol=2e-10)
    direction = jax.device_put(jnp.array([.2, -.1, .3, .01, -.02, .01, .1]), selected_device)
    _, tangent = jax.jvp(evaluate, (params,), (direction,))
    np.testing.assert_allclose(tangent, np.asarray(jacobian) @ np.asarray(direction), atol=2e-8, rtol=2e-10)
    step = 1e-5
    difference = (evaluate(params + step * direction) - evaluate(params - step * direction)) / (2 * step)
    np.testing.assert_allclose(tangent, difference, atol=2e-5, rtol=2e-6)
