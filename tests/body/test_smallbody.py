from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from difforb.body.ephbody import EphemerisBody
from difforb.body.smallbody import SmallBody
from difforb.core.constants import GM_SUN
from difforb.core.element import KepElement
from difforb.core.state.frame import BCRS, HELIO_ICRS
from difforb.core.state.state import State
from difforb.core.time.timescale import Time
from difforb.dynamics.force_model import Force, ForceModel
from difforb.dynamics.two_body import kepler_propagate
from difforb.integrator.integrator import NumericalIntegrator
from difforb.spk.spk import Ephemeris
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)

SPK_PATH = Path(__file__).resolve().parents[1] / "data" / "spk" / "de441_2017_2025_excerpt.bsp"
pytestmark = pytest.mark.skipif(
    not SPK_PATH.exists(),
    reason="local DE441 SPK excerpt is not installed",
)


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


@pytest.fixture
def sun():
    return EphemerisBody("sun", eph=Ephemeris(str(SPK_PATH)))


@pytest.fixture
def force_model():
    return ForceModel([CentralGravityForce(GM_SUN)])


@pytest.fixture
def integrator():
    return NumericalIntegrator(method="IAS15", tol=1.0e-12, initial_step=0.05, max_steps=4096)


def test_state_after_propagate_against_kepler(force_model, integrator, sun):
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
    orbit0 = element.state().to(BCRS, sun=sun)
    body = SmallBody(orbit0)
    t_start = Time.from_tdb_jd(2460741.5, -12.0).tdb()
    t_end = Time.from_tdb_jd(2460741.5, 18.0).tdb()
    propagated = body.propagate(t_start, t_end, force_model, integrator)
    offsets = jnp.asarray([-10.0, 0.0, 15.0], dtype=float)
    query_tdb = Time.from_tdb_jd(2460741.5, offsets).tdb()

    actual = propagated.state(query_tdb)
    expected_pos, expected_vel = kepler_propagate(
        jnp.broadcast_to(orbit0.pos, (offsets.shape[0], 3)),
        jnp.broadcast_to(orbit0.vel, (offsets.shape[0], 3)),
        offsets,
        GM_SUN,
    )

    print(
        "[smallbody.state.kepler] "
        f"pos_max_abs_diff={float(jnp.max(jnp.abs(actual.pos - expected_pos))):+.12e} au "
        f"vel_max_abs_diff={float(jnp.max(jnp.abs(actual.vel - expected_vel))):+.12e} au/day"
    )

    assert actual.frame == BCRS
    assert actual.shape == (3,)
    assert_allclose(actual.pos, expected_pos, atol=3.0e-11, rtol=0.0)
    assert_allclose(actual.vel, expected_vel, atol=3.0e-12, rtol=0.0)


def test_state_frame_conversion(force_model, integrator, sun):
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    orbit0 = State(
        tdb=tdb,
        pos=jnp.asarray([1.2, -0.4, 0.3], dtype=float),
        vel=jnp.asarray([0.01, 0.015, -0.005], dtype=float),
        frame=BCRS,
    )
    body = SmallBody(orbit0).propagate(
        Time.from_tdb_jd(2460741.5, -1.0).tdb(),
        Time.from_tdb_jd(2460741.5, 1.0).tdb(),
        force_model,
        integrator,
    )
    query_tdb = Time.from_tdb_jd(2460741.5, 0.5).tdb()

    bcrs_state = body.state(query_tdb)
    actual = body.state(query_tdb, frame=HELIO_ICRS, sun=sun)
    expected = bcrs_state.to(HELIO_ICRS, sun=sun)

    print(
        "[smallbody.state.frame] "
        f"pos_max_abs_diff={float(jnp.max(jnp.abs(actual.pos - expected.pos))):+.12e} au "
        f"vel_max_abs_diff={float(jnp.max(jnp.abs(actual.vel - expected.vel))):+.12e} au/day"
    )

    assert actual.frame == HELIO_ICRS
    assert_allclose(actual.pos, expected.pos, atol=1.0e-15, rtol=0.0)
    assert_allclose(actual.vel, expected.vel, atol=1.0e-15, rtol=0.0)


def test_batch_state_shape(force_model, integrator):
    tdb = Time.from_tdb_jd(jnp.asarray([2460741.5, 2460741.5], dtype=float), jnp.asarray([0.0, 0.0], dtype=float)).tdb()
    pos0 = jnp.asarray([[1.0, 0.2, -0.1], [1.5, -0.3, 0.4]], dtype=float)
    vel0 = jnp.asarray([[0.001, 0.017, 0.002], [-0.005, 0.012, 0.004]], dtype=float)
    body = SmallBody(State(tdb=tdb, pos=pos0, vel=vel0, frame=BCRS)).propagate(
        Time.from_tdb_jd(jnp.asarray([2460741.5, 2460741.5], dtype=float), jnp.asarray([-1.0, -2.0], dtype=float)).tdb(),
        Time.from_tdb_jd(jnp.asarray([2460741.5, 2460741.5], dtype=float), jnp.asarray([2.0, 3.0], dtype=float)).tdb(),
        force_model,
        integrator,
    )
    query_offsets = jnp.asarray([0.5, -1.0], dtype=float)
    actual = body.state(Time.from_tdb_jd(jnp.asarray([2460741.5, 2460741.5], dtype=float), query_offsets).tdb())

    print(
        "[smallbody.state.batch] "
        f"body_shape={body.shape} "
        f"state_shape={actual.shape} "
        f"pos_shape={actual.pos.shape}"
    )

    assert body.shape == (2,)
    assert actual.shape == (2,)
    assert actual.pos.shape == (2, 3)
    assert actual.vel.shape == (2, 3)


def test_grid_state_shape(force_model, integrator):
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    orbit0 = State(
        tdb=tdb,
        pos=jnp.asarray([1.0, 0.2, -0.1], dtype=float),
        vel=jnp.asarray([0.001, 0.017, 0.002], dtype=float),
        frame=BCRS,
    )
    body = SmallBody(orbit0).propagate(
        Time.from_tdb_jd(2460741.5, -2.0).tdb(),
        Time.from_tdb_jd(2460741.5, 2.0).tdb(),
        force_model,
        integrator,
    )
    query_tdb = Time.from_tdb_jd(2460741.5, jnp.asarray([-1.0, 0.0, 1.5], dtype=float)).tdb()
    actual = body.state(query_tdb, grid=True)

    print(
        "[smallbody.state.grid] "
        f"state_shape={actual.shape} "
        f"pos_shape={actual.pos.shape}"
    )

    assert actual.shape == (3,)
    assert actual.pos.shape == (3, 3)
    assert actual.vel.shape == (3, 3)


def test_propagate_returns_new_body(force_model, integrator):
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    orbit0 = State(
        tdb=tdb,
        pos=jnp.asarray([1.0, -0.2, 0.3], dtype=float),
        vel=jnp.asarray([0.01, 0.02, -0.01], dtype=float),
        frame=BCRS,
    )
    body = SmallBody(orbit0)
    propagated = body.propagate(
        Time.from_tdb_jd(2460741.5, -1.0).tdb(),
        Time.from_tdb_jd(2460741.5, 1.0).tdb(),
        force_model,
        integrator,
    )

    print(
        "[smallbody.propagate] "
        f"before={repr(body)} "
        f"after={repr(propagated)}"
    )

    assert body.trajectory is None
    assert propagated.trajectory is not None
    assert "trajectory=uninitialized" in repr(body)
    assert "trajectory=ready" in repr(propagated)


def test_state_before_propagate_raises():
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    orbit0 = State(
        tdb=tdb,
        pos=jnp.asarray([1.0, 0.0, 0.0], dtype=float),
        vel=jnp.asarray([0.0, 0.01, 0.0], dtype=float),
        frame=BCRS,
    )
    body = SmallBody(orbit0)

    with pytest.raises(RuntimeError, match="Trajectory is not initialized"):
        body.state(tdb)


def test_state_outside_coverage_raises(force_model, integrator):
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    orbit0 = State(
        tdb=tdb,
        pos=jnp.asarray([1.0, 0.0, 0.0], dtype=float),
        vel=jnp.asarray([0.0, 0.01, 0.0], dtype=float),
        frame=BCRS,
    )
    body = SmallBody(orbit0).propagate(
        Time.from_tdb_jd(2460741.5, -1.0).tdb(),
        Time.from_tdb_jd(2460741.5, 1.0).tdb(),
        force_model,
        integrator,
    )

    with pytest.raises(Exception, match="outside the coverage"):
        body.state(Time.from_tdb_jd(2460741.5, 2.0).tdb())


def test_create_from_bcrs_state():
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    orbit0 = State(
        tdb=tdb,
        pos=jnp.asarray([1.0, -2.0, 0.5], dtype=float),
        vel=jnp.asarray([0.01, 0.02, -0.03], dtype=float),
        frame=BCRS,
    )

    body = SmallBody.create(orbit0)

    assert body.orbit0.frame == BCRS
    assert_allclose(body.orbit0.pos, orbit0.pos, atol=0.0, rtol=0.0)
    assert_allclose(body.orbit0.vel, orbit0.vel, atol=0.0, rtol=0.0)


def test_create_from_helio_state(sun):
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    helio = State(
        tdb=tdb,
        pos=jnp.asarray([1.0, -2.0, 0.5], dtype=float),
        vel=jnp.asarray([0.01, 0.02, -0.03], dtype=float),
        frame=HELIO_ICRS,
    )
    expected = helio.to(BCRS, sun=sun)

    body = SmallBody.create(helio, sun=sun)

    print(
        "[smallbody.create.helio] "
        f"pos_max_abs_diff={float(jnp.max(jnp.abs(body.orbit0.pos - expected.pos))):+.12e} au "
        f"vel_max_abs_diff={float(jnp.max(jnp.abs(body.orbit0.vel - expected.vel))):+.12e} au/day"
    )

    assert body.orbit0.frame == BCRS
    assert_allclose(body.orbit0.pos, expected.pos, atol=1.0e-15, rtol=0.0)
    assert_allclose(body.orbit0.vel, expected.vel, atol=1.0e-15, rtol=0.0)


def test_create_from_kep_element(sun):
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    element = KepElement.from_classical(
        tdb=tdb,
        a=2.0,
        e=0.2,
        inc=5.0,
        node=30.0,
        peri=50.0,
        m=70.0,
    )
    expected = element.state().to(BCRS, sun=sun)

    body = SmallBody.create(element, sun=sun)

    print(
        "[smallbody.create.element] "
        f"pos_max_abs_diff={float(jnp.max(jnp.abs(body.orbit0.pos - expected.pos))):+.12e} au "
        f"vel_max_abs_diff={float(jnp.max(jnp.abs(body.orbit0.vel - expected.vel))):+.12e} au/day"
    )

    assert body.orbit0.frame == BCRS
    assert_allclose(body.orbit0.pos, expected.pos, atol=1.0e-15, rtol=0.0)
    assert_allclose(body.orbit0.vel, expected.vel, atol=1.0e-15, rtol=0.0)


def test_smallbody_rejects_invalid_inputs():
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    helio = State(
        tdb=tdb,
        pos=jnp.asarray([1.0, 0.0, 0.0], dtype=float),
        vel=jnp.asarray([0.0, 0.01, 0.0], dtype=float),
        frame=HELIO_ICRS,
    )

    with pytest.raises(TypeError, match="orbit0"):
        SmallBody("not a state")
    with pytest.raises(ValueError, match="BCRS"):
        SmallBody(helio)
    with pytest.raises(TypeError, match="KepElement.*State"):
        SmallBody.create("not an orbit")


def test_smallbody_rejects_non_tdb(force_model, integrator):
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    tt = Time.from_tdb_jd(2460741.5, 0.0).tt
    orbit0 = State(
        tdb=tdb,
        pos=jnp.asarray([1.0, 0.0, 0.0], dtype=float),
        vel=jnp.asarray([0.0, 0.01, 0.0], dtype=float),
        frame=BCRS,
    )
    body = SmallBody(orbit0)

    with pytest.raises(TypeError, match="t_start"):
        body.propagate(tt, tdb, force_model, integrator)

    propagated = body.propagate(
        Time.from_tdb_jd(2460741.5, -1.0).tdb(),
        Time.from_tdb_jd(2460741.5, 1.0).tdb(),
        force_model,
        integrator,
    )
    with pytest.raises(TypeError, match="tdb"):
        propagated.state(tt)
