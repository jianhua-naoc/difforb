import jax
import jax.numpy as jnp
import pytest

from difforb.core.constants import GM_SUN
from difforb.core.element import KepElement
from difforb.core.time.timescale import Time
from difforb.dynamics.two_body import C2C3, kepler_propagate, lambert_solver
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("psi", "expected_c2", "expected_c3", "label"),
    [
        (
            0.25,
            (1.0 - jnp.cos(jnp.sqrt(0.25))) / 0.25,
            (jnp.sqrt(0.25) - jnp.sin(jnp.sqrt(0.25))) / (0.25 * jnp.sqrt(0.25)),
            "elliptic",
        ),
        (
            -0.25,
            (1.0 - jnp.cosh(jnp.sqrt(0.25))) / -0.25,
            (jnp.sinh(jnp.sqrt(0.25)) - jnp.sqrt(0.25)) / (0.25 * jnp.sqrt(0.25)),
            "hyperbolic",
        ),
        (
            1.0e-8,
            0.5 - 1.0e-8 / 24.0 + (1.0e-8 ** 2) / 720.0 - (1.0e-8 ** 3) / 40320.0,
            1.0 / 6.0 - 1.0e-8 / 120.0 + (1.0e-8 ** 2) / 5040.0 - (1.0e-8 ** 3) / 362880.0,
            "series",
        ),
    ],
)
def test_c2c3_against_closed_form(psi, expected_c2, expected_c3, label):
    actual_c2, actual_c3 = C2C3(jnp.asarray([psi], dtype=float))

    print(
        "[two_body.c2c3] "
        f"label={label:<10} "
        f"c2_diff={float(actual_c2[0] - expected_c2):+.12e} "
        f"c3_diff={float(actual_c3[0] - expected_c3):+.12e}"
    )

    assert_allclose(actual_c2[0], expected_c2, atol=1.0e-15, rtol=0.0)
    assert_allclose(actual_c3[0], expected_c3, atol=1.0e-15, rtol=0.0)


def test_kepler_propagate_zero_dt():
    init_pos = jnp.asarray([[1.0, 0.2, -0.1]], dtype=float)
    init_vel = jnp.asarray([[0.001, 0.017, 0.002]], dtype=float)
    dt = jnp.asarray([0.0], dtype=float)

    final_pos, final_vel = kepler_propagate(init_pos, init_vel, dt, GM_SUN)

    print(
        "[two_body.kepler.zero_dt] "
        f"pos_max_abs_diff={float(jnp.max(jnp.abs(final_pos - init_pos))):+.12e} au "
        f"vel_max_abs_diff={float(jnp.max(jnp.abs(final_vel - init_vel))):+.12e} au/day"
    )

    assert_allclose(final_pos, init_pos, atol=1.0e-14, rtol=0.0)
    assert_allclose(final_vel, init_vel, atol=1.0e-14, rtol=0.0)


def test_kepler_propagate_one_elliptic_period():
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    element = KepElement.from_classical(
        tdb=tdb,
        a=2.4,
        e=0.35,
        inc=28.0,
        node=140.0,
        peri=250.0,
        m=70.0,
    )
    state = element.state()
    dt = jnp.asarray([element.period], dtype=float)

    final_pos, final_vel = kepler_propagate(state.pos[jnp.newaxis, :], state.vel[jnp.newaxis, :], dt, GM_SUN)

    print(
        "[two_body.kepler.one_period] "
        f"pos_max_abs_diff={float(jnp.max(jnp.abs(final_pos[0] - state.pos))):+.12e} au "
        f"vel_max_abs_diff={float(jnp.max(jnp.abs(final_vel[0] - state.vel))):+.12e} au/day"
    )

    assert_allclose(final_pos[0], state.pos, atol=3.0e-12, rtol=0.0)
    assert_allclose(final_vel[0], state.vel, atol=3.0e-14, rtol=0.0)


def test_kepler_propagate_matches_kep_element_phase_advance():
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    a = 2.1
    e = 0.42
    inc = 35.0
    node = 80.0
    peri = 120.0
    mean_anomaly = 40.0
    dt = 123.0
    initial = KepElement.from_classical(
        tdb=tdb,
        a=a,
        e=e,
        inc=inc,
        node=node,
        peri=peri,
        m=mean_anomaly,
    )
    n = jnp.sqrt(GM_SUN / a ** 3)
    expected = KepElement.from_classical(
        tdb=tdb,
        a=a,
        e=e,
        inc=inc,
        node=node,
        peri=peri,
        m=jnp.rad2deg(jnp.deg2rad(mean_anomaly) + n * dt),
    ).state()

    state = initial.state()
    final_pos, final_vel = kepler_propagate(
        state.pos[jnp.newaxis, :],
        state.vel[jnp.newaxis, :],
        jnp.asarray([dt], dtype=float),
        GM_SUN,
    )

    print(
        "[two_body.kepler.phase_advance] "
        f"pos_max_abs_diff={float(jnp.max(jnp.abs(final_pos[0] - expected.pos))):+.12e} au "
        f"vel_max_abs_diff={float(jnp.max(jnp.abs(final_vel[0] - expected.vel))):+.12e} au/day"
    )

    assert_allclose(final_pos[0], expected.pos, atol=1.0e-12, rtol=0.0)
    assert_allclose(final_vel[0], expected.vel, atol=1.0e-14, rtol=0.0)


def test_kepler_propagate_hyperbolic_roundtrip():
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    element = KepElement.from_true_anomaly(
        tdb=tdb,
        p=0.9,
        e=1.4,
        inc=55.0,
        node=210.0,
        peri=130.0,
        v=80.0,
    )
    state = element.state()
    dt = jnp.asarray([25.0], dtype=float)

    forward_pos, forward_vel = kepler_propagate(state.pos[jnp.newaxis, :], state.vel[jnp.newaxis, :], dt, GM_SUN)
    back_pos, back_vel = kepler_propagate(forward_pos, forward_vel, -dt, GM_SUN)

    print(
        "[two_body.kepler.hyperbolic_roundtrip] "
        f"pos_max_abs_diff={float(jnp.max(jnp.abs(back_pos[0] - state.pos))):+.12e} au "
        f"vel_max_abs_diff={float(jnp.max(jnp.abs(back_vel[0] - state.vel))):+.12e} au/day"
    )

    assert_allclose(back_pos[0], state.pos, atol=1.0e-12, rtol=0.0)
    assert_allclose(back_vel[0], state.vel, atol=1.0e-13, rtol=0.0)


def test_kepler_propagate_batch_shape():
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    first = KepElement.from_classical(tdb=tdb, a=2.0, e=0.1, inc=5.0, node=30.0, peri=50.0, m=70.0).state()
    second = KepElement.from_classical(tdb=tdb, a=3.0, e=0.2, inc=6.0, node=40.0, peri=60.0, m=80.0).state()

    final_pos, final_vel = kepler_propagate(
        jnp.stack([first.pos, second.pos], axis=0),
        jnp.stack([first.vel, second.vel], axis=0),
        jnp.asarray([10.0, 20.0], dtype=float),
        GM_SUN,
    )

    print(
        "[two_body.kepler.batch_shape] "
        f"pos_shape={final_pos.shape} "
        f"vel_shape={final_vel.shape}"
    )

    assert final_pos.shape == (2, 3)
    assert final_vel.shape == (2, 3)


def test_lambert_solver_reconstructs_two_body_transfer():
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    element = KepElement.from_classical(
        tdb=tdb,
        a=1.8,
        e=0.2,
        inc=20.0,
        node=70.0,
        peri=40.0,
        m=30.0,
    )
    initial = element.state()
    dt = jnp.asarray([80.0], dtype=float)
    final_pos, final_vel = kepler_propagate(initial.pos[jnp.newaxis, :], initial.vel[jnp.newaxis, :], dt, GM_SUN)

    lambert_init_vel, lambert_final_vel = lambert_solver(initial.pos[jnp.newaxis, :], final_pos, dt, GM_SUN)

    print(
        "[two_body.lambert.reconstruct_transfer] "
        f"init_vel_max_abs_diff={float(jnp.max(jnp.abs(lambert_init_vel[0] - initial.vel))):+.12e} au/day "
        f"final_vel_max_abs_diff={float(jnp.max(jnp.abs(lambert_final_vel[0] - final_vel[0]))):+.12e} au/day"
    )

    assert_allclose(lambert_init_vel[0], initial.vel, atol=2.0e-10, rtol=0.0)
    assert_allclose(lambert_final_vel[0], final_vel[0], atol=2.0e-10, rtol=0.0)
