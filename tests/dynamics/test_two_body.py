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


@pytest.mark.parametrize(
    ("psi", "expected_c2", "expected_c3", "expected_dc2", "expected_dc3"),
    [
        (-4.0, 0.6905489227709078, 0.20335755098087735, -0.054041544797711706, -0.010059533728534478),
        (-1.00000001, 0.5430806352604434, 0.17520119373118673, -0.04451996203595631, -0.00873852694605634),
        (-1.0, 0.5430806348152438, 0.17520119364380146, -0.04451996200665695, -0.008738526941919704),
        (-0.99999999, 0.5430806343700442, 0.1752011935564162, -0.04451996197735759, -0.008738526937783068),
        (-1.001e-6, 0.5000000417083347, 0.16666667500833354, -0.0416666694472223, -0.008333333730555564),
        (-1.0e-6, 0.500000041666668, 0.1666666750000002, -0.04166666944444452, -0.008333333730158739),
        (0.0, 0.5, 0.16666666666666666, -0.041666666666666664, -0.008333333333333333),
        (1.0e-6, 0.4999999583333347, 0.16666665833333352, -0.04166666388888896, -0.008333332936507945),
        (1.001e-6, 0.49999995829166805, 0.1666666583250002, -0.041666663886111184, -0.00833333293611112),
        (0.99999999, 0.4596976945214823, 0.15852901527155025, -0.038962201754234364, -0.007944675726030984),
        (1.0, 0.4596976941318603, 0.1585290151921035, -0.03896220172791203, -0.0079446757222251),
        (1.00000001, 0.4596976937422383, 0.15852901511265674, -0.03896220170158969, -0.007944675718419215),
        (4.0, 0.3540367091367856, 0.1363378216467898, -0.03167808810759129, -0.0068720944754479705),
    ],
)
def test_c2c3_values_and_derivatives_against_high_precision(psi, expected_c2, expected_c3, expected_dc2, expected_dc3):
    # Reference values and derivatives were evaluated with 80-digit arithmetic.
    psi = jnp.asarray([psi], dtype=float)
    (c2, c3), (dc2, dc3) = jax.jvp(C2C3, (psi,), (jnp.ones_like(psi),))

    assert_allclose(c2[0], expected_c2, atol=3.0e-16, rtol=0.0)
    assert_allclose(c3[0], expected_c3, atol=3.0e-16, rtol=0.0)
    assert_allclose(dc2[0], expected_dc2, atol=3.0e-16, rtol=0.0)
    assert_allclose(dc3[0], expected_dc3, atol=3.0e-16, rtol=0.0)


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


@pytest.mark.parametrize("d", [-0.5, 0.0, 0.5])
def test_kepler_propagate_parabolic_state_and_time_derivative(d):
    init_pos = jnp.asarray([[1.0, 0.0, 0.0]], dtype=float)
    init_vel = jnp.asarray([[0.0, jnp.sqrt(2.0 * GM_SUN), 0.0]], dtype=float)
    # Barker's equation with perihelion distance 1 au and d = tan(true_anomaly / 2).
    dt = jnp.sqrt(2.0 / GM_SUN) * (d + d ** 3 / 3.0)
    expected_pos = jnp.asarray([[1.0 - d ** 2, 2.0 * d, 0.0]], dtype=float)
    expected_vel = jnp.sqrt(2.0 * GM_SUN) / (1.0 + d ** 2) * jnp.asarray([[-d, 1.0, 0.0]])
    expected_acc = -GM_SUN * expected_pos / jnp.linalg.norm(expected_pos, axis=-1, keepdims=True) ** 3

    def propagate(offset):
        return kepler_propagate(init_pos, init_vel, offset[jnp.newaxis], GM_SUN)

    (final_pos, final_vel), (pos_rate, vel_rate) = jax.jvp(propagate, (dt,), (jnp.ones_like(dt),))

    assert_allclose(final_pos, expected_pos, atol=1.0e-14, rtol=0.0)
    assert_allclose(final_vel, expected_vel, atol=1.0e-14, rtol=0.0)
    assert_allclose(pos_rate, expected_vel, atol=1.0e-14, rtol=0.0)
    assert_allclose(vel_rate, expected_acc, atol=1.0e-14, rtol=0.0)


@pytest.mark.parametrize("alpha", [-5.0e-7, 5.0e-7])
def test_kepler_propagate_near_parabolic_roundtrip(alpha):
    init_pos = jnp.asarray([[1.0, 0.0, 0.0]], dtype=float)
    init_vel = jnp.asarray([[0.001, jnp.sqrt(GM_SUN * (2.0 - alpha) - 0.001 ** 2), 0.0]], dtype=float)
    dt = jnp.asarray([20.0], dtype=float)

    forward_pos, forward_vel = kepler_propagate(init_pos, init_vel, dt, GM_SUN)
    back_pos, back_vel = kepler_propagate(forward_pos, forward_vel, -dt, GM_SUN)

    assert_allclose(back_pos, init_pos, atol=1.0e-12, rtol=0.0)
    assert_allclose(back_vel, init_vel, atol=1.0e-14, rtol=0.0)


def test_kepler_propagate_near_parabolic_against_high_precision():
    init_pos = jnp.asarray([[0.6434738321730742, 0.0, 0.0]], dtype=float)
    init_vel = jnp.asarray([[-0.011013061710245001, 0.02825678640666542, 0.0]], dtype=float)
    dt = jnp.asarray(417.05165763133306, dtype=float)
    # Independent 80-digit solution of the universal Kepler equation.
    expected_pos = jnp.asarray([[-5.474438386066792, -0.5189269009809977, 0.0]], dtype=float)
    expected_vel = jnp.asarray([[-0.009477266127358091, -0.004219704259349992, 0.0]], dtype=float)
    expected_acc = -GM_SUN * expected_pos / jnp.linalg.norm(expected_pos, axis=-1, keepdims=True) ** 3

    def propagate(offset):
        return kepler_propagate(init_pos, init_vel, offset[jnp.newaxis], GM_SUN)

    (final_pos, final_vel), (pos_rate, vel_rate) = jax.jvp(propagate, (dt,), (jnp.ones_like(dt),))

    assert_allclose(final_pos, expected_pos, atol=2.0e-14, rtol=0.0)
    assert_allclose(final_vel, expected_vel, atol=2.0e-16, rtol=0.0)
    assert_allclose(pos_rate, expected_vel, atol=2.0e-16, rtol=0.0)
    assert_allclose(vel_rate, expected_acc, atol=2.0e-16, rtol=0.0)


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


@pytest.mark.parametrize("tol", [1.0e-15, 1.0e-6])
def test_kepler_propagate_matches_kep_element_phase_advance(tol):
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
        tol=tol,
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
