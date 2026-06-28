import jax
import jax.numpy as jnp
import pytest

from difforb.core.constants import GM_SUN
from difforb.core.element import KepElement
from difforb.core.state.frame import HELIO_ECLIP_J2000
from difforb.core.state.state import State
from difforb.core.time.timescale import Time
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)

# Hard-coded JPL Horizons references. The ELEMENTS and VECTORS values were
# queried at JD 2460741.5 TDB with center=500@10 and ref_plane=ECLIPTIC.
HORIZONS_ELEMENT_STATE_CASES = [
    (
        "nominal elliptical",
        "248370;",
        "248370 (2005 QN173)",
        2460741.5,
        0.224593838298332,
        2.377724384684545,
        3.066424413582824,
        0.0681061863257785,
        174.1167794361279,
        145.9885401784695,
        256.1452706576621,
        233.3446049734478,
        [-3.27039054658403, -0.7821281721242387, 0.001323268943432758],
        [0.003796968652966308, -0.008067431255172916, 9.076440396715248e-06],
        1.0e-12,
    ),
    (
        "near-parabolic elliptical",
        "C/2020 F3;",
        "NEOWISE (C/2020 F3)",
        2460741.5,
        0.9992561761311389,
        0.2946012281907313,
        396.0631548998308,
        129.1682836884898,
        61.10018582955719,
        37.3623218553757,
        0.2133641516441716,
        164.2311820396671,
        [-10.02668885904143, -10.77669121549351, -4.38204550048232],
        [-0.004262850860982517, -0.00375010853558257, -0.00235636166194909],
        5.0e-12,
    ),
    (
        "hyperbolic",
        "1I;",
        "1I/'Oumuamua (A/2017 U1)",
        2460741.5,
        1.204622538233297,
        0.260693196364617,
        -1.274019952129535,
        122.1944816214854,
        24.29880160833165,
        241.8937792839734,
        1876.122297595634,
        145.054566045318,
        [41.67855597592661, 6.70783220382005, 17.53009429054784],
        [0.01421377521032147, 0.002120884472152698, 0.006219675775502764],
        2.0e-11,
    ),
]

CLASSICAL_ARRAY_ROUNDTRIP_CASES = [
    ("degrees", True, 2.5, 0.4, 15.0, 35.0, 55.0, 75.0),
    ("radians", False, 2.5, 0.4, 0.25, 0.5, 0.75, 1.0),
]


@pytest.mark.parametrize(
    (
        "label",
        "target_command",
        "target_name",
        "expected_epoch_tdb_jd",
        "expected_e",
        "expected_perihelion_distance",
        "expected_a",
        "expected_inc_deg",
        "expected_node_deg",
        "expected_peri_deg",
        "expected_mean_anomaly_deg",
        "expected_true_anomaly_deg",
        "expected_pos",
        "expected_vel",
        "atol",
    ),
    HORIZONS_ELEMENT_STATE_CASES,
)
def test_from_classical_state_against_horizons_vectors(
        label,
        target_command,
        target_name,
        expected_epoch_tdb_jd,
        expected_e,
        expected_perihelion_distance,
        expected_a,
        expected_inc_deg,
        expected_node_deg,
        expected_peri_deg,
        expected_mean_anomaly_deg,
        expected_true_anomaly_deg,
        expected_pos,
        expected_vel,
        atol,
):
    tdb = Time.from_tdb_jd(expected_epoch_tdb_jd, 0.0).tdb()
    element = KepElement.from_classical(
        tdb=tdb,
        a=expected_a,
        e=expected_e,
        inc=expected_inc_deg,
        node=expected_node_deg,
        peri=expected_peri_deg,
        m=expected_mean_anomaly_deg,
    )
    state = element.state()
    expected_pos = jnp.asarray(expected_pos, dtype=float)
    expected_vel = jnp.asarray(expected_vel, dtype=float)
    pos_diff = jnp.max(jnp.abs(state.pos - expected_pos))
    vel_diff = jnp.max(jnp.abs(state.vel - expected_vel))

    print(
        "[kepelement.from_classical.state.horizons_vectors] "
        f"label={label:<25} "
        f"target={target_command:<11} "
        f"name={target_name} "
        f"pos_max_abs_diff={float(pos_diff):+.12e} au "
        f"vel_max_abs_diff={float(vel_diff):+.12e} au/day"
    )

    assert state.frame == HELIO_ECLIP_J2000
    assert state.shape == ()
    anomaly_diff = jnp.arctan2(
        jnp.sin(element.v - jnp.deg2rad(expected_true_anomaly_deg)),
        jnp.cos(element.v - jnp.deg2rad(expected_true_anomaly_deg)),
    )
    assert_allclose(element.p, expected_perihelion_distance * (1.0 + expected_e), atol=atol, rtol=0.0)
    assert_allclose(element.e, expected_e, atol=0.0, rtol=0.0)
    assert_allclose(anomaly_diff, 0.0, atol=atol, rtol=0.0)
    assert_allclose(state.tdb.jd, tdb.jd, atol=0.0, rtol=0.0)
    assert_allclose(state.pos, expected_pos, atol=atol, rtol=0.0)
    assert_allclose(state.vel, expected_vel, atol=atol, rtol=0.0)


@pytest.mark.parametrize(
    (
        "label",
        "target_command",
        "target_name",
        "expected_epoch_tdb_jd",
        "expected_e",
        "expected_perihelion_distance",
        "expected_a",
        "expected_inc_deg",
        "expected_node_deg",
        "expected_peri_deg",
        "expected_mean_anomaly_deg",
        "expected_true_anomaly_deg",
        "expected_pos",
        "expected_vel",
        "atol",
    ),
    HORIZONS_ELEMENT_STATE_CASES,
)
def test_from_true_anomaly_state_against_horizons_vectors(
        label,
        target_command,
        target_name,
        expected_epoch_tdb_jd,
        expected_e,
        expected_perihelion_distance,
        expected_a,
        expected_inc_deg,
        expected_node_deg,
        expected_peri_deg,
        expected_mean_anomaly_deg,
        expected_true_anomaly_deg,
        expected_pos,
        expected_vel,
        atol,
):
    tdb = Time.from_tdb_jd(expected_epoch_tdb_jd, 0.0).tdb()
    element = KepElement.from_true_anomaly(
        tdb=tdb,
        p=expected_perihelion_distance * (1.0 + expected_e),
        e=expected_e,
        inc=expected_inc_deg,
        node=expected_node_deg,
        peri=expected_peri_deg,
        v=expected_true_anomaly_deg,
    )
    state = element.state()
    expected_pos = jnp.asarray(expected_pos, dtype=float)
    expected_vel = jnp.asarray(expected_vel, dtype=float)
    pos_diff = jnp.max(jnp.abs(state.pos - expected_pos))
    vel_diff = jnp.max(jnp.abs(state.vel - expected_vel))

    print(
        "[kepelement.from_true_anomaly.state.horizons_vectors] "
        f"label={label:<25} "
        f"target={target_command:<11} "
        f"name={target_name} "
        f"pos_max_abs_diff={float(pos_diff):+.12e} au "
        f"vel_max_abs_diff={float(vel_diff):+.12e} au/day"
    )

    assert state.frame == HELIO_ECLIP_J2000
    assert state.shape == ()
    mean_anomaly_diff = jnp.arctan2(
        jnp.sin(element.m - jnp.deg2rad(expected_mean_anomaly_deg)),
        jnp.cos(element.m - jnp.deg2rad(expected_mean_anomaly_deg)),
    )
    assert_allclose(element.p, expected_perihelion_distance * (1.0 + expected_e), atol=0.0, rtol=0.0)
    assert_allclose(element.e, expected_e, atol=0.0, rtol=0.0)
    assert_allclose(mean_anomaly_diff, 0.0, atol=atol, rtol=0.0)
    assert_allclose(state.tdb.jd, tdb.jd, atol=0.0, rtol=0.0)
    assert_allclose(state.pos, expected_pos, atol=atol, rtol=0.0)
    assert_allclose(state.vel, expected_vel, atol=atol, rtol=0.0)


@pytest.mark.parametrize(
    ("label", "degrees", "a", "e", "inc", "node", "peri", "mean_anomaly"),
    CLASSICAL_ARRAY_ROUNDTRIP_CASES,
)
def test_kepelement_from_classical_and_from_array_roundtrip(label, degrees, a, e, inc, node, peri, mean_anomaly):
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    element = KepElement.from_classical(
        tdb=tdb,
        a=a,
        e=e,
        inc=inc,
        node=node,
        peri=peri,
        m=mean_anomaly,
        degrees=degrees,
    )
    recovered = KepElement.from_array(tdb, element.array)
    factor = jnp.pi / 180.0 if degrees else 1.0
    expected_array = jnp.asarray(
        [
            a * (1.0 - e ** 2),
            e,
            inc * factor,
            node * factor,
            peri * factor,
            mean_anomaly * factor,
        ],
        dtype=float,
    )
    array_diff = jnp.max(jnp.abs(element.array - expected_array))

    print(
        "[kepelement.from_classical.from_array_roundtrip] "
        f"label={label:<7} "
        f"array_max_abs_diff={float(array_diff):+.12e}"
    )

    assert element.shape == ()
    assert element.array.shape == (6,)
    assert_allclose(element.array, expected_array, atol=1.0e-15, rtol=0.0)
    assert_allclose(recovered.array, element.array, atol=0.0, rtol=0.0)
    assert_allclose(recovered.tdb.jd, element.tdb.jd, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    (
        "label",
        "target_command",
        "target_name",
        "expected_epoch_tdb_jd",
        "expected_e",
        "expected_perihelion_distance",
        "expected_a",
        "expected_inc_deg",
        "expected_node_deg",
        "expected_peri_deg",
        "expected_mean_anomaly_deg",
        "expected_true_anomaly_deg",
        "input_pos",
        "input_vel",
        "atol",
    ),
    HORIZONS_ELEMENT_STATE_CASES,
)
def test_from_state_against_horizons_elements(
        label,
        target_command,
        target_name,
        expected_epoch_tdb_jd,
        expected_e,
        expected_perihelion_distance,
        expected_a,
        expected_inc_deg,
        expected_node_deg,
        expected_peri_deg,
        expected_mean_anomaly_deg,
        expected_true_anomaly_deg,
        input_pos,
        input_vel,
        atol,
):
    tdb = Time.from_tdb_jd(expected_epoch_tdb_jd, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.asarray(input_pos, dtype=float),
        vel=jnp.asarray(input_vel, dtype=float),
        frame=HELIO_ECLIP_J2000,
    )
    element = KepElement.from_state(state)
    expected_p = expected_perihelion_distance * (1.0 + expected_e)
    expected_angles = {
        "inc": jnp.deg2rad(expected_inc_deg),
        "node": jnp.deg2rad(expected_node_deg),
        "peri": jnp.deg2rad(expected_peri_deg),
        "m": jnp.deg2rad(expected_mean_anomaly_deg),
        "v": jnp.deg2rad(expected_true_anomaly_deg),
    }
    p_diff = element.p - expected_p
    e_diff = element.e - expected_e

    print(
        "[kepelement.from_state.horizons_elements] "
        f"label={label:<25} "
        f"target={target_command:<11} "
        f"name={target_name} "
        f"p_diff={float(p_diff):+.12e} au "
        f"e_diff={float(e_diff):+.12e}"
    )

    assert element.shape == ()
    assert_allclose(element.tdb.jd, tdb.jd, atol=0.0, rtol=0.0)
    assert_allclose(element.p, expected_p, atol=atol, rtol=0.0)
    assert_allclose(element.e, expected_e, atol=max(atol, 2.0e-14), rtol=0.0)
    for key, expected_angle in expected_angles.items():
        actual_angle = element.v if key == "v" else getattr(element, key)
        diff = jnp.arctan2(
            jnp.sin(actual_angle - expected_angle),
            jnp.cos(actual_angle - expected_angle),
        )
        angle_atol = max(atol, 5.0e-11) if key == "m" else max(atol, 1.0e-14)
        print(
            "[kepelement.from_state.horizons_elements.angle] "
            f"label={label:<25} "
            f"target={target_command:<11} "
            f"angle={key:<4} "
            f"diff={float(diff):+.12e} rad"
        )
        assert_allclose(diff, 0.0, atol=angle_atol, rtol=0.0)


def test_kepelement_period_and_perihelion_time_for_elliptical_orbit():
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    element = KepElement.from_classical(
        tdb=tdb,
        a=3.0,
        e=0.1,
        inc=10.0,
        node=20.0,
        peri=30.0,
        m=40.0,
    )
    expected_period = 2.0 * jnp.pi * jnp.sqrt(element.a ** 3 / GM_SUN)
    expected_perihelion_jd = element.tdb.jd + (2.0 * jnp.pi - element.m) / (2.0 * jnp.pi / expected_period)

    print(
        "[kepelement.period_perit.elliptical] "
        f"period_diff={float(element.period - expected_period):+.12e} day "
        f"perit_diff={float(element.perit_jd - expected_perihelion_jd):+.12e} day"
    )

    assert_allclose(element.period, expected_period, atol=1.0e-12, rtol=0.0)
    assert_allclose(element.perit_jd, expected_perihelion_jd, atol=1.0e-12, rtol=0.0)


def test_kepelement_period_and_perihelion_time_for_nonperiodic_orbit():
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    element = KepElement.from_true_anomaly(
        tdb=tdb,
        p=2.0,
        e=1.3,
        inc=10.0,
        node=20.0,
        peri=30.0,
        v=40.0,
    )

    print(
        "[kepelement.period_perit.nonperiodic] "
        f"period={float(element.period):+.12e} day "
        f"perit_jd={float(element.perit_jd):+.12e}"
    )

    assert bool(jnp.isinf(element.period))
    assert bool(jnp.isnan(element.perit_jd))


def test_kepelement_from_classical_state_batch_shapes():
    tdb = Time.from_tdb_jd(
        jnp.asarray([2460741.5, 2460742.5], dtype=float),
        jnp.asarray([0.0, 0.0], dtype=float),
    ).tdb()
    element = KepElement.from_classical(
        tdb=tdb,
        a=jnp.asarray([2.0, 3.0], dtype=float),
        e=jnp.asarray([0.1, 0.2], dtype=float),
        inc=jnp.asarray([5.0, 6.0], dtype=float),
        node=jnp.asarray([30.0, 40.0], dtype=float),
        peri=jnp.asarray([50.0, 60.0], dtype=float),
        m=jnp.asarray([70.0, 80.0], dtype=float),
    )
    state = element.state()

    print(
        "[kepelement.batch_shapes] "
        f"element_shape={element.shape} "
        f"state_shape={state.shape}"
    )

    assert element.shape == (2,)
    assert element.array.shape == (2, 6)
    assert state.frame == HELIO_ECLIP_J2000
    assert state.shape == (2,)
    assert state.pos.shape == (2, 3)
    assert state.vel.shape == (2, 3)


def test_from_equinoctial_elements_orbfit_case():
    tt = Time.from_tt_jd(2460741.0, 0.5).tt
    element = KepElement.from_equinoctial_elements(
        tt=tt,
        a=2.1109381975974553,
        g=-0.704801207025459,
        f=0.250060102768097,
        k=0.012996744980151,
        h=0.041540167717717,
        M=jnp.deg2rad(265.2262785103422),
    )
    expected = {
        "a": 2.1109381975974553,
        "e": 0.747846773357307,
        "inc": jnp.deg2rad(4.9845505587451),
        "node": jnp.deg2rad(17.3734273102467),
        "peri": jnp.deg2rad(272.1610147278425),
        "m": jnp.deg2rad(335.6918364722530),
    }
    print(
        "[kepelement.from_equinoctial_elements.orbfit] "
        f"a_diff={float(element.a - expected['a']):+.12e} au "
        f"e_diff={float(element.e - expected['e']):+.12e}"
    )

    assert_allclose(element.a, expected["a"], atol=1.0e-12, rtol=0.0)
    assert_allclose(element.e, expected["e"], atol=1.0e-12, rtol=0.0)
    for key in ["inc", "node", "peri", "m"]:
        diff = jnp.arctan2(
            jnp.sin(getattr(element, key) - expected[key]),
            jnp.cos(getattr(element, key) - expected[key]),
        )
        print(
            "[kepelement.from_equinoctial_elements.orbfit.angle] "
            f"angle={key:<4} "
            f"diff={float(diff):+.12e} rad"
        )
        assert_allclose(diff, 0.0, atol=1.0e-10, rtol=0.0)


def test_from_classical_rejects_non_tdb_view():
    tt = Time.from_tt_jd(2460741.5, 0.0).tt

    with pytest.raises(TypeError, match=r"argument `tdb` must be an instance of `TDBView`"):
        KepElement.from_classical(
            tdb=tt,
            a=2.0,
            e=0.1,
            inc=10.0,
            node=20.0,
            peri=30.0,
            m=40.0,
        )


def test_from_array_rejects_non_tdb_view():
    tt = Time.from_tt_jd(2460741.5, 0.0).tt

    with pytest.raises(TypeError, match=r"argument `tdb` must be an instance of `TDBView`"):
        KepElement.from_array(tt, jnp.ones(6, dtype=float))


def test_from_equinoctial_elements_rejects_non_tt_view():
    tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()

    with pytest.raises(TypeError, match=r"argument `tt` must be an instance of `TTView`"):
        KepElement.from_equinoctial_elements(
            tt=tdb,
            a=2.0,
            g=0.1,
            f=0.2,
            k=0.01,
            h=0.02,
            M=0.3,
        )
