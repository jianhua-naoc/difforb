import jax
import jax.numpy as jnp
import pytest

from difforb.core.element import cart_to_kep, cart_to_kep_single, kep_to_cart, kep_to_cart_single
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)

# Hard-coded JPL Horizons references. The ELEMENTS and VECTORS values were
# queried at JD 2460741.5 TDB with center=500@10 and ref_plane=ECLIPTIC.
HORIZONS_CARTESIAN_CASES = [
    (
        "nominal elliptical",
        "248370;",
        "248370 (2005 QN173)",
        0.224593838298332,
        2.377724384684545,
        0.0681061863257785,
        174.1167794361279,
        145.9885401784695,
        233.3446049734478,
        [-3.27039054658403, -0.7821281721242387, 0.001323268943432758],
        [0.003796968652966308, -0.008067431255172916, 9.076440396715248e-06],
        1.0e-12,
    ),
    (
        "near-parabolic elliptical",
        "C/2020 F3;",
        "NEOWISE (C/2020 F3)",
        0.9992561761311389,
        0.2946012281907313,
        129.1682836884898,
        61.10018582955719,
        37.3623218553757,
        164.2311820396671,
        [-10.02668885904143, -10.77669121549351, -4.38204550048232],
        [-0.004262850860982517, -0.00375010853558257, -0.00235636166194909],
        5.0e-12,
    ),
    (
        "hyperbolic",
        "1I;",
        "1I/'Oumuamua (A/2017 U1)",
        1.204622538233297,
        0.260693196364617,
        122.1944816214854,
        24.29880160833165,
        241.8937792839734,
        145.054566045318,
        [41.67855597592661, 6.70783220382005, 17.53009429054784],
        [0.01421377521032147, 0.002120884472152698, 0.006219675775502764],
        2.0e-11,
    ),
]

ROUNDTRIP_ELEMENT_CASES = [
    ("low-e elliptical", 1.7, 0.05, 12.0, 35.0, 80.0, 45.0),
    ("high-e elliptical", 2.4, 0.82, 28.0, 140.0, 250.0, 170.0),
    ("retrograde elliptical", 3.1, 0.35, 142.0, 70.0, 40.0, 300.0),
    ("hyperbolic", 0.8, 1.4, 55.0, 210.0, 130.0, 80.0),
]


@pytest.mark.parametrize(
    (
        "label",
        "target_command",
        "target_name",
        "expected_e",
        "expected_perihelion_distance",
        "expected_inc_deg",
        "expected_node_deg",
        "expected_peri_deg",
        "expected_true_anomaly_deg",
        "expected_pos",
        "expected_vel",
        "atol",
    ),
    HORIZONS_CARTESIAN_CASES,
)
def test_kep_to_cart_single_against_horizons_vectors(
        label,
        target_command,
        target_name,
        expected_e,
        expected_perihelion_distance,
        expected_inc_deg,
        expected_node_deg,
        expected_peri_deg,
        expected_true_anomaly_deg,
        expected_pos,
        expected_vel,
        atol,
):
    expected_p = expected_perihelion_distance * (1.0 + expected_e)
    actual_pos, actual_vel = kep_to_cart_single(
        expected_p,
        expected_e,
        jnp.deg2rad(expected_inc_deg),
        jnp.deg2rad(expected_node_deg),
        jnp.deg2rad(expected_peri_deg),
        jnp.deg2rad(expected_true_anomaly_deg),
    )
    expected_pos = jnp.asarray(expected_pos, dtype=float)
    expected_vel = jnp.asarray(expected_vel, dtype=float)
    pos_diff = jnp.max(jnp.abs(actual_pos - expected_pos))
    vel_diff = jnp.max(jnp.abs(actual_vel - expected_vel))

    print(
        "[kep_to_cart_single.horizons_vectors] "
        f"label={label:<25} "
        f"target={target_command:<11} "
        f"name={target_name} "
        f"pos_max_abs_diff={float(pos_diff):+.12e} au "
        f"vel_max_abs_diff={float(vel_diff):+.12e} au/day"
    )

    assert_allclose(
        actual_pos,
        expected_pos,
        atol=atol,
        rtol=0.0,
        msg=f"Keplerian-to-Cartesian position mismatch for {label}",
    )
    assert_allclose(
        actual_vel,
        expected_vel,
        atol=atol,
        rtol=0.0,
        msg=f"Keplerian-to-Cartesian velocity mismatch for {label}",
    )


@pytest.mark.parametrize(
    (
        "label",
        "target_command",
        "target_name",
        "expected_e",
        "expected_perihelion_distance",
        "expected_inc_deg",
        "expected_node_deg",
        "expected_peri_deg",
        "expected_true_anomaly_deg",
        "input_pos",
        "input_vel",
        "atol",
    ),
    HORIZONS_CARTESIAN_CASES,
)
def test_cart_to_kep_single_against_horizons_elements(
        label,
        target_command,
        target_name,
        expected_e,
        expected_perihelion_distance,
        expected_inc_deg,
        expected_node_deg,
        expected_peri_deg,
        expected_true_anomaly_deg,
        input_pos,
        input_vel,
        atol,
):
    actual = cart_to_kep_single(
        jnp.asarray(input_pos, dtype=float),
        jnp.asarray(input_vel, dtype=float),
    )
    expected_p = expected_perihelion_distance * (1.0 + expected_e)
    expected_angles = {
        "inc": jnp.deg2rad(expected_inc_deg),
        "node": jnp.deg2rad(expected_node_deg),
        "peri": jnp.deg2rad(expected_peri_deg),
        "v": jnp.deg2rad(expected_true_anomaly_deg),
    }
    p_diff = actual["p"] - expected_p
    e_diff = actual["e"] - expected_e

    print(
        "[cart_to_kep_single.horizons_elements] "
        f"label={label:<25} "
        f"target={target_command:<11} "
        f"name={target_name} "
        f"p_diff={float(p_diff):+.12e} au "
        f"e_diff={float(e_diff):+.12e}"
    )

    assert_allclose(actual["p"], expected_p, atol=atol, rtol=0.0)
    assert_allclose(actual["e"], expected_e, atol=atol, rtol=0.0)
    for key, expected_angle in expected_angles.items():
        diff = jnp.arctan2(
            jnp.sin(actual[key] - expected_angle),
            jnp.cos(actual[key] - expected_angle),
        )
        print(
            "[cart_to_kep_single.horizons_elements.angle] "
            f"label={label:<25} "
            f"target={target_command:<11} "
            f"angle={key:<4} "
            f"diff={float(diff):+.12e} rad"
        )
        assert_allclose(diff, 0.0, atol=atol, rtol=0.0)


@pytest.mark.parametrize(
    ("label", "p", "e", "inc_deg", "node_deg", "peri_deg", "true_anomaly_deg"),
    ROUNDTRIP_ELEMENT_CASES,
)
def test_kep_cart_roundtrip_single_for_non_singular_cases(
        label,
        p,
        e,
        inc_deg,
        node_deg,
        peri_deg,
        true_anomaly_deg,
):
    inc = jnp.deg2rad(inc_deg)
    node = jnp.deg2rad(node_deg)
    peri = jnp.deg2rad(peri_deg)
    true_anomaly = jnp.deg2rad(true_anomaly_deg)

    pos, vel = kep_to_cart_single(p, e, inc, node, peri, true_anomaly)
    actual = cart_to_kep_single(pos, vel)
    print(
        "[kep_cart_roundtrip_single] "
        f"label={label:<22} "
        f"pos_norm={float(jnp.linalg.norm(pos)):+.12e} au "
        f"vel_norm={float(jnp.linalg.norm(vel)):+.12e} au/day"
    )

    assert_allclose(actual["p"], p, atol=1.0e-12, rtol=0.0)
    assert_allclose(actual["e"], e, atol=1.0e-12, rtol=0.0)
    for key, expected_angle in {
        "inc": inc,
        "node": node,
        "peri": peri,
        "v": true_anomaly,
    }.items():
        diff = jnp.arctan2(
            jnp.sin(actual[key] - expected_angle),
            jnp.cos(actual[key] - expected_angle),
        )
        print(
            "[kep_cart_roundtrip_single.angle] "
            f"label={label:<22} "
            f"angle={key:<4} "
            f"diff={float(diff):+.12e} rad"
        )
        assert_allclose(diff, 0.0, atol=1.0e-12, rtol=0.0)


def test_kep_to_cart_and_cart_to_kep_batch_shapes():
    p = jnp.asarray([1.7, 2.4], dtype=float)
    e = jnp.asarray([0.05, 0.82], dtype=float)
    inc = jnp.deg2rad(jnp.asarray([12.0, 28.0], dtype=float))
    node = jnp.deg2rad(jnp.asarray([35.0, 140.0], dtype=float))
    peri = jnp.deg2rad(jnp.asarray([80.0, 250.0], dtype=float))
    true_anomaly = jnp.deg2rad(jnp.asarray([45.0, 170.0], dtype=float))

    pos, vel = kep_to_cart(p, e, inc, node, peri, true_anomaly)
    recovered = cart_to_kep(pos, vel)
    print(
        "[kep_cart_batch_shapes] "
        f"pos_shape={pos.shape} "
        f"vel_shape={vel.shape}"
    )

    assert pos.shape == (2, 3)
    assert vel.shape == (2, 3)
    for value in recovered.values():
        assert value.shape == (2,)
