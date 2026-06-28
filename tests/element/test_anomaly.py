import jax
import jax.numpy as jnp
import pytest

from difforb.core.element import m_to_v_single, v_to_m_single
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)

# Hard-coded JPL Horizons ELEMENTS references. The values were queried at
# JD 2460741.5 TDB with center=500@10 and ref_plane=ECLIPTIC.
HORIZONS_ANOMALY_CASES = [
    (
        "nominal elliptical",
        "248370;",
        "248370 (2005 QN173)",
        0.224593838298332,
        256.1452706576621,
        233.3446049734478,
        1.0e-12,
    ),
    (
        "near-parabolic elliptical",
        "C/2020 F3;",
        "NEOWISE (C/2020 F3)",
        0.9992561761311389,
        0.2133641516441716,
        164.2311820396671,
        1.0e-12,
    ),
    (
        "hyperbolic",
        "1I;",
        "1I/'Oumuamua (A/2017 U1)",
        1.204622538233297,
        1876.122297595634,
        145.054566045318,
        2.0e-11,
    ),
]


@pytest.mark.parametrize(
    (
        "label",
        "target_command",
        "target_name",
        "expected_e",
        "expected_mean_anomaly_deg",
        "expected_true_anomaly_deg",
        "atol",
    ),
    HORIZONS_ANOMALY_CASES,
)
def test_m_to_v_single_against_horizons_elements(
        label,
        target_command,
        target_name,
        expected_e,
        expected_mean_anomaly_deg,
        expected_true_anomaly_deg,
        atol,
):
    actual = m_to_v_single(jnp.deg2rad(expected_mean_anomaly_deg), expected_e)
    expected = jnp.deg2rad(expected_true_anomaly_deg)
    diff = jnp.arctan2(jnp.sin(actual - expected), jnp.cos(actual - expected))

    print(
        "[m_to_v_single.horizons_elements] "
        f"label={label:<25} "
        f"target={target_command:<11} "
        f"name={target_name} "
        f"diff={float(diff):+.12e} rad"
    )

    assert_allclose(
        diff,
        0.0,
        atol=atol,
        rtol=0.0,
        msg=f"Mean-to-true anomaly mismatch for {label}",
    )


@pytest.mark.parametrize(
    (
        "label",
        "target_command",
        "target_name",
        "expected_e",
        "expected_mean_anomaly_deg",
        "expected_true_anomaly_deg",
        "atol",
    ),
    HORIZONS_ANOMALY_CASES,
)
def test_v_to_m_single_against_horizons_elements(
        label,
        target_command,
        target_name,
        expected_e,
        expected_mean_anomaly_deg,
        expected_true_anomaly_deg,
        atol,
):
    actual = v_to_m_single(jnp.deg2rad(expected_true_anomaly_deg), expected_e)
    expected = jnp.deg2rad(expected_mean_anomaly_deg)
    diff = actual - expected

    print(
        "[v_to_m_single.horizons_elements] "
        f"label={label:<25} "
        f"target={target_command:<11} "
        f"name={target_name} "
        f"diff={float(diff):+.12e} rad"
    )

    assert_allclose(
        actual,
        expected,
        atol=atol,
        rtol=0.0,
        msg=f"True-to-mean anomaly mismatch for {label}",
    )


@pytest.mark.parametrize(
    ("true_anomaly_rad",),
    [
        (0.0,),
        (1.234,),
        (5.6,),
    ],
)
def test_anomaly_single_exact_circular(true_anomaly_rad):
    mean_anomaly = v_to_m_single(true_anomaly_rad, 0.0)
    recovered_true_anomaly = m_to_v_single(mean_anomaly, 0.0)
    mean_diff = mean_anomaly - true_anomaly_rad
    roundtrip_diff = recovered_true_anomaly - true_anomaly_rad

    print(
        "[anomaly_single.circular] "
        f"true_anomaly={true_anomaly_rad:+.12e} rad "
        f"mean_diff={float(mean_diff):+.12e} rad "
        f"roundtrip_diff={float(roundtrip_diff):+.12e} rad"
    )

    assert_allclose(mean_diff, 0.0, atol=0.0, rtol=0.0)
    assert_allclose(roundtrip_diff, 0.0, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("true_anomaly_rad",),
    [
        (-1.0,),
        (0.0,),
        (1.0,),
    ],
)
def test_anomaly_single_exact_parabolic(true_anomaly_rad):
    expected_mean_anomaly = jnp.tan(true_anomaly_rad / 2.0)
    expected_mean_anomaly = expected_mean_anomaly + expected_mean_anomaly ** 3 / 3.0

    actual_mean_anomaly = v_to_m_single(true_anomaly_rad, 1.0)
    recovered_true_anomaly = m_to_v_single(actual_mean_anomaly, 1.0)
    recovered_diff = jnp.arctan2(
        jnp.sin(recovered_true_anomaly - true_anomaly_rad),
        jnp.cos(recovered_true_anomaly - true_anomaly_rad),
    )

    print(
        "[anomaly_single.parabolic] "
        f"true_anomaly={true_anomaly_rad:+.12e} rad "
        f"mean_diff={float(actual_mean_anomaly - expected_mean_anomaly):+.12e} rad "
        f"roundtrip_diff={float(recovered_diff):+.12e} rad"
    )

    assert_allclose(actual_mean_anomaly, expected_mean_anomaly, atol=1.0e-15, rtol=0.0)
    assert_allclose(recovered_diff, 0.0, atol=1.0e-14, rtol=0.0)
