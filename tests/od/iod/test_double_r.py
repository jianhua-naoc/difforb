import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from difforb.core.constants import GM_SUN
from difforb.od.iod.double_r import double_r_iod
from tests.assertions import assert_allclose
from tests.od.iod.reference import (
    EDGE_OBSERVATION_INDICES,
    PRIMARY_EPOCH_TDB_JD,
    PRIMARY_OFFSETS,
    PRIMARY_POS_T2,
    PRIMARY_VEL_T2,
    SECONDARY_OFFSETS,
    SECONDARY_POS_T2,
    SECONDARY_VEL_T2,
    propagate_reference_arc,
)


def build_double_r_case(
        pos_t2=PRIMARY_POS_T2,
        vel_t2=PRIMARY_VEL_T2,
        offsets=PRIMARY_OFFSETS,
        phase=0.4,
        epoch_tdb_jd=PRIMARY_EPOCH_TDB_JD,
):
    target_pos, target_vel = propagate_reference_arc(pos_t2, vel_t2, offsets)

    earth_mean_motion = jnp.sqrt(GM_SUN)
    site_phase = phase + earth_mean_motion * offsets
    site_pos = jnp.stack(
        [
            jnp.cos(site_phase),
            jnp.sin(site_phase),
            0.02 * jnp.sin(0.5 * site_phase),
        ],
        axis=1,
    )
    topocentric_pos = target_pos - site_pos
    topocentric_distance = jnp.linalg.norm(topocentric_pos, axis=1)
    los_unit = topocentric_pos / topocentric_distance[:, None]

    tdb_jd = epoch_tdb_jd + offsets
    tdb_jd1 = jnp.floor(tdb_jd)
    tdb_jd2 = tdb_jd - tdb_jd1

    return {
        "site_pos": site_pos,
        "los_unit": los_unit,
        "tdb_jd1": tdb_jd1,
        "tdb_jd2": tdb_jd2,
        "init_rho": jnp.take(topocentric_distance, EDGE_OBSERVATION_INDICES),
        "expected_pos_t2": target_pos[1],
        "expected_vel_t2": target_vel[1],
        "expected_tdb_jd1": tdb_jd1[1],
        "expected_tdb_jd2": tdb_jd2[1],
    }


def test_double_r_recovers_triplet():
    case = build_double_r_case()

    result = double_r_iod(
        case["site_pos"][None, :, :],
        case["los_unit"][None, :, :],
        case["tdb_jd1"][None, :],
        case["tdb_jd2"][None, :],
        mu=GM_SUN,
        init_rho=case["init_rho"][None, :],
        tol=1.0e-10,
        max_iter=20,
    )

    pos_diff = result.pos_t2[0] - case["expected_pos_t2"]
    vel_diff = result.vel_t2[0] - case["expected_vel_t2"]
    print(
        "[od.iod.double_r.triplet] "
        f"residual={float(result.residual_norm[0]):.12e} "
        f"pos_norm_diff={float(jnp.linalg.norm(pos_diff)):.12e} au "
        f"vel_norm_diff={float(jnp.linalg.norm(vel_diff)):.12e} au/day"
    )

    assert result.residual_norm[0] < 1.0e-9
    assert_allclose(result.pos_t2[0], case["expected_pos_t2"], atol=1.0e-8, rtol=0.0)
    assert_allclose(result.vel_t2[0], case["expected_vel_t2"], atol=1.0e-9, rtol=0.0)


def test_double_r_batch():
    cases = [
        build_double_r_case(),
        build_double_r_case(
            pos_t2=SECONDARY_POS_T2,
            vel_t2=SECONDARY_VEL_T2,
            offsets=SECONDARY_OFFSETS,
        ),
    ]
    site_pos = jnp.stack([case["site_pos"] for case in cases])
    los_unit = jnp.stack([case["los_unit"] for case in cases])
    tdb_jd1 = jnp.stack([case["tdb_jd1"] for case in cases])
    tdb_jd2 = jnp.stack([case["tdb_jd2"] for case in cases])
    init_rho = jnp.stack([case["init_rho"] for case in cases])
    expected_pos = jnp.stack([case["expected_pos_t2"] for case in cases])
    expected_vel = jnp.stack([case["expected_vel_t2"] for case in cases])

    result = double_r_iod(
        site_pos,
        los_unit,
        tdb_jd1,
        tdb_jd2,
        mu=GM_SUN,
        init_rho=init_rho,
        tol=1.0e-10,
        max_iter=20,
    )

    print(
        "[od.iod.double_r.batch] "
        f"shape={result.pos_t2.shape} "
        f"pos_max_abs_diff={float(jnp.max(jnp.abs(result.pos_t2 - expected_pos))):.12e} au "
        f"vel_max_abs_diff={float(jnp.max(jnp.abs(result.vel_t2 - expected_vel))):.12e} au/day"
    )

    assert result.pos_t2.shape == (2, 3)
    assert result.vel_t2.shape == (2, 3)
    assert result.residual_norm.shape == (2,)
    assert_allclose(result.pos_t2, expected_pos, atol=1.0e-8, rtol=0.0)
    assert_allclose(result.vel_t2, expected_vel, atol=1.0e-9, rtol=0.0)


def test_double_r_result_contract():
    case = build_double_r_case()

    result = double_r_iod(
        case["site_pos"][None, :, :],
        case["los_unit"][None, :, :],
        case["tdb_jd1"][None, :],
        case["tdb_jd2"][None, :],
        mu=GM_SUN,
        init_rho=case["init_rho"][None, :],
        tol=1.0e-10,
        max_iter=20,
    )

    assert result.pos_t2.shape == (1, 3)
    assert result.vel_t2.shape == (1, 3)
    assert result.epoch_tdb_jd1.shape == (1,)
    assert result.epoch_tdb_jd2.shape == (1,)
    assert result.residual_norm.shape == (1,)
    assert int(result.iter_num) <= 20
    assert bool(jnp.all(jnp.isfinite(result.pos_t2)))
    assert bool(jnp.all(jnp.isfinite(result.vel_t2)))
    assert bool(jnp.all(jnp.isfinite(result.residual_norm)))
    assert_allclose(result.epoch_tdb_jd1[0], case["expected_tdb_jd1"], atol=0.0, rtol=0.0)
    assert_allclose(result.epoch_tdb_jd2[0], case["expected_tdb_jd2"], atol=0.0, rtol=0.0)
