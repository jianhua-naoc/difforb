import jax
import jax.numpy as jnp
import pytest

from difforb.core.state.origins import ORIGIN_IN_SSB_ICRS, Origin, origin_in_ssb_icrs
from difforb.core.time.timescale import Time
from tests.assertions import assert_allclose, assert_array_equal

jax.config.update("jax_enable_x64", True)


# -------------------------------------------------------------------------
# Registry And Basic Interface
# -------------------------------------------------------------------------


def test_origin_in_ssb_icrs_registry_covers_all_origins():
    assert set(ORIGIN_IN_SSB_ICRS) == set(Origin)


# -------------------------------------------------------------------------
# SSB Origin
# -------------------------------------------------------------------------


def test_origin_in_ssb_icrs_returns_zero_state_for_ssb_scalar_epoch(fake_sun, fake_earth):
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    actual_pos, actual_vel = origin_in_ssb_icrs(Origin.SSB, tdb, sun=fake_sun, earth=fake_earth)

    assert_array_equal(actual_pos, jnp.zeros(3, dtype=jnp.float64))
    assert_array_equal(actual_vel, jnp.zeros(3, dtype=jnp.float64))


def test_origin_in_ssb_icrs_returns_zero_state_for_ssb_batched_epoch():
    tdb = Time.from_tdb_jd(
        jnp.array([2451545.0, 2451546.0, 2451544.5], dtype=jnp.float64),
        jnp.array([0.0, 0.25, -0.125], dtype=jnp.float64),
    ).tdb()

    actual_pos, actual_vel = origin_in_ssb_icrs(Origin.SSB, tdb)

    print(
        "[state.origin_in_ssb_icrs.SSB.batch] "
        f"pos_shape={actual_pos.shape} "
        f"vel_shape={actual_vel.shape}"
    )

    assert actual_pos.shape == (3, 3)
    assert actual_vel.shape == (3, 3)
    assert_array_equal(actual_pos, jnp.zeros((3, 3), dtype=jnp.float64))
    assert_array_equal(actual_vel, jnp.zeros((3, 3), dtype=jnp.float64))


# -------------------------------------------------------------------------
# SUN Origin
# -------------------------------------------------------------------------


def test_origin_in_ssb_icrs_returns_fake_sun_reference_state_at_jd_ref(fake_sun):
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    actual_pos, actual_vel = origin_in_ssb_icrs(Origin.SUN, tdb, sun=fake_sun)

    assert_allclose(actual_pos, fake_sun.pos0, atol=0.0, rtol=0.0)
    assert_allclose(actual_vel, fake_sun.vel0, atol=0.0, rtol=0.0)


def test_origin_in_ssb_icrs_returns_linearly_shifted_fake_sun_state(fake_sun):
    tdb = Time.from_tdb_jd(2451547.0, 0.5).tdb()
    dt = tdb.jd - fake_sun.jd_ref
    expected_pos = fake_sun.pos0 + dt * fake_sun.vel0
    expected_vel = fake_sun.vel0
    actual_pos, actual_vel = origin_in_ssb_icrs(Origin.SUN, tdb, sun=fake_sun)
    pos_max_abs_diff = jnp.max(jnp.abs(actual_pos - expected_pos))
    vel_max_abs_diff = jnp.max(jnp.abs(actual_vel - expected_vel))

    print(
        "[state.origin_in_ssb_icrs.SUN.scalar] "
        f"pos_max_abs_diff={float(pos_max_abs_diff):+.12e} "
        f"vel_max_abs_diff={float(vel_max_abs_diff):+.12e}"
    )

    assert_allclose(actual_pos, expected_pos, atol=1e-15, rtol=0.0)
    assert_allclose(actual_vel, expected_vel, atol=1e-15, rtol=0.0)


# -------------------------------------------------------------------------
# EARTH Origin
# -------------------------------------------------------------------------


def test_origin_in_ssb_icrs_returns_fake_earth_reference_state_at_jd_ref(fake_earth):
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    actual_pos, actual_vel = origin_in_ssb_icrs(Origin.EARTH, tdb, earth=fake_earth)

    assert_allclose(actual_pos, fake_earth.pos0, atol=0.0, rtol=0.0)
    assert_allclose(actual_vel, fake_earth.vel0, atol=0.0, rtol=0.0)


def test_origin_in_ssb_icrs_returns_linearly_shifted_fake_earth_state_for_batched_epochs(fake_earth):
    tdb = Time.from_tdb_jd(
        jnp.array([2451545.0, 2451546.0, 2451544.5], dtype=jnp.float64),
        jnp.array([0.0, 0.25, -0.125], dtype=jnp.float64),
    ).tdb()
    dt = tdb.jd - fake_earth.jd_ref
    expected_pos = fake_earth.pos0 + dt[..., None] * fake_earth.vel0
    expected_vel = jnp.broadcast_to(fake_earth.vel0, expected_pos.shape)
    actual_pos, actual_vel = origin_in_ssb_icrs(Origin.EARTH, tdb, earth=fake_earth)
    pos_max_abs_diff = jnp.max(jnp.abs(actual_pos - expected_pos))
    vel_max_abs_diff = jnp.max(jnp.abs(actual_vel - expected_vel))

    print(
        "[state.origin_in_ssb_icrs.EARTH.batch] "
        f"pos_max_abs_diff={float(pos_max_abs_diff):+.12e} "
        f"vel_max_abs_diff={float(vel_max_abs_diff):+.12e}"
    )

    assert actual_pos.shape == (3, 3)
    assert actual_vel.shape == (3, 3)
    assert_allclose(actual_pos, expected_pos, atol=1e-15, rtol=0.0)
    assert_allclose(actual_vel, expected_vel, atol=1e-15, rtol=0.0)


# -------------------------------------------------------------------------
# Error Handling
# -------------------------------------------------------------------------


def test_origin_in_ssb_icrs_raises_for_missing_sun_ephemeris_body():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()

    with pytest.raises(ValueError, match=r"Origin ``SUN`` requires the ``sun`` ephemeris body\."):
        origin_in_ssb_icrs(Origin.SUN, tdb)


def test_origin_in_ssb_icrs_raises_for_missing_earth_ephemeris_body():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()

    with pytest.raises(ValueError, match=r"Origin ``EARTH`` requires the ``earth`` ephemeris body\."):
        origin_in_ssb_icrs(Origin.EARTH, tdb)


def test_origin_in_ssb_icrs_raises_for_missing_registered_provider(monkeypatch):
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    monkeypatch.delitem(ORIGIN_IN_SSB_ICRS, Origin.SSB)

    with pytest.raises(KeyError, match=r"No provider in ``SSB`` and ``ICRS`` is registered for origin 'SSB'"):
        origin_in_ssb_icrs(Origin.SSB, tdb)
