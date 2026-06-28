import jax
import jax.numpy as jnp
import pytest

from difforb.core.constants import DAY_S
from difforb.core.time.tai import ttdtai, tai_to_tt, tt_to_tai
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)

TAI_TT_CASES = [
    (2451545.0, 0.0, "J2000"),
    (2451727.0, 0.625, "Post J2000"),
    (1721057.0, 0.5, "1 BC"),
    (2816787.0, 0.5, "3000-01-01"),
    (2451544.0, 0.4999, "Near split boundary"),
]


def test_ttdtai_value():
    assert ttdtai() == 32.184


@pytest.mark.parametrize("tai_jd1, tai_jd2, label", TAI_TT_CASES)
def test_tai_to_tt_constant_offset(tai_jd1, tai_jd2, label):
    actual_tt_jd1, actual_tt_jd2 = tai_to_tt(tai_jd1, tai_jd2)

    assert_allclose(actual_tt_jd1, tai_jd1, atol=0.0, rtol=0.0)
    assert_allclose(actual_tt_jd2, tai_jd2 + ttdtai() / DAY_S, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("tt_jd1, tt_jd2, label", TAI_TT_CASES)
def test_tt_to_tai_constant_offset(tt_jd1, tt_jd2, label):
    actual_tai_jd1, actual_tai_jd2 = tt_to_tai(tt_jd1, tt_jd2)

    assert_allclose(actual_tai_jd1, tt_jd1, atol=0.0, rtol=0.0)
    assert_allclose(actual_tai_jd2, tt_jd2 - ttdtai() / DAY_S, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("tai_jd1, tai_jd2, label", TAI_TT_CASES)
def test_tai_tt_roundtrip(tai_jd1, tai_jd2, label):
    tt_jd1, tt_jd2 = tai_to_tt(tai_jd1, tai_jd2)
    actual_tai_jd1, actual_tai_jd2 = tt_to_tai(tt_jd1, tt_jd2)

    assert_allclose(actual_tai_jd1, tai_jd1, atol=0.0, rtol=0.0)
    assert_allclose(actual_tai_jd2, tai_jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("tt_jd1, tt_jd2, label", TAI_TT_CASES)
def test_tt_tai_roundtrip(tt_jd1, tt_jd2, label):
    tai_jd1, tai_jd2 = tt_to_tai(tt_jd1, tt_jd2)
    actual_tt_jd1, actual_tt_jd2 = tai_to_tt(tai_jd1, tai_jd2)

    assert_allclose(actual_tt_jd1, tt_jd1, atol=0.0, rtol=0.0)
    assert_allclose(actual_tt_jd2, tt_jd2, atol=0.0, rtol=0.0)


def test_tai_to_tt_shapes():
    tai_jd1 = jnp.array([2451545.0, 2451727.0], dtype=float)
    tai_jd2 = jnp.array([0.0, 0.625], dtype=float)

    tt_jd1, tt_jd2 = tai_to_tt(tai_jd1, tai_jd2)

    assert tt_jd1.shape == (2,)
    assert tt_jd2.shape == (2,)


def test_tt_to_tai_shapes():
    tt_jd1 = jnp.array([2451545.0, 2451727.0], dtype=float)
    tt_jd2 = jnp.array([0.0, 0.625], dtype=float)

    tai_jd1, tai_jd2 = tt_to_tai(tt_jd1, tt_jd2)

    assert tai_jd1.shape == (2,)
    assert tai_jd2.shape == (2,)
