import jax.numpy as jnp

from numpy.testing import assert_allclose

from difforb.core.constants import DAY_S
from difforb.core.time.timedelta import TimeDelta


def test_timedelta_from_days():
    delta = TimeDelta.from_days(1.25)

    assert delta.shape == ()
    assert_allclose(delta.jd, 1.25, atol=0.0, rtol=0.0)
    assert_allclose(delta.days, 1.25, atol=0.0, rtol=0.0)
    assert_allclose(delta.seconds, 1.25 * DAY_S, atol=0.0, rtol=0.0)


def test_timedelta_from_seconds():
    delta = TimeDelta.from_seconds(12345.6789)

    assert_allclose(delta.jd, 12345.6789 / DAY_S, atol=0.0, rtol=0.0)
    assert_allclose(delta.seconds, 12345.6789, atol=1e-12, rtol=0.0)


def test_timedelta_arithmetic():
    left = TimeDelta.from_days(1.25)
    right = TimeDelta.from_seconds(3600.0)

    summed = left + right
    diffed = left - right
    negated = -right

    assert_allclose(summed.jd, 1.25 + 3600.0 / DAY_S, atol=0.0, rtol=0.0)
    assert_allclose(diffed.jd, 1.25 - 3600.0 / DAY_S, atol=0.0, rtol=0.0)
    assert_allclose(negated.jd, -3600.0 / DAY_S, atol=0.0, rtol=0.0)


def test_timedelta_shapes():
    delta = TimeDelta.from_seconds(jnp.array([0.0, 60.0, 120.0]))

    assert delta.shape == (3,)
    assert delta.jd1.shape == (3,)
    assert delta.jd2.shape == (3,)
    assert delta.jd.shape == (3,)
    assert delta.seconds.shape == (3,)
