import jax
import jax.numpy as jnp

from difforb.core.eop.interpolate import lagrangian_interpolate, lagrangian_interpolate_single
from tests.assertions import assert_allclose


jax.config.update("jax_enable_x64", True)


def _lagrange4_reference(x_window, y_window, xint):
    total = 0.0
    for i in range(4):
        basis = 1.0
        for j in range(4):
            if i != j:
                basis *= (xint - x_window[j]) / (x_window[i] - x_window[j])
        total += y_window[i] * basis
    return total


def test_lagrangian_interpolate_single_recovers_cubic_exactly():
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    y = x ** 3 - 2.0 * x ** 2 + x - 5.0
    xint = 1.5
    expected = xint ** 3 - 2.0 * xint ** 2 + xint - 5.0

    actual = lagrangian_interpolate_single(x, y, xint)

    assert_allclose(actual, expected, atol=1.0e-12, rtol=0.0)


def test_lagrangian_interpolate_single_matches_sample_node_exactly():
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = jnp.asarray([1.0, -2.0, 0.5, 4.5, -1.5, 3.0])
    xint = x[3]
    expected = y[3]

    actual = lagrangian_interpolate_single(x, y, xint)

    assert_allclose(actual, expected, atol=0.0, rtol=0.0)


def test_lagrangian_interpolate_single_uses_first_window_near_start():
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = jnp.asarray([1.0, 2.0, 0.5, 4.0, -1.0, 3.0])
    xint = 0.2
    expected = _lagrange4_reference(x[:4], y[:4], xint)

    actual = lagrangian_interpolate_single(x, y, xint)

    assert_allclose(actual, expected, atol=1.0e-12, rtol=0.0)


def test_lagrangian_interpolate_single_uses_centered_window_interior():
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = jnp.asarray([1.0, 2.0, 0.5, 4.0, -1.0, 3.0])
    xint = 2.2
    expected = _lagrange4_reference(x[1:5], y[1:5], xint)

    actual = lagrangian_interpolate_single(x, y, xint)

    assert_allclose(actual, expected, atol=1.0e-12, rtol=0.0)


def test_lagrangian_interpolate_single_uses_last_window_near_end():
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = jnp.asarray([1.0, 2.0, 0.5, 4.0, -1.0, 3.0])
    xint = 4.8
    expected = _lagrange4_reference(x[2:6], y[2:6], xint)

    actual = lagrangian_interpolate_single(x, y, xint)

    assert_allclose(actual, expected, atol=1.0e-12, rtol=0.0)


def test_lagrangian_interpolate_clamps_queries_below_sample_range():
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    y = jnp.asarray([1.0, -2.0, 0.5, 4.5])
    xint = -1.5
    expected = lagrangian_interpolate_single(x, y, x[0])

    actual = lagrangian_interpolate(x, y, xint)

    assert_allclose(actual, expected, atol=0.0, rtol=0.0)


def test_lagrangian_interpolate_clamps_queries_above_sample_range():
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    y = jnp.asarray([1.0, -2.0, 0.5, 4.5])
    xint = 4.5
    expected = lagrangian_interpolate_single(x, y, x[-1])

    actual = lagrangian_interpolate(x, y, xint)

    assert_allclose(actual, expected, atol=0.0, rtol=0.0)


def test_lagrangian_interpolate_preserves_batch_shape():
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    y = jnp.asarray([1.0, -2.0, 0.5, 4.5])
    xint = jnp.asarray([[0.25, 1.25], [2.25, 3.25], [-0.5, 4.0]])

    actual = lagrangian_interpolate(x, y, xint)

    assert actual.shape == xint.shape


def test_lagrangian_interpolate_batch_matches_pointwise_single():
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = jnp.asarray([1.0, 2.0, 0.5, 4.0, -1.0, 3.0])
    xint = jnp.asarray([[0.2, 2.2], [4.8, -1.0], [5.5, 3.0]])
    clamped = jnp.clip(xint, x[0], x[-1])
    expected = jax.vmap(lambda row: jax.vmap(lambda value: lagrangian_interpolate_single(x, y, value))(row))(clamped)

    actual = lagrangian_interpolate(x, y, xint)

    assert_allclose(actual, expected, atol=1.0e-12, rtol=0.0)
