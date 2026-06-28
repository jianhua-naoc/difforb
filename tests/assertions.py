"""Shared assertion helpers for JAX-first test cases."""

import pytest
import jax.numpy as jnp


def assert_allclose(actual, expected, *, atol=1e-12, rtol=0.0, equal_nan=False, msg=None):
    """Fail the test when two arrays are not numerically close."""
    actual_arr = jnp.asarray(actual)
    expected_arr = jnp.asarray(expected)
    if bool(jnp.allclose(actual_arr, expected_arr, atol=atol, rtol=rtol, equal_nan=equal_nan)):
        return

    pytest.fail(
        msg
        or (
            "Arrays are not close.\n"
            f"actual={actual_arr}\n"
            f"expected={expected_arr}\n"
            f"atol={atol}, rtol={rtol}, equal_nan={equal_nan}"
        )
    )


def assert_array_equal(actual, expected, *, msg=None):
    """Fail the test when two arrays are not exactly equal."""
    actual_arr = jnp.asarray(actual)
    expected_arr = jnp.asarray(expected)
    if bool(jnp.array_equal(actual_arr, expected_arr)):
        return

    pytest.fail(msg or f"Arrays are not equal.\nactual={actual_arr}\nexpected={expected_arr}")
