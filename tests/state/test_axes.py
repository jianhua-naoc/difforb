import jax
import jax.numpy as jnp
import pytest

from difforb.core.state.axes import (
    AXES_TO_ICRS_ROT,
    ECLIP_J2000_TO_ICRS,
    ICRS_TO_ICRS,
    J2000_TO_ICRS,
    Axes,
    axes_to_icrs_rotation,
    icrs_to_axes_rotation,
)
from tests.assertions import assert_allclose, assert_array_equal

jax.config.update("jax_enable_x64", True)

AXES_CASES = [
    (Axes.ICRS, "ICRS"),
    (Axes.J2000, "J2000"),
    (Axes.ECLIP_J2000, "ECLIP_J2000"),
]

REFERENCE_ECLIP_J2000_TO_ICRS = jnp.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.9174820620691818, 0.3977771559319137],
    [0.0, -0.3977771559319137, 0.9174820620691818],
], dtype=jnp.float64)

REFERENCE_J2000_TO_ICRS = jnp.array([
    [9.9999999999999423e-01, 7.0782794778595927e-08, -8.0561491730079863e-08],
    [-7.0782797441991980e-08, 9.9999999999999689e-01, -3.3060408839853798e-08],
    [8.0561489389971497e-08, 3.3060414542221364e-08, 9.9999999999999623e-01],
], dtype=jnp.float64)

REFERENCE_VECTOR = jnp.array([0.3812, -1.4725, 2.1834], dtype=jnp.float64)
REFERENCE_BATCH_VECTORS = jnp.array([
    [0.3812, -1.4725, 2.1834],
    [-0.9044, 0.1188, 1.7042],
    [2.1140, -0.6633, -0.2457],
], dtype=jnp.float64)

# -------------------------------------------------------------------------
# Registry And Basic Interface
# -------------------------------------------------------------------------


def test_axes_to_icrs_rotation_registry_covers_all_axes():
    assert set(AXES_TO_ICRS_ROT) == set(Axes)


@pytest.mark.parametrize("axes, label", AXES_CASES)
def test_axes_to_icrs_rotation_shape(axes, label):
    actual = axes_to_icrs_rotation(axes)

    print(
        "[state.axes_to_icrs_rotation.shape] "
        f"label={label:<16} "
        f"shape={actual.shape}"
    )

    assert actual.shape == (3, 3)


@pytest.mark.parametrize("axes, label", AXES_CASES)
def test_icrs_to_axes_rotation_shape(axes, label):
    actual = icrs_to_axes_rotation(axes)

    print(
        "[state.icrs_to_axes_rotation.shape] "
        f"label={label:<16} "
        f"shape={actual.shape}"
    )

    assert actual.shape == (3, 3)


# -------------------------------------------------------------------------
# Fixed Matrix Regression
# -------------------------------------------------------------------------


def test_icrs_to_icrs_rotation_is_identity():
    assert_array_equal(ICRS_TO_ICRS, jnp.eye(3, dtype=jnp.float64))


def test_eclip_j2000_to_icrs_matches_reference_matrix():
    max_abs_diff = jnp.max(jnp.abs(ECLIP_J2000_TO_ICRS - REFERENCE_ECLIP_J2000_TO_ICRS))

    print(
        "[state.ECLIP_J2000_TO_ICRS] "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        ECLIP_J2000_TO_ICRS,
        REFERENCE_ECLIP_J2000_TO_ICRS,
        atol=1e-15,
        rtol=0.0,
        msg="ECLIP_J2000_TO_ICRS does not match the fixed reference matrix.",
    )


def test_j2000_to_icrs_matches_reference_matrix():
    max_abs_diff = jnp.max(jnp.abs(J2000_TO_ICRS - REFERENCE_J2000_TO_ICRS))

    print(
        "[state.J2000_TO_ICRS] "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        J2000_TO_ICRS,
        REFERENCE_J2000_TO_ICRS,
        atol=1e-20,
        rtol=0.0,
        msg="J2000_TO_ICRS does not match the fixed reference matrix.",
    )


# -------------------------------------------------------------------------
# Rotation Matrix Contracts
# -------------------------------------------------------------------------


@pytest.mark.parametrize("axes, label", AXES_CASES)
def test_axes_to_icrs_rotation_is_orthogonal(axes, label):
    actual = axes_to_icrs_rotation(axes)
    ident = jnp.eye(3, dtype=jnp.float64)
    left = actual @ actual.T
    right = actual.T @ actual
    left_max_abs_diff = jnp.max(jnp.abs(left - ident))
    right_max_abs_diff = jnp.max(jnp.abs(right - ident))

    print(
        "[state.axes_to_icrs_rotation.orthogonal] "
        f"label={label:<16} "
        f"left_max_abs_diff={float(left_max_abs_diff):+.12e} "
        f"right_max_abs_diff={float(right_max_abs_diff):+.12e}"
    )

    assert_allclose(left, ident, atol=1e-15, rtol=0.0, msg=f"{label} left orthogonality check failed.")
    assert_allclose(right, ident, atol=1e-15, rtol=0.0, msg=f"{label} right orthogonality check failed.")


@pytest.mark.parametrize("axes, label", AXES_CASES)
def test_axes_to_icrs_rotation_has_positive_unit_determinant(axes, label):
    actual = axes_to_icrs_rotation(axes)
    det = jnp.linalg.det(actual)
    diff = det - 1.0

    print(
        "[state.axes_to_icrs_rotation.det] "
        f"label={label:<16} "
        f"det_diff={float(diff):+.12e}"
    )

    assert_allclose(det, 1.0, atol=1e-15, rtol=0.0, msg=f"{label} determinant is not +1.")


@pytest.mark.parametrize("axes, label", AXES_CASES)
def test_icrs_to_axes_rotation_is_transpose_of_forward_rotation(axes, label):
    actual = icrs_to_axes_rotation(axes)
    expected = axes_to_icrs_rotation(axes).T
    max_abs_diff = jnp.max(jnp.abs(actual - expected))

    print(
        "[state.icrs_to_axes_rotation.transpose] "
        f"label={label:<16} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        actual,
        expected,
        atol=0.0,
        rtol=0.0,
        msg=f"{label} inverse rotation is not the transpose of the forward rotation.",
    )


# -------------------------------------------------------------------------
# Rotation Semantics
# -------------------------------------------------------------------------


@pytest.mark.parametrize("axes, label", AXES_CASES)
def test_axis_rotation_roundtrip_preserves_single_vector(axes, label):
    forward = axes_to_icrs_rotation(axes)
    backward = icrs_to_axes_rotation(axes)
    recovered = REFERENCE_VECTOR @ forward @ backward
    max_abs_diff = jnp.max(jnp.abs(recovered - REFERENCE_VECTOR))

    print(
        "[state.axis_rotation.roundtrip_single] "
        f"label={label:<16} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        recovered,
        REFERENCE_VECTOR,
        atol=1e-15,
        rtol=0.0,
        msg=f"{label} roundtrip did not recover the original single vector.",
    )


@pytest.mark.parametrize("axes, label", AXES_CASES)
def test_axis_rotation_roundtrip_preserves_batched_vectors(axes, label):
    forward = axes_to_icrs_rotation(axes)
    backward = icrs_to_axes_rotation(axes)
    recovered = (REFERENCE_BATCH_VECTORS @ forward) @ backward
    max_abs_diff = jnp.max(jnp.abs(recovered - REFERENCE_BATCH_VECTORS))

    print(
        "[state.axis_rotation.roundtrip_batch] "
        f"label={label:<16} "
        f"max_abs_diff={float(max_abs_diff):+.12e}"
    )

    assert_allclose(
        recovered,
        REFERENCE_BATCH_VECTORS,
        atol=1e-15,
        rtol=0.0,
        msg=f"{label} roundtrip did not recover the original batch vectors.",
    )


def test_eclip_j2000_rotation_preserves_x_axis():
    rotated = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64) @ ECLIP_J2000_TO_ICRS
    assert_allclose(
        rotated,
        jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        atol=1e-15,
        rtol=0.0,
        msg="ECLIP_J2000_TO_ICRS should preserve the x-axis for an x-axis rotation.",
    )


def test_j2000_rotation_is_not_identity():
    assert not bool(jnp.array_equal(J2000_TO_ICRS, jnp.eye(3, dtype=jnp.float64)))


# -------------------------------------------------------------------------
# Error Handling
# -------------------------------------------------------------------------


def test_axes_to_icrs_rotation_raises_for_missing_registration(monkeypatch):
    monkeypatch.delitem(AXES_TO_ICRS_ROT, Axes.ICRS)

    with pytest.raises(KeyError, match="No rotation to ``ICRS`` is registered for axes 'ICRS'"):
        axes_to_icrs_rotation(Axes.ICRS)
