import jax
import jax.numpy as jnp
import pytest

from difforb.core.earth_rotation import (
    bias_precession_nutation_matrix,
    cip_xyz,
    cirs_to_gcrs_matrix,
    cio_locator,
    earth_rotation_angle,
    gcrs_to_cirs_matrix,
    inversed_polar_motion_matrix,
    mean_obliquity,
    nutation_matrix,
    polar_motion_matrix,
    precession_bias_matrix,
    tio_locator,
)
from difforb.core.earth_rotation.unified import (
    bias_precession_nutation_matrix_single,
    cirs_to_gcrs_matrix_single,
    cip_xyz_single,
    cio_locator_single,
    earth_rotation_angle_single,
    gcrs_to_cirs_matrix_single,
    inversed_polar_motion_matrix_single,
    mean_obliquity_single,
    nutation_matrix_single,
    polar_motion_matrix_single,
    precession_bias_matrix_single,
    tio_locator_single,
)
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)

@pytest.mark.parametrize(
    "t, expected_shape",
    [
        (0.0, ()),
        (jnp.array([0.0, 0.1, 0.2]), (3,)),
        (jnp.array([[0.0], [0.1], [0.2]]), (3, 1)),
    ],
)
def test_mean_obliquity_shape(t, expected_shape):
    actual = mean_obliquity(t)
    assert actual.shape == expected_shape


@pytest.mark.parametrize(
    "t, expected_shape",
    [
        (0.0, ()),
        (jnp.array([0.0, 0.1, 0.2]), (3,)),
        (jnp.array([[0.0], [0.1], [0.2]]), (3, 1)),
    ],
)
def test_tio_locator_shape(t, expected_shape):
    actual = tio_locator(t)
    assert actual.shape == expected_shape


@pytest.mark.parametrize(
    "t, xp, yp, expected_shape",
    [
        (0.0, 0.0, 0.0, (3, 3)),
        (jnp.array([0.0, 0.1, 0.2]), jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]), (3, 3, 3)),
        (jnp.array([[0.0], [0.1], [0.2]]), jnp.array([[0.0], [0.0], [0.0]]), jnp.array([[0.0], [0.0], [0.0]]), (3, 1, 3, 3)),
    ],
)
def test_polar_motion_matrix_shape(t, xp, yp, expected_shape):
    actual = polar_motion_matrix(t, xp, yp)
    assert actual.shape == expected_shape


@pytest.mark.parametrize(
    "t, xp, yp, expected_shape",
    [
        (0.0, 0.0, 0.0, (3, 3)),
        (jnp.array([0.0, 0.1, 0.2]), jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]), (3, 3, 3)),
        (jnp.array([[0.0], [0.1], [0.2]]), jnp.array([[0.0], [0.0], [0.0]]), jnp.array([[0.0], [0.0], [0.0]]), (3, 1, 3, 3)),
    ],
)
def test_inversed_polar_motion_matrix_shape(t, xp, yp, expected_shape):
    actual = inversed_polar_motion_matrix(t, xp, yp)
    assert actual.shape == expected_shape


@pytest.mark.parametrize(
    "ut1_jd1, ut1_jd2, expected_shape",
    [
        (2451545.0, 0.0, ()),
        (jnp.array([2451545.0, 2451546.0, 2451547.0]), jnp.array([0.0, 0.0, 0.0]), (3,)),
        (
            jnp.array([[2451545.0], [2451546.0], [2451547.0]]),
            jnp.array([[0.0], [0.0], [0.0]]),
            (3, 1),
        ),
    ],
)
def test_earth_rotation_angle_shape(ut1_jd1, ut1_jd2, expected_shape):
    actual = earth_rotation_angle(ut1_jd1, ut1_jd2)
    assert actual.shape == expected_shape


@pytest.mark.parametrize(
    "t, cor_delta_obliquity, cor_delta_longitude, expected_shape",
    [
        (0.0, 0.0, 0.0, ()),
        (jnp.array([0.0, 0.1, 0.2]), jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]), (3,)),
        (jnp.array([[0.0], [0.1], [0.2]]), jnp.array([[0.0], [0.0], [0.0]]), jnp.array([[0.0], [0.0], [0.0]]), (3, 1)),
    ],
)
def test_cip_xyz_shape(t, cor_delta_obliquity, cor_delta_longitude, expected_shape):
    x, y, z = cip_xyz(t, cor_delta_obliquity, cor_delta_longitude)
    assert x.shape == expected_shape
    assert y.shape == expected_shape
    assert z.shape == expected_shape


@pytest.mark.parametrize(
    "t, x, y, expected_shape",
    [
        (0.0, 0.0, 0.0, ()),
        (jnp.array([0.0, 0.1, 0.2]), jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]), (3,)),
        (jnp.array([[0.0], [0.1], [0.2]]), jnp.array([[0.0], [0.0], [0.0]]), jnp.array([[0.0], [0.0], [0.0]]), (3, 1)),
    ],
)
def test_cio_locator_shape(t, x, y, expected_shape):
    actual = cio_locator(t, x, y)
    assert actual.shape == expected_shape


@pytest.mark.parametrize(
    "t, expected_shape",
    [
        (0.0, (3, 3)),
        (jnp.array([0.0, 0.1, 0.2]), (3, 3, 3)),
        (jnp.array([[0.0], [0.1], [0.2]]), (3, 1, 3, 3)),
    ],
)
def test_precession_bias_matrix_shape(t, expected_shape):
    actual = precession_bias_matrix(t)
    assert actual.shape == expected_shape


@pytest.mark.parametrize(
    "t, expected_shape",
    [
        (0.0, (3, 3)),
        (jnp.array([0.0, 0.1, 0.2]), (3, 3, 3)),
        (jnp.array([[0.0], [0.1], [0.2]]), (3, 1, 3, 3)),
    ],
)
def test_nutation_matrix_shape(t, expected_shape):
    actual = nutation_matrix(t)
    assert actual.shape == expected_shape


@pytest.mark.parametrize(
    "t, expected_shape",
    [
        (0.0, (3, 3)),
        (jnp.array([0.0, 0.1, 0.2]), (3, 3, 3)),
        (jnp.array([[0.0], [0.1], [0.2]]), (3, 1, 3, 3)),
    ],
)
def test_bias_precession_nutation_matrix_shape(t, expected_shape):
    actual = bias_precession_nutation_matrix(t)
    assert actual.shape == expected_shape


@pytest.mark.parametrize(
    "t, cor_delta_obliquity, cor_delta_longitude, expected_shape",
    [
        (0.0, 0.0, 0.0, (3, 3)),
        (jnp.array([0.0, 0.1, 0.2]), jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]), (3, 3, 3)),
        (jnp.array([[0.0], [0.1], [0.2]]), jnp.array([[0.0], [0.0], [0.0]]), jnp.array([[0.0], [0.0], [0.0]]), (3, 1, 3, 3)),
    ],
)
def test_gcrs_to_cirs_matrix_shape(t, cor_delta_obliquity, cor_delta_longitude, expected_shape):
    actual = gcrs_to_cirs_matrix(t, cor_delta_obliquity, cor_delta_longitude)
    assert actual.shape == expected_shape


@pytest.mark.parametrize(
    "t, cor_delta_obliquity, cor_delta_longitude, expected_shape",
    [
        (0.0, 0.0, 0.0, (3, 3)),
        (jnp.array([0.0, 0.1, 0.2]), jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]), (3, 3, 3)),
        (jnp.array([[0.0], [0.1], [0.2]]), jnp.array([[0.0], [0.0], [0.0]]), jnp.array([[0.0], [0.0], [0.0]]), (3, 1, 3, 3)),
    ],
)
def test_cirs_to_gcrs_matrix_shape(t, cor_delta_obliquity, cor_delta_longitude, expected_shape):
    actual = cirs_to_gcrs_matrix(t, cor_delta_obliquity, cor_delta_longitude)
    assert actual.shape == expected_shape


def test_mean_obliquity_scalar_matches_single():
    actual = mean_obliquity(0.0)
    expected = mean_obliquity_single(0.0)
    assert_allclose(actual, expected, atol=1e-15, rtol=0.0)


def test_tio_locator_scalar_matches_single():
    actual = tio_locator(0.0)
    expected = tio_locator_single(0.0)
    assert_allclose(actual, expected, atol=1e-15, rtol=0.0)


def test_polar_motion_matrix_scalar_matches_single():
    actual = polar_motion_matrix(0.0, 0.0, 0.0)
    expected = polar_motion_matrix_single(0.0, 0.0, 0.0)
    assert_allclose(actual, expected, atol=1e-15, rtol=0.0)


def test_inversed_polar_motion_matrix_scalar_matches_single():
    actual = inversed_polar_motion_matrix(0.0, 0.0, 0.0)
    expected = inversed_polar_motion_matrix_single(0.0, 0.0, 0.0)
    assert_allclose(actual, expected, atol=1e-15, rtol=0.0)


def test_earth_rotation_angle_scalar_matches_single():
    actual = earth_rotation_angle(2451545.0, 0.0)
    expected = earth_rotation_angle_single(2451545.0, 0.0)
    assert_allclose(actual, expected, atol=1e-15, rtol=0.0)


def test_cip_xyz_scalar_matches_single():
    actual = cip_xyz(0.0, 0.0, 0.0)
    expected = cip_xyz_single(0.0, 0.0, 0.0)
    for actual_component, expected_component in zip(actual, expected):
        assert_allclose(actual_component, expected_component, atol=1e-15, rtol=0.0)


def test_cio_locator_scalar_matches_single():
    x, y, _ = cip_xyz_single(0.0, 0.0, 0.0)
    actual = cio_locator(0.0, x, y)
    expected = cio_locator_single(0.0, x, y)
    assert_allclose(actual, expected, atol=1e-15, rtol=0.0)


def test_precession_bias_matrix_scalar_matches_single():
    actual = precession_bias_matrix(0.0)
    expected = precession_bias_matrix_single(0.0)
    assert_allclose(actual, expected, atol=1e-15, rtol=0.0)


def test_nutation_matrix_scalar_matches_single():
    actual = nutation_matrix(0.0)
    expected = nutation_matrix_single(0.0)
    assert_allclose(actual, expected, atol=1e-15, rtol=0.0)


def test_bias_precession_nutation_matrix_scalar_matches_single():
    actual = bias_precession_nutation_matrix(0.0)
    expected = bias_precession_nutation_matrix_single(0.0)
    assert_allclose(actual, expected, atol=1e-15, rtol=0.0)


def test_gcrs_to_cirs_matrix_scalar_matches_single():
    actual = gcrs_to_cirs_matrix(0.0, 0.0, 0.0)
    expected = gcrs_to_cirs_matrix_single(0.0, 0.0, 0.0)
    assert_allclose(actual, expected, atol=1e-15, rtol=0.0)


def test_cirs_to_gcrs_matrix_scalar_matches_single():
    actual = cirs_to_gcrs_matrix(0.0, 0.0, 0.0)
    expected = cirs_to_gcrs_matrix_single(0.0, 0.0, 0.0)
    assert_allclose(actual, expected, atol=1e-15, rtol=0.0)
