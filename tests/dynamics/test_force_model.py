import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from difforb.dynamics.force_model import (
    CometOutgassingEffect,
    EarthJ2Perturbation,
    EmpiricalRadiationPressure,
    EmpiricalYarkovskyEffect,
    Force,
    ForceModel,
    ParametrizedForce,
    RTNDistanceLawNonGravEffect,
    SolarJ2Perturbation,
    compute_newtonian_acceleration,
    compute_planetary_potentials,
    compute_rtn_distance_law_non_grav_acceleration,
)
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)


class FakeEphemerisBody(eqx.Module):
    pos: jax.Array
    vel: jax.Array
    acc: jax.Array
    gm: float

    def __init__(self, pos, vel=(0.0, 0.0, 0.0), acc=(0.0, 0.0, 0.0), gm=1.0):
        self.pos = jnp.asarray(pos, dtype=float)
        self.vel = jnp.asarray(vel, dtype=float)
        self.acc = jnp.asarray(acc, dtype=float)
        self.gm = float(gm)

    def _bcrs_pos_jd(self, tdb_jd1, tdb_jd2):
        return self.pos + jnp.zeros_like(jnp.asarray(tdb_jd1) + jnp.asarray(tdb_jd2))[..., None]

    def _bcrs_pv_jd(self, tdb_jd1, tdb_jd2):
        pos = self._bcrs_pos_jd(tdb_jd1, tdb_jd2)
        vel = self.vel + jnp.zeros_like(jnp.asarray(tdb_jd1) + jnp.asarray(tdb_jd2))[..., None]
        return pos, vel

    def _bcrs_pva_jd(self, tdb_jd1, tdb_jd2):
        pos, vel = self._bcrs_pv_jd(tdb_jd1, tdb_jd2)
        acc = self.acc + jnp.zeros_like(jnp.asarray(tdb_jd1) + jnp.asarray(tdb_jd2))[..., None]
        return pos, vel, acc


class ConstantForce(Force):
    acceleration: jax.Array

    def __init__(self, acceleration):
        self.acceleration = jnp.asarray(acceleration, dtype=float)

    def __call__(self, tdb_jd1, tdb_jd2, state, args):
        return self.acceleration

    @property
    def shape(self):
        return self.acceleration.shape[:-1]


class SimpleParametrizedForce(ParametrizedForce):
    params: jax.Array
    scales: jax.Array
    param_names: tuple = eqx.field(static=True)

    def __init__(self, params, scales, param_names):
        self.params = jnp.asarray(params, dtype=float)
        self.scales = jnp.asarray(scales, dtype=float)
        self.param_names = tuple(param_names)

    @property
    def n_estimated_params(self):
        return self.params.shape[-1]

    def get_estimated_params(self):
        return self.params

    def get_estimated_param_scales(self):
        return self.scales

    def update_estimated_params(self, new_params):
        return SimpleParametrizedForce(new_params, self.scales, self.param_names)

    def get_estimated_param_names(self):
        return list(self.param_names)

    def __call__(self, tdb_jd1, tdb_jd2, state, args):
        return jnp.zeros(3, dtype=float)

    @property
    def shape(self):
        return self.params.shape[:-1]


def test_compute_newtonian_acceleration_closed_form():
    target_pos = jnp.asarray([0.5, -0.25, 0.125], dtype=float)
    body_pos = jnp.asarray(
        [
            [2.5, -0.25, 0.125],
            [0.5, 2.75, 0.125],
            [0.5, -0.25, -3.875],
        ],
        dtype=float,
    )
    body_gm = jnp.asarray([8.0, 27.0, 64.0], dtype=float)
    expected = jnp.asarray([2.0, 3.0, -4.0], dtype=float)

    actual = compute_newtonian_acceleration(target_pos, body_pos, body_gm)

    print(
        "[force_model.newtonian] "
        f"max_abs_diff={float(jnp.max(jnp.abs(actual - expected))):+.12e} au/day^2"
    )

    assert_allclose(actual, expected, atol=1.0e-15, rtol=0.0)


def test_compute_planetary_potentials_pairwise():
    body_pos = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    body_gm = jnp.asarray([5.0, 7.0, 11.0], dtype=float)
    expected = jnp.asarray(
        [
            7.0 / 2.0 + 11.0 / 3.0,
            5.0 / 2.0 + 11.0 / jnp.sqrt(13.0),
            5.0 / 3.0 + 7.0 / jnp.sqrt(13.0),
        ],
        dtype=float,
    )

    actual = compute_planetary_potentials(body_pos, body_gm)

    print(
        "[force_model.planetary_potentials] "
        f"max_abs_diff={float(jnp.max(jnp.abs(actual - expected))):+.12e} au^2/day^2"
    )

    assert_allclose(actual, expected, atol=1.0e-15, rtol=0.0)


@pytest.mark.parametrize(
    ("label", "pos", "expected"),
    [
        ("equator", [2.0, 0.0, 0.0], [-0.01875, 0.0, 0.0]),
        ("pole", [0.0, 0.0, 2.0], [0.0, 0.0, 0.0375]),
    ],
)
def test_solar_j2_perturbation_axis_cases(label, pos, expected):
    body = FakeEphemerisBody(pos=(0.0, 0.0, 0.0), gm=2.0)
    force = SolarJ2Perturbation(
        body=body,
        j2=0.1,
        radius=1.0,
        pole_unit_vec=jnp.asarray([0.0, 0.0, 1.0], dtype=float),
    )
    actual = force(
        tdb_jd1=jnp.asarray(2460741.0, dtype=float),
        tdb_jd2=jnp.asarray(0.5, dtype=float),
        state=(jnp.asarray(pos, dtype=float), jnp.zeros(3, dtype=float)),
        args=None,
    )
    expected = jnp.asarray(expected, dtype=float)

    print(
        "[force_model.j2.axis] "
        f"label={label:<7} "
        f"max_abs_diff={float(jnp.max(jnp.abs(actual - expected))):+.12e} au/day^2"
    )

    assert_allclose(actual, expected, atol=1.0e-15, rtol=0.0)


def test_earth_j2_perturbation_distance_cutoff():
    body = FakeEphemerisBody(pos=(0.0, 0.0, 0.0), gm=2.0)
    force = EarthJ2Perturbation(
        body=body,
        j2=0.1,
        radius=1.0,
        max_distance=3.0,
        fixed_pole_unit_vec=jnp.asarray([0.0, 0.0, 1.0], dtype=float),
    )
    tdb_jd1 = jnp.asarray(2460741.0, dtype=float)
    tdb_jd2 = jnp.asarray(0.5, dtype=float)
    vel = jnp.zeros(3, dtype=float)
    near_pos = jnp.asarray([2.0, 0.0, 0.0], dtype=float)
    far_pos = jnp.asarray([5.0, 0.0, 0.0], dtype=float)

    near_actual = force(tdb_jd1, tdb_jd2, (near_pos, vel), args=None)
    near_expected = jnp.asarray([-0.01875, 0.0, 0.0], dtype=float)
    far_actual = force(tdb_jd1, tdb_jd2, (far_pos, vel), args=None)

    print(
        "[force_model.j2.cutoff] "
        f"near_diff={float(jnp.max(jnp.abs(near_actual - near_expected))):+.12e} au/day^2 "
        f"far_norm={float(jnp.linalg.norm(far_actual)):+.12e} au/day^2"
    )

    assert_allclose(near_actual, near_expected, atol=1.0e-15, rtol=0.0)
    assert_allclose(far_actual, jnp.zeros(3, dtype=float), atol=0.0, rtol=0.0)


def test_compute_rtn_distance_law_non_grav_acceleration_axis_case():
    pos = jnp.asarray([2.0, 0.0, 0.0], dtype=float)
    vel = jnp.asarray([0.0, 0.03, 0.0], dtype=float)
    pos_sun = jnp.zeros(3, dtype=float)
    vel_sun = jnp.zeros(3, dtype=float)
    a1, a2, a3 = 1.0e-10, -2.0e-10, 3.0e-10
    expected = 0.25 * jnp.asarray([a1, a2, a3], dtype=float)

    actual = compute_rtn_distance_law_non_grav_acceleration(
        pos=pos,
        vel=vel,
        pos_sun=pos_sun,
        vel_sun=vel_sun,
        A1=a1,
        A2=a2,
        A3=a3,
        alpha=1.0,
        r0=1.0,
        m=2.0,
        n=1.0,
        k=0.0,
    )

    print(
        "[force_model.rtn.axis] "
        f"max_abs_diff={float(jnp.max(jnp.abs(actual - expected))):+.12e} au/day^2"
    )

    assert_allclose(actual, expected, atol=1.0e-24, rtol=0.0)


def test_rtn_distance_law_effect_call_and_estimated_params():
    sun = FakeEphemerisBody(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0))
    force = RTNDistanceLawNonGravEffect(
        sun=sun,
        estimated_params=("A3", "A1"),
        A1=1.0e-10,
        A2=-2.0e-10,
        A3=3.0e-10,
        alpha=1.0,
        r0=1.0,
        m=2.0,
        n=1.0,
        k=0.0,
        param_prefix="Test",
    )
    pos = jnp.asarray([2.0, 0.0, 0.0], dtype=float)
    vel = jnp.asarray([0.0, 0.03, 0.0], dtype=float)

    actual = force(jnp.asarray(2460741.0), jnp.asarray(0.5), (pos, vel), args=None)
    expected = 0.25 * jnp.asarray([1.0e-10, -2.0e-10, 3.0e-10], dtype=float)
    updated = force.update_estimated_params(jnp.asarray([9.0e-10, -8.0e-10], dtype=float))

    print(
        "[force_model.rtn.effect] "
        f"max_abs_diff={float(jnp.max(jnp.abs(actual - expected))):+.12e} au/day^2 "
        f"names={force.get_estimated_param_names()}"
    )

    assert force.shape == ()
    assert force.get_estimated_param_names() == ["Test_A1", "Test_A3"]
    assert_allclose(force.get_estimated_params(), jnp.asarray([1.0e-10, 3.0e-10], dtype=float), atol=0.0, rtol=0.0)
    assert_allclose(force.get_estimated_param_scales(), jnp.asarray([1.0e-12, 1.0e-12], dtype=float), atol=0.0, rtol=0.0)
    assert_allclose(actual, expected, atol=1.0e-24, rtol=0.0)
    assert_allclose(updated.params, jnp.asarray([9.0e-10, -2.0e-10, -8.0e-10], dtype=float), atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("label", "force", "expected_names", "expected_params", "expected_scales"),
    [
        (
            "outgassing",
            CometOutgassingEffect(
                sun=FakeEphemerisBody(pos=(0.0, 0.0, 0.0)),
                estimated_params=("A1", "A2", "A3"),
                A1=1.0e-8,
                A2=2.0e-8,
                A3=3.0e-8,
            ),
            ["Outgassing_A1", "Outgassing_A2", "Outgassing_A3"],
            [1.0e-8, 2.0e-8, 3.0e-8],
            [1.0e-8, 1.0e-8, 1.0e-8],
        ),
        (
            "yarkovsky",
            EmpiricalYarkovskyEffect(
                sun=FakeEphemerisBody(pos=(0.0, 0.0, 0.0)),
                A2=5.0e-14,
            ),
            ["Yarkovsky_A2"],
            [5.0e-14],
            [1.0e-13],
        ),
        (
            "radiation_pressure",
            EmpiricalRadiationPressure(
                sun=FakeEphemerisBody(pos=(0.0, 0.0, 0.0)),
                A1=7.0e-13,
            ),
            ["RadiationPressure_A1"],
            [7.0e-13],
            [1.0e-12],
        ),
    ],
)
def test_specialized_rtn_effect_param_contracts(label, force, expected_names, expected_params, expected_scales):
    print(
        "[force_model.rtn.specialized] "
        f"label={label:<18} "
        f"names={force.get_estimated_param_names()}"
    )

    assert force.get_estimated_param_names() == expected_names
    assert_allclose(force.get_estimated_params(), jnp.asarray(expected_params, dtype=float), atol=0.0, rtol=0.0)
    assert_allclose(force.get_estimated_param_scales(), jnp.asarray(expected_scales, dtype=float), atol=0.0, rtol=0.0)


def test_force_model_sums_forces():
    model = ForceModel(
        [
            ConstantForce([1.0e-6, 2.0e-6, -3.0e-6]),
            ConstantForce([-4.0e-7, 5.0e-7, 6.0e-7]),
        ]
    )
    state = (jnp.asarray([1.0, 0.0, 0.0], dtype=float), jnp.asarray([0.0, 0.01, 0.0], dtype=float))
    actual = model(0.25, state, args=(jnp.asarray(2460741.0), jnp.asarray(0.5)))
    expected = jnp.asarray([6.0e-7, 2.5e-6, -2.4e-6], dtype=float)

    print(
        "[force_model.sum] "
        f"max_abs_diff={float(jnp.max(jnp.abs(actual - expected))):+.12e} au/day^2"
    )

    assert_allclose(actual, expected, atol=1.0e-18, rtol=0.0)


def test_force_model_estimated_param_contract():
    first = SimpleParametrizedForce(
        params=[1.0e-10, 2.0e-10],
        scales=[1.0e-12, 1.0e-12],
        param_names=("F1_A", "F1_B"),
    )
    second = SimpleParametrizedForce(
        params=[3.0e-10],
        scales=[1.0e-13],
        param_names=("F2_C",),
    )
    model = ForceModel([ConstantForce([0.0, 0.0, 0.0]), first, second])
    updated = model.update_estimated_params(jnp.asarray([4.0e-10, 5.0e-10, 6.0e-10], dtype=float))

    print(
        "[force_model.params] "
        f"names={model.get_all_estimated_param_names()} "
        f"shape={model.shape}"
    )

    assert model.get_all_estimated_param_names() == ["F1_A", "F1_B", "F2_C"]
    assert_allclose(model.get_all_estimated_params(), jnp.asarray([1.0e-10, 2.0e-10, 3.0e-10], dtype=float), atol=0.0, rtol=0.0)
    assert_allclose(model.get_all_estimated_param_scales(), jnp.asarray([1.0e-12, 1.0e-12, 1.0e-13], dtype=float), atol=0.0, rtol=0.0)
    assert_allclose(updated.get_all_estimated_params(), jnp.asarray([4.0e-10, 5.0e-10, 6.0e-10], dtype=float), atol=0.0, rtol=0.0)
