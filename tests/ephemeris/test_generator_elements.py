import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from difforb.body.ephbody import EphemerisBody
from difforb.body.smallbody import SmallBody
from difforb.core.element import KepElement
from difforb.core.state.frame import HELIO_ECLIP_J2000
from difforb.core.time.timescale import Time
from difforb.dynamics.dynamic_system import DynamicSystem
from difforb.ephemeris.generator import EphemerisGenerator
from difforb.integrator.integrator import NumericalIntegrator
from tests.ephemeris.generator_reference import (
    EPOCH_TDB_JD,
    HORIZONS_GENERATOR_ELEMENT_CASES,
    HORIZONS_TARGET_TDB_JD,
)
from tests.assertions import assert_allclose


@pytest.mark.parametrize(
    ("label", "target_command", "target_name", "initial_elements", "expected_elements"),
    HORIZONS_GENERATOR_ELEMENT_CASES,
)
def test_elements_table_against_horizons(
        default_ephemeris,
        label,
        target_command,
        target_name,
        initial_elements,
        expected_elements,
):
    sun = EphemerisBody("sun", eph=default_ephemeris)
    force_model = DynamicSystem.from_extended_system(default_ephemeris).build_force_model()
    integrator = NumericalIntegrator(method="IAS15", tol=1.0e-12, initial_step=0.05, max_steps=4096)
    tdb0 = Time.from_tdb_jd(EPOCH_TDB_JD, 0.0).tdb()
    tdb1 = Time.from_tdb_jd(HORIZONS_TARGET_TDB_JD, 0.0).tdb()
    initial = KepElement.from_classical(tdb0, *initial_elements)
    expected = KepElement.from_classical(tdb1, *expected_elements)

    target = SmallBody.create(initial, sun=sun).propagate(
        Time.from_tdb_jd(EPOCH_TDB_JD, -0.5).tdb(),
        Time.from_tdb_jd(HORIZONS_TARGET_TDB_JD, 0.5).tdb(),
        force_model,
        integrator,
    )
    actual = EphemerisGenerator(target).elements_table(tdb1)
    p_diff = actual.p - expected.p
    e_diff = actual.e - expected.e

    print(
        "[ephemeris.elements.horizons] "
        f"label={label:<25} "
        f"target={target_command:<11} "
        f"name={target_name} "
        f"p_diff={float(p_diff):+.12e} au "
        f"e_diff={float(e_diff):+.12e}"
    )

    assert_allclose(actual.p, expected.p, atol=1.0e-11, rtol=0.0)
    assert_allclose(actual.e, expected.e, atol=1.0e-12, rtol=0.0)
    for angle_name, actual_angle, expected_angle in [
        ("inc", actual.inc, expected.inc),
        ("node", actual.node, expected.node),
        ("peri", actual.peri, expected.peri),
        ("m", actual.m, expected.m),
    ]:
        angle_diff = jnp.arctan2(jnp.sin(actual_angle - expected_angle), jnp.cos(actual_angle - expected_angle))
        print(
            "[ephemeris.elements.horizons.angle] "
            f"label={label:<25} "
            f"target={target_command:<11} "
            f"angle={angle_name:<4} "
            f"diff={float(angle_diff):+.12e} rad"
        )
        assert_allclose(angle_diff, 0.0, atol=2.0e-10, rtol=0.0)


def test_elements_table_matches_target_state(generator):
    tdb = Time.from_tdb_jd(EPOCH_TDB_JD, 0.75).tdb()

    actual = generator.elements_table(tdb)
    expected_state = generator.target.state(tdb, frame=HELIO_ECLIP_J2000, sun=generator.sun)
    expected = KepElement.from_state(expected_state, sun=generator.sun)

    print(
        "[ephemeris.elements] "
        f"p_diff={float(actual.p - expected.p):+.12e} au "
        f"e_diff={float(actual.e - expected.e):+.12e}"
    )

    assert actual.shape == ()
    assert_allclose(actual.array, expected.array, atol=1.0e-12, rtol=0.0)
