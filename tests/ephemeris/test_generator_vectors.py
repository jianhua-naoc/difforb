import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from difforb.body.ephbody import EphemerisBody
from difforb.body.smallbody import SmallBody
from difforb.core.element import KepElement
from difforb.core.time.timescale import Time
from difforb.dynamics.dynamic_system import DynamicSystem
from difforb.ephemeris.core import VectorTable
from difforb.ephemeris.generator import EphemerisGenerator
from difforb.integrator.integrator import NumericalIntegrator
from tests.ephemeris.generator_reference import (
    EPOCH_TDB_JD,
    HORIZONS_GENERATOR_VECTOR_CASES,
    HORIZONS_TARGET_TDB_JD,
)
from tests.assertions import assert_allclose


@pytest.mark.parametrize(
    (
        "label",
        "target_command",
        "target_name",
        "initial_elements",
        "expected_geometric",
        "expected_astrometric",
        "expected_apparent",
    ),
    HORIZONS_GENERATOR_VECTOR_CASES,
)
def test_vector_table_against_horizons(
        default_ephemeris,
        ground_site,
        label,
        target_command,
        target_name,
        initial_elements,
        expected_geometric,
        expected_astrometric,
        expected_apparent,
):
    sun = EphemerisBody("sun", eph=default_ephemeris)
    force_model = DynamicSystem.from_extended_system(default_ephemeris).build_force_model()
    integrator = NumericalIntegrator(method="IAS15", tol=1.0e-12, initial_step=0.05, max_steps=4096)
    tdb0 = Time.from_tdb_jd(EPOCH_TDB_JD, 0.0).tdb()
    initial = KepElement.from_classical(tdb0, *initial_elements)

    target = SmallBody.create(initial, sun=sun).propagate(
        Time.from_tdb_jd(EPOCH_TDB_JD, -0.5).tdb(),
        Time.from_tdb_jd(HORIZONS_TARGET_TDB_JD, 0.5).tdb(),
        force_model,
        integrator,
    )
    table = EphemerisGenerator(target).vector_table(Time.from_tdb_jd(HORIZONS_TARGET_TDB_JD, 0.0), ground_site)

    for layer_name, actual, expected in [
        ("geometric", table.geometric, expected_geometric),
        ("astrometric", table.astrometric, expected_astrometric),
        ("apparent", table.apparent, expected_apparent),
    ]:
        expected_pos = jnp.asarray(expected[0], dtype=float)
        expected_vel = jnp.asarray(expected[1], dtype=float)
        pos_diff = jnp.max(jnp.abs(actual.pos - expected_pos))
        vel_diff = jnp.max(jnp.abs(actual.vel - expected_vel))

        print(
            "[ephemeris.vector.horizons] "
            f"label={label:<25} "
            f"target={target_command:<11} "
            f"name={target_name} "
            f"layer={layer_name:<11} "
            f"pos_max_abs_diff={float(pos_diff):+.12e} au "
            f"vel_max_abs_diff={float(vel_diff):+.12e} au/day"
        )

        assert_allclose(actual.pos, expected_pos, atol=2.0e-10, rtol=0.0)
        assert_allclose(actual.vel, expected_vel, atol=1.0e-10, rtol=0.0)


def test_vector_table_shapes(generator, ground_site):
    t_obs = Time.from_tt_jd(EPOCH_TDB_JD, 0.0)

    table = generator.vector_table(t_obs, ground_site)
    geom_dist = jnp.linalg.norm(table.geometric.pos)
    astro_dist = jnp.linalg.norm(table.astrometric.pos)
    app_dist = jnp.linalg.norm(table.apparent.pos)

    print(
        "[ephemeris.vector] "
        f"shape={table.shape} "
        f"light_time={float(table.light_time):+.12e} day "
        f"geom_dist={float(geom_dist):+.12e} au "
        f"astro_dist={float(astro_dist):+.12e} au "
        f"app_dist={float(app_dist):+.12e} au"
    )

    assert isinstance(table, VectorTable)
    assert table.shape == ()
    assert table.geometric.shape == ()
    assert table.astrometric.shape == ()
    assert table.apparent.shape == ()
    assert table.geometric.pos.shape == (3,)
    assert table.astrometric.pos.shape == (3,)
    assert table.apparent.pos.shape == (3,)
    assert jnp.isfinite(table.light_time)
    assert float(table.light_time) > 0.0
    assert float(table.light_time - table.astrometric.lt) >= 0.0
    assert_allclose(table.astrometric.lt, table.light_time, atol=1.0e-8, rtol=0.0)
