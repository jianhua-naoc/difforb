import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

import difforb.spk as spk
from difforb.body.site import Site
from difforb.core.time.timescale import Time
from difforb.ephemeris.generator import EphemerisGenerator
from tests.ephemeris.generator_reference import EPOCH_TDB_JD


def test_generator_requires_default_spk(target):
    spk.clear_default_ephemeris()

    with pytest.raises(RuntimeError, match="No default ephemeris set"):
        EphemerisGenerator(target)


def test_generator_pointwise_batch_shapes(generator, ground_site):
    t_obs = Time.from_tt_jd(
        jnp.asarray([EPOCH_TDB_JD, EPOCH_TDB_JD], dtype=float),
        jnp.asarray([0.0, 0.25], dtype=float),
    )

    vector_table = generator.vector_table(t_obs, ground_site)
    optical_table = generator.optical_table(t_obs, ground_site)
    elements = generator.elements_table(t_obs.tdb())

    print(
        "[ephemeris.batch] "
        f"vector_shape={vector_table.shape} "
        f"optical_shape={optical_table.shape} "
        f"elements_shape={elements.shape}"
    )

    assert vector_table.shape == (2,)
    assert vector_table.geometric.pos.shape == (2, 3)
    assert vector_table.light_time.shape == (2,)
    assert optical_table.shape == (2,)
    assert optical_table.astrometric_ra.shape == (2,)
    assert elements.shape == (2,)
    assert elements.array.shape == (2, 6)


def test_generator_grid_shapes(generator):
    sites = Site.from_code(["568", "G96"]).require_ground()
    t_obs = Time.from_tt_jd(
        jnp.asarray([EPOCH_TDB_JD, EPOCH_TDB_JD], dtype=float),
        jnp.asarray([0.0, 0.25], dtype=float),
    )

    vector_table = generator.vector_table(t_obs, sites, grid=True)
    optical_table = generator.optical_table(t_obs, sites, grid=True)

    print(
        "[ephemeris.grid] "
        f"vector_shape={vector_table.shape} "
        f"optical_shape={optical_table.shape}"
    )

    assert vector_table.shape == (2, 2)
    assert vector_table.geometric.pos.shape == (2, 2, 3)
    assert vector_table.light_time.shape == (2, 2)
    assert optical_table.shape == (2, 2)
    assert optical_table.astrometric_ra.shape == (2, 2)


def test_generator_rejects_invalid_time_views(generator, ground_site):
    tdb = Time.from_tdb_jd(EPOCH_TDB_JD, 0.0).tdb()
    tt = Time.from_tdb_jd(EPOCH_TDB_JD, 0.0).tt

    with pytest.raises(TypeError, match="t_obs"):
        generator.vector_table(tdb, ground_site)

    with pytest.raises(TypeError, match="tdb"):
        generator.elements_table(tt)
