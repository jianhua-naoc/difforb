import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from difforb.body.ephbody import EphemerisBody
from difforb.body.smallbody import SmallBody
from difforb.core.element import KepElement
from difforb.core.time.timescale import Time
from difforb.dynamics.dynamic_system import DynamicSystem
from difforb.ephemeris.core import OpticalTable
from difforb.ephemeris.generator import EphemerisGenerator
from difforb.integrator.integrator import NumericalIntegrator
from tests.ephemeris.generator_reference import (
    EPOCH_TDB_JD,
    HORIZONS_GENERATOR_OBSERVER_CASES,
    HORIZONS_OBSERVER_UTC_JD,
    HORIZONS_TARGET_TDB_JD,
)
from tests.assertions import assert_allclose


@pytest.mark.parametrize(
    ("label", "target_command", "target_name", "initial_elements", "expected_observer"),
    HORIZONS_GENERATOR_OBSERVER_CASES,
)
def test_optical_table_against_horizons(
        default_ephemeris,
        ground_site,
        label,
        target_command,
        target_name,
        initial_elements,
        expected_observer,
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
    table = EphemerisGenerator(target).optical_table(
        Time.from_utc_jd(HORIZONS_OBSERVER_UTC_JD, 0.0),
        ground_site,
    )
    expected_ra, expected_dec, expected_az, expected_el, expected_delta = expected_observer
    ra_diff = table.astrometric_ra - expected_ra
    dec_diff = table.astrometric_dec - expected_dec
    az_diff = table.azimuth - expected_az
    el_diff = table.elevation - expected_el
    delta_diff = table.delta - expected_delta

    print(
        "[ephemeris.optical.horizons] "
        f"label={label:<25} "
        f"target={target_command:<11} "
        f"name={target_name} "
        f"ra_diff={float(ra_diff):+.12e} deg "
        f"dec_diff={float(dec_diff):+.12e} deg "
        f"delta_diff={float(delta_diff):+.12e} au"
    )

    assert_allclose(table.astrometric_ra, expected_ra, atol=5.0e-5, rtol=0.0)
    assert_allclose(table.astrometric_dec, expected_dec, atol=5.0e-5, rtol=0.0)
    assert_allclose(table.azimuth, expected_az, atol=3.0e-4, rtol=0.0)
    assert_allclose(table.elevation, expected_el, atol=3.0e-4, rtol=0.0)
    assert_allclose(table.delta, expected_delta, atol=1.0e-10, rtol=0.0)


def test_optical_table_basic_contract(generator, ground_site):
    t_obs = Time.from_tt_jd(EPOCH_TDB_JD, 0.0)

    table = generator.optical_table(t_obs, ground_site)

    print(
        "[ephemeris.optical] "
        f"shape={table.shape} "
        f"astrometric_ra={float(table.astrometric_ra):+.12e} deg "
        f"apparent_ra={float(table.apparent_ra):+.12e} deg "
        f"delta={float(table.delta):+.12e} au"
    )

    assert isinstance(table, OpticalTable)
    assert table.shape == ()
    assert 0.0 <= float(table.astrometric_ra) < 360.0
    assert -90.0 <= float(table.astrometric_dec) <= 90.0
    assert 0.0 <= float(table.apparent_ra) < 360.0
    assert -90.0 <= float(table.apparent_dec) <= 90.0
    assert 0.0 <= float(table.azimuth) < 360.0
    assert -90.0 <= float(table.elevation) <= 90.0
    assert float(table.delta) > 0.0
    assert float(table.r) > 0.0
    assert 0.0 <= float(table.phase_angle) <= 180.0
    assert 0.0 <= float(table.elongation) <= 180.0
    assert bool(jnp.isnan(table.mag))
