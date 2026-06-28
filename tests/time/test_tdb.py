import erfa
import pytest
import jax
from jax import numpy as jnp

from difforb.core.constants import DAY_S
from difforb.core.eop import load_default_eop_file
from difforb.core.time.tdb import tt_to_tdb_single, tdb_to_tt_single, tt_to_tdb, tdb_to_tt
from difforb.core.time.ut1 import tt_to_ut1_single
from difforb.core.time.utils import ut1_fraction, renormalize_split_jd
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)

eop = load_default_eop_file()

TT_CASES = [
    (2451545.0, 0.0, "J2000.0"),
    (2451727.0, 0.625, "Half year after J2000.0"),
    (1721057.0, 0.5, "1 B.C. 01-01.0"),
    (2816787.0, 0.5, "3000-01-01.0"),
]
LOCATION_CASES = [
    (0.0, 0.0, 0.0, "Geocenter"),
    (0.0, 6378.1, 0.0, "Equator"),
    (0.0, 0.0, 6356.7, "North Pole"),
    (1.2, 3800.0, 4200.0, "Nominal"),
]


@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TT_CASES)
@pytest.mark.parametrize("lon, u, v, loc_label", LOCATION_CASES)
def test_tt_to_tdb_single(tt_jd1, tt_jd2, time_label, lon, u, v, loc_label):
    lon_rad = jnp.deg2rad(lon)
    ut1_jd1, ut1_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)
    ut1_frac = ut1_fraction(ut1_jd1, ut1_jd2)
    actual_tdb_jd1, actual_tdb_jd2 = tt_to_tdb_single(lon_rad, u, v, tt_jd1, tt_jd2, ut1_frac)

    dtdb = erfa.dtdb(tt_jd1, tt_jd2, ut1_frac, lon_rad, u, v)
    expected_tdb_jd1, expected_tdb_jd2 = erfa.tttdb(tt_jd1, tt_jd2, dtdb)

    print(
        "[tt_to_tdb_single] "
        f"time_label={time_label:<22} "
        f"loc_label={loc_label:<10} "
        f"diff=({float(actual_tdb_jd1 - expected_tdb_jd1):+.12e}, {float(actual_tdb_jd2 - expected_tdb_jd2):+.12e}) s "
    )

    assert_allclose(actual_tdb_jd1, expected_tdb_jd1, atol=0., rtol=0.)
    assert_allclose(actual_tdb_jd2, expected_tdb_jd2, atol=0., rtol=0.)


@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TT_CASES)
@pytest.mark.parametrize("lon, u, v, loc_label", LOCATION_CASES)
def test_tdb_to_tt_single_against_erfa_inverse(tt_jd1, tt_jd2, time_label, lon, u, v, loc_label):
    lon_rad = jnp.deg2rad(lon)

    ut1_jd1, ut1_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)
    ut1_frac = ut1_fraction(ut1_jd1, ut1_jd2)

    dtdb = erfa.dtdb(tt_jd1, tt_jd2, ut1_frac, lon_rad, u, v)
    tdb_jd1, tdb_jd2 = erfa.tttdb(tt_jd1, tt_jd2, dtdb)

    actual_tt_jd1, actual_tt_jd2 = tdb_to_tt_single(lon_rad, u, v, tdb_jd1, tdb_jd2, eop)

    actual_tt_jd = actual_tt_jd1 + actual_tt_jd2
    expected_tt_jd = tt_jd1 + tt_jd2

    print(
        "[tdb_to_tt_single] "
        f"time_label={time_label:<22} "
        f"loc_label={loc_label:<10} "
        f"diff={float((actual_tt_jd - expected_tt_jd) * DAY_S):+.12e} s "
    )

    assert_allclose(actual_tt_jd, expected_tt_jd, atol=0., rtol=0.0)


@pytest.mark.parametrize("tt_jd1, tt_jd2, time_label", TT_CASES)
@pytest.mark.parametrize("lon, u, v, loc_label", LOCATION_CASES)
def test_tdb_tt_roundtrip(tt_jd1, tt_jd2, time_label, lon, u, v, loc_label):
    lon_rad = jnp.deg2rad(lon)
    ut1_jd1, ut1_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)
    ut1_frac = ut1_fraction(ut1_jd1, ut1_jd2)

    tdb_jd1, tdb_jd2 = tt_to_tdb_single(lon_rad, u, v, tt_jd1, tt_jd2, ut1_frac)
    actual_tt_jd1, actual_tt_jd2 = tdb_to_tt_single(lon_rad, u, v, tdb_jd1, tdb_jd2, eop)
    actual_tt_jd1, actual_tt_jd2 = renormalize_split_jd(actual_tt_jd1, actual_tt_jd2)

    actual_tt_jd = actual_tt_jd1 + actual_tt_jd2
    expected_tt_jd = tt_jd1 + tt_jd2

    print(
        "[tdb_tt_roundtrip] "
        f"time_label={time_label:<22} "
        f"loc_label={loc_label:<10} "
        f"roundtrip diff={float((actual_tt_jd - expected_tt_jd) * DAY_S):+.12e} s "
    )

    assert_allclose(actual_tt_jd, expected_tt_jd, atol=0., rtol=0.)


def test_tt_to_tdb_shapes():
    lon = jnp.array([0.0, jnp.deg2rad(1.2)])
    u = jnp.array([6378.1, 3800.0])
    v = jnp.array([0.0, 4200.0])
    tt_jd1 = jnp.array([2451545.0, 2451727.0])
    tt_jd2 = jnp.array([0.0, 0.625])
    ut1_frac = jnp.array([0.99926124, 0.624259490])

    tdb_jd1, tdb_jd2 = tt_to_tdb(lon, u, v, tt_jd1, tt_jd2, ut1_frac)

    assert tdb_jd1.shape == (2,)
    assert tdb_jd2.shape == (2,)


def test_tt_to_tdb_shapes_grid():
    lon = jnp.array([0.0, jnp.deg2rad(1.2)])
    u = jnp.array([6378.1, 3800.0])
    v = jnp.array([0.0, 4200.0])
    tt_jd1 = jnp.array([2451545.0, 2451727.0, 2816787.0])
    tt_jd2 = jnp.array([0.0, 0.625, 0.5])
    ut1_frac = jnp.array([0.99926124, 0.624259490, 0.5])

    tdb_jd1, tdb_jd2 = tt_to_tdb(lon, u, v, tt_jd1, tt_jd2, ut1_frac, grid=True)

    assert tdb_jd1.shape == (2, 3)
    assert tdb_jd2.shape == (2, 3)


def test_tdb_to_tt_shapes():
    lon = jnp.array([0.0, jnp.deg2rad(1.2)])
    u = jnp.array([6378.1, 3800.0])
    v = jnp.array([0.0, 4200.0])
    tdb_jd1 = jnp.array([2451545.0, 2451727.0])
    tdb_jd2 = jnp.array([-1.1396458541945606e-08, 0.6249999886173912])

    tt_jd1, tt_jd2 = tdb_to_tt(lon, u, v, tdb_jd1, tdb_jd2, eop)

    assert tt_jd1.shape == (2,)
    assert tt_jd2.shape == (2,)


def test_tdb_to_tt_shapes_grid():
    lon = jnp.array([0.0, jnp.deg2rad(1.2)])
    u = jnp.array([6378.1, 3800.0])
    v = jnp.array([0.0, 4200.0])
    tdb_jd1 = jnp.array([2451545.0, 2451727.0, 2816787.0])
    tdb_jd2 = jnp.array([-1.1396458541945606e-08, 0.6249999886173912, 0.4999999886173912])

    tt_jd1, tt_jd2 = tdb_to_tt(lon, u, v, tdb_jd1, tdb_jd2, eop, grid=True)

    assert tt_jd1.shape == (2, 3)
    assert tt_jd2.shape == (2, 3)
