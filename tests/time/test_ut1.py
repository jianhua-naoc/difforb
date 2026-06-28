import pytest
import jax
import jax.numpy as jnp

from difforb.core.constants import DAY_S
from difforb.core.eop import load_default_eop_file
from difforb.core.time.ut1 import _historical_delta_t_single, tt_to_ut1_single, ut1_to_tt_single, tt_to_ut1, ut1_to_tt
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)

eop = load_default_eop_file()
HISTORICAL_TT_CASES = [
    (1355807.0, 0.5, "-1000-01-01"),
    (1721057.0, 0.5, "0000-01-01"),
    (2104570.0, 0.5, "1050-01-01"),
    (2323710.0, 0.5, "1650-01-01"),
    (2360234.0, 0.5, "1750-01-01"),
    (2389453.0, 0.5, "1830-01-01"),
    (2397488.0, 0.0, "1852.0 Julian year"),
    (2407715.0, 0.5, "1880-01-01"),
    (2418672.0, 0.5, "1910-01-01"),
    (2425977.0, 0.5, "1930-01-01"),
    (2433282.0, 0.5, "1950-01-01"),
    (2436934.0, 0.5, "1960-01-01"),
]
MODERN_EOP_TT_CASES = [
    (eop.tt_jds[10], 0.0, "Modern EOP sample 1"),
    (eop.tt_jds[100], 0.0, "Modern EOP sample 2"),
]
UT1_BOUNDARY_TT_CASES = [
    (2437665.0, 0.499999999, "Before EOP boundary"),
    (2437665.0, 0.5, "At EOP boundary"),
]


@pytest.mark.parametrize(
    ("tt_jd1", "tt_jd2", "expected_ttdut1", "label"),
    [
        (1355807.0, 0.5, 25616.402831492990, "-1000-01-01"),
        (1721057.0, 0.5, 10440.945611383888, "0000-01-01"),
        (2104570.0, 0.5, 1418.767331803928, "1050-01-01"),
        (2323710.0, 0.5, 43.944011205377, "1650-01-01"),
        (2360234.0, 0.5, 16.877075742830, "1750-01-01"),
        (2389453.0, 0.5, 10.802352582072, "1830-01-01"),
        (2397488.0, 0.0, 9.980048000000, "1852.0 Julian year"),
        (2407715.0, 0.5, -3.210518263749, "1880-01-01"),
        (2418672.0, 0.5, 11.142000000000, "1910-01-01"),
        (2425977.0, 0.5, 24.418000000000, "1930-01-01"),
        (2433282.0, 0.5, 28.932000000000, "1950-01-01"),
        (2436934.0, 0.5, 33.072097903624, "1960-01-01"),
    ]
)
def test_tt_to_ut1_single_delta_t_branch(tt_jd1, tt_jd2, expected_ttdut1, label):
    actual_ut1_jd1, actual_ut1_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)
    expected_ut1_jd1, expected_ut1_jd2 = tt_jd1, tt_jd2 - expected_ttdut1 / DAY_S
    print(
        "[tt_to_ut1_single] "
        f"label={label:<22} "
        f"diff=({float(actual_ut1_jd1 - expected_ut1_jd1):+.12e}, {float(actual_ut1_jd2 - expected_ut1_jd2):+.12e}) s "
    )
    assert_allclose(actual_ut1_jd1, expected_ut1_jd1, atol=0., rtol=0.)
    assert_allclose(actual_ut1_jd2, expected_ut1_jd2, atol=1e-15, rtol=0.)


@pytest.mark.parametrize(
    ("tt_jd1", "tt_jd2", "expected_ttdut1", "label"),
    [
        (1458065.0, 0.0, 20371.848000000002, "Julian year -720"),
        (1501895.0, 0.0, 18468.489849417612, "Julian year -600"),
        (1684520.0, 0.0, 11557.668000000000, "Julian year -100"),
        (1721045.0, 0.0, 10441.312576000000, "Julian year 0"),
        (2104557.0, 0.5, 1418.918814814815, "Julian year 1050"),
        (2323707.0, 0.5, 43.952000000000, "Julian year 1650"),
        (2360232.0, 0.5, 16.875857421875, "Julian year 1750"),
        (2389452.0, 0.5, 10.804000000000, "Julian year 1830"),
        (2397488.0, 0.0, 9.980048000000, "Julian year 1852"),
        (2407715.0, 0.0, -3.210000000000, "Julian year 1880"),
        (2418672.0, 0.5, 11.142000000000, "Julian year 1910"),
        (2425977.0, 0.5, 24.418000000000, "Julian year 1930"),
        (2433282.0, 0.5, 28.932000000000, "Julian year 1950"),
        (2436935.0, 0.0, 33.072555555556, "Julian year 1960"),
        (2458302.0, 0.125, 69.087865740741, "Julian year 2018.5"),
        (2458484.0, 0.75, 69.240000000000, "Julian year 2019"),
    ]
)
def test_historical_delta_t_single_addendum_table(tt_jd1, tt_jd2, expected_ttdut1, label):
    actual_ttdut1 = _historical_delta_t_single(tt_jd1, tt_jd2)
    print(
        "[historical_delta_t_single] "
        f"label={label:<22} "
        f"diff={float(actual_ttdut1 - expected_ttdut1):+.12e} s "
    )
    assert_allclose(actual_ttdut1, expected_ttdut1, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(("tt_jd1", "tt_jd2", "label"), MODERN_EOP_TT_CASES)
def test_tt_to_ut1_single_eop_branch(tt_jd1, tt_jd2, label):
    actual_ut1_jd1, actual_ut1_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)
    expected_ut1_jd1 = tt_jd1
    expected_ut1_jd2 = tt_jd2 + eop.ut1dtt(tt_jd1, tt_jd2) / DAY_S

    assert_allclose(actual_ut1_jd1, expected_ut1_jd1, atol=0.0, rtol=0.0)
    assert_allclose(actual_ut1_jd2, expected_ut1_jd2, atol=0.0, rtol=0.0)


def test_tt_to_ut1_single_branch_boundary():
    boundary_tt_jd = float(eop.final_date_range[0])

    boundary_tt_jd1 = float(int(boundary_tt_jd))
    boundary_tt_jd2 = boundary_tt_jd - boundary_tt_jd1
    before_tt_jd1 = boundary_tt_jd1
    before_tt_jd2 = boundary_tt_jd2 - 1e-9

    actual_boundary_ut1_jd1, actual_boundary_ut1_jd2 = tt_to_ut1_single(boundary_tt_jd1, boundary_tt_jd2, eop)
    expected_boundary_ut1_jd1 = boundary_tt_jd1
    expected_boundary_ut1_jd2 = boundary_tt_jd2 + eop.ut1dtt(boundary_tt_jd1, boundary_tt_jd2) / DAY_S

    actual_before_ut1_jd1, actual_before_ut1_jd2 = tt_to_ut1_single(before_tt_jd1, before_tt_jd2, eop)
    expected_before_ut1_jd1 = before_tt_jd1
    expected_before_ut1_jd2 = before_tt_jd2 - _historical_delta_t_single(before_tt_jd1, before_tt_jd2) / DAY_S

    assert_allclose(actual_boundary_ut1_jd1, expected_boundary_ut1_jd1, atol=0.0, rtol=0.0)
    assert_allclose(actual_boundary_ut1_jd2, expected_boundary_ut1_jd2, atol=0.0, rtol=0.0)

    assert_allclose(actual_before_ut1_jd1, expected_before_ut1_jd1, atol=0.0, rtol=0.0)
    assert_allclose(actual_before_ut1_jd2, expected_before_ut1_jd2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("tt_jd1", "tt_jd2", "label"),
    HISTORICAL_TT_CASES
    + UT1_BOUNDARY_TT_CASES
    + MODERN_EOP_TT_CASES
)
def test_tt_ut1_single_roundtrip(tt_jd1, tt_jd2, label):
    ut1_jd1, ut1_jd2 = tt_to_ut1_single(tt_jd1, tt_jd2, eop)
    actual_tt_jd1, actual_tt_jd2 = ut1_to_tt_single(ut1_jd1, ut1_jd2, eop)

    actual_tt_jd = actual_tt_jd1 + actual_tt_jd2
    expected_tt_jd = tt_jd1 + tt_jd2

    print(
        "[tt_ut1_roundtrip] "
        f"label={label:<22} "
        f"diff={float((actual_tt_jd - expected_tt_jd) * DAY_S):+.12e} s "
    )

    assert_allclose(actual_tt_jd, expected_tt_jd, atol=1e-15, rtol=0.0)


def test_tt_to_ut1_shapes():
    tt_jd1 = jnp.array([2436934.0, eop.tt_jds[10]])
    tt_jd2 = jnp.array([0.5, 0.0])

    ut1_jd1, ut1_jd2 = tt_to_ut1(tt_jd1, tt_jd2, eop)

    assert ut1_jd1.shape == (2,)
    assert ut1_jd2.shape == (2,)


def test_ut1_to_tt_shapes():
    tt_jd1 = jnp.array([2436934.0, eop.tt_jds[10]])
    tt_jd2 = jnp.array([0.5, 0.0])
    ut1_jd1, ut1_jd2 = tt_to_ut1(tt_jd1, tt_jd2, eop)

    recovered_tt_jd1, recovered_tt_jd2 = ut1_to_tt(ut1_jd1, ut1_jd2, eop)

    assert recovered_tt_jd1.shape == (2,)
    assert recovered_tt_jd2.shape == (2,)
