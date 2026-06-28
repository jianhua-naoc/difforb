import jax
import jax.numpy as jnp
import pytest

from difforb.core.eop.container import EarthOrientationData
from difforb.core.eop.loaders import parse_iers_eopc04
from tests.assertions import assert_allclose


jax.config.update("jax_enable_x64", True)


@pytest.fixture
def manual_eop():
    return EarthOrientationData(
        mjds=jnp.asarray([37665.0, 37666.0, 37667.0, 37668.0, 37669.0, 37670.0]),
        xpoles=jnp.asarray([-0.012700, -0.015900, -0.019000, -0.021999, -0.024799, -0.027599]),
        ypoles=jnp.asarray([0.213000, 0.214100, 0.215200, 0.216301, 0.217301, 0.218301]),
        ut1dutcs=jnp.asarray([0.0326338, 0.0320547, 0.0315526, 0.0311435, 0.0308154, 0.0305353]),
        dpsis=jnp.asarray([0.100000, 0.110000, 0.120000, 0.130000, 0.140000, 0.150000]),
        depss=jnp.asarray([-0.200000, -0.190000, -0.180000, -0.170000, -0.160000, -0.150000]),
        final_date_range=jnp.asarray([2437665.5, 2437670.5]),
    )


def test_manual_container_exact_sample_queries_match_table_values(manual_eop):
    idx = 2
    tt_jd = manual_eop.tt_jds[idx]

    assert_allclose(manual_eop.xpole(tt_jd, 0.0), manual_eop.xpoles[idx], atol=0.0, rtol=0.0)
    assert_allclose(manual_eop.ypole(tt_jd, 0.0), manual_eop.ypoles[idx], atol=0.0, rtol=0.0)
    assert_allclose(manual_eop.ut1dtt(tt_jd, 0.0), manual_eop.ut1dtts[idx], atol=0.0, rtol=0.0)
    assert_allclose(manual_eop.ut1dutc(tt_jd, 0.0), manual_eop.ut1dutcs[idx], atol=1.0e-12, rtol=0.0)
    assert_allclose(manual_eop.cor_delta_longitude(tt_jd, 0.0), manual_eop.dpsis[idx], atol=0.0, rtol=0.0)
    assert_allclose(manual_eop.cor_delta_obliquity(tt_jd, 0.0), manual_eop.depss[idx], atol=0.0, rtol=0.0)


def test_manual_container_ut1dtt_table_matches_internal_samples(manual_eop):
    actual = manual_eop.ut1dtt(manual_eop.tt_jds, jnp.zeros_like(manual_eop.tt_jds))

    assert_allclose(actual, manual_eop.ut1dtts, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "query_shape",
    [
        (3,),
        (2, 2),
    ],
)
def test_manual_container_query_methods_preserve_batch_shape(manual_eop, query_shape):
    flat = manual_eop.tt_jds[: jnp.prod(jnp.asarray(query_shape))]
    tt_jd1 = jnp.reshape(flat, query_shape)
    tt_jd2 = jnp.zeros(query_shape)

    assert manual_eop.xpole(tt_jd1, tt_jd2).shape == query_shape
    assert manual_eop.ypole(tt_jd1, tt_jd2).shape == query_shape
    assert manual_eop.ut1dtt(tt_jd1, tt_jd2).shape == query_shape
    assert manual_eop.ut1dutc(tt_jd1, tt_jd2).shape == query_shape
    assert manual_eop.cor_delta_longitude(tt_jd1, tt_jd2).shape == query_shape
    assert manual_eop.cor_delta_obliquity(tt_jd1, tt_jd2).shape == query_shape


def test_manual_container_batch_queries_match_selected_samples(manual_eop):
    indices = jnp.asarray([[0, 2], [3, 5]])
    tt_jd1 = manual_eop.tt_jds[indices]
    tt_jd2 = jnp.zeros_like(tt_jd1)

    assert_allclose(manual_eop.xpole(tt_jd1, tt_jd2), manual_eop.xpoles[indices], atol=0.0, rtol=0.0)
    assert_allclose(manual_eop.ypole(tt_jd1, tt_jd2), manual_eop.ypoles[indices], atol=0.0, rtol=0.0)
    assert_allclose(manual_eop.ut1dtt(tt_jd1, tt_jd2), manual_eop.ut1dtts[indices], atol=0.0, rtol=0.0)
    assert_allclose(manual_eop.ut1dutc(tt_jd1, tt_jd2), manual_eop.ut1dutcs[indices], atol=1.0e-12, rtol=0.0)
    assert_allclose(manual_eop.cor_delta_longitude(tt_jd1, tt_jd2), manual_eop.dpsis[indices], atol=0.0, rtol=0.0)
    assert_allclose(manual_eop.cor_delta_obliquity(tt_jd1, tt_jd2), manual_eop.depss[indices], atol=0.0, rtol=0.0)


def test_manual_container_pre_coverage_pole_and_nutation_queries_return_zero(manual_eop):
    tt_jd = manual_eop.tt_jds[0] - 10.0

    assert_allclose(manual_eop.xpole(tt_jd, 0.0), 0.0, atol=0.0, rtol=0.0)
    assert_allclose(manual_eop.ypole(tt_jd, 0.0), 0.0, atol=0.0, rtol=0.0)
    assert_allclose(manual_eop.cor_delta_longitude(tt_jd, 0.0), 0.0, atol=0.0, rtol=0.0)
    assert_allclose(manual_eop.cor_delta_obliquity(tt_jd, 0.0), 0.0, atol=0.0, rtol=0.0)


def test_real_sample_container_exact_sample_queries_match_table_values(eopc04_sample_path):
    eop = parse_iers_eopc04(str(eopc04_sample_path))
    idx = 4
    tt_jd = eop.tt_jds[idx]

    assert_allclose(eop.xpole(tt_jd, 0.0), eop.xpoles[idx], atol=0.0, rtol=0.0)
    assert_allclose(eop.ypole(tt_jd, 0.0), eop.ypoles[idx], atol=0.0, rtol=0.0)
    assert_allclose(eop.ut1dtt(tt_jd, 0.0), eop.ut1dtts[idx], atol=0.0, rtol=0.0)
    assert_allclose(eop.ut1dutc(tt_jd, 0.0), eop.ut1dutcs[idx], atol=1.0e-12, rtol=0.0)
    assert_allclose(eop.cor_delta_longitude(tt_jd, 0.0), eop.dpsis[idx], atol=0.0, rtol=0.0)
    assert_allclose(eop.cor_delta_obliquity(tt_jd, 0.0), eop.depss[idx], atol=0.0, rtol=0.0)
