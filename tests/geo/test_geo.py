import erfa
import jax
import jax.numpy as jnp
import pytest

from difforb.core.constants import AU_M, DAY_S
from difforb.core.eop import load_default_eop_file
from difforb.core.geo import WGS84, itrs_to_gcrs_single
from difforb.core.state.frame import GCRS
from difforb.core.time.timescale import Time
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)

eop = load_default_eop_file()


@pytest.mark.parametrize(
    ("xp_arcsec", "yp_arcsec", "use_tio_locator", "label"),
    [
        (0.0, 0.0, False, "no polar motion"),
        (0.12, -0.25, True, "with polar motion"),
    ],
)
def test_itrs_to_gcrs_single_against_erfa_pvtob(xp_arcsec, yp_arcsec, use_tio_locator, label):
    tt_jd1, tt_jd2 = 2453005.25, 0.0
    ut1_jd1, ut1_jd2 = 2453005.0, 0.24925712963
    lon = float(jnp.deg2rad(203.7441))
    lat = float(jnp.deg2rad(20.7071888))
    height = 3076.38
    xp = float(jnp.deg2rad(xp_arcsec / 3600.0))
    yp = float(jnp.deg2rad(yp_arcsec / 3600.0))
    sp = erfa.sp00(tt_jd1, tt_jd2) if use_tio_locator else 0.0
    era = erfa.era00(ut1_jd1, ut1_jd2)

    itrs_pos = jnp.asarray(erfa.gd2gc(1, lon, lat, height), dtype=float)
    # ERFA ``pom00`` returns TIRS -> ITRS; ``itrs_to_gcrs_single`` needs ITRS -> TIRS.
    W_T = jnp.asarray(erfa.pom00(xp, yp, sp), dtype=float).T
    x, y, cio_s = erfa.xys06a(tt_jd1, tt_jd2)
    C_T = jnp.asarray(erfa.c2ixys(x, y, cio_s), dtype=float).T

    delta_days = float(1.0e-2 / DAY_S)
    x_before, y_before, cio_s_before = erfa.xys06a(tt_jd1, tt_jd2 - delta_days)
    x_after, y_after, cio_s_after = erfa.xys06a(tt_jd1, tt_jd2 + delta_days)
    C_T_before = jnp.asarray(erfa.c2ixys(x_before, y_before, cio_s_before), dtype=float).T
    C_T_after = jnp.asarray(erfa.c2ixys(x_after, y_after, cio_s_after), dtype=float).T
    C_T_deriv = (C_T_after - C_T_before) / (2.0 * delta_days)

    actual_pos, actual_vel = itrs_to_gcrs_single(itrs_pos, W_T, era, C_T, C_T_deriv)

    erfa_pv = erfa.pvtob(lon, lat, height, xp, yp, sp, era)
    cirs_pos = jnp.asarray(erfa_pv["p"], dtype=float) / AU_M
    cirs_vel = jnp.asarray(erfa_pv["v"], dtype=float) * DAY_S / AU_M
    expected_pos = C_T @ cirs_pos
    expected_vel = C_T @ cirs_vel + C_T_deriv @ cirs_pos

    pos_diff = jnp.max(jnp.abs(actual_pos - expected_pos))
    vel_diff = jnp.max(jnp.abs(actual_vel - expected_vel))
    print(
        "[itrs_to_gcrs_single.erfa_pvtob] "
        f"label={label:<17} "
        f"pos_max_abs_diff={float(pos_diff):+.12e} au "
        f"vel_max_abs_diff={float(vel_diff):+.12e} au/day"
    )

    assert_allclose(
        actual_pos,
        expected_pos,
        atol=1.0e-18,
        rtol=0.0,
        msg=f"ITRS-to-GCRS position mismatch for {label}",
    )
    assert_allclose(
        actual_vel,
        expected_vel,
        atol=1.0e-18,
        rtol=0.0,
        msg=f"ITRS-to-GCRS velocity mismatch for {label}",
    )


def test_wgs84_from_geodetic_against_erfa_gd2gc():
    lon_deg = jnp.asarray([0.0, 203.7441, -70.8065], dtype=float)
    lat_deg = jnp.asarray([0.0, 20.7071888, -30.1690], dtype=float)
    height = jnp.asarray([0.0, 3076.38, 2380.0], dtype=float)

    actual = WGS84.from_geodetic(lon_deg, lat_deg, height)
    expected = jnp.stack(
        [
            jnp.asarray(
                erfa.gd2gc(1, float(jnp.deg2rad(lon)), float(jnp.deg2rad(lat)), float(alt)),
                dtype=float,
            )
            for lon, lat, alt in zip(lon_deg, lat_deg, height)
        ],
        axis=0,
    )

    max_abs_diff = jnp.max(jnp.abs(actual.pos - expected))
    print(
        "[wgs84.from_geodetic.erfa_gd2gc] "
        f"pos_max_abs_diff={float(max_abs_diff):+.12e} m"
    )

    assert actual.shape == (3,)
    assert actual.pos.shape == (3, 3)
    assert_allclose(actual.pos, expected, atol=1.0e-9, rtol=0.0)


def test_wgs84_from_geocentric_matches_mpc_parallax_definition():
    lon_deg = jnp.asarray([0.0, 90.0, 203.74409], dtype=float)
    parallax_const1 = jnp.asarray([1.0, 0.936241, 0.815913], dtype=float)
    parallax_const2 = jnp.asarray([0.0, 0.351543, 0.576510], dtype=float)

    actual = WGS84.from_geocentric(lon_deg, parallax_const1, parallax_const2)
    lon = jnp.deg2rad(lon_deg)
    expected = jnp.stack(
        [
            WGS84.a * parallax_const1 * jnp.cos(lon),
            WGS84.a * parallax_const1 * jnp.sin(lon),
            WGS84.a * parallax_const2,
        ],
        axis=-1,
    )

    max_abs_diff = jnp.max(jnp.abs(actual.pos - expected))
    print(
        "[wgs84.from_geocentric.mpc_definition] "
        f"pos_max_abs_diff={float(max_abs_diff):+.12e} m"
    )

    assert actual.shape == (3,)
    assert actual.pos.shape == (3, 3)
    assert_allclose(actual.lon, lon, atol=0.0, rtol=0.0)
    assert_allclose(actual.pos, expected, atol=0.0, rtol=0.0)


def test_itrs_state_scalar_uses_tdb_epoch():
    site = WGS84.from_geodetic(203.7441, 20.7071888, 3076.38)
    time = Time.from_tt_jd(2451545.0, 0.0, eop=eop)

    actual = site.state(time)
    expected_tdb = time.tdb(site)
    epoch_diff_s = (actual.tdb.jd - expected_tdb.jd) * DAY_S
    print(
        "[itrs.state.scalar_tdb_epoch] "
        f"epoch_diff={float(epoch_diff_s):+.12e} s"
    )

    assert actual.frame == GCRS
    assert actual.shape == ()
    assert actual.pos.shape == (3,)
    assert actual.vel.shape == (3,)
    assert actual.tdb.shape == ()
    assert_allclose(actual.tdb.jd, expected_tdb.jd, atol=0.0, rtol=0.0)


def test_itrs_state_broadcast_and_grid_shapes():
    sites = WGS84.from_geodetic(
        lon=jnp.asarray([0.0, 203.7441], dtype=float),
        lat=jnp.asarray([0.0, 20.7071888], dtype=float),
        alt=jnp.asarray([0.0, 3076.38], dtype=float),
    )
    times = Time.from_tt_jd(
        tt_jd1=jnp.asarray([2451545.0, 2451727.0], dtype=float),
        tt_jd2=jnp.asarray([0.0, 0.625], dtype=float),
        eop=eop,
    )

    broadcast_state = sites.state(times)
    grid_state = sites.state(times, grid=True)

    assert broadcast_state.frame == GCRS
    assert broadcast_state.shape == (2,)
    assert broadcast_state.pos.shape == (2, 3)
    assert broadcast_state.vel.shape == (2, 3)

    assert grid_state.frame == GCRS
    assert grid_state.shape == (2, 2)
    assert grid_state.pos.shape == (2, 2, 3)
    assert grid_state.vel.shape == (2, 2, 3)
    assert grid_state.tdb.shape == (2, 2)
