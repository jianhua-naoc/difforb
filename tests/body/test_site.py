import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difforb.body.site import Site, format_site_key, parse_iau_obs_codes, parse_jpl_radar_obs_codes, parse_site_keys
from difforb.core.geo import WGS84
from difforb.core.state.frame import BCRS, GCRS
from difforb.core.state.state import State
from difforb.core.time.timescale import Time
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)


MPC_GROUND_SITE_CASES = [
    ("000", "Greenwich", 0.0, 0.62411, 0.77873),
    ("568", "Maunakea", 204.5278, 0.94171, 0.33725),
    ("G96", "Mt. Lemmon Survey", 249.21128, 0.845107, 0.533611),
]

JPL_RADAR_SITE_CASES = [
    ("-1", "Arecibo", 293.2473068, 18.3442199, 453.34),
    ("-14", "DSS-14", 243.1104618, 35.4259009, 1001.39),
    ("-35", "DSS-35", 148.9814558, -35.3957955, 694.90),
]


@pytest.mark.parametrize(
    ("code", "name", "expected_lon_deg", "expected_rho_cos_phi", "expected_rho_sin_phi"),
    MPC_GROUND_SITE_CASES,
)
def test_site_from_code_mpc(
        code,
        name,
        expected_lon_deg,
        expected_rho_cos_phi,
        expected_rho_sin_phi,
):
    site = Site.from_code(code)
    expected = WGS84.from_geocentric(expected_lon_deg, expected_rho_cos_phi, expected_rho_sin_phi)
    pos_diff = jnp.max(jnp.abs(site.ground_itrs.pos - expected.pos))

    print(
        "[site.from_code.mpc] "
        f"code={code:<3} "
        f"name={name:<16} "
        f"pos_max_abs_diff={float(pos_diff):+.12e} m"
    )

    assert site.shape == ()
    assert bool(site.is_fixed_ground)
    assert site.raw_keys == (code,)
    assert site.identity_keys == (code,)
    assert_allclose(site.ground_itrs.lon, expected.lon, atol=0.0, rtol=0.0)
    assert_allclose(site.ground_itrs.pos, expected.pos, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("code", "name", "expected_lon_deg", "expected_lat_deg", "expected_alt_m"),
    JPL_RADAR_SITE_CASES,
)
def test_site_from_code_radar(
        code,
        name,
        expected_lon_deg,
        expected_lat_deg,
        expected_alt_m,
):
    site = Site.from_code(code)
    expected = WGS84.from_geodetic(expected_lon_deg, expected_lat_deg, expected_alt_m)
    pos_diff = jnp.max(jnp.abs(site.ground_itrs.pos - expected.pos))

    print(
        "[site.from_code.radar] "
        f"code={code:<3} "
        f"name={name:<8} "
        f"pos_max_abs_diff={float(pos_diff):+.12e} m"
    )

    assert site.shape == ()
    assert bool(site.is_fixed_ground)
    assert_allclose(site.ground_itrs.lon, expected.lon, atol=0.0, rtol=0.0)
    assert_allclose(site.ground_itrs.pos, expected.pos, atol=0.0, rtol=0.0)


def test_site_from_code_batch():
    site = Site.from_code(["000", "568", "-14"])
    maunakea = Site.from_code("568")
    sliced = site[1]

    print(
        "[site.from_code.batch] "
        f"site_shape={site.shape} "
        f"sliced_shape={sliced.shape}"
    )

    assert site.shape == (3,)
    assert site.ground_itrs.pos.shape == (3, 3)
    assert sliced.shape == ()
    assert sliced.raw_keys == ("568",)
    assert_allclose(sliced.ground_itrs.lon, maunakea.ground_itrs.lon, atol=0.0, rtol=0.0)
    assert_allclose(sliced.ground_itrs.pos, maunakea.ground_itrs.pos, atol=0.0, rtol=0.0)


def test_site_from_code_empty_batch():
    site = Site.from_code(np.asarray([], dtype=str))

    assert site.shape == (0,)
    assert site.raw_keys == ()
    assert site.ground_itrs.pos.shape == (0, 3)
    assert site.gcrs_pos.shape == (0, 3)


def test_site_from_code_roving(monkeypatch):
    patched_codes = dict(Site._iau_obs_code)
    patched_codes["R01"] = (0.0, 0.0, 0.0, "TEMPORARY ROVING STATION", Site.TYPE_ROVING_GROUND)
    monkeypatch.setattr(Site, "_iau_obs_code", patched_codes)

    roving_payload = [10.0, 20.0, 30.0]
    roving_key = format_site_key("R01", Site.TYPE_ROVING_GROUND, roving_payload)
    roving_itrs = WGS84.from_geodetic(*roving_payload)
    site = Site.from_code(["000", roving_key])
    greenwich = Site.from_code("000")

    print(
        "[site.from_code.roving] "
        f"site_shape={site.shape} "
        f"roving_lon_deg={float(jnp.rad2deg(site.ground_itrs.lon[1])):+.12e}"
    )

    assert site.shape == (2,)
    assert site.identity_keys[1] == roving_key
    assert_allclose(site.ground_itrs.lon[0], greenwich.ground_itrs.lon, atol=0.0, rtol=0.0)
    assert_allclose(site.ground_itrs.pos[0], greenwich.ground_itrs.pos, atol=0.0, rtol=0.0)
    assert_allclose(site.ground_itrs.lon[1], roving_itrs.lon, atol=0.0, rtol=0.0)
    assert_allclose(site.ground_itrs.pos[1], roving_itrs.pos, atol=0.0, rtol=0.0)

    with pytest.raises(ValueError, match=r"R01 is a roving site and requires '@' coordinates"):
        Site.from_code("R01")


def test_site_from_code_rejects_roving_coordinates_for_fixed_ground_site():
    fixed_ground_key = format_site_key("568", Site.TYPE_ROVING_GROUND, [10.0, 20.0, 30.0])

    with pytest.raises(ValueError, match=r"568 is a fixed ground site and cannot use '@' roving coordinates"):
        Site.from_code(fixed_ground_key)


def test_site_state_shapes():
    site = Site.from_code(["000", "568"])
    time = Time.from_tt_jd(
        jnp.asarray([2460741.5, 2460742.5], dtype=float),
        jnp.asarray([0.0, 0.0], dtype=float),
    )

    state = site.state(time, frame=GCRS, grid=False)
    expected = site.ground_itrs.state(time, frame=GCRS, grid=False)
    grid_state = site.state(time, frame=GCRS, grid=True)
    expected_grid = site.ground_itrs.state(time, frame=GCRS, grid=True)

    print(
        "[site.state.shapes] "
        f"state_shape={state.shape} "
        f"grid_shape={grid_state.shape}"
    )

    assert state.frame == GCRS
    assert state.shape == (2,)
    assert state.pos.shape == (2, 3)
    assert state.vel.shape == (2, 3)
    assert_allclose(state.tdb.jd, expected.tdb.jd, atol=0.0, rtol=0.0)
    assert_allclose(state.pos, expected.pos, atol=0.0, rtol=0.0)
    assert_allclose(state.vel, expected.vel, atol=0.0, rtol=0.0)

    assert grid_state.frame == GCRS
    assert grid_state.shape == (2, 2)
    assert grid_state.pos.shape == (2, 2, 3)
    assert grid_state.vel.shape == (2, 2, 3)
    assert_allclose(grid_state.tdb.jd, expected_grid.tdb.jd, atol=0.0, rtol=0.0)
    assert_allclose(grid_state.pos, expected_grid.pos, atol=0.0, rtol=0.0)
    assert_allclose(grid_state.vel, expected_grid.vel, atol=0.0, rtol=0.0)


def test_site_from_geocentric():
    actual = Site.from_geocentric(204.5278, 0.94171, 0.33725)
    expected = WGS84.from_geocentric(204.5278, 0.94171, 0.33725)

    assert actual.shape == ()
    assert_allclose(actual.ground_itrs.lon, expected.lon, atol=0.0, rtol=0.0)
    assert_allclose(actual.ground_itrs.pos, expected.pos, atol=0.0, rtol=0.0)


def test_site_from_geodetic():
    actual = Site.from_geodetic(243.1104618, 35.4259009, 1001.39)
    expected = WGS84.from_geodetic(243.1104618, 35.4259009, 1001.39)

    assert actual.shape == ()
    assert_allclose(actual.ground_itrs.lon, expected.lon, atol=0.0, rtol=0.0)
    assert_allclose(actual.ground_itrs.pos, expected.pos, atol=0.0, rtol=0.0)


def test_site_from_gcrs_state_shapes():
    pos = jnp.asarray([[1.0e-4, 2.0e-4, 3.0e-4], [4.0e-4, 5.0e-4, 6.0e-4]], dtype=float)
    vel = jnp.asarray([[1.0e-6, 2.0e-6, 3.0e-6], [4.0e-6, 5.0e-6, 6.0e-6]], dtype=float)
    site = Site.from_gcrs(pos, vel)
    time = Time.from_tt_jd(
        jnp.asarray([2460743.5, 2460744.5], dtype=float),
        jnp.asarray([0.0, 0.0], dtype=float),
    )

    state = site.state(time, frame=GCRS, grid=False)
    grid_state = site.state(time, frame=GCRS, grid=True)

    print(
        "[site.from_gcrs.state.shapes] "
        f"state_shape={state.shape} "
        f"grid_shape={grid_state.shape}"
    )

    assert site.shape == (2,)
    assert state.frame == GCRS
    assert state.shape == (2,)
    assert_allclose(state.pos, pos, atol=0.0, rtol=0.0)
    assert_allclose(state.vel, vel, atol=0.0, rtol=0.0)
    assert_allclose(state.tdb.jd, time.tdb().jd, atol=0.0, rtol=0.0)

    assert grid_state.frame == GCRS
    assert grid_state.shape == (2, 2)
    assert grid_state.pos.shape == (2, 2, 3)
    assert grid_state.vel.shape == (2, 2, 3)
    assert_allclose(grid_state.pos[:, 0, :], pos, atol=0.0, rtol=0.0)
    assert_allclose(grid_state.vel[:, 0, :], vel, atol=0.0, rtol=0.0)


def test_site_from_code_space_key():
    scalar_pos = [1.0e-4, 2.0e-4, 3.0e-4]
    batch_pos = [[1.0e-4, 2.0e-4, 3.0e-4], [4.0e-4, 5.0e-4, 6.0e-4]]
    scalar_key = format_site_key("C51", Site.TYPE_SATELLITE, scalar_pos)
    batch_keys = [
        scalar_key,
        format_site_key("250", Site.TYPE_SATELLITE, batch_pos[1]),
    ]

    scalar_site = Site.from_code(scalar_key)
    batch_site = Site.from_code(batch_keys)

    assert scalar_site.shape == ()
    assert bool(scalar_site.is_space)
    assert scalar_site.identity_keys == ("C51",)
    assert_allclose(scalar_site.gcrs_pos, jnp.asarray(scalar_pos), atol=0.0, rtol=0.0)
    assert_allclose(scalar_site.gcrs_vel, jnp.zeros(3), atol=0.0, rtol=0.0)
    assert batch_site.shape == (2,)
    assert batch_site.identity_keys == ("C51", "250")
    assert_allclose(batch_site.gcrs_pos, jnp.asarray(batch_pos), atol=0.0, rtol=0.0)

    with pytest.raises(ValueError, match=r"568 is a ground site"):
        Site.from_code(format_site_key("568", Site.TYPE_SATELLITE, scalar_pos))
    with pytest.raises(ValueError, match=r"C51 is a satellite site and requires '#' coordinates"):
        Site.from_code("C51")


def test_site_from_state_contract():
    scalar_tdb = Time.from_tdb_jd(2460741.5, 0.0).tdb()
    gcrs_state = State(
        tdb=scalar_tdb,
        pos=jnp.asarray([1.0e-4, 2.0e-4, 3.0e-4], dtype=float),
        vel=jnp.asarray([1.0e-6, 2.0e-6, 3.0e-6], dtype=float),
        frame=GCRS,
    )
    bcrs_state = State(
        tdb=scalar_tdb,
        pos=jnp.asarray([1.0e-4, 2.0e-4, 3.0e-4], dtype=float),
        vel=jnp.asarray([1.0e-6, 2.0e-6, 3.0e-6], dtype=float),
        frame=BCRS,
    )

    site = Site.from_state(gcrs_state)

    assert site.shape == ()
    assert_allclose(site.gcrs_pos, gcrs_state.pos, atol=0.0, rtol=0.0)
    with pytest.raises(TypeError, match=r"`state` must be an instance of `State`"):
        Site.from_state("not a state")
    with pytest.raises(ValueError, match=r"`state.frame` must be `GCRS`"):
        Site.from_state(bcrs_state)


def test_parse_site_keys():
    roving_key = format_site_key("R01", Site.TYPE_ROVING_GROUND, [1.0, 2.0, 3.0])
    space_key = format_site_key("C51", Site.TYPE_SATELLITE, [1.0e-4, 2.0e-4, 3.0e-4])

    parsed = parse_site_keys(["568", roving_key, space_key])

    assert parsed.raw_keys == ("568", roving_key, space_key)
    assert parsed.codes == ("568", "R01", "C51")
    assert parsed.identity_keys == ("568", roving_key, "C51")
    assert parsed.display_labels == ("568", roving_key, space_key)


def test_parse_iau_codes(tmp_path):
    filepath = tmp_path / "iau_obs_codes.txt"
    filepath.write_text(
        "Code  Long.   cos      sin    Name\n"
        f"{'T01':<4}{123.4567:>9.4f}{0.12345:>8.5f}{0.98765:>+9.5f}Test Observatory\n"
        f"{'R01':<4}{'':>9}{'':>8}{'':>9}Temporary Roving Station\n"
        f"{'S01':<4}{'':>9}{'':>8}{'':>9}Space Telescope\n"
    )

    parsed = parse_iau_obs_codes(str(filepath))

    assert parsed["T01"] == (123.4567, 0.12345, 0.98765, "TEST OBSERVATORY", Site.TYPE_COMMON_GROUND)
    assert parsed["R01"] == (0.0, 0.0, 0.0, "TEMPORARY ROVING STATION", Site.TYPE_ROVING_GROUND)
    assert parsed["S01"] == (0.0, 0.0, 0.0, "SPACE TELESCOPE", Site.TYPE_SATELLITE)


def test_parse_iau_codes_rejects_html_download(tmp_path):
    filepath = tmp_path / "iau_obs_codes.txt"
    filepath.write_text("<!DOCTYPE html>\n<html><body>List Of Observatory Codes</body></html>\n")

    with pytest.raises(ValueError, match="Invalid MPC observatory code table"):
        parse_iau_obs_codes(str(filepath))


def test_parse_radar_codes(tmp_path):
    filepath = tmp_path / "jpl_radar_codes.txt"
    missing_filepath = tmp_path / "missing_radar_codes.txt"
    filepath.write_text(
        "# Code,Name,Longitude,Latitude,Altitude\n"
        "-14,DSS-14 (70-m),243.1104618,35.4259009,1.00139\n"
        "C51,WISE,,,\n"
    )

    parsed = parse_jpl_radar_obs_codes(str(filepath))

    assert parsed["-14"] == (
        243.1104618,
        35.4259009,
        1001.39,
        "DSS-14 (70-m)",
        Site.TYPE_COMMON_GROUND,
    )
    assert parsed["C51"] == (0.0, 0.0, 0.0, "WISE", Site.TYPE_SATELLITE)
    assert parse_jpl_radar_obs_codes(str(missing_filepath)) == {}


def test_site_from_code_errors():
    with pytest.raises(RuntimeError, match=r"Not found observatory code: NOPE"):
        Site.from_code("NOPE")
    with pytest.raises(ValueError, match=r"C51 is a satellite site"):
        Site.from_code("C51")
