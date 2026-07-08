import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from difforb.body.site import Site
from difforb.core.time.timescale import Time
from difforb.ephemeris.core import RadarTable
from tests.assertions import assert_allclose
from tests.ephemeris.generator_reference import EPOCH_TDB_JD


def test_radar_table_basic_contract(generator):
    t = Time.from_tt_jd(EPOCH_TDB_JD, 0.0)
    radar_site = Site.from_code("-14").require_ground()

    table = generator.radar_table(t, radar_site, tx_freq=8.56e9)

    print(
        "[ephemeris.radar] "
        f"shape={table.shape} "
        f"delay={float(table.radar_delay):+.12e} us "
        f"range={float(table.radar_range):+.12e} au "
        f"doppler={float(table.radar_doppler):+.12e} Hz "
        f"tx_az={float(table.tx_azimuth):+.12e} deg "
        f"rx_az={float(table.rx_azimuth):+.12e} deg"
    )

    assert isinstance(table, RadarTable)
    assert table.shape == ()
    assert table.epoch_at == "receive"
    assert_allclose(table.t.tt.jd, t.tt.jd, atol=0.0, rtol=0.0)
    assert float(table.radar_delay) > 0.0
    assert float(table.radar_range) > 0.0
    assert jnp.isfinite(table.radar_rate)
    assert jnp.isfinite(table.radar_doppler)
    assert jnp.isfinite(table.tx_azimuth)
    assert jnp.isfinite(table.tx_elevation)
    assert jnp.isfinite(table.rx_azimuth)
    assert jnp.isfinite(table.rx_elevation)


def test_radar_table_transmit_epoch_contract(generator):
    t = Time.from_tt_jd(EPOCH_TDB_JD, 0.0)
    radar_site = Site.from_code("-14").require_ground()

    table = generator.radar_table(t, radar_site, tx_freq=8.56e9, epoch_at="transmit")

    print(
        "[ephemeris.radar.transmit] "
        f"shape={table.shape} "
        f"delay={float(table.radar_delay):+.12e} us "
        f"range={float(table.radar_range):+.12e} au "
        f"doppler={float(table.radar_doppler):+.12e} Hz "
        f"tx_el={float(table.tx_elevation):+.12e} deg "
        f"rx_el={float(table.rx_elevation):+.12e} deg"
    )

    assert isinstance(table, RadarTable)
    assert table.shape == ()
    assert table.epoch_at == "transmit"
    assert_allclose(table.t.tt.jd, t.tt.jd, atol=0.0, rtol=0.0)
    assert float(table.radar_delay) > 0.0
    assert float(table.radar_range) > 0.0
    assert jnp.isfinite(table.radar_rate)
    assert jnp.isfinite(table.radar_doppler)
    assert jnp.isfinite(table.tx_azimuth)
    assert jnp.isfinite(table.tx_elevation)
    assert jnp.isfinite(table.rx_azimuth)
    assert jnp.isfinite(table.rx_elevation)


def test_radar_table_bistatic_pointing_fields(generator):
    t = Time.from_tt_jd(EPOCH_TDB_JD, 0.0)
    rx = Site.from_code("-14").require_ground()
    tx = Site.from_code("-13").require_ground()

    table = generator.radar_table(t, rx=rx, tx=tx, tx_freq=8.56e9)

    assert table.epoch_at == "receive"
    assert jnp.isfinite(table.tx_azimuth)
    assert jnp.isfinite(table.tx_elevation)
    assert jnp.isfinite(table.rx_azimuth)
    assert jnp.isfinite(table.rx_elevation)


def test_radar_table_rejects_invalid_epoch_at(generator):
    t = Time.from_tt_jd(EPOCH_TDB_JD, 0.0)
    radar_site = Site.from_code("-14").require_ground()

    with pytest.raises(ValueError, match="epoch_at"):
        generator.radar_table(t, radar_site, tx_freq=8.56e9, epoch_at="start")
