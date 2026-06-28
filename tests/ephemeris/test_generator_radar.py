import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from difforb.body.site import Site
from difforb.core.time.timescale import Time
from difforb.ephemeris.core import RadarTable
from tests.ephemeris.generator_reference import EPOCH_TDB_JD


def test_radar_table_basic_contract(generator):
    t_rec = Time.from_tt_jd(EPOCH_TDB_JD, 0.0)
    radar_site = Site.from_code("-14").require_ground()

    table = generator.radar_table(t_rec, radar_site, tx_freq=8.56e9)

    print(
        "[ephemeris.radar] "
        f"shape={table.shape} "
        f"delay={float(table.radar_delay):+.12e} us "
        f"range={float(table.radar_range):+.12e} au "
        f"doppler={float(table.radar_doppler):+.12e} Hz"
    )

    assert isinstance(table, RadarTable)
    assert table.shape == ()
    assert float(table.radar_delay) > 0.0
    assert float(table.radar_range) > 0.0
    assert jnp.isfinite(table.radar_rate)
    assert jnp.isfinite(table.radar_doppler)
