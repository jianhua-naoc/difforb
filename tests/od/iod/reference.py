import jax.numpy as jnp

from difforb.core.constants import GM_SUN
from difforb.core.state.frame import HELIO_ICRS
from difforb.core.state.state import State
from difforb.core.time.timescale import Time
from difforb.dynamics.two_body import kepler_propagate


PRIMARY_EPOCH_TDB_JD = 2460000.0
PRIMARY_OFFSETS = jnp.asarray([-2.0, 0.0, 2.0], dtype=jnp.float64)
PRIMARY_POS_T2 = jnp.asarray([1.25, 0.60, -0.10], dtype=jnp.float64)
PRIMARY_VEL_T2 = jnp.asarray([-0.010, 0.012, 0.001], dtype=jnp.float64)

SECONDARY_OFFSETS = jnp.asarray([-3.0, 0.0, 3.0], dtype=jnp.float64)
SECONDARY_POS_T2 = jnp.asarray([1.70, 0.30, 0.15], dtype=jnp.float64)
SECONDARY_VEL_T2 = jnp.asarray([-0.003, 0.011, 0.0015], dtype=jnp.float64)

EDGE_OBSERVATION_INDICES = jnp.asarray([0, 2], dtype=jnp.int32)


def propagate_reference_arc(pos_t2=PRIMARY_POS_T2, vel_t2=PRIMARY_VEL_T2, offsets=PRIMARY_OFFSETS):
    return kepler_propagate(
        jnp.broadcast_to(pos_t2, (3, 3)),
        jnp.broadcast_to(vel_t2, (3, 3)),
        offsets,
        mu=GM_SUN,
    )


def reference_tdb(offsets=PRIMARY_OFFSETS, epoch_tdb_jd=PRIMARY_EPOCH_TDB_JD):
    return Time.from_tdb_jd(epoch_tdb_jd + offsets, jnp.zeros_like(offsets, dtype=float))


def reference_t2_state(pos_t2=PRIMARY_POS_T2, vel_t2=PRIMARY_VEL_T2, epoch_tdb_jd=PRIMARY_EPOCH_TDB_JD):
    return State(
        Time.from_tdb_jd(epoch_tdb_jd, 0.0).tdb(),
        pos_t2,
        vel_t2,
        HELIO_ICRS,
    )
