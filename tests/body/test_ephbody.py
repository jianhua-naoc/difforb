from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from difforb.body.ephbody import EphemerisBody
from difforb.body.gm import gms
from difforb.core.constants import AU_KM
from difforb.core.state.frame import BCRS
from difforb.core.time.timescale import Time
from difforb.spk.spk import Ephemeris
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)

SPK_PATH = Path(__file__).resolve().parents[1] / "data" / "spk" / "de441_2017_2025_excerpt.bsp"
HORIZONS_EPOCH_TDB_JD = 2460690.5
pytestmark = pytest.mark.skipif(
    not SPK_PATH.exists(),
    reason="local DE441 SPK excerpt is not installed",
)

# Hard-coded JPL Horizons references in ICRF axes. These cases intentionally
# use the SSB as the center so they exercise EphemerisBody's path composition
# for bodies such as Earth and Moon. The VECTORS values were queried with:
# COMMAND=<target>, CENTER="@0", TLIST="2460690.5", TIME_TYPE="TDB",
# OUT_UNITS="KM-D", REF_PLANE="FRAME", VEC_TABLE="3".
HORIZONS_DE441_BODY_CASES = [
    (
        "sun",
        "10",
        [-8.422394503539632e05, -6.914504691158463e05, -2.707819684110850e05],
        [1.078070502708862e03, -4.720054389792868e02, -2.241565178777602e02],
    ),
    (
        "earth",
        "399",
        [-6.241148145809648e07, 1.219305041320531e08, 5.288413766533313e07],
        [-2.377552490962821e06, -9.968394922706377e05, -4.320609732950881e05],
    ),
    (
        "moon",
        "301",
        [-6.264736809505305e07, 1.221989540731991e08, 5.302944667568619e07],
        [-2.450090042925509e06, -1.041316223421215e06, -4.563139463729434e05],
    ),
    (
        "mars barycenter",
        "4",
        [-1.049132066669511e08, 1.979707027498702e08, 9.365797405521458e07],
        [-1.811639806064567e06, -6.711873526700274e05, -2.589681336873384e05],
    ),
]


@pytest.mark.parametrize(
    ("body_name", "command", "expected_pos_km", "expected_vel_km_per_day"),
    HORIZONS_DE441_BODY_CASES,
)
def test_ephbody_state_against_horizons(
        body_name,
        command,
        expected_pos_km,
        expected_vel_km_per_day,
):
    eph = Ephemeris(str(SPK_PATH))
    body = EphemerisBody(body_name, eph=eph)
    tdb = Time.from_tdb_jd(HORIZONS_EPOCH_TDB_JD, 0.0).tdb()

    state = body.state(tdb)
    expected_pos = jnp.asarray(expected_pos_km, dtype=float) / AU_KM
    expected_vel = jnp.asarray(expected_vel_km_per_day, dtype=float) / AU_KM
    pos_diff = jnp.max(jnp.abs(state.pos - expected_pos))
    vel_diff = jnp.max(jnp.abs(state.vel - expected_vel))

    print(
        "[ephbody.state.horizons] "
        f"body={body.naif_name:<16} "
        f"command={command:<3} "
        f"pos_max_abs_diff={float(pos_diff):+.12e} au "
        f"vel_max_abs_diff={float(vel_diff):+.12e} au/day"
    )

    assert state.frame == BCRS
    assert state.shape == ()
    assert_allclose(state.tdb.jd, tdb.jd, atol=0.0, rtol=0.0)
    assert_allclose(state.pos, expected_pos, atol=1.0e-15, rtol=0.0)
    assert_allclose(state.vel, expected_vel, atol=1.0e-17, rtol=0.0)


def test_ephbody_state_batch_shape():
    eph = Ephemeris(str(SPK_PATH))
    body = EphemerisBody("earth", eph=eph)
    tdb = Time.from_tdb_jd(
        jnp.asarray([2460690.5, 2460691.5], dtype=float),
        jnp.asarray([0.0, 0.0], dtype=float),
    ).tdb()

    state = body.state(tdb)

    print(
        "[ephbody.state.batch_shape] "
        f"body={body.naif_name:<8} "
        f"state_shape={state.shape}"
    )

    assert state.frame == BCRS
    assert state.shape == (2,)
    assert state.pos.shape == (2, 3)
    assert state.vel.shape == (2, 3)
    assert_allclose(state.tdb.jd, tdb.jd, atol=0.0, rtol=0.0)


def test_ephbody_name_gm_and_repr():
    eph = Ephemeris(str(SPK_PATH))
    body = EphemerisBody("sun", eph=eph)
    text = repr(body)

    assert body.naif_name == "SUN"
    assert body.gm == gms["SUN"]
    assert len(body.segments) == 1
    assert body.signs == (1.0,)
    assert "SUN" in text
    assert "segment_count" in text


def test_ephbody_rejects_non_tdb():
    eph = Ephemeris(str(SPK_PATH))
    body = EphemerisBody("sun", eph=eph)
    tt = Time.from_tdb_jd(HORIZONS_EPOCH_TDB_JD, 0.0).tt

    with pytest.raises(TypeError, match=r"argument `tdb` must be an instance of `TDBView`"):
        body.state(tt)


def test_ephbody_rejects_missing_gm():
    class FakeEphemeris:
        def load_body(self, target_name):
            assert target_name == "FAKE BODY"
            return (), ()

    with pytest.raises(RuntimeError, match=r"Invalid object name: FAKE BODY"):
        EphemerisBody("fake body", eph=FakeEphemeris())
