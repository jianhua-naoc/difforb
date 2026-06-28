from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from difforb.core.constants import AU_KM
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

# Hard-coded JPL Horizons references in ICRF axes. Each case compares one
# raw DE441 SPK segment against a Horizons vector with the same target and
# center. The raw DE441 SPK coefficients are stored in the same axes as
# DiffOrb's ``BCRS`` frame.
# Do not use Horizons' default reference plane here; without
# ``REF_PLANE="FRAME"``, Horizons returns Ecliptic of J2000.0 vectors.
# The VECTORS values were queried with:
# COMMAND=<target>, CENTER="@<center>", TLIST="2460690.5", TIME_TYPE="TDB",
# OUT_UNITS="KM-D", REF_PLANE="FRAME", VEC_TABLE="3".
HORIZONS_DE441_SEGMENT_CASES = [
    (
        "SSB to Sun",
        "SOLAR SYSTEM BARYCENTER",
        "SUN",
        0,
        10,
        [-8.422394503539632e05, -6.914504691158463e05, -2.707819684110850e05],
        [1.078070502708862e03, -4.720054389792868e02, -2.241565178777602e02],
    ),
    (
        "SSB to Earth-Moon barycenter",
        "SOLAR SYSTEM BARYCENTER",
        "EARTH BARYCENTER",
        0,
        3,
        [-6.241434761858667e07, 1.219337659557191e08, 5.288590325472690e07],
        [-2.378433864609811e06, -9.973799105461333e05, -4.323556610913204e05],
    ),
    (
        "Earth-Moon barycenter to Earth",
        "EARTH BARYCENTER",
        "EARTH",
        3,
        399,
        [2.866160490189099e03, -3.261823665949350e03, -1.765589393769178e03],
        [8.813736469894861e02, 5.404182754956970e02, 2.946877962322544e02],
    ),
    (
        "Earth-Moon barycenter to Moon",
        "EARTH BARYCENTER",
        "MOON",
        3,
        301,
        [-2.330204764663788e05, 2.651881174800093e05, 1.435434209592829e05],
        [-7.165617831569853e04, -4.393631287508181e04, -2.395828528162307e04],
    ),
    (
        "SSB to Mars barycenter",
        "SOLAR SYSTEM BARYCENTER",
        "MARS BARYCENTER",
        0,
        4,
        [-1.049132066669511e08, 1.979707027498702e08, 9.365797405521458e07],
        [-1.811639806064567e06, -6.711873526700274e05, -2.589681336873384e05],
    ),
]


def test_de441_loads_expected_bodies():
    eph = Ephemeris(str(SPK_PATH))
    expected_bodies = {
        "SOLAR SYSTEM BARYCENTER",
        "SUN",
        "MERCURY BARYCENTER",
        "MERCURY",
        "VENUS BARYCENTER",
        "VENUS",
        "EARTH BARYCENTER",
        "EARTH",
        "MOON",
        "MARS BARYCENTER",
        "JUPITER BARYCENTER",
        "SATURN BARYCENTER",
        "URANUS BARYCENTER",
        "NEPTUNE BARYCENTER",
        "PLUTO BARYCENTER",
    }

    assert set(eph.available_bodies) == expected_bodies


def test_de441_body_paths():
    eph = Ephemeris(str(SPK_PATH))
    expected_paths = {
        "SUN": (((0, 10),), (1.0,)),
        "MARS BARYCENTER": (((0, 4),), (1.0,)),
        "EARTH": (((0, 3), (3, 399)), (1.0, 1.0)),
        "MOON": (((0, 3), (3, 301)), (1.0, 1.0)),
    }

    for target_name, (expected_ids, expected_signs) in expected_paths.items():
        segments, signs = eph.load_body(target_name)
        actual_ids = tuple((int(segment.center_ids[0]), int(segment.target_ids[0])) for segment in segments)

        print(
            "[spk.de441.body_paths] "
            f"target={target_name:<16} "
            f"segment_count={len(segments)} "
            f"signs={signs}"
        )

        assert actual_ids == expected_ids
        assert signs == expected_signs


@pytest.mark.parametrize(
    (
        "label",
        "center_name",
        "target_name",
        "center_id",
        "target_id",
        "expected_pos",
        "expected_vel",
    ),
    HORIZONS_DE441_SEGMENT_CASES,
)
def test_de441_segment_state_against_horizons(
        label,
        center_name,
        target_name,
        center_id,
        target_id,
        expected_pos,
        expected_vel,
):
    eph = Ephemeris(str(SPK_PATH))
    tdb = Time.from_tdb_jd(HORIZONS_EPOCH_TDB_JD, 0.0).tdb()

    segments, signs = eph.load_body(target_name, center_name=center_name)
    assert len(segments) == 1
    assert signs == (1.0,)
    segment = segments[0]
    assert (int(segment.center_ids[0]), int(segment.target_ids[0])) == (center_id, target_id)
    actual_pos, actual_vel = segment.state(tdb.jd1, tdb.jd2)

    expected_pos = jnp.asarray(expected_pos, dtype=float)
    expected_vel = jnp.asarray(expected_vel, dtype=float)
    pos_diff = jnp.max(jnp.abs(actual_pos - expected_pos))
    vel_diff = jnp.max(jnp.abs(actual_vel - expected_vel))
    pos_diff_au = pos_diff / AU_KM
    vel_diff_au_per_day = vel_diff / AU_KM

    print(
        "[spk.de441.horizons] "
        f"segment={label:<32} "
        f"ids={center_id}->{target_id:<3} "
        f"pos_max_abs_diff={float(pos_diff_au):+.12e} au "
        f"vel_max_abs_diff={float(vel_diff_au_per_day):+.12e} au/day"
    )

    assert_allclose(actual_pos, expected_pos, atol=1.0e-7, rtol=0.0)
    assert_allclose(actual_vel, expected_vel, atol=1.0e-9, rtol=0.0)
