import jax
import jax.numpy as jnp
import pytest

from difforb.core.constants import C, LC
from difforb.core.state.axes import Axes, axes_to_icrs_rotation, icrs_to_axes_rotation
from difforb.core.state.frame import BCRS, GCRS, HELIO_ICRS, Frame
from difforb.core.state.origins import Origin, origin_in_ssb_icrs
from difforb.core.state.state import State
from difforb.core.time.timescale import Time
from tests.assertions import assert_allclose

jax.config.update("jax_enable_x64", True)


# -------------------------------------------------------------------------
# Construction And Basic Properties
# -------------------------------------------------------------------------


def test_state_init_broadcasts_position_and_velocity():
    tdb = Time.from_tdb_jd(jnp.array([2451545.0, 2451546.25], dtype=jnp.float64), jnp.array([0.0, 0.0], dtype=jnp.float64)).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float64),
        vel=jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64),
        frame=BCRS,
    )

    assert state.shape == (2,)
    assert state.pos.shape == (2, 3)
    assert state.vel.shape == (2, 3)
    assert_allclose(state.pos, jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float64), atol=0.0, rtol=0.0)
    assert_allclose(state.vel, jnp.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], dtype=jnp.float64), atol=0.0, rtol=0.0)


def test_state_array_stacks_position_and_velocity():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=BCRS,
    )

    assert_allclose(
        state.array,
        jnp.array([1.0, -2.0, 0.5, 0.01, 0.02, -0.03], dtype=jnp.float64),
        atol=0.0,
        rtol=0.0,
    )


def test_state_from_array_builds_state_with_expected_fields():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State.from_array(
        tdb=tdb,
        array=jnp.array([1.0, -2.0, 0.5, 0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=BCRS,
    )

    assert state.frame == BCRS
    assert state.shape == ()
    assert_allclose(state.pos, jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64), atol=0.0, rtol=0.0)
    assert_allclose(state.vel, jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64), atol=0.0, rtol=0.0)


def test_state_array_and_from_array_are_inverse_for_batched_state():
    tdb = Time.from_tdb_jd(
        jnp.array([2451545.0, 2451546.25], dtype=jnp.float64),
        jnp.array([0.0, -0.125], dtype=jnp.float64),
    ).tdb()
    original = State(
        tdb=tdb,
        pos=jnp.array([[1.0, -2.0, 0.5], [3.5, 4.0, -1.2]], dtype=jnp.float64),
        vel=jnp.array([[0.01, 0.02, -0.03], [-0.04, 0.05, 0.06]], dtype=jnp.float64),
        frame=BCRS,
    )
    rebuilt = State.from_array(tdb=tdb, array=original.array, frame=BCRS)

    assert rebuilt.frame == BCRS
    assert rebuilt.shape == original.shape
    assert_allclose(rebuilt.pos, original.pos, atol=0.0, rtol=0.0)
    assert_allclose(rebuilt.vel, original.vel, atol=0.0, rtol=0.0)
    assert_allclose(rebuilt.array, original.array, atol=0.0, rtol=0.0)


def test_state_dist_returns_position_norm():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, 2.0, 2.0], dtype=jnp.float64),
        vel=jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64),
        frame=BCRS,
    )

    assert_allclose(state.dist, 3.0, atol=1e-16, rtol=0.0)


def test_state_lt_returns_distance_divided_by_speed_of_light():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, 2.0, 2.0], dtype=jnp.float64),
        vel=jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64),
        frame=BCRS,
    )

    assert_allclose(state.lt, state.dist / C, atol=1e-15, rtol=0.0)


def test_state_getitem_slices_all_batched_fields_consistently():
    tdb = Time.from_tdb_jd(
        jnp.array([2451545.0, 2451546.25, 2451547.5], dtype=jnp.float64),
        jnp.array([0.0, -0.125, 0.25], dtype=jnp.float64),
    ).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array(
            [
                [1.0, -2.0, 0.5],
                [3.5, 4.0, -1.2],
                [-0.4, 0.7, 2.1],
            ],
            dtype=jnp.float64,
        ),
        vel=jnp.array(
            [
                [0.01, 0.02, -0.03],
                [-0.04, 0.05, 0.06],
                [0.07, -0.08, 0.09],
            ],
            dtype=jnp.float64,
        ),
        frame=BCRS,
    )
    sliced = state[1]

    assert sliced.frame == BCRS
    assert sliced.shape == ()
    assert_allclose(sliced.tdb.jd, 2451546.125, atol=1e-15, rtol=0.0)
    assert_allclose(sliced.pos, jnp.array([3.5, 4.0, -1.2], dtype=jnp.float64), atol=0.0, rtol=0.0)
    assert_allclose(sliced.vel, jnp.array([-0.04, 0.05, 0.06], dtype=jnp.float64), atol=0.0, rtol=0.0)


def test_len_raises_for_scalar_state():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=BCRS,
    )

    with pytest.raises(TypeError, match=r"len\(\) of unsized object: State"):
        len(state)


def test_len_returns_for_batched_state():
    tdb = Time.from_tdb_jd(
        jnp.array([2451545.0, 2451546.25, 2451547.5], dtype=jnp.float64),
        jnp.array([0.0, -0.125, 0.25], dtype=jnp.float64),
    ).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array(
            [
                [1.0, -2.0, 0.5],
                [3.5, 4.0, -1.2],
                [-0.4, 0.7, 2.1],
            ],
            dtype=jnp.float64,
        ),
        vel=jnp.array(
            [
                [0.01, 0.02, -0.03],
                [-0.04, 0.05, 0.06],
                [0.07, -0.08, 0.09],
            ],
            dtype=jnp.float64,
        ),
        frame=BCRS,
    )

    assert len(state) == 3


def test_state_init_raises_when_position_last_dimension_is_not_three():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()

    with pytest.raises(ValueError, match=r"State position last dimension must be 3"):
        State(
            tdb=tdb,
            pos=jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64),
            vel=jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float64),
            frame=BCRS,
        )


def test_state_init_raises_when_position_and_velocity_cannot_be_broadcast_together():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()

    with pytest.raises(ValueError, match=r"Incompatible shapes for broadcasting"):
        State(
            tdb=tdb,
            pos=jnp.array([1.0, 2.0], dtype=jnp.float64),
            vel=jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64),
            frame=BCRS,
        )


def test_state_init_raises_when_tdb_shape_does_not_match_position_and_velocity_shape():
    tdb = Time.from_tdb_jd(
        jnp.array([2451545.0, 2451546.25], dtype=jnp.float64),
        jnp.array([0.0, 0.0], dtype=jnp.float64),
    ).tdb()

    with pytest.raises(ValueError, match=r"``tdb`` and ``pos``/``vel`` must have same shape\."):
        State(
            tdb=tdb,
            pos=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64),
            vel=jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64),
            frame=BCRS,
        )


def test_state_from_array_raises_when_array_last_dimension_is_not_six():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()

    with pytest.raises(ValueError, match=r"State array last dimension must be 6"):
        State.from_array(
            tdb=tdb,
            array=jnp.array([1.0, 2.0, 3.0, 0.1, 0.2], dtype=jnp.float64),
            frame=BCRS,
        )


# -------------------------------------------------------------------------
# Axis-Only Conversion
# -------------------------------------------------------------------------


def test_state_to_same_frame_preserves_state_values():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=BCRS,
    )

    converted = state.to(BCRS)

    assert converted.frame == state.frame
    assert_allclose(converted.tdb.jd, state.tdb.jd, atol=0.0, rtol=0.0)
    assert_allclose(converted.pos, state.pos, atol=0.0, rtol=0.0)
    assert_allclose(converted.vel, state.vel, atol=0.0, rtol=0.0)


def test_state_to_j2000_rotates_icrs_state_with_expected_matrix():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=BCRS,
    )
    target_frame = Frame(axes=Axes.J2000, origin=Origin.SSB)
    expected_pos = state.pos @ icrs_to_axes_rotation(Axes.J2000)
    expected_vel = state.vel @ icrs_to_axes_rotation(Axes.J2000)
    converted = state.to(target_frame)

    assert converted.frame == target_frame
    assert_allclose(converted.pos, expected_pos, atol=1e-15, rtol=0.0)
    assert_allclose(converted.vel, expected_vel, atol=1e-15, rtol=0.0)
    assert_allclose(converted.tdb.jd, state.tdb.jd, atol=0.0, rtol=0.0)


def test_state_to_eclip_j2000_rotates_icrs_state_with_expected_matrix():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=BCRS,
    )
    target_frame = Frame(axes=Axes.ECLIP_J2000, origin=Origin.SSB)
    expected_pos = state.pos @ icrs_to_axes_rotation(Axes.ECLIP_J2000)
    expected_vel = state.vel @ icrs_to_axes_rotation(Axes.ECLIP_J2000)
    converted = state.to(target_frame)

    assert converted.frame == target_frame
    assert_allclose(converted.pos, expected_pos, atol=1e-15, rtol=0.0)
    assert_allclose(converted.vel, expected_vel, atol=1e-15, rtol=0.0)
    assert_allclose(converted.tdb.jd, state.tdb.jd, atol=0.0, rtol=0.0)


def test_state_to_bcrs_rotates_j2000_state_with_expected_matrix():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    source_frame = Frame(axes=Axes.J2000, origin=Origin.SSB)
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=source_frame,
    )
    expected_pos = state.pos @ axes_to_icrs_rotation(Axes.J2000)
    expected_vel = state.vel @ axes_to_icrs_rotation(Axes.J2000)
    converted = state.to(BCRS)

    assert converted.frame == BCRS
    assert_allclose(converted.pos, expected_pos, atol=1e-15, rtol=0.0)
    assert_allclose(converted.vel, expected_vel, atol=1e-15, rtol=0.0)
    assert_allclose(converted.tdb.jd, state.tdb.jd, atol=0.0, rtol=0.0)


def test_state_to_bcrs_rotates_eclip_j2000_state_with_expected_matrix():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    source_frame = Frame(axes=Axes.ECLIP_J2000, origin=Origin.SSB)
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=source_frame,
    )
    expected_pos = state.pos @ axes_to_icrs_rotation(Axes.ECLIP_J2000)
    expected_vel = state.vel @ axes_to_icrs_rotation(Axes.ECLIP_J2000)
    converted = state.to(BCRS)

    assert converted.frame == BCRS
    assert_allclose(converted.pos, expected_pos, atol=1e-15, rtol=0.0)
    assert_allclose(converted.vel, expected_vel, atol=1e-15, rtol=0.0)
    assert_allclose(converted.tdb.jd, state.tdb.jd, atol=0.0, rtol=0.0)


def test_state_axis_roundtrip_recovers_original_state_for_j2000():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=BCRS,
    )
    j2000_frame = Frame(axes=Axes.J2000, origin=Origin.SSB)
    recovered = state.to(j2000_frame).to(BCRS)

    assert recovered.frame == BCRS
    assert_allclose(recovered.pos, state.pos, atol=1e-15, rtol=0.0)
    assert_allclose(recovered.vel, state.vel, atol=1e-15, rtol=0.0)
    assert_allclose(recovered.tdb.jd, state.tdb.jd, atol=0.0, rtol=0.0)


def test_state_axis_roundtrip_recovers_original_state_for_eclip_j2000():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=BCRS,
    )
    eclip_frame = Frame(axes=Axes.ECLIP_J2000, origin=Origin.SSB)
    recovered = state.to(eclip_frame).to(BCRS)

    assert recovered.frame == BCRS
    assert_allclose(recovered.pos, state.pos, atol=1e-15, rtol=0.0)
    assert_allclose(recovered.vel, state.vel, atol=1e-15, rtol=0.0)
    assert_allclose(recovered.tdb.jd, state.tdb.jd, atol=0.0, rtol=0.0)


def test_state_axis_only_conversion_matches_expected_rotation_for_batched_state():
    tdb = Time.from_tdb_jd(
        jnp.array([2451545.0, 2451546.25, 2451544.875], dtype=jnp.float64),
        jnp.array([0.0, -0.125, 0.25], dtype=jnp.float64),
    ).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array(
            [
                [1.0, -2.0, 0.5],
                [3.5, 4.0, -1.2],
                [-0.4, 0.7, 2.1],
            ],
            dtype=jnp.float64,
        ),
        vel=jnp.array(
            [
                [0.01, 0.02, -0.03],
                [-0.04, 0.05, 0.06],
                [0.07, -0.08, 0.09],
            ],
            dtype=jnp.float64,
        ),
        frame=BCRS,
    )
    target_frame = Frame(axes=Axes.J2000, origin=Origin.SSB)
    expected_pos = state.pos @ icrs_to_axes_rotation(Axes.J2000)
    expected_vel = state.vel @ icrs_to_axes_rotation(Axes.J2000)
    converted = state.to(target_frame)

    assert converted.shape == (3,)
    assert converted.frame == target_frame
    assert_allclose(converted.pos, expected_pos, atol=1e-15, rtol=0.0)
    assert_allclose(converted.vel, expected_vel, atol=1e-15, rtol=0.0)
    assert_allclose(converted.tdb.jd, state.tdb.jd, atol=0.0, rtol=0.0)


# -------------------------------------------------------------------------
# Origin-Only Conversion
# -------------------------------------------------------------------------


def test_state_to_helio_icrs_translates_bcrs_state_with_expected_sun_offset(fake_sun):
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=BCRS,
    )
    sun_pos, sun_vel = origin_in_ssb_icrs(Origin.SUN, tdb, sun=fake_sun)
    expected_pos = state.pos - sun_pos
    expected_vel = state.vel - sun_vel
    converted = state.to(HELIO_ICRS, sun=fake_sun)

    assert converted.frame == HELIO_ICRS
    assert_allclose(converted.pos, expected_pos, atol=1e-15, rtol=0.0)
    assert_allclose(converted.vel, expected_vel, atol=1e-15, rtol=0.0)
    assert_allclose(converted.tdb.jd, state.tdb.jd, atol=0.0, rtol=0.0)


def test_state_to_bcrs_translates_helio_icrs_state_with_expected_sun_offset(fake_sun):
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=HELIO_ICRS,
    )
    sun_pos, sun_vel = origin_in_ssb_icrs(Origin.SUN, tdb, sun=fake_sun)
    expected_pos = state.pos + sun_pos
    expected_vel = state.vel + sun_vel
    converted = state.to(BCRS, sun=fake_sun)

    assert converted.frame == BCRS
    assert_allclose(converted.pos, expected_pos, atol=1e-15, rtol=0.0)
    assert_allclose(converted.vel, expected_vel, atol=1e-15, rtol=0.0)
    assert_allclose(converted.tdb.jd, state.tdb.jd, atol=0.0, rtol=0.0)


def test_state_sun_origin_roundtrip_recovers_original_bcrs_state(fake_sun):
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=BCRS,
    )
    recovered = state.to(HELIO_ICRS, sun=fake_sun).to(BCRS, sun=fake_sun)

    assert recovered.frame == BCRS
    assert_allclose(recovered.pos, state.pos, atol=1e-15, rtol=0.0)
    assert_allclose(recovered.vel, state.vel, atol=1e-15, rtol=0.0)
    assert_allclose(recovered.tdb.jd, state.tdb.jd, atol=0.0, rtol=0.0)


def test_state_to_gcrs_translates_bcrs_state_with_expected_earth_offset_and_lc_correction(fake_earth):
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=BCRS,
    )
    earth_pos, earth_vel = origin_in_ssb_icrs(Origin.EARTH, tdb, earth=fake_earth)
    pos0 = state.pos - earth_pos
    vel0 = state.vel - earth_vel
    v_earth_c = earth_vel / C
    v_dot_p = jnp.sum(v_earth_c * pos0, axis=-1, keepdims=True)
    expected_pos = pos0 * (1.0 + LC) + 0.5 * v_dot_p * v_earth_c
    expected_vel = vel0 * (1.0 + LC)
    converted = state.to(GCRS, earth=fake_earth)

    assert converted.frame == GCRS
    assert_allclose(converted.pos, expected_pos, atol=1e-15, rtol=0.0)
    assert_allclose(converted.vel, expected_vel, atol=1e-15, rtol=0.0)
    assert_allclose(converted.tdb.jd, state.tdb.jd, atol=0.0, rtol=0.0)


def test_state_to_bcrs_translates_gcrs_state_with_expected_earth_offset_and_lc_correction(fake_earth):
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=GCRS,
    )
    earth_pos, earth_vel = origin_in_ssb_icrs(Origin.EARTH, tdb, earth=fake_earth)
    v_earth_c = earth_vel / C
    v_dot_p = jnp.sum(v_earth_c * state.pos, axis=-1, keepdims=True)
    pos_corr = state.pos * (1.0 - LC) - 0.5 * v_dot_p * v_earth_c
    vel_corr = state.vel * (1.0 - LC)
    expected_pos = pos_corr + earth_pos
    expected_vel = vel_corr + earth_vel
    converted = state.to(BCRS, earth=fake_earth)

    assert converted.frame == BCRS
    assert_allclose(converted.pos, expected_pos, atol=1e-15, rtol=0.0)
    assert_allclose(converted.vel, expected_vel, atol=1e-15, rtol=0.0)
    assert_allclose(converted.tdb.jd, state.tdb.jd, atol=0.0, rtol=0.0)


def test_state_earth_origin_roundtrip_recovers_original_bcrs_state(fake_earth):
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=BCRS,
    )
    recovered = state.to(GCRS, earth=fake_earth).to(BCRS, earth=fake_earth)

    assert recovered.frame == BCRS
    assert_allclose(recovered.pos, state.pos, atol=1e-12, rtol=0.0)
    assert_allclose(recovered.vel, state.vel, atol=1e-12, rtol=0.0)
    assert_allclose(recovered.tdb.jd, state.tdb.jd, atol=0.0, rtol=0.0)


def test_state_sun_origin_conversion_matches_expected_translation_for_batched_state(fake_sun):
    tdb = Time.from_tdb_jd(
        jnp.array([2451545.0, 2451546.25, 2451544.875], dtype=jnp.float64),
        jnp.array([0.0, -0.125, 0.25], dtype=jnp.float64),
    ).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array(
            [
                [1.0, -2.0, 0.5],
                [3.5, 4.0, -1.2],
                [-0.4, 0.7, 2.1],
            ],
            dtype=jnp.float64,
        ),
        vel=jnp.array(
            [
                [0.01, 0.02, -0.03],
                [-0.04, 0.05, 0.06],
                [0.07, -0.08, 0.09],
            ],
            dtype=jnp.float64,
        ),
        frame=BCRS,
    )
    sun_pos, sun_vel = origin_in_ssb_icrs(Origin.SUN, tdb, sun=fake_sun)
    expected_pos = state.pos - sun_pos
    expected_vel = state.vel - sun_vel
    converted = state.to(HELIO_ICRS, sun=fake_sun)

    assert converted.shape == (3,)
    assert converted.frame == HELIO_ICRS
    assert_allclose(converted.pos, expected_pos, atol=1e-15, rtol=0.0)
    assert_allclose(converted.vel, expected_vel, atol=1e-15, rtol=0.0)


def test_state_earth_origin_conversion_matches_expected_translation_for_batched_state(fake_earth):
    tdb = Time.from_tdb_jd(
        jnp.array([2451545.0, 2451546.25, 2451544.875], dtype=jnp.float64),
        jnp.array([0.0, -0.125, 0.25], dtype=jnp.float64),
    ).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array(
            [
                [1.0, -2.0, 0.5],
                [3.5, 4.0, -1.2],
                [-0.4, 0.7, 2.1],
            ],
            dtype=jnp.float64,
        ),
        vel=jnp.array(
            [
                [0.01, 0.02, -0.03],
                [-0.04, 0.05, 0.06],
                [0.07, -0.08, 0.09],
            ],
            dtype=jnp.float64,
        ),
        frame=BCRS,
    )
    earth_pos, earth_vel = origin_in_ssb_icrs(Origin.EARTH, tdb, earth=fake_earth)
    pos0 = state.pos - earth_pos
    vel0 = state.vel - earth_vel
    v_earth_c = earth_vel / C
    v_dot_p = jnp.sum(v_earth_c * pos0, axis=-1, keepdims=True)
    expected_pos = pos0 * (1.0 + LC) + 0.5 * v_dot_p * v_earth_c
    expected_vel = vel0 * (1.0 + LC)
    converted = state.to(GCRS, earth=fake_earth)

    assert converted.shape == (3,)
    assert converted.frame == GCRS
    assert_allclose(converted.pos, expected_pos, atol=1e-15, rtol=0.0)
    assert_allclose(converted.vel, expected_vel, atol=1e-15, rtol=0.0)


def test_state_to_helio_icrs_raises_when_sun_ephemeris_is_missing():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=BCRS,
    )

    with pytest.raises(ValueError, match=r"Origin ``SUN`` requires the ``sun`` ephemeris body\."):
        state.to(HELIO_ICRS)


def test_state_to_bcrs_from_helio_icrs_raises_when_sun_ephemeris_is_missing():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=HELIO_ICRS,
    )

    with pytest.raises(ValueError, match=r"Origin ``SUN`` requires the ``sun`` ephemeris body\."):
        state.to(BCRS)


def test_state_to_gcrs_raises_when_earth_ephemeris_is_missing():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=BCRS,
    )

    with pytest.raises(ValueError, match=r"Origin ``EARTH`` requires the ``earth`` ephemeris body\."):
        state.to(GCRS)


def test_state_to_bcrs_from_gcrs_raises_when_earth_ephemeris_is_missing():
    tdb = Time.from_tdb_jd(2451545.0, 0.0).tdb()
    state = State(
        tdb=tdb,
        pos=jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64),
        vel=jnp.array([0.01, 0.02, -0.03], dtype=jnp.float64),
        frame=GCRS,
    )

    with pytest.raises(ValueError, match=r"Origin ``EARTH`` requires the ``earth`` ephemeris body\."):
        state.to(BCRS)
