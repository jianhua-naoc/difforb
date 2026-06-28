from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from difforb.body.ephbody import EphemerisBody
from difforb.body.smallbody import SmallBody
from difforb.core.constants import AU_KM, DAY_S
from difforb.core.element import KepElement
from difforb.core.time.timescale import Time
from difforb.dynamics.dynamic_system import DynamicSystem
from difforb.ephemeris import core as ephemeris_core
from difforb.ephemeris.generator import EphemerisGenerator
from difforb.integrator.integrator import NumericalIntegrator
from tests.ephemeris.generator_reference import (
    HORIZONS_GENERATOR_APSIDES_CASES,
    HORIZONS_GENERATOR_CLOSE_APPROACH_CASES,
)
from tests.assertions import assert_allclose


@pytest.mark.parametrize(
    (
        "label",
        "target_command",
        "target_name",
        "initial_tdb_jd",
        "initial_elements",
        "search_start_jd",
        "search_end_jd",
        "expected_events",
        "time_atol",
        "distance_atol",
    ),
    HORIZONS_GENERATOR_APSIDES_CASES,
)
def test_apsides_against_horizons_vectors(
        default_ephemeris,
        label,
        target_command,
        target_name,
        initial_tdb_jd,
        initial_elements,
        search_start_jd,
        search_end_jd,
        expected_events,
        time_atol,
        distance_atol,
):
    sun = EphemerisBody("sun", eph=default_ephemeris)
    force_model = DynamicSystem.from_extended_system(default_ephemeris).build_force_model()
    integrator = NumericalIntegrator(method="IAS15", tol=1.0e-12, initial_step=0.05, max_steps=4096)
    tdb0 = Time.from_tdb_jd(initial_tdb_jd, 0.0).tdb()
    initial = KepElement.from_classical(tdb0, *initial_elements)

    target = SmallBody.create(initial, sun=sun).propagate(
        Time.from_tdb_jd(search_start_jd, 0.0).tdb(),
        Time.from_tdb_jd(search_end_jd, 0.0).tdb(),
        force_model,
        integrator,
    )
    table = EphemerisGenerator(target).find_apsides(
        Time.from_tdb_jd(search_start_jd, 0.0).tdb(),
        Time.from_tdb_jd(search_end_jd, 0.0).tdb(),
        sun,
        max_events=max(4, len(expected_events)),
    )
    actual = table.valid
    expected_event_type = jnp.asarray([event[0] for event in expected_events])
    expected_tdb_jd = jnp.asarray([event[1] for event in expected_events])
    expected_distance = jnp.asarray([event[2] for event in expected_events])

    print(
        "[ephemeris.apsides.horizons] "
        f"label={label:<16} "
        f"target={target_command:<14} "
        f"name={target_name} "
        f"t_max_abs_diff={float(jnp.max(jnp.abs(actual.t_apsides.jd - expected_tdb_jd))):.12e} day "
        f"distance_max_abs_diff={float(jnp.max(jnp.abs(actual.distance - expected_distance))):.12e} au"
    )

    assert actual.shape == (len(expected_events),)
    assert jnp.count_nonzero(table.is_valid) == len(expected_events)
    assert table.periapsis.shape == (int(jnp.count_nonzero(expected_event_type == 0)),)
    assert table.apoapsis.shape == (int(jnp.count_nonzero(expected_event_type == 1)),)
    assert jnp.array_equal(actual.event_type, expected_event_type)
    assert_allclose(actual.t_apsides.jd, expected_tdb_jd, atol=time_atol, rtol=0.0)
    assert_allclose(actual.distance, expected_distance, atol=distance_atol, rtol=0.0)


@pytest.mark.parametrize(
    (
        "label",
        "target_command",
        "target_name",
        "initial_tdb_jd",
        "initial_elements",
        "search_start_jd",
        "search_end_jd",
        "center_name",
        "max_distance",
        "expected_tdb_jd",
        "expected_distance",
        "expected_relative_velocity_km_s",
        "time_atol",
        "distance_atol",
        "relative_velocity_atol",
    ),
    HORIZONS_GENERATOR_CLOSE_APPROACH_CASES,
)
def test_close_approaches_against_horizons(
        default_ephemeris,
        label,
        target_command,
        target_name,
        initial_tdb_jd,
        initial_elements,
        search_start_jd,
        search_end_jd,
        center_name,
        max_distance,
        expected_tdb_jd,
        expected_distance,
        expected_relative_velocity_km_s,
        time_atol,
        distance_atol,
        relative_velocity_atol,
):
    sun = EphemerisBody("sun", eph=default_ephemeris)
    center = EphemerisBody(center_name, eph=default_ephemeris)
    force_model = DynamicSystem.from_extended_system(default_ephemeris).build_force_model()
    integrator = NumericalIntegrator(method="IAS15", tol=1.0e-12, initial_step=0.05, max_steps=4096)
    tdb0 = Time.from_tdb_jd(initial_tdb_jd, 0.0).tdb()
    initial = KepElement.from_classical(tdb0, *initial_elements)

    target = SmallBody.create(initial, sun=sun).propagate(
        Time.from_tdb_jd(search_start_jd, 0.0).tdb(),
        Time.from_tdb_jd(search_end_jd, 0.0).tdb(),
        force_model,
        integrator,
    )
    table = EphemerisGenerator(target).find_close_approaches(
        Time.from_tdb_jd(search_start_jd, 0.0).tdb(),
        Time.from_tdb_jd(search_end_jd, 0.0).tdb(),
        center,
        max_distance=max_distance,
        max_events=3,
    )
    actual = table.valid
    expected_relative_velocity = expected_relative_velocity_km_s * DAY_S / AU_KM

    print(
        "[ephemeris.close_approach.horizons] "
        f"label={label:<22} "
        f"target={target_command:<7} "
        f"name={target_name} "
        f"t_diff={float(actual.t_close.jd[0] - expected_tdb_jd):+.12e} day "
        f"distance_diff={float(actual.distance[0] - expected_distance):+.12e} au "
        f"relative_velocity_diff={float(actual.relative_velocity[0] - expected_relative_velocity):+.12e} au/day"
    )

    assert actual.shape == (1,)
    assert bool(actual.is_valid[0])
    assert jnp.count_nonzero(table.is_valid) == 1
    assert_allclose(actual.t_close.jd[0], expected_tdb_jd, atol=time_atol, rtol=0.0)
    assert_allclose(actual.distance[0], expected_distance, atol=distance_atol, rtol=0.0)
    assert_allclose(actual.relative_velocity[0], expected_relative_velocity, atol=relative_velocity_atol, rtol=0.0)


def test_close_approach_search_receives_distance_limit(monkeypatch):
    captured: dict[str, float] = {}

    def fake_find_distance_extrema_single(target, center, t_start, t_end, max_events, extrema_type="min", max_distance=jnp.inf):
        captured["max_distance"] = float(max_distance)
        t_close = Time.from_tdb_jd(jnp.asarray([2460000.0, 2460001.0, 2460002.0]), jnp.zeros(3)).tdb()
        return jnp.asarray([True, True, False]), t_close, jnp.zeros(3, dtype=int)

    class FakeTarget:
        def state(self, tdb, frame=None):
            distance = jnp.asarray([0.01, 0.03, 0.04])
            zeros = jnp.zeros_like(distance)
            return SimpleNamespace(
                pos=jnp.stack([distance, zeros, zeros], axis=-1),
                vel=jnp.stack([zeros, distance, zeros], axis=-1),
            )

    class FakeCenter:
        def state(self, tdb, frame=None):
            zeros = jnp.zeros((3, 3))
            return SimpleNamespace(pos=zeros, vel=zeros)

    monkeypatch.setattr(ephemeris_core, "find_distance_extrema_single", fake_find_distance_extrema_single)

    table = ephemeris_core.find_close_approaches_single(
        Time.from_tdb_jd(2460000.0, 0.0).tdb(),
        Time.from_tdb_jd(2460003.0, 0.0).tdb(),
        FakeTarget(),
        FakeCenter(),
        max_distance=0.02,
        max_events=3,
    )

    assert captured["max_distance"] == 0.02
    assert jnp.count_nonzero(table.is_valid) == 1
    assert_allclose(table.valid.distance, jnp.asarray([0.01]), atol=0.0, rtol=0.0)


def test_close_approach_distance_filter_precedes_event_limit():
    t0 = 2460000.0
    amplitude = 0.3

    def relative_state(jd1, jd2):
        elapsed = jd1 + jd2 - t0
        baseline = 1.0 - 0.1 * elapsed
        phase = 2.0 * jnp.pi * elapsed
        radius = baseline + amplitude * jnp.cos(phase)
        radial_velocity = -0.1 - amplitude * 2.0 * jnp.pi * jnp.sin(phase)
        zeros = jnp.zeros_like(radius)
        return (
            jnp.stack([radius, zeros, zeros], axis=-1),
            jnp.stack([radial_velocity, zeros, zeros], axis=-1),
        )

    class FakeTarget:
        def _bcrs_pv_jd(self, jd1, jd2):
            return relative_state(jd1, jd2)

        def state(self, tdb, frame=None):
            pos, vel = relative_state(tdb.jd1, tdb.jd2)
            return SimpleNamespace(pos=pos, vel=vel)

    class FakeCenter:
        def _bcrs_pv_jd(self, jd1, jd2):
            shape = jnp.shape(jd1)
            return jnp.zeros((*shape, 3)), jnp.zeros((*shape, 3))

        def state(self, tdb, frame=None):
            shape = jnp.shape(tdb.jd1)
            return SimpleNamespace(pos=jnp.zeros((*shape, 3)), vel=jnp.zeros((*shape, 3)))

    table = ephemeris_core.find_close_approaches_single(
        Time.from_tdb_jd(t0, 0.0).tdb(),
        Time.from_tdb_jd(t0 + 6.0, 0.0).tdb(),
        FakeTarget(),
        FakeCenter(),
        max_distance=0.5,
        max_events=1,
    )

    assert jnp.count_nonzero(table.is_valid) == 1
    assert float(table.valid.t_close.jd[0]) > t0 + 2.0
    assert float(table.valid.distance[0]) <= 0.5
