"""Batch contracts for ordinary arrays and shared dynamic field subtrees."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difforb.core.batch import BatchableObject, safe_cartesian_dispatch, safe_dispatch
from difforb.core.eop.container import EarthOrientationData
from difforb.core.state.frame import BCRS
from difforb.core.state.state import State
from difforb.core.time.timescale import Time


class Samples(BatchableObject):
    values: jax.Array
    shared: dict = eqx.field(metadata={"batch_shared": True})
    label: str = eqx.field(static=True, default="samples")

    @property
    def shape(self):
        return self.values.shape


def make_eop(offset=0.):
    days = jnp.arange(8., dtype=jnp.float64)
    return EarthOrientationData(
        60000. + days, .1 + days * .01 + offset, .2 + days * .01,
        days * 0. - .1, days * 0., days * 0., jnp.array([2460000.5, 2460007.5]),
    )


def assert_eop_equal(actual, expected):
    for x, y in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("left,right", [
    ((), ()), ((), (3,)), ((3,), ()), ((3,), (3,)),
    ((2, 1), (1, 3)), ((2, 1, 3), (4, 1)), ((0,), (1,)),
])
def test_pointwise_ordinary_array_broadcast(left, right):
    x = jnp.arange(np.prod(left), dtype=float).reshape(left)
    y = jnp.arange(np.prod(right) * 3, dtype=float).reshape(right + (3,))
    result = safe_dispatch(lambda a, b: {"value": a + b, "label": "fixed"}, (0, 1), x, y)
    expected = np.broadcast_to(np.asarray(x)[..., None], np.broadcast_shapes(left, right) + (3,)) + y
    np.testing.assert_array_equal(result["value"], expected)
    assert result["label"] == "fixed"


def test_pointwise_preserves_intrinsic_array_broadcast():
    # The original dispatcher broadcasts a field lacking the object's batch prefix.
    class Vectors(BatchableObject):
        scale: jax.Array
        direction: jax.Array

        @property
        def shape(self):
            return self.scale.shape

    obj = Vectors(jnp.arange(2.), jnp.arange(3.))
    result = safe_dispatch(lambda s: s.scale * s.direction, (0,), obj)
    np.testing.assert_array_equal(result, obj.scale[:, None] * obj.direction)


@pytest.mark.parametrize("shape", [(), (4,), (2, 1), (2, 3), (0,)])
def test_shared_input_and_output_with_jit_and_forward_ad(shape):
    values = jnp.arange(np.prod(shape), dtype=float).reshape(shape)
    shared = {"table": jnp.arange(4.), "nested": (jnp.arange(6.).reshape(2, 3), None)}
    samples = Samples(values, shared)

    @jax.jit
    def evaluate(obj):
        def single(s):
            table = s.shared["table"]
            output_shared = {"table": table + 1., "nested": s.shared["nested"]}
            return {"samples": Samples(s.values * 2. + table.sum(), output_shared)}
        return safe_dispatch(single, (0,), obj)

    result = evaluate(samples)["samples"]
    np.testing.assert_array_equal(result.values, values * 2. + 6.)
    np.testing.assert_array_equal(result.shared["table"], shared["table"] + 1.)
    np.testing.assert_array_equal(result.shared["nested"][0], shared["nested"][0])
    assert result.shared["nested"][1] is None
    assert result.label == samples.label
    tangent = jax.jvp(evaluate, (samples,), (Samples(jnp.ones_like(values), jax.tree_util.tree_map(jnp.zeros_like, shared)),))[1]
    np.testing.assert_array_equal(tangent["samples"].values, jnp.full_like(values, 2.))
    shared_tangent = Samples(jnp.zeros_like(values), jax.tree_util.tree_map(jnp.ones_like, shared))
    tangent = jax.jvp(evaluate, (samples,), (shared_tangent,))[1]["samples"]
    np.testing.assert_array_equal(tangent.values, jnp.full_like(values, 4.))
    np.testing.assert_array_equal(tangent.shared["table"], jnp.ones(4))


def test_shared_output_cannot_depend_on_a_batch_row():
    obj = Samples(jnp.arange(4.), {"table": jnp.arange(4.)})
    with pytest.raises(ValueError, match="out_axes"):
        safe_dispatch(lambda s: Samples(s.values, {"table": s.shared["table"] + s.values}), (0,), obj)


@pytest.mark.parametrize("left,right", [
    ((), (3,)), ((2,), (3,)), ((2, 1), (3,)), ((0,), (3,)),
])
def test_cartesian_group_order_and_shared_outputs(left, right):
    x = Samples(jnp.arange(np.prod(left), dtype=float).reshape(left), {"table": jnp.arange(3.)})
    y = Samples(jnp.arange(np.prod(right), dtype=float).reshape(right), {"table": jnp.arange(3.) + 10.})
    def single(a, b, scale):
        return Samples(a.values + b.values * scale, a.shared), Samples(b.values, b.shared)
    first, second = safe_cartesian_dispatch(single, ((0,), (x,)), ((0, 0), (y, 2.)))
    expected = x.values.reshape(left + (1,) * len(right)) + y.values.reshape((1,) * len(left) + right) * 2.
    np.testing.assert_array_equal(first.values, expected)
    assert first.shape == left + right
    np.testing.assert_array_equal(first.shared["table"], x.shared["table"])
    np.testing.assert_array_equal(second.shared["table"], y.shared["table"])


def test_broadcast_errors_remain_errors():
    with pytest.raises(ValueError, match="Broadcast alignment failed"):
        safe_dispatch(lambda a, b: a + b, (0, 0), jnp.zeros(2), jnp.zeros(3))
    with pytest.raises(ValueError, match="Insufficient dimensions"):
        safe_dispatch(lambda a: a, (1,), jnp.asarray(1.))


@pytest.mark.parametrize("index", [1, slice(1, 4, 2), np.array([3, 1]), np.array([True, False, True, False])])
def test_nested_time_state_slicing_keeps_complete_eop(index):
    eop = make_eop()
    time = Time.from_tt_jd(jnp.full(4, 2460003.5), jnp.arange(4.) * .1, eop=eop)
    state = State(time.tdb(), jnp.arange(12.).reshape(4, 3), jnp.ones((4, 3)), BCRS)
    for original in (time, time.tt, state):
        sliced = original[index]
        inner_time = sliced if isinstance(sliced, Time) else (sliced.time if hasattr(sliced, "time") else sliced.tdb.time)
        assert_eop_equal(inner_time.eop, eop)
        np.testing.assert_array_equal(inner_time.tt.jd2, time.tt.jd2[index])
    np.testing.assert_array_equal(state[index].pos, state.pos[index])


@pytest.mark.parametrize("count", [1, 2, 8])
def test_time_batch_including_eop_length_collision(count):
    eop = make_eop()
    time = Time.from_tt_jd(jnp.full(count, 2460003.5), jnp.arange(count) * .01, eop=eop)
    result, pole = safe_dispatch(lambda t: (t.tt, t.xpole), (0,), time)
    np.testing.assert_array_equal(pole, time.xpole)
    np.testing.assert_array_equal(result.jd2, time.tt.jd2)
    assert_eop_equal(result.time.eop, eop)


def test_time_grid_retains_distinct_eop_tables():
    left = Time.from_tt_jd(jnp.full((2, 1), 2460003.5), jnp.array([[.1], [.2]]), eop=make_eop())
    right = Time.from_tt_jd(jnp.full(3, 2460003.5), jnp.array([.1, .2, .3]), eop=make_eop(.3))
    @eqx.filter_jit
    def evaluate(a, b):
        return safe_cartesian_dispatch(
            lambda x, y: (x.tt, y.tt, x.xpole + y.xpole), ((0,), (a,)), ((0,), (b,)),
        )

    a, b, pole = evaluate(left, right)
    assert a.shape == b.shape == pole.shape == (2, 1, 3)
    np.testing.assert_array_equal(pole, left.xpole[..., None] + right.xpole[None, None, :])
    np.testing.assert_array_equal(a.jd2, jnp.broadcast_to(left.tt.jd2[..., None], (2, 1, 3)))
    np.testing.assert_array_equal(b.jd2, jnp.broadcast_to(right.tt.jd2, (2, 1, 3)))
    assert_eop_equal(a.time.eop, left.eop)
    assert_eop_equal(b.time.eop, right.eop)


def test_nested_jit_accepts_replaced_eop_without_static_metadata_comparison():
    traces = []
    @eqx.filter_jit
    def query(t):
        traces.append(True)
        return t.xpole

    first = make_eop()
    second = make_eop(.3)
    query(Time.from_tt_jd(2460003.5, .1, eop=first))

    @eqx.filter_jit
    def outer(eop, offset):
        return query(Time.from_tt_jd(2460003.5, offset, eop=eop))

    for eop in (first, second):
        actual = outer(eop, jnp.asarray(.1))
        np.testing.assert_array_equal(actual, Time.from_tt_jd(2460003.5, .1, eop=eop).xpole)
    assert len(traces) == 1


def test_time_device_placement_includes_eop():
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("use JAX_NUM_CPU_DEVICES=2 to exercise device migration")
    time = jax.device_put(Time.from_tt_jd(2460003.5, .1, eop=make_eop()), devices[0])
    placed = jax.device_put(time, devices[1])
    assert len(jax.tree_util.tree_leaves(placed)) == 11
    for leaf in jax.tree_util.tree_leaves(placed):
        assert leaf.devices() == {devices[1]}
    assert time.eop.tt_jds.devices() == {devices[0]}
