# Batch Inputs And Shapes

DiffOrb uses the same user interfaces for scalar and batch calculations. You do not switch to a separate batch API.

## One Interface

The same operation can work on one case or many cases. A scalar input stays scalar. A batch input produces a batch
output. A scalar input can also be combined with a batch input; in that case, the scalar value is reused for each case
in the batch.

Most DiffOrb [Guides](../guides/index.md) and [Workflows](../workflows/index.md) start with scalar code. The same
pattern usually works when scalar values are replaced by arrays or batched objects.

## Two Kinds Of Inputs

Batch-aware interfaces usually receive two kinds of inputs.

- Numerical inputs are plain values such as a scalar number, a Python list, a NumPy array, or a JAX array. For
  scalar-valued quantities, the array shape is the batch shape. For vector-valued quantities, such as Cartesian
  positions, the component axis belongs to one value and is not counted as batch shape.
- DiffOrb objects are registered as JAX PyTrees. Their numerical fields carry arrays, so the object can also carry
  batch data.

## PyTrees

Many DiffOrb objects are JAX and Equinox PyTrees. A PyTree object has two kinds of fields.

- PyTree leaves are data fields. JAX can map over them. In DiffOrb, they are usually arrays, such as times, positions,
  velocities, or computed values.
- Static fields are settings. JAX does not map over them. Examples include frame labels, ephemeris settings, solver
  settings, and policy objects.

During batch dispatch, DiffOrb maps over PyTree leaves, not static fields.

### PyTree Object Shape

A PyTree object's `.shape` is defined by its PyTree leaves. It tells how many cases are stored in the object. It does
not include component axes inside one case.

For a `State`, `pos` and `vel` include the Cartesian component axis. If `pos.shape == (N, 3)`, the final `3` stores
`x, y, z` for one state, and `state.shape` is `(N,)`. For a higher-level object, such as a small body or an observer
site, the shape comes from the stored state or geometry.

Examples:

| Object data | Object `.shape` | Meaning |
| --- | --- | --- |
| one epoch | `()` | one scalar time |
| `N` epochs | `(N,)` | `N` times |
| one Cartesian state with `pos.shape == (3,)` | `()` | one state |
| `N` Cartesian states with `pos.shape == (N, 3)` | `(N,)` | `N` states |

Objects can also carry more than one batch dimension. A shape such as `(N, M)` means the object stores an `N` by `M`
array of cases.

## Point-Wise Broadcasting

Point-wise broadcasting is the default batch mode. It follows the usual broadcasting behavior in NumPy and JAX.

Numerical inputs and DiffOrb objects can be mixed in one point-wise call. If an object has shape `()` and a numerical
input has shape `(N,)`, the scalar object is reused and the result has shape `(N,)`. If an object has shape `(N,)` and a
numerical input is scalar, the scalar value is reused for each object case.

If one input has shape `(N,)` and another input has shape `(N,)`, output row `i` uses input row `i` from both inputs. If
one input is scalar, DiffOrb reuses it for all `N` rows. Compatible shapes follow the usual JAX and NumPy broadcasting
rules.

This mode is useful when each row describes one complete case: one target with one epoch, one site with one time, or one
state with one matching time.

## Cartesian-Product Grids

Cartesian-product mode is for every combination of the inputs. It is useful when you want all targets against all
observers at all times, instead of matching row `i` with row `i`.

Grid dimensions use this order:

1. Target.
2. Observer position or observer geometry.
3. Time.

Only the grid dimensions used by a calculation appear in the output shape. A calculation with targets and times returns
dimensions in target-time order. A calculation with observers and times returns dimensions in observer-time order.

Inputs that belong to the same grid dimension are aligned before DiffOrb builds the grid. In a radar calculation, the
receiver, transmitter, and transmit frequency all belong to the observer-geometry dimension. DiffOrb aligns them first,
then combines that observer geometry with target and time dimensions.

As a shape example, target shape `(2,)`, observer shape `(3,)`, and time shape `(4,)` produce grid shape `(2, 3, 4)`.

## How Dispatch Works

DiffOrb dispatch starts by reading the batch shape of each argument. For a numerical input, this comes from the array
shape after removing any trailing dimensions that store one physical value. For a PyTree object, this comes from the
object `.shape`.

For a point-wise call, DiffOrb checks that the input shapes can broadcast. It then uses JAX `vmap` to map the scalar
calculation over the broadcast shape. This lets the mapped cases run in parallel on JAX-supported backends.

For a Cartesian-product call, DiffOrb first separates inputs into target, observer-geometry, and time dimensions. It
broadcasts inputs inside each dimension. Then it reshapes those dimensions and uses the same `vmap` mechanism so the
final output has target dimensions first, observer dimensions second, and time dimensions last.

This explains two common results:

- Shape errors usually mean that two inputs cannot be aligned.
- Vector-valued fields have one extra trailing dimension after the batch shape.

## Read Next

- Use [Use Batch Inputs Across DiffOrb APIs](../guides/use-batch-inputs-across-difforb-apis.md) for concrete interfaces
  and runnable examples.
- Read [Frames And State Representation](frames-and-state-representation.md) for the shape rules behind Cartesian
  vector fields.
- Use the [State API](../api/state.md) and [Time API](../api/time.md) for details on common
  batched objects.
