"""Strategy preparation and dispatch for differential correction."""

from itertools import product
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from difforb.astrometry.data import ObservationData, ObservationLayout
from difforb.astrometry.reduction.photocenter import PhotocenterCorrection
from difforb.astrometry.weight import WeightPolicy
from difforb.dynamics.force_model import ForceModel
from difforb.integrator.integrator import NumericalIntegrator
from difforb.od.dc.core import (
    bind_linearization,
    solve_differential_correction_batch,
    solve_differential_correction_single,
)
from difforb.od.dc.lsq import LeastSquares, RobustLeastSquares, RobustResult
from difforb.od.dc.prediction import AstrometryMeasurementModel
from difforb.od.outlier.policy import CompiledOutlierPolicy, InteractiveOutlierPolicy
from difforb.od.progress import SolverReporter


DCStrategy = tuple[ForceModel, WeightPolicy, InteractiveOutlierPolicy]
DCStrategyResult = tuple[RobustResult, ForceModel]


class PreparedStrategyInputs(NamedTuple):
    """Array inputs prepared for one differential-correction strategy."""

    force_model: ForceModel
    base_optical_covariances: Float[Array, "N_optical 2 2"]
    optical_time_uncertainties: Float[Array, "N_optical"]
    radar_weights: Float[Array, "N_radar"]
    outlier_policy: CompiledOutlierPolicy
    init_params: Float[Array, "N_param"]


def expand_dc_strategies(
        force_model: ForceModel | list[ForceModel] | tuple[ForceModel, ...],
        weight_policy: WeightPolicy | list[WeightPolicy] | tuple[WeightPolicy, ...],
        outlier_policy: InteractiveOutlierPolicy | list[InteractiveOutlierPolicy] | tuple[InteractiveOutlierPolicy, ...],
        grid: bool,
) -> tuple[tuple[DCStrategy, ...], tuple[int, ...], bool]:
    """Expand scalar or sequence-valued strategy arguments."""
    values = []
    batched = []
    for value in (force_model, weight_policy, outlier_policy):
        is_batched = isinstance(value, (list, tuple))
        options = tuple(value) if is_batched else (value,)
        if not options:
            raise ValueError("Batched solver arguments must not be empty.")
        values.append(options)
        batched.append(is_batched)

    if not any(batched):
        return ((force_model, weight_policy, outlier_policy),), (), False

    if grid:
        shape = tuple(len(options) for options, explicit in zip(values, batched) if explicit)
        return tuple(product(*values)), shape, True

    size = max(map(len, values))
    lengths = tuple(map(len, values))
    if any(length not in (1, size) for length in lengths):
        raise ValueError(
            "Point-wise strategy broadcasting requires every sequence length to be 1 "
            f"or {size}; got {lengths}."
        )
    strategies = tuple(
        tuple(options[0 if len(options) == 1 else index] for options in values)
        for index in range(size)
    )
    return strategies, (size,), True


def prepare_dc_strategy(
        strategy: DCStrategy,
        data: ObservationData,
        layout: ObservationLayout,
        initial_state_params: Float[Array, "6"],
        photocenter_correction: PhotocenterCorrection,
) -> PreparedStrategyInputs:
    """Prepare array inputs for one differential-correction strategy."""
    force_model, weight_policy, outlier_policy = strategy
    weights = weight_policy.weights(data)
    init_params = jnp.concatenate([
        initial_state_params,
        force_model.get_all_estimated_params(),
        photocenter_correction.get_estimated_params(),
    ])
    return PreparedStrategyInputs(
        force_model,
        jnp.asarray(weights.optical_covariances),
        jnp.asarray(weights.optical_time_uncertainties),
        jnp.asarray(weights.radar_weights),
        outlier_policy.compiled(layout),
        init_params,
    )


def dispatch_dc_strategies(
        prepared_inputs: tuple[PreparedStrategyInputs, ...],
        is_batched: bool,
        measure_model: AstrometryMeasurementModel,
        integrator: NumericalIntegrator,
        least_squares: LeastSquares,
        batch_size: int | None,
        reporter: SolverReporter | None,
) -> tuple[DCStrategyResult, ...]:
    """Solve prepared strategies sequentially or in compatible mapped batches."""
    mapped = is_batched and least_squares.solver_jit
    if not mapped:
        robust_solver = RobustLeastSquares(least_squares)
        compiled = least_squares.solver_jit
        results = []
        for inputs in prepared_inputs:
            if compiled:
                result = solve_differential_correction_single(
                    inputs.base_optical_covariances,
                    inputs.optical_time_uncertainties,
                    inputs.radar_weights,
                    inputs.outlier_policy,
                    inputs.init_params,
                    inputs.force_model,
                    measure_model=measure_model,
                    integrator=integrator,
                    solver_options=least_squares.options,
                )
            else:
                linearize = bind_linearization(
                    measure_model,
                    inputs.force_model,
                    integrator,
                    inputs.base_optical_covariances,
                    inputs.optical_time_uncertainties,
                    inputs.radar_weights,
                )
                result = robust_solver.solve(
                    inputs.init_params,
                    inputs.outlier_policy,
                    linearize,
                    verbose=False if reporter is None else reporter,
                )
            results.append((result, inputs.force_model))
        return tuple(results)

    groups = {}
    for index, inputs in enumerate(prepared_inputs):
        force_arrays = tuple(
            (leaf.shape, leaf.dtype)
            for leaf in jax.tree_util.tree_leaves(inputs.force_model)
            if eqx.is_array(leaf)
        )
        signature = (
            jax.tree_util.tree_structure(inputs.force_model),
            force_arrays,
            inputs.force_model.get_all_estimated_params().shape,
            jax.tree_util.tree_structure(inputs.outlier_policy),
        )
        groups.setdefault(signature, []).append((index, inputs))

    results = [None] * len(prepared_inputs)
    for group in groups.values():
        chunk_size = len(group) if batch_size is None else batch_size
        for start in range(0, len(group), chunk_size):
            chunk = group[start:start + chunk_size]
            chunk_force_models = tuple(item[1].force_model for item in chunk)
            force_leaves = [jax.tree_util.tree_flatten(model)[0] for model in chunk_force_models]
            force_tree = jax.tree_util.tree_structure(chunk_force_models[0])
            stacked_leaves = []
            force_axes_leaves = []
            for leaves in zip(*force_leaves):
                first = leaves[0]
                if eqx.is_array(first) and not all(leaf is first for leaf in leaves[1:]):
                    stacked_leaves.append(jnp.stack(leaves))
                    force_axes_leaves.append(0)
                else:
                    stacked_leaves.append(first)
                    force_axes_leaves.append(None)
            force_models = jax.tree_util.tree_unflatten(force_tree, stacked_leaves)
            force_model_axes = jax.tree_util.tree_unflatten(force_tree, force_axes_leaves)
            outlier_policies = jax.tree_util.tree_map(
                lambda *leaves: jnp.stack(leaves),
                *(item[1].outlier_policy for item in chunk),
            )

            mapped_results = solve_differential_correction_batch(
                jnp.stack([item[1].base_optical_covariances for item in chunk]),
                jnp.stack([item[1].optical_time_uncertainties for item in chunk]),
                jnp.stack([item[1].radar_weights for item in chunk]),
                outlier_policies,
                jnp.stack([item[1].init_params for item in chunk]),
                force_models,
                force_model_axes,
                measure_model,
                integrator,
                least_squares.options,
            )
            jax.block_until_ready(mapped_results)
            for local_index, item in enumerate(chunk):
                results[item[0]] = (
                    jax.tree_util.tree_map(
                        lambda leaf: leaf[local_index], mapped_results,
                    ),
                    item[1].force_model,
                )

    return tuple(results)
