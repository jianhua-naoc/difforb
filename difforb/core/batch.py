import numpy as np
import equinox as eqx
import jax
from jax import numpy as jnp
from abc import abstractmethod
from typing import Any, Callable, Tuple


def is_array(x: Any) -> bool:
    """
    Check if an object is a valid JAX or NumPy array.

    Parameters
    ----------
    x : Any
        Object to check.

    Returns
    -------
    bool
        True if the object is an instance of a JAX or NumPy array.
    """
    return isinstance(x, (jax.Array, jnp.ndarray, np.ndarray))


class BatchableObject(eqx.Module):
    """
    Base class for objects supporting automatic batch slicing and PyTree mapping.
    """

    def __getitem__(self, key: Any) -> 'BatchableObject':
        """
        Slice the object across its batch dimensions.

        Parameters
        ----------
        key : Any
            The slice key (index, slice object, or mask).

        Returns
        -------
        BatchableObject
            A new instance with sliced array leaves.
        """

        def _slice_leaf(x):
            if is_array(x) and x.ndim > 0:
                return x[key]
            return x

        return jax.tree_util.tree_map(_slice_leaf, self)

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """
        Return the batch shape of the object.
        """
        pass

    @property
    def ndim(self) -> int:
        """
        Return the number of batch dimensions.
        """
        return len(self.shape)

    def __len__(self) -> int:
        """
        Return the size of the leading batch dimension.
        """
        if self.ndim == 0:
            raise TypeError(f"len() of unsized object: {self.__class__.__name__}")
        return self.shape[0]


def get_tree_ndim(obj: Any) -> int:
    """
    Determine the dimensionality of a PyTree based on its leaves.

    Parameters
    ----------
    obj : Any
        PyTree or array-like object.

    Returns
    -------
    int
        Dimensionality of the object.
    """
    if hasattr(obj, "ndim"):
        return obj.ndim
    if hasattr(obj, "shape"):
        return len(obj.shape)
    leaves = jax.tree_util.tree_leaves(obj)
    if not leaves:
        return 0
    return jnp.ndim(leaves[0])


def safe_dispatch(single_func: Callable, core_ndims_in: Tuple, *args: Any) -> Any:
    """
    Point-wise dispatcher for PyTree arguments with automatic broadcasting.

    References
    ----------
    JAX Broadcaster Logic: https://jax.readthedocs.io/en/latest/notebooks/how-jax-thinks.html

    Parameters
    ----------
    single_func : Callable
        The core function to be executed on single-point data.
    core_ndims_in : Tuple
        Number of intrinsic (non-batch) dimensions for each input argument.
    *args : Any
        PyTree arguments to be batched.

    Returns
    -------
    Any
        Broadcasted and batched results in the original PyTree structure.
    """

    # 1. Normalize supported Python scalar leaves into JAX arrays before dispatch.
    def _ensure_array(leaf):
        if isinstance(leaf, (int, float, bool, complex)) and not isinstance(leaf, str):
            return jnp.asarray(leaf)
        return leaf

    args = tuple(jax.tree_util.tree_map(_ensure_array, arg) for arg in args)

    # 2. Extract and validate raw batch shapes.
    raw_batch_shapes = []
    for arg, c_dim in zip(args, core_ndims_in):
        ndim = get_tree_ndim(arg)
        # Reject inputs that do not provide the required intrinsic dimensions.
        if ndim < c_dim:
            raise ValueError(f"Insufficient dimensions: Argument requires "
                             f"{c_dim} intrinsic dimensions, but got {ndim}.")

        if hasattr(arg, "shape"):
            s = arg.shape
        else:
            leaves = jax.tree_util.tree_leaves(arg)
            s = jnp.shape(leaves[0]) if leaves else ()

        raw_batch_shapes.append(s[:-c_dim] if c_dim > 0 else s)

    max_batch_ndim = max((len(s) for s in raw_batch_shapes), default=0)

    # Fast scalar path: no mapping is needed.
    if max_batch_ndim == 0:
        return single_func(*args)

    # 3. Align broadcast dimensions.
    normalized_batch_shapes = []
    batch_axis_offsets = []
    for b_shape in raw_batch_shapes:
        diff = max_batch_ndim - len(b_shape)
        normalized_batch_shapes.append((1,) * diff + b_shape)
        batch_axis_offsets.append(diff)

    try:
        common_batch_shape = jnp.broadcast_shapes(*normalized_batch_shapes)
    except ValueError as e:
        raise ValueError(f"Broadcast alignment failed for batch shapes: {normalized_batch_shapes}") from e

    K = len(common_batch_shape)
    flat_args = []
    in_axes_args_per_k = [[] for _ in range(K)]

    # 4. Build zero-copy views and exact ``in_axes`` control trees.
    for arg, norm_b_shape, raw_b_shape, axis_offset in zip(
            args,
            normalized_batch_shapes,
            raw_batch_shapes,
            batch_axis_offsets,
    ):
        kept_b_shape = []
        for k in range(K):
            D_k = common_batch_shape[k]
            s_jk = norm_b_shape[k]
            has_real_axis = k >= axis_offset

            if s_jk == D_k:
                if has_real_axis:
                    kept_b_shape.append(D_k)

            # Build the control tree for this ``vmap`` level; defaults avoid late binding.
            def get_in_axes_leaf(leaf, s_jk=s_jk, D_k=D_k, has_real_axis=has_real_axis):
                if is_array(leaf):
                    if has_real_axis and s_jk == D_k:
                        return 0
                return None

            in_axes_args_per_k[k].append(jax.tree_util.tree_map(get_in_axes_leaf, arg))

        kept_b_shape = tuple(kept_b_shape)

        def prepare_leaf(leaf, raw_b_shape=raw_b_shape, kept_b_shape=kept_b_shape):
            if is_array(leaf):
                batch_ndim = len(raw_b_shape)
                if batch_ndim > 0:
                    # Fill missing batch axes for isolated core-array leaves.
                    if leaf.ndim < batch_ndim or leaf.shape[:batch_ndim] != raw_b_shape:
                        leaf = jnp.broadcast_to(leaf, raw_b_shape + leaf.shape)
                intrinsic_shape = leaf.shape[batch_ndim:]
                # Drop fake broadcast axes to keep the mapped view minimal.
                return leaf.reshape(kept_b_shape + intrinsic_shape)
            return leaf

        flat_args.append(jax.tree_util.tree_map(prepare_leaf, arg))

    # 5. Apply nested ``vmap``.
    vmapped_func = single_func
    for k in reversed(range(K)):
        vmapped_func = eqx.filter_vmap(vmapped_func, in_axes=tuple(in_axes_args_per_k[k]))

    res = vmapped_func(*flat_args)

    # 6. Restore the output shape.
    def reshape_output(leaf):
        if is_array(leaf):
            output_intrinsic_shape = jnp.shape(leaf)[K:]
            return leaf.reshape(common_batch_shape + output_intrinsic_shape)
        return leaf

    return jax.tree_util.tree_map(reshape_output, res)


def safe_cartesian_dispatch(single_func: Callable, *arg_groups: Tuple[Tuple, Tuple]) -> Any:
    """
    Cartesian product (Grid) dispatcher for N-dimensional argument groups.

    References
    ----------
    Equinox filter_vmap: https://docs.kidger.site/equinox/api/transformations/#equinox.filter_vmap

    Parameters
    ----------
    single_func : Callable
        The core function to be executed on single-point data.
    *arg_groups : Tuple[Tuple[int], Tuple[Any]]
        Variable number of groups. Each group is a tuple (core_ndims, arguments).

    Returns
    -------
    Any
        Results with combined leading batch dimensions from all groups.
    """

    def _ensure_array(leaf):
        if isinstance(leaf, (int, float, bool, complex)) and not isinstance(leaf, str):
            return jnp.asarray(leaf)
        return leaf

    processed_groups = []
    group_batch_shapes = []

    # Extract and validate the batch dimensions for each argument group.
    for core_ndims, args in arg_groups:
        args = tuple(jax.tree_util.tree_map(_ensure_array, arg) for arg in args)

        group_b_shape = ()
        arg_b_shapes = []  # Raw batch dimensions for each argument in this group.
        for arg, c_dim in zip(args, core_ndims):
            ndim = get_tree_ndim(arg)
            if ndim < c_dim:
                raise ValueError(f"Cartesian Dispatch Error: Argument requires "
                                 f"{c_dim} intrinsic dimensions, but got {ndim}.")

            if hasattr(arg, "shape"):
                s = arg.shape
            else:
                leaves = jax.tree_util.tree_leaves(arg)
                s = jnp.shape(leaves[0]) if leaves else ()

            b_shape = s[:-c_dim] if c_dim > 0 else s
            arg_b_shapes.append(b_shape)
            if len(b_shape) > 0:
                group_b_shape = b_shape

        processed_groups.append((core_ndims, args, arg_b_shapes))
        group_batch_shapes.append(group_b_shape)

    flat_args = []
    flat_core_ndims = []

    # Wrap groups into mutually orthogonal grid dimensions before calling ``safe_dispatch``.
    for i, (core_ndims, args, arg_b_shapes) in enumerate(processed_groups):
        left_pad = sum(len(b) for b in group_batch_shapes[:i])
        right_pad = sum(len(b) for b in group_batch_shapes[i + 1:])
        g_b_shape = group_batch_shapes[i]

        for arg, c_dim, raw_b_shape in zip(args, core_ndims, arg_b_shapes):
            def reshape_leaf(leaf, raw_b_shape=raw_b_shape, g_b_shape=g_b_shape, left_pad=left_pad, right_pad=right_pad):
                if is_array(leaf):
                    batch_ndim = len(raw_b_shape)

                    # 1. Ensure this leaf carries its own argument batch dimensions.
                    if batch_ndim > 0:
                        if leaf.ndim < batch_ndim or leaf.shape[:batch_ndim] != raw_b_shape:
                            leaf = jnp.broadcast_to(leaf, raw_b_shape + leaf.shape)

                    # 2. Slice after the argument batch dimensions to get the intrinsic shape.
                    intrinsic_shape = leaf.shape[batch_ndim:]

                    # 3. Broadcast within the group when the argument batch shape is smaller.
                    if raw_b_shape != g_b_shape:
                        leaf = jnp.broadcast_to(leaf, g_b_shape + intrinsic_shape)

                    # 4. Pad both sides with singleton axes so each group occupies orthogonal grid axes.
                    new_shape = (1,) * left_pad + g_b_shape + (1,) * right_pad + intrinsic_shape
                    return leaf.reshape(new_shape)
                return leaf

            flat_args.append(jax.tree_util.tree_map(reshape_leaf, arg))
            flat_core_ndims.append(c_dim)

    # Delegate the multi-level ``vmap`` mapping to the base safe dispatcher.
    return safe_dispatch(single_func, tuple(flat_core_ndims), *flat_args)
