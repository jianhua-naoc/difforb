"""Array placement for mixed Equinox PyTrees."""

import equinox as eqx
import jax


def put_arrays(tree, device: jax.Device):
    """Place array leaves on one device, preserving dtypes and non-array leaves."""
    arrays, static = eqx.partition(tree, eqx.is_array)
    return eqx.combine(jax.device_put(arrays, device), static)
