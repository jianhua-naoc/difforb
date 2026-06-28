"""Lagrange interpolation helpers for ``EOP`` tables.

This module implements the 4-point Lagrange interpolation kernel used by :mod:`difforb.core.eop.container`. The scalar kernel follows the IERS reference windowing rule, and the public helper applies the same scheme to batched query shapes.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from difforb.core.batch import safe_dispatch


def lagrangian_interpolate_single(x: Float[Array, "n"], y: Float[Array, "n"], xint: Float[Array, ""]) -> Float[Array, ""]:
    """Interpolate one scalar query with a 4-point Lagrange scheme.

    Parameters
    ----------
    x : Float[Array, "n"]
        Monotonic sample coordinates.
    y : Float[Array, "n"]
        Sample values for ``x``.
    xint : Float[Array, ""]
        Scalar query point in the same unit as ``x``.

    Returns
    -------
    Float[Array, ""]
        Interpolated value at ``xint``.

    References
    ----------
    1. IERS interpolation reference code, ``interp.f``, https://hpiers.obspm.fr/iers/models/interp.f
    """
    n = x.shape[0]

    # Find the table interval around the query.
    idx = jnp.searchsorted(x, xint, side="right") - 1

    # Choose a valid 4-point window.
    raw_start_idx = idx - 1

    # Keep the window inside the table.
    start_idx = jnp.clip(raw_start_idx, 0, n - 4)

    # Read the local window with a fixed shape.
    x_window = jax.lax.dynamic_slice(x, (start_idx,), (4,))
    y_window = jax.lax.dynamic_slice(y, (start_idx,), (4,))

    # Build the Lagrange basis terms.
    xm = x_window[:, None]
    xj = x_window[None, :]
    denom = xm - xj
    numer = xint - xj

    # Skip the diagonal terms in the product.
    eye_mask = jnp.eye(4, dtype=bool)
    denom_safe = jnp.where(eye_mask, 1.0, denom)
    numer_safe = jnp.where(eye_mask, 1.0, numer)
    basis_polys = jnp.prod(numer_safe / denom_safe, axis=1)
    yout = jnp.sum(y_window * basis_polys)

    return yout


@jax.jit
def lagrangian_interpolate(x: Float[Array, "n"], y: Float[Array, "n"], xint: Float[Array, "..."]) -> Float[Array, "..."]:
    """Interpolate query points with a 4-point Lagrange scheme.

    Parameters
    ----------
    x : Float[Array, "n"]
        Monotonic sample coordinates.
    y : Float[Array, "n"]
        Sample values for ``x``.
    xint : Float[Array, "..."]
        Query points in the same unit as ``x``.

    Returns
    -------
    Float[Array, "..."]
        Interpolated values with the same shape as ``xint``.

    Notes
    -----
    Query points are clamped to the sample range before interpolation.
    Vectorize :func:`lagrangian_interpolate_single`.
    """
    xint_clamped = jnp.clip(xint, x[0], x[-1])
    wrapper = lambda _xint: lagrangian_interpolate_single(x, y, _xint)
    return safe_dispatch(wrapper, (0,), xint_clamped)
