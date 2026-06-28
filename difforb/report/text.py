"""Shared plain-text formatting helpers for object repr output."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


_ARRAY_THRESHOLD = 6
_ARRAY_EDGEITEMS = 1
_MAX_LINE_WIDTH = 10**9


def _as_numpy(value) -> np.ndarray:
    return np.asarray(value)


def format_shape(shape: tuple[int, ...]) -> str:
    """Format a batch shape using Python tuple syntax."""
    return str(tuple(shape))


def format_float_scalar(value, precision: int = 15, scientific: bool = True, signed: bool = True) -> str:
    """Format a scalar float for repr output."""
    scalar = float(_as_numpy(value).item())
    if scientific:
        fmt = f"{{:{'+' if signed else ''}.{precision}E}}"
    else:
        fmt = f"{{:.{precision}f}}"
    return fmt.format(scalar)


def format_int_scalar(value) -> str:
    """Format a scalar integer for repr output."""
    return str(int(_as_numpy(value).item()))


def _array_to_string(arr: np.ndarray) -> str:
    rendered = np.array2string(
        arr,
        separator=", ",
        threshold=_ARRAY_THRESHOLD,
        edgeitems=_ARRAY_EDGEITEMS,
        max_line_width=_MAX_LINE_WIDTH,
        formatter={"all": lambda x: x},
    )
    return " ".join(rendered.splitlines())


def format_float_array(value, precision: int = 15, scientific: bool = True, signed: bool = True) -> str:
    """Format a float scalar or array using a stable truncated representation."""
    arr = _as_numpy(value)
    if arr.ndim == 0:
        return format_float_scalar(arr.item(), precision=precision, scientific=scientific, signed=signed)

    formatted = np.vectorize(
        lambda x: format_float_scalar(x, precision=precision, scientific=scientific, signed=signed),
        otypes=[object],
    )(arr)
    return _array_to_string(formatted)


def format_int_array(value) -> str:
    """Format an integer scalar or array using a stable truncated representation."""
    arr = _as_numpy(value)
    if arr.ndim == 0:
        return format_int_scalar(arr.item())

    formatted = np.vectorize(lambda x: format_int_scalar(x), otypes=[object])(arr)
    return _array_to_string(formatted)


def format_string_array(value, quote: bool = True) -> str:
    """Format a string scalar or array using a stable truncated representation."""
    arr = np.asarray(value, dtype=object)

    def _format_string(item) -> str:
        string = str(item)
        return repr(string) if quote else string

    if arr.ndim == 0:
        return _format_string(arr.item())

    formatted = np.vectorize(_format_string, otypes=[object])(arr)
    return _array_to_string(formatted)


def format_count(current: int, total: int) -> str:
    """Format a count pair like ``3/5``."""
    return f"{int(current)}/{int(total)}"


def format_class_name_array(values) -> str:
    """Format a sequence of objects using their class names."""
    class_names = [value.__class__.__name__ for value in values]
    return format_string_array(class_names, quote=False)


def build_repr(class_name: str, fields: Iterable[tuple[str, Optional[str]]]) -> str:
    """Assemble a canonical one-line repr string."""
    parts = [f"{key}={value}" for key, value in fields if value is not None]
    if not parts:
        return f"<{class_name}>"
    return f"<{class_name} {' '.join(parts)}>"


def format_optional_shape(shape: tuple[int, ...], include_scalar: bool = True) -> Optional[str]:
    """Format shape or return None when scalar shapes should be omitted."""
    if shape == () and not include_scalar:
        return None
    return format_shape(shape)
