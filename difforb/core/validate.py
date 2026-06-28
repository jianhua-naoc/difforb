"""Validation helpers for runtime scientific input contracts."""

from typing import Type, Any, Tuple, Optional

import jax.numpy as jnp


def _scalar_item(name: str, value: Any):
    """Return one Python scalar from a scalar-like input."""
    arr = jnp.asarray(value)
    if arr.shape != ():
        raise ValueError(f"{name} must be a scalar.")
    return arr.item()


def coerce_scalar_float(name: str, value: Any) -> float:
    """Validate and convert one scalar-like input to a Python ``float``."""
    return float(_scalar_item(name, value))


def coerce_optional_scalar_float(name: str, value: Any | None) -> Optional[float]:
    """Validate and convert one optional scalar-like input to a Python ``float``."""
    if value is None:
        return None
    return coerce_scalar_float(name, value)


def coerce_scalar_int(name: str, value: Any) -> int:
    """Validate and convert one scalar-like input to a Python ``int``."""
    item = _scalar_item(name, value)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ValueError(f"{name} must be an integer.")
    return int(item)


def coerce_scalar_bool(name: str, value: Any) -> bool:
    """Validate and convert one scalar-like input to a Python ``bool``."""
    item = _scalar_item(name, value)
    if not isinstance(item, bool):
        raise ValueError(f"{name} must be a boolean.")
    return bool(item)


def validate_timeview(obj: Any, expected_cls: Type, arg_name: str):
    """
    Validates that the input object is of the expected timescale class.
    """
    if not isinstance(obj, expected_cls):
        raise TypeError(
            f"Invalid time view: argument `{arg_name}` must be an instance of "
            f"`{expected_cls.__name__}`, but got `{type(obj).__name__}`. "
        )


def validate_timeview_union(obj: Any, expected_classes: Tuple[Type, ...], arg_name: str):
    if not isinstance(obj, expected_classes):
        class_names = " or ".join([c.__name__ for c in expected_classes])
        raise TypeError(
            f"Invalid time view: `{arg_name}` must be an instance of `{class_names}`, "
            f"but got `{type(obj).__name__}`. "
        )


def validate_initial_orbit(obj: Any, expected_cls: Type, arg_name: str):
    """Validate that an initial orbit matches the expected runtime class.

    Parameters
    ----------
    obj : Any
        Object passed as an initial orbit.
    expected_cls : Type
        Required concrete orbit class.
    arg_name : str
        Name of the validated argument in the caller.

    Raises
    ------
    TypeError
        If ``obj`` is not an instance of ``expected_cls``.
    """
    if not isinstance(obj, expected_cls):
        raise TypeError(
            f"Invalid initial orbit: argument `{arg_name}` must be an instance of "
            f"`{expected_cls.__name__}`, but got `{type(obj).__name__}`."
        )


def validate_initial_orbit_union(obj: Any, expected_classes: Tuple[Type, ...], arg_name: str):
    """Validate that an initial orbit matches one of the supported runtime classes.

    Parameters
    ----------
    obj : Any
        Object passed as an initial orbit.
    expected_classes : tuple of Type
        Allowed concrete orbit classes.
    arg_name : str
        Name of the validated argument in the caller.

    Raises
    ------
    TypeError
        If ``obj`` is not an instance of any class in ``expected_classes``.
    """
    if not isinstance(obj, expected_classes):
        class_names = " or ".join([c.__name__ for c in expected_classes])
        raise TypeError(
            f"Invalid initial orbit: `{arg_name}` must be an instance of `{class_names}`, "
            f"but got `{type(obj).__name__}`."
        )
