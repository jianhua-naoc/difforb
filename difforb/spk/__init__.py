"""SPK ephemeris access and default-kernel management.

This module exposes one small process-local registry for the default
``Ephemeris`` instance used by high-level helpers such as
``difforb.body.EphemerisBody``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from difforb.core.time.timescale import TDBView
    from difforb.spk.spk import Ephemeris

_DEFAULT_EPHEMERIS: Optional["Ephemeris"] = None


def set_default_ephemeris(
        ephemeris: "Ephemeris | str | Sequence[str]",
        *,
        load_window: "tuple[TDBView, TDBView] | None" = None,
) -> "Ephemeris":
    """Register one process-local default ephemeris.

    Parameters
    ----------
    ephemeris : Ephemeris or str or sequence[str]
        Default ephemeris source. This may be an already constructed
        :class:`Ephemeris` object, one kernel path, or a sequence of kernel
        paths.
    load_window : tuple[TDBView, TDBView], optional
        Inclusive scalar ``TDB`` load window forwarded to :class:`Ephemeris`
        when ``ephemeris`` is a path or path sequence. This argument must be
        omitted when ``ephemeris`` is already an :class:`Ephemeris` object.

    Returns
    -------
    Ephemeris
        The registered default ephemeris object.

    Raises
    ------
    ValueError
        If ``load_window`` is provided together with an existing
        :class:`Ephemeris` object.
    """
    Ephemeris = getattr(import_module("difforb.spk.spk"), "Ephemeris")
    global _DEFAULT_EPHEMERIS
    if isinstance(ephemeris, Ephemeris):
        if load_window is not None:
            raise ValueError(
                "`load_window` cannot be provided when registering an existing Ephemeris object."
            )
        _DEFAULT_EPHEMERIS = ephemeris
    else:
        _DEFAULT_EPHEMERIS = Ephemeris(ephemeris, load_window=load_window)
    return _DEFAULT_EPHEMERIS


def clear_default_ephemeris() -> None:
    """Clear the process-local default ephemeris."""
    global _DEFAULT_EPHEMERIS
    _DEFAULT_EPHEMERIS = None


def load_default_ephemeris() -> "Ephemeris":
    """Return the process-local default ephemeris."""
    global _DEFAULT_EPHEMERIS
    if _DEFAULT_EPHEMERIS:
        return _DEFAULT_EPHEMERIS
    else:
        raise RuntimeError(
            "No default ephemeris set. Please set a default ephemeris by 'difforb.spk.set_default_ephemeris'.")


__all__ = [
    "Ephemeris",
    "clear_default_ephemeris",
    "load_default_ephemeris",
    "set_default_ephemeris",
]


def __getattr__(name):
    if name == "Ephemeris":
        value = getattr(import_module("difforb.spk.spk"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
