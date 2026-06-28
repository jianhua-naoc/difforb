from typing import NamedTuple

from jax import Array
from jaxtyping import Int

from difforb.body.smallbody import Orbit
from difforb.report.display_units import orbit_element_specs, repr_fields_from_specs, STATE_REPR_SPECS
from difforb.report.text import build_repr, format_float_array, format_int_array
from difforb.core.state.state import State


class IODResult(NamedTuple):
    """Initial orbit solution summary.

    The stored orbit keeps its canonical internal units. Human-facing display
    layers append unit suffixes to labels instead of changing runtime
    attribute names. ``used_indices`` stores the original observation indices
    of the selected triplet in the mixed input order of the source
    :class:`difforb.astrometry.data2.ObservationData`.
    """

    initial_orbit: Orbit
    iter_num: int
    err: float
    used_indices: Int[Array, "3"]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(getattr(self.initial_orbit, "shape", ()))

    def __repr__(self) -> str:
        orbit = self.initial_orbit
        if isinstance(orbit, State):
            orbit_type = orbit.frame.name or f"{orbit.frame.origin.value}+{orbit.frame.axes.value}"
        else:
            orbit_type = orbit.__class__.__name__
        orbit_fields = [("orbit_type", orbit_type)]
        if "Kep" in orbit.__class__.__name__:
            orbit_fields.extend(repr_fields_from_specs(orbit, orbit_element_specs(orbit)))
        else:
            orbit_fields.extend(repr_fields_from_specs(orbit, STATE_REPR_SPECS))

        return build_repr(
            self.__class__.__name__,
            [
                ("epoch_jd", format_float_array(orbit.tdb.jd, precision=9, scientific=False, signed=False)),
                *orbit_fields,
                ("err_rad", format_float_array(self.err)),
                ("iters", str(self.iter_num)),
                ("indices", format_int_array(self.used_indices)),
            ],
        )
