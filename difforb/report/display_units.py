"""Shared display-label metadata for repr and rich output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from difforb.core.time import tdb
from difforb.report.text import format_float_array

DisplayAccessor = str | Callable[[Any], Any]
DisplayTransform = Callable[[Any], Any]


@dataclass(frozen=True)
class DisplayFieldSpec:
    """Describe how one semantic field should be rendered for humans."""

    key: str
    label: str
    accessor: DisplayAccessor
    transform: DisplayTransform | None = None
    scientific: bool = True
    signed: bool = True


def _get_value(source: Any, accessor: DisplayAccessor) -> Any:
    if callable(accessor):
        return accessor(source)
    return getattr(source, accessor)


def resolve_display_value(source: Any, spec: DisplayFieldSpec) -> Any:
    """Extract and transform a display value according to the field spec."""
    value = _get_value(source, spec.accessor)
    if spec.transform is not None:
        return spec.transform(value)
    return value


def repr_field_from_spec(source: Any, spec: DisplayFieldSpec) -> tuple[str, str]:
    """Format one repr field using shared display metadata."""
    return (
        spec.label,
        format_float_array(
            resolve_display_value(source, spec),
            scientific=spec.scientific,
            signed=spec.signed,
        ),
    )


def repr_fields_from_specs(source: Any, specs: Iterable[DisplayFieldSpec]) -> list[tuple[str, str]]:
    """Format multiple repr fields from a shared spec list."""
    return [repr_field_from_spec(source, spec) for spec in specs]


def prefixed_repr_fields_from_specs(
        source: Any,
        specs: Iterable[DisplayFieldSpec],
        prefix: str,
) -> list[tuple[str, str]]:
    """Format repr fields while prefixing the shared display labels."""
    fields = []
    for spec in specs:
        label, value = repr_field_from_spec(source, spec)
        fields.append((f"{prefix}_{label}", value))
    return fields


def _rad_to_deg(value: Any) -> Any:
    return np.rad2deg(value)


STATE_REPR_SPECS: tuple[DisplayFieldSpec, ...] = (
    DisplayFieldSpec("pos", "pos_au", "pos"),
    DisplayFieldSpec("vel", "vel_au_per_d", "vel"),
)

STATE_COMPONENT_SPECS: tuple[DisplayFieldSpec, ...] = (
    DisplayFieldSpec("x", "x_au", lambda state: state.pos[0]),
    DisplayFieldSpec("y", "y_au", lambda state: state.pos[1]),
    DisplayFieldSpec("z", "z_au", lambda state: state.pos[2]),
    DisplayFieldSpec("vx", "vx_au_per_d", lambda state: state.vel[0]),
    DisplayFieldSpec("vy", "vy_au_per_d", lambda state: state.vel[1]),
    DisplayFieldSpec("vz", "vz_au_per_d", lambda state: state.vel[2]),
)


def orbit_element_specs(orbit) -> tuple[DisplayFieldSpec, ...]:
    """Return shared display specs for an orbit-like object."""
    a = np.asarray(orbit.a)
    primary_label = "a_au" if np.all(np.isfinite(a)) else "p_au"
    primary_accessor: DisplayAccessor = "a" if primary_label == "a_au" else "p"
    return (
        DisplayFieldSpec("a" if primary_label == "a_au" else "p", primary_label, primary_accessor),
        DisplayFieldSpec("e", "e", "e"),
        DisplayFieldSpec("inc", "i_deg", "inc", transform=_rad_to_deg, scientific=False, signed=False),
        DisplayFieldSpec("node", "node_deg", "node", transform=_rad_to_deg, scientific=False, signed=False),
        DisplayFieldSpec("peri", "peri_deg", "peri", transform=_rad_to_deg, scientific=False, signed=False),
        DisplayFieldSpec("m", "M_deg", "m", transform=_rad_to_deg, scientific=False, signed=False),
    )


def orbit_element_table_specs(orbit) -> tuple[DisplayFieldSpec, ...]:
    """Return the element-table display specs, including derived fields."""
    specs = list(orbit_element_specs(orbit))
    if not any(spec.label == "p_au" for spec in specs):
        specs.append(DisplayFieldSpec("p", "p_au", "p"))
    specs.extend(
        (
            DisplayFieldSpec("q", "q_au", lambda elem: np.asarray(elem.p) / (1.0 + np.asarray(elem.e))),
            DisplayFieldSpec(
                "Q",
                "Q_au",
                lambda elem: np.where(
                    np.asarray(elem.e) < 1.0,
                    np.asarray(elem.p) / (1.0 - np.asarray(elem.e)),
                    np.nan,
                ),
            ),
            DisplayFieldSpec("v", "v_deg", "v", transform=_rad_to_deg, scientific=False, signed=False),
            DisplayFieldSpec("period", "period_d", "period", scientific=False, signed=False),
            DisplayFieldSpec("perit_jd", "perit_tdb_jd", "perit_jd", scientific=False, signed=False),
        )
    )
    return tuple(specs)


OPTICAL_TABLE_SPECS: tuple[DisplayFieldSpec, ...] = (
    DisplayFieldSpec("astrometric_ra", "astrometric_ra_deg", "astrometric_ra", scientific=False, signed=False),
    DisplayFieldSpec("astrometric_dec", "astrometric_dec_deg", "astrometric_dec", scientific=False, signed=True),
    DisplayFieldSpec("apparent_ra", "apparent_ra_deg", "apparent_ra", scientific=False, signed=False),
    DisplayFieldSpec("apparent_dec", "apparent_dec_deg", "apparent_dec", scientific=False, signed=True),
    DisplayFieldSpec("azimuth", "azimuth_deg", "azimuth", scientific=False, signed=False),
    DisplayFieldSpec("elevation", "elevation_deg", "elevation", scientific=False, signed=True),
    DisplayFieldSpec("delta", "delta_au", "delta"),
    DisplayFieldSpec("r", "r_au", "r"),
    DisplayFieldSpec("phase_angle", "phase_angle_deg", "phase_angle", scientific=False, signed=False),
    DisplayFieldSpec("elongation", "elongation_deg", "elongation", scientific=False, signed=False),
    DisplayFieldSpec("mag", "mag", "mag")
)
RADAR_TABLE_SPECS: tuple[DisplayFieldSpec, ...] = (
    DisplayFieldSpec("radar_delay", "radar_delay_us", "radar_delay"),
    DisplayFieldSpec("radar_doppler", "radar_doppler_hz", "radar_doppler"),
    DisplayFieldSpec("radar_range", "radar_range_au", "radar_range"),
    DisplayFieldSpec("radar_rate", "radar_rate_au_per_d", "radar_rate"),
)


def compact_orbit_summary(orbit) -> str:
    """Create a compact one-line orbit summary using shared display labels."""
    orbit_type = orbit.__class__.__name__
    try:
        if "Kep" in orbit_type:
            primary_spec, e_spec, inc_spec, *_ = orbit_element_specs(orbit)
            primary = resolve_display_value(orbit, primary_spec)
            e_val = resolve_display_value(orbit, e_spec)
            inc_val = resolve_display_value(orbit, inc_spec)
            return (
                f"{orbit_type}("
                f"{primary_spec.label}={float(np.asarray(primary).reshape(-1)[0]):.6f}, "
                f"{e_spec.label}={float(np.asarray(e_val).reshape(-1)[0]):.6f}, "
                f"{inc_spec.label}={float(np.asarray(inc_val).reshape(-1)[0]):.3f})"
            )

        values = [resolve_display_value(orbit, spec) for spec in STATE_COMPONENT_SPECS]
        labels = [spec.label for spec in STATE_COMPONENT_SPECS]
        parts = [
            f"{label}={float(np.asarray(value).reshape(-1)[0]):.6f}"
            for label, value in zip(labels, values)
        ]
        return f"{orbit_type}({', '.join(parts)})"
    except Exception:
        return orbit_type
