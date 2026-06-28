"""Unified observer site containers and observatory-code loaders.

This module defines the ``Site`` container used for fixed ground, roving
ground, and space observers. Ground rows store positions in ``ITRS``. Space
rows store canonical ``GCRS`` positions and velocities. The module also loads
observatory code tables from the International Astronomical Union (``IAU``)
Minor Planet Center and the Jet Propulsion Laboratory (``JPL``).
"""

import os
from typing import Dict, List, Tuple, Union, ClassVar, TypeVar, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Int
import equinox as eqx

from difforb.core.batch import BatchableObject
from difforb.core.config import get_data_path, missing_data_message
from difforb.core.geo import ITRS, WGS84
from difforb.core.state.frame import GCRS, Frame
from difforb.core.state.origins import Origin
from difforb.core.state.state import State
from difforb.core.time.timescale import Time
from difforb.report.text import build_repr, format_float_array, format_shape

jax.config.update("jax_enable_x64", True)

DEFAULT_IAU_CODES_FILENAME = str(get_data_path("obs_code/iau_obs_codes.txt", dataset="mpc-obs-codes", must_exist=False))
DEFAULT_JPL_RADAR_CODES_FILENAME = str(get_data_path("obs_code/jpl_radar_codes.txt", dataset="jpl-radar-codes", must_exist=False))
_SITE_TYPE_COMMON_GROUND = 0
_SITE_TYPE_ROVING_GROUND = 1
_SITE_TYPE_SATELLITE = 2

I = TypeVar("I", bound='ITRS')


def parse_iau_obs_codes(
        filename: str = DEFAULT_IAU_CODES_FILENAME) -> Dict[str, Tuple[float, float, float, str, int]]:
    """Parse the IAU Minor Planet Center observatory code table.

    Parameters
    ----------
    filename : str, optional
        Path to the observatory code table.

    Returns
    -------
    dict[str, tuple[float, float, float, str, int]]
        Mapping from observatory code to ``(lon, parallax_const1, parallax_const2, name, type)``. Longitude is in degrees. The two parallax constants are in Earth-radius units.

    Notes
    -----
    Common ground sites use the published geocentric constants. Roving sites and satellite sites are marked by a site-type flag and use zeros in the numeric fields.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(missing_data_message("mpc-obs-codes", filename))
    roving_label = "ROVING"
    with open(filename, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        rows = f.readlines()
        if not header.startswith("Code"):
            raise ValueError(
                f"Invalid MPC observatory code table at {filename!r}. "
                "Expected a fixed-width table whose first line starts with 'Code'. "
                "Refresh it with `python -m difforb.data install mpc-obs-codes --force`."
            )
        obs = {}
        for lineno, row in enumerate(rows, start=2):
            row = row.rstrip("\n")
            if not row.strip():
                continue
            parts = row.split(maxsplit=4)
            code = parts[0] if parts else row[:4].strip()
            lon = cos = sin = name = ""
            if len(parts) >= 5:
                lon, cos, sin, name = parts[1], parts[2], parts[3], parts[4]
                try:
                    lon = float(lon)
                    cos = float(cos)
                    sin = float(sin)
                except ValueError:
                    lon = cos = sin = name = ""
            if name == "":
                code = row[:4].strip()
                lon = row[4:13].strip()
                cos = row[13:21].strip()
                sin = row[21:30].strip()
                name = row[30:].strip()
            name = name.upper()
            if roving_label in name:
                lon = 0.
                cos = 0.
                sin = 0.
                type = _SITE_TYPE_ROVING_GROUND
            elif not isinstance(lon, float) and len(lon) == 0 and len(cos) == 0 and len(sin) == 0:
                lon = 0.
                cos = 0.
                sin = 0.
                type = _SITE_TYPE_SATELLITE
            else:
                if not isinstance(lon, float):
                    try:
                        lon = float(lon)
                        cos = float(cos)
                        sin = float(sin)
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid MPC observatory code row {lineno} in {filename!r}: {row!r}. "
                            "Refresh the table with `python -m difforb.data install mpc-obs-codes --force`."
                        ) from exc
                type = _SITE_TYPE_COMMON_GROUND
            obs[code] = (lon, cos, sin, name, type)
        return obs


def parse_jpl_radar_obs_codes(
        filename: str = DEFAULT_JPL_RADAR_CODES_FILENAME) -> Dict[str, Tuple[float, float, float, str, int]]:
    """Parse the optional JPL radar observatory code table.

    Parameters
    ----------
    filename : str, optional
        Path to the radar observatory table.

    Returns
    -------
    dict[str, tuple[float, float, float, str, int]]
        Mapping from observatory code to ``(lon, lat, alt, name, type)``.

    Notes
    -----
    This table is optional. If the file is missing, this function returns an empty mapping so that optical-only workflows still work.
    """
    if not os.path.exists(filename):
        return {}
    with open(filename, "r") as f:
        f.readline()
        rows = f.readlines()
        obs = {}
        for row in rows:
            row = row.strip()
            code, name, lon, lat, alt = row.split(",")
            if len(lon) == 0 and len(lat) == 0 and len(alt) == 0:
                lon = 0.
                lat = 0.
                alt = 0.
                type = _SITE_TYPE_SATELLITE
            else:
                lon = float(lon)
                lat = float(lat)
                alt = float(alt)
                alt = alt * 1000.0
                type = _SITE_TYPE_COMMON_GROUND
            obs[code] = (lon, lat, alt, name, type)
        return obs


class SiteKeyBatch(NamedTuple):
    """Parsed observer keys used to build and group observing sites.

    Attributes
    ----------
    raw_keys : tuple[str, ...]
        Canonical observer key strings.
    codes : tuple[str, ...]
        Observatory codes without payload suffixes.
    kind_ids : numpy.ndarray
        Site-type identifiers. Values follow :class:`Site` type constants.
    identity_keys : tuple[str, ...]
        Keys used to group observations by physical observing identity.
    display_labels : tuple[str, ...]
        Human-facing labels for reports and tables.
    payload_pos : numpy.ndarray
        Payload coordinates with shape ``(N, 3)``. Fixed ground rows contain
        ``NaN``. Roving ground rows store WGS84 longitude, latitude, and
        altitude. Space rows store canonical ``GCRS`` position in ``au``.
    """
    raw_keys: tuple[str, ...]
    codes: tuple[str, ...]
    kind_ids: np.ndarray
    identity_keys: tuple[str, ...]
    display_labels: tuple[str, ...]
    payload_pos: np.ndarray


def format_site_coordinate(value: float) -> str:
    """Format one site-key coordinate with round-trip precision."""
    return f"{float(value):.17g}"


def format_site_key(code: str, kind_id: int = _SITE_TYPE_COMMON_GROUND,
                    payload_pos: Float[ArrayLike, "3"] | None = None) -> str:
    """Build a canonical observer key string.

    Parameters
    ----------
    code : str
        Observatory code.
    kind_id : int, default=``Site.TYPE_COMMON_GROUND``
        Site kind identifier.
    payload_pos : array-like of float, optional
        Three-coordinate payload. Roving ground payloads are WGS84
        ``lon_deg, lat_deg, alt_m``. Space payloads are ``GCRS`` position in
        ``au``.

    Returns
    -------
    str
        Canonical observer key.
    """
    code = str(code).strip().upper()
    if kind_id == _SITE_TYPE_COMMON_GROUND:
        return code
    if payload_pos is None:
        raise ValueError("Payload coordinates are required for roving and space site keys.")
    payload = np.asarray(payload_pos, dtype=float)
    if payload.shape != (3,):
        raise ValueError(f"Site-key payload must have shape (3,), got {payload.shape}.")
    coordinates = ", ".join(format_site_coordinate(value) for value in payload)
    if kind_id == _SITE_TYPE_ROVING_GROUND:
        return f"{code} @ {coordinates}"
    if kind_id == _SITE_TYPE_SATELLITE:
        return f"{code} # {coordinates}"
    raise ValueError(f"Unknown site kind id: {kind_id}.")


def parse_site_payload(text: str) -> np.ndarray:
    """Parse a three-coordinate site-key payload."""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Site-key payload must contain three comma-separated coordinates: {text!r}.")
    try:
        return np.array([float(part) for part in parts], dtype=float)
    except ValueError as exc:
        raise ValueError(f"Invalid site-key payload coordinates: {text!r}.") from exc


def parse_site_keys(keys: Union[str, List[str], np.ndarray]) -> SiteKeyBatch:
    """Parse fixed-ground, roving-ground, and space observer keys.

    Parameters
    ----------
    keys : str or list[str] or numpy.ndarray
        Observer keys. The accepted formats are ``"568"``,
        ``"249 @ lon_deg, lat_deg, alt_m"``, and
        ``"C57 # x_au, y_au, z_au"``.

    Returns
    -------
    SiteKeyBatch
        Parsed site-key metadata and payload coordinates.
    """
    raw_key_array = np.char.upper(np.asarray(keys, dtype=str).reshape(-1))
    raw_keys = []
    codes = []
    kind_ids = []
    identity_keys = []
    display_labels = []
    payload_rows = []
    for raw in raw_key_array:
        raw_key = str(raw).strip()
        if "@" in raw_key and "#" in raw_key:
            raise ValueError(f"Site key cannot contain both '@' and '#': {raw_key!r}.")
        if "#" in raw_key:
            code, payload_text = raw_key.split("#", 1)
            code = code.strip()
            payload = parse_site_payload(payload_text)
            kind_id = _SITE_TYPE_SATELLITE
            identity_key = code
            display_label = f"{code} # {', '.join(format_site_coordinate(value) for value in payload)}"
            raw_key = display_label
        elif "@" in raw_key:
            code, payload_text = raw_key.split("@", 1)
            code = code.strip()
            payload = parse_site_payload(payload_text)
            kind_id = _SITE_TYPE_ROVING_GROUND
            identity_key = f"{code} @ {', '.join(format_site_coordinate(value) for value in payload)}"
            display_label = identity_key
            raw_key = identity_key
        else:
            code = raw_key
            payload = np.full(3, np.nan, dtype=float)
            kind_id = _SITE_TYPE_COMMON_GROUND
            identity_key = code
            display_label = code
        if not code:
            raise ValueError(f"Site key is missing an observatory code: {raw_key!r}.")
        raw_keys.append(raw_key)
        codes.append(code)
        kind_ids.append(kind_id)
        identity_keys.append(identity_key)
        display_labels.append(display_label)
        payload_rows.append(payload)
    return SiteKeyBatch(
        raw_keys=tuple(raw_keys),
        codes=tuple(codes),
        kind_ids=np.asarray(kind_ids, dtype=int),
        identity_keys=tuple(identity_keys),
        display_labels=tuple(display_labels),
        payload_pos=np.stack(payload_rows, axis=0) if payload_rows else np.empty((0, 3), dtype=float),
    )


def site_identity_keys(keys: Union[str, List[str], np.ndarray]) -> np.ndarray:
    """Return station-identity keys for one or more observer keys."""
    return np.asarray(parse_site_keys(keys).identity_keys, dtype=str)


def site_display_labels(keys: Union[str, List[str], np.ndarray]) -> np.ndarray:
    """Return display labels for one or more observer keys."""
    return np.asarray(parse_site_keys(keys).display_labels, dtype=str)


class Site(BatchableObject):
    """Observer sites stored as a uniform numerical batch.

    A ``Site`` may contain fixed ground, roving ground, and space observers in
    the same batch. Ground rows store ``ITRS`` positions. Space rows store
    canonical ``GCRS`` position and velocity. String metadata is static PyTree
    data; numerical state is stored in JAX arrays.
    """
    TYPE_COMMON_GROUND: ClassVar[int] = _SITE_TYPE_COMMON_GROUND
    TYPE_ROVING_GROUND: ClassVar[int] = _SITE_TYPE_ROVING_GROUND
    TYPE_SATELLITE: ClassVar[int] = _SITE_TYPE_SATELLITE
    _iau_obs_code: ClassVar[dict] = parse_iau_obs_codes()
    _radar_obs_code: ClassVar[dict] = parse_jpl_radar_obs_codes()

    kind_ids: Int[Array, "..."]
    itrs_pos: Float[Array, "... 3"]
    itrs_lon: Float[Array, "..."]
    gcrs_pos: Float[Array, "... 3"]
    gcrs_vel: Float[Array, "... 3"]
    raw_keys: tuple[str, ...] = eqx.field(static=True)
    codes: tuple[str, ...] = eqx.field(static=True)
    identity_keys: tuple[str, ...] = eqx.field(static=True)
    display_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
            self,
            kind_ids: Int[ArrayLike, "..."],
            itrs_pos: Float[ArrayLike, "... 3"],
            itrs_lon: Float[ArrayLike, "..."],
            gcrs_pos: Float[ArrayLike, "... 3"],
            gcrs_vel: Float[ArrayLike, "... 3"],
            *,
            raw_keys: tuple[str, ...] | None = None,
            codes: tuple[str, ...] | None = None,
            identity_keys: tuple[str, ...] | None = None,
            display_labels: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize a site batch from canonical numerical fields."""
        self.kind_ids = jnp.asarray(kind_ids, dtype=int)
        self.itrs_pos = jnp.asarray(itrs_pos, dtype=float)
        self.itrs_lon = jnp.asarray(itrs_lon, dtype=float)
        self.gcrs_pos = jnp.asarray(gcrs_pos, dtype=float)
        self.gcrs_vel = jnp.asarray(gcrs_vel, dtype=float)
        if self.itrs_pos.shape[-1] != 3:
            raise ValueError(f"`itrs_pos` must end with dimension 3, got {self.itrs_pos.shape}.")
        if self.gcrs_pos.shape[-1] != 3:
            raise ValueError(f"`gcrs_pos` must end with dimension 3, got {self.gcrs_pos.shape}.")
        if self.gcrs_vel.shape[-1] != 3:
            raise ValueError(f"`gcrs_vel` must end with dimension 3, got {self.gcrs_vel.shape}.")
        if self.itrs_pos.shape[:-1] != self.kind_ids.shape:
            raise ValueError("`itrs_pos` batch shape must match `kind_ids`.")
        if self.itrs_lon.shape != self.kind_ids.shape:
            raise ValueError("`itrs_lon` shape must match `kind_ids`.")
        if self.gcrs_pos.shape[:-1] != self.kind_ids.shape:
            raise ValueError("`gcrs_pos` batch shape must match `kind_ids`.")
        if self.gcrs_vel.shape[:-1] != self.kind_ids.shape:
            raise ValueError("`gcrs_vel` batch shape must match `kind_ids`.")
        n = 1 if self.kind_ids.shape == () else int(self.kind_ids.shape[0])
        self.raw_keys = tuple(raw_keys) if raw_keys is not None else tuple([""] * n)
        self.codes = tuple(codes) if codes is not None else tuple([""] * n)
        self.identity_keys = tuple(identity_keys) if identity_keys is not None else self.raw_keys
        self.display_labels = tuple(display_labels) if display_labels is not None else self.raw_keys

    @classmethod
    def from_code(cls, keys: Union[str, List[str], np.ndarray]) -> "Site":
        """Build sites from observer keys.

        Parameters
        ----------
        keys : str or list[str] or numpy.ndarray
            Observer keys. Fixed ground keys are plain observatory codes.
            Roving ground keys use ``"code @ lon_deg, lat_deg, alt_m"``.
            Space keys use ``"code # x_au, y_au, z_au"``.

        Returns
        -------
        Site
            Site batch represented by uniform numerical arrays.
        """
        key_is_str = isinstance(keys, str)
        parsed = parse_site_keys(keys)
        n = len(parsed.raw_keys)
        kind_ids = np.zeros(n, dtype=int)
        itrs_pos = np.zeros((n, 3), dtype=float)
        itrs_lon = np.zeros(n, dtype=float)
        gcrs_pos = np.zeros((n, 3), dtype=float)
        gcrs_vel = np.zeros((n, 3), dtype=float)

        for i, code in enumerate(parsed.codes):
            optical_site = cls._iau_obs_code.get(code)
            radar_site = cls._radar_obs_code.get(code)
            if radar_site:
                site = radar_site
                lon, lat, alt, name, table_kind = site
                table_itrs = WGS84.from_geodetic(lon, lat, alt)
            elif optical_site:
                site = optical_site
                lon, p1, p2, name, table_kind = site
                table_itrs = WGS84.from_geocentric(lon, p1, p2)
            else:
                raise RuntimeError(f"Not found observatory code: {code}.")

            key_kind = int(parsed.kind_ids[i])
            if key_kind == cls.TYPE_COMMON_GROUND:
                if table_kind == cls.TYPE_ROVING_GROUND:
                    raise ValueError(f"{code} is a roving site and requires '@' coordinates.")
                if table_kind == cls.TYPE_SATELLITE:
                    raise ValueError(f"{code} is a satellite site and requires '#' coordinates.")
                kind_ids[i] = cls.TYPE_COMMON_GROUND
                itrs_pos[i] = np.asarray(table_itrs.pos)
                itrs_lon[i] = float(np.asarray(table_itrs.lon))
            elif key_kind == cls.TYPE_ROVING_GROUND:
                if table_kind == cls.TYPE_SATELLITE:
                    raise ValueError(f"{code} is a satellite site and cannot use '@' ground coordinates.")
                if table_kind != cls.TYPE_ROVING_GROUND:
                    raise ValueError(
                        f"{code} is a fixed ground site and cannot use '@' roving coordinates; "
                        "use Site.from_geodetic(...) for custom ground coordinates."
                    )
                lon, lat, alt = parsed.payload_pos[i]
                roving_itrs = WGS84.from_geodetic(lon, lat, alt)
                kind_ids[i] = cls.TYPE_ROVING_GROUND
                itrs_pos[i] = np.asarray(roving_itrs.pos)
                itrs_lon[i] = float(np.asarray(roving_itrs.lon))
            elif key_kind == cls.TYPE_SATELLITE:
                if table_kind != cls.TYPE_SATELLITE:
                    raise ValueError(f"{code} is a ground site and cannot use '#' space coordinates.")
                kind_ids[i] = cls.TYPE_SATELLITE
                gcrs_pos[i] = parsed.payload_pos[i]
            else:
                raise ValueError(f"Unknown site kind id: {key_kind}.")

        if key_is_str:
            kind_ids = np.squeeze(kind_ids, axis=0)
            itrs_pos = np.squeeze(itrs_pos, axis=0)
            itrs_lon = np.squeeze(itrs_lon, axis=0)
            gcrs_pos = np.squeeze(gcrs_pos, axis=0)
            gcrs_vel = np.squeeze(gcrs_vel, axis=0)

        return cls(
            kind_ids=kind_ids,
            itrs_pos=itrs_pos,
            itrs_lon=itrs_lon,
            gcrs_pos=gcrs_pos,
            gcrs_vel=gcrs_vel,
            raw_keys=parsed.raw_keys,
            codes=parsed.codes,
            identity_keys=parsed.identity_keys,
            display_labels=parsed.display_labels,
        )

    @classmethod
    def from_geodetic(
            cls,
            lon: Float[ArrayLike, '...'],
            lat: Float[ArrayLike, '...'],
            alt: Float[ArrayLike, '...'],
            type: "I" = WGS84,
    ) -> "Site":
        """Build ground sites from geodetic coordinates."""
        return cls.from_itrs(type.from_geodetic(lon, lat, alt))

    @classmethod
    def from_geocentric(
            cls,
            lon: Float[ArrayLike, '...'],
            parallax_const1: Float[ArrayLike, '...'],
            parallax_const2: Float[ArrayLike, '...'],
            type: "I" = WGS84,
    ) -> "Site":
        """Build ground sites from geocentric observatory constants."""
        return cls.from_itrs(type.from_geocentric(lon, parallax_const1, parallax_const2))

    @classmethod
    def from_itrs(cls, itrs: ITRS) -> "Site":
        """Build ground sites from ``ITRS`` coordinates."""
        if not isinstance(itrs, ITRS):
            raise TypeError(f"`itrs` must be an instance of `ITRS`, got `{type(itrs).__name__}`.")
        kind_ids = jnp.full(itrs.shape, cls.TYPE_COMMON_GROUND, dtype=int)
        gcrs_pos = jnp.zeros(itrs.shape + (3,), dtype=float)
        gcrs_vel = jnp.zeros(itrs.shape + (3,), dtype=float)
        return cls(
            kind_ids=kind_ids,
            itrs_pos=itrs.pos,
            itrs_lon=itrs.lon,
            gcrs_pos=gcrs_pos,
            gcrs_vel=gcrs_vel,
        )

    @classmethod
    def from_gcrs(
            cls,
            pos_au: Float[ArrayLike, "... 3"],
            vel_au_per_d: Float[ArrayLike, "... 3"] | None = None,
    ) -> "Site":
        """Build space sites from canonical ``GCRS`` coordinates."""
        pos = jnp.asarray(pos_au, dtype=float)
        if pos.shape[-1] != 3:
            raise ValueError(f"`pos_au` must end with dimension 3, got {pos.shape}.")
        vel = jnp.zeros_like(pos) if vel_au_per_d is None else jnp.asarray(vel_au_per_d, dtype=float)
        if vel.shape != pos.shape:
            raise ValueError(f"`vel_au_per_d` shape {vel.shape} must match `pos_au` shape {pos.shape}.")
        batch_shape = pos.shape[:-1]
        return cls(
            kind_ids=jnp.full(batch_shape, cls.TYPE_SATELLITE, dtype=int),
            itrs_pos=jnp.zeros(batch_shape + (3,), dtype=float),
            itrs_lon=jnp.zeros(batch_shape, dtype=float),
            gcrs_pos=pos,
            gcrs_vel=vel,
        )

    @classmethod
    def from_state(cls, state: State) -> "Site":
        """Build space sites from a canonical ``GCRS`` state."""
        if not isinstance(state, State):
            raise TypeError(f"`state` must be an instance of `State`, got `{type(state).__name__}`.")
        if state.frame != GCRS:
            raise ValueError("`state.frame` must be `GCRS`.")
        return cls.from_gcrs(state.pos, state.vel)

    @property
    def shape(self):
        """Batch shape of the stored site set."""
        return self.kind_ids.shape

    @property
    def is_fixed_ground(self):
        """Mask fixed ground rows."""
        return self.kind_ids == self.TYPE_COMMON_GROUND

    @property
    def is_roving_ground(self):
        """Mask roving ground rows."""
        return self.kind_ids == self.TYPE_ROVING_GROUND

    @property
    def is_ground(self):
        """Mask all ground rows."""
        return self.kind_ids != self.TYPE_SATELLITE

    @property
    def is_space(self):
        """Mask space rows."""
        return self.kind_ids == self.TYPE_SATELLITE

    @property
    def ground_itrs(self) -> WGS84:
        """Ground coordinates stored as ``WGS84``.

        Space rows contain dummy zeros. Use :meth:`require_ground` when a
        caller requires every row to be terrestrial.
        """
        return WGS84(self.itrs_pos, self.itrs_lon)

    def require_ground(self) -> "Site":
        """Return ``self`` after validating that all rows are ground sites."""
        if bool(np.any(np.asarray(self.kind_ids == self.TYPE_SATELLITE))):
            raise ValueError("This operation requires ground sites, but at least one site is space-based.")
        return self

    @eqx.filter_jit
    def state(
            self,
            t: Time,
            frame: Frame = GCRS,
            *,
            sun: "EphemerisBody | None" = None,
            earth: "EphemerisBody | None" = None,
            grid: bool = False,
    ) -> State:
        """Return site states in one requested frame.

        Ground rows are evaluated through the stored ``ITRS`` coordinates and
        Earth-rotation model. Space rows are broadcast from their canonical
        ``GCRS`` position and velocity payloads.
        """
        ground_state = self.ground_itrs.state(t, frame=GCRS, grid=grid)
        site_shape = self.shape
        time_shape = t.shape
        if not grid:
            diff = len(time_shape) - len(site_shape)
            if diff > 0:
                normalized_site_batch = site_shape + (1,) * diff
            else:
                normalized_site_batch = site_shape
            target_time_shape = jnp.broadcast_shapes(normalized_site_batch, time_shape)
        else:
            diff = 0
            target_time_shape = site_shape + time_shape

        def broadcast_site_array(arr, core_dim):
            curr_core_shape = arr.shape[arr.ndim - core_dim:] if core_dim > 0 else ()
            if not grid and diff > 0:
                arr = arr.reshape(site_shape + (1,) * diff + curr_core_shape)
            elif grid:
                arr = jnp.expand_dims(arr, axis=tuple(range(len(site_shape), len(site_shape) + len(time_shape))))
            return jnp.broadcast_to(arr, target_time_shape + curr_core_shape)

        space_pos = broadcast_site_array(self.gcrs_pos, 1)
        space_vel = broadcast_site_array(self.gcrs_vel, 1)
        space_mask = broadcast_site_array(self.is_space, 0)
        gcrs_pos = jnp.where(space_mask[..., None], space_pos, ground_state.pos)
        gcrs_vel = jnp.where(space_mask[..., None], space_vel, ground_state.vel)
        state = State(tdb=ground_state.tdb, pos=gcrs_pos, vel=gcrs_vel, frame=GCRS)
        if frame == GCRS:
            return state
        from difforb.body.ephbody import EphemerisBody
        if frame.origin is not Origin.EARTH and earth is None:
            earth = EphemerisBody("earth")
        if frame.origin is Origin.SUN and sun is None:
            sun = EphemerisBody("sun")
        return state.to(frame, sun=sun, earth=earth)

    def __getitem__(self, idx: ArrayLike) -> "Site":
        """Return a sliced site object."""
        indices = np.arange(1 if self.shape == () else self.shape[0])[idx]
        indices = np.asarray(indices).reshape(-1)
        raw_keys = tuple(self.raw_keys[int(i)] for i in indices)
        codes = tuple(self.codes[int(i)] for i in indices)
        identity_keys = tuple(self.identity_keys[int(i)] for i in indices)
        display_labels = tuple(self.display_labels[int(i)] for i in indices)
        return self.__class__(
            kind_ids=self.kind_ids[idx],
            itrs_pos=self.itrs_pos[idx],
            itrs_lon=self.itrs_lon[idx],
            gcrs_pos=self.gcrs_pos[idx],
            gcrs_vel=self.gcrs_vel[idx],
            raw_keys=raw_keys,
            codes=codes,
            identity_keys=identity_keys,
            display_labels=display_labels,
        )

    def __repr__(self) -> str:
        return build_repr(
            self.__class__.__name__,
            [
                ("shape", format_shape(self.shape)),
                ("n_ground", str(int(np.sum(np.asarray(self.is_ground))))),
                ("n_space", str(int(np.sum(np.asarray(self.is_space))))),
                ("labels", repr(self.display_labels) if self.display_labels and self.shape != () else None),
                ("lon_deg", format_float_array(jnp.rad2deg(self.itrs_lon)) if np.any(np.asarray(self.is_ground)) else None),
                ("gcrs_pos_au", format_float_array(self.gcrs_pos) if np.any(np.asarray(self.is_space)) else None),
            ],
        )
