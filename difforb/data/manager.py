"""Manage optional runtime data files used by DiffOrb.

DiffOrb source distributions may omit third-party data files whose
redistribution terms are separate from the package license. This module keeps
the path policy, manifest lookup, and download entry points in one place.
"""

from __future__ import annotations

import io
import sys
import tarfile
import tomllib
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests

PACKAGE_DATA_DIR = Path(__file__).resolve().parent
MANIFEST_FILENAME = "manifest.toml"


class DataNotInstalledError(FileNotFoundError):
    """Raised when a required DiffOrb data set is not available locally."""


@dataclass(frozen=True)
class DataSetStatus:
    """Installation state for one manifest data set."""

    name: str
    installed: bool
    files: tuple[tuple[str, str, str], ...]
    installable: bool


def get_user_data_dir() -> Path:
    """Return the fixed platform data directory used for user-installed DiffOrb data."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path.home() / "AppData" / "Local"
    else:
        base = Path.home() / ".local" / "share"
    return base / "difforb"


def get_data_dir() -> str:
    """Return the writable DiffOrb data directory as a string path."""
    return str(get_user_data_dir())


def get_writable_data_path(relative_path: str | Path, *, create_parent: bool = False) -> Path:
    """Return the writable path for one data file.

    Parameters
    ----------
    relative_path : str or Path
        Path relative to the DiffOrb data directory.
    create_parent : bool, default=False
        If ``True``, create the parent directory before returning.

    Returns
    -------
    Path
        Path under the platform data directory.
    """
    path = get_user_data_dir() / Path(relative_path)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _candidate_existing_path(relative_path: str | Path) -> tuple[Path, str]:
    relative = Path(relative_path)
    user_path = get_user_data_dir() / relative
    if user_path.exists():
        return user_path, "user"

    packaged_path = PACKAGE_DATA_DIR / relative
    if packaged_path.exists():
        return packaged_path, "packaged"

    return user_path, "missing"


def get_data_path(
        relative_path: str | Path,
        *,
        dataset: str | None = None,
        must_exist: bool = True) -> Path:
    """Resolve one DiffOrb data file.

    Parameters
    ----------
    relative_path : str or Path
        Path relative to the DiffOrb data directory.
    dataset : str, optional
        Manifest data set name used in the missing-data error message.
    must_exist : bool, default=True
        If ``True``, raise :class:`DataNotInstalledError` when the file is not
        found.

    Returns
    -------
    Path
        Existing user or packaged fallback path. If ``must_exist`` is
        ``False`` and no file is found, the writable user path is returned.
    """
    path, state = _candidate_existing_path(relative_path)
    if state != "missing" or not must_exist:
        return path
    raise DataNotInstalledError(missing_data_message(dataset, path))


def _load_manifest() -> dict[str, Any]:
    manifest_path = PACKAGE_DATA_DIR / MANIFEST_FILENAME
    with manifest_path.open("rb") as f:
        return tomllib.load(f)


def _datasets() -> dict[str, dict[str, Any]]:
    return _load_manifest().get("datasets", {})


def list_datasets() -> tuple[str, ...]:
    """Return manifest data set names."""
    return tuple(sorted(_datasets()))


def get_dataset(name: str) -> dict[str, Any]:
    """Return one manifest data set entry."""
    datasets = _datasets()
    try:
        return datasets[name]
    except KeyError as exc:
        known = ", ".join(sorted(datasets))
        raise KeyError(f"Unknown DiffOrb data set {name!r}. Known data sets: {known}") from exc


def missing_data_message(dataset: str | None, expected_path: str | Path | None = None) -> str:
    """Build a user-facing missing-data message."""
    data_dir = get_user_data_dir()
    lines = [
        f"DiffOrb data set {dataset!r} is not installed." if dataset else "A required DiffOrb data file is not installed.",
    ]
    if expected_path is not None:
        lines.append(f"Expected file: {Path(expected_path)}")
    lines.append(f"Data directory: {data_dir}")

    if dataset:
        try:
            entry = get_dataset(dataset)
        except KeyError:
            entry = {}
        if entry.get("installable", False):
            lines.extend([
                "",
                "Install it with:",
                f"    python -m difforb.data install {dataset}",
            ])
        else:
            instructions = entry.get("instructions")
            lines.append("")
            lines.append("This data set is not automatically downloadable by DiffOrb.")
            if instructions:
                lines.append(str(instructions))
    return "\n".join(lines)


def dataset_status(name: str) -> DataSetStatus:
    """Return the installation state for one data set."""
    entry = get_dataset(name)
    rows = []
    installed = True
    for relative_path in entry.get("files", []):
        path, state = _candidate_existing_path(relative_path)
        if state == "missing":
            installed = False
        rows.append((relative_path, state, str(path)))
    return DataSetStatus(
        name=name,
        installed=installed,
        files=tuple(rows),
        installable=bool(entry.get("installable", False)),
    )


def ensure_data(name: str, *, download: bool = False, force: bool = False) -> tuple[Path, ...]:
    """Ensure that one data set is locally available.

    Parameters
    ----------
    name : str
        Manifest data set name.
    download : bool, default=False
        If ``True``, download installable missing files before checking.
    force : bool, default=False
        If ``True`` and ``download`` is also ``True``, replace existing
        downloaded files.

    Returns
    -------
    tuple[Path, ...]
        Resolved data file paths.
    """
    if download:
        install_dataset(name, force=force)

    status = dataset_status(name)
    if not status.installed:
        expected = next(Path(path) for _, state, path in status.files if state == "missing")
        raise DataNotInstalledError(missing_data_message(name, expected))
    return tuple(Path(path) for _, _, path in status.files)


def install_dataset(name: str, *, force: bool = False) -> tuple[Path, ...]:
    """Download one installable manifest data set into the user data directory."""
    entry = get_dataset(name)
    if not entry.get("installable", False):
        raise DataNotInstalledError(missing_data_message(name))

    written = []
    completed_targets: set[Path] = set()
    errors: list[tuple[str, Exception]] = []
    required_targets = {
        get_writable_data_path(path)
        for path in entry.get("files", [])
    }
    for source in entry.get("sources", []):
        targets = _source_targets(source)
        if targets and all(target in completed_targets for target in targets):
            continue
        try:
            if "archive" in source:
                source_written = _install_archive_source(source, force=force)
            else:
                source_written = (_install_file_source(source, force=force),)
        except Exception as exc:
            errors.append((str(source.get("url", "<unknown source>")), exc))
            continue
        written.extend(source_written)
        completed_targets.update(targets or source_written)

    if force and required_targets and not required_targets.issubset(completed_targets):
        if errors:
            detail = "\n".join(
                f"    - {url}: {type(exc).__name__}: {exc}"
                for url, exc in errors
            )
            raise DataNotInstalledError(
                f"Could not refresh DiffOrb data set {name!r} from the configured sources.\n"
                f"Tried:\n{detail}"
            ) from errors[-1][1]
        missing = required_targets - completed_targets
        expected = next(iter(missing))
        raise DataNotInstalledError(missing_data_message(name, expected))

    status = dataset_status(name)
    if not status.installed:
        if errors:
            detail = "\n".join(
                f"    - {url}: {type(exc).__name__}: {exc}"
                for url, exc in errors
            )
            raise DataNotInstalledError(
                f"Could not install DiffOrb data set {name!r} from the configured sources.\n"
                f"Tried:\n{detail}"
            ) from errors[-1][1]
        expected = next(Path(path) for _, state, path in status.files if state == "missing")
        raise DataNotInstalledError(missing_data_message(name, expected))
    return tuple(written)


def _source_targets(source: dict[str, Any]) -> tuple[Path, ...]:
    if "archive" in source:
        return tuple(
            get_writable_data_path(member["path"])
            for member in source.get("members", [])
        )
    if "path" in source:
        return (get_writable_data_path(source["path"]),)
    return ()


def _install_file_source(source: dict[str, Any], *, force: bool = False) -> Path:
    relative_path = source["path"]
    target = get_writable_data_path(relative_path, create_parent=True)
    if target.exists() and not force:
        return target

    response = requests.get(source["url"], timeout=60)
    response.raise_for_status()
    target.write_bytes(_transform_file_content(response.content, source))
    return target


class _FirstPreExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_pre = False
        self.done = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "pre" and not self.done:
            self.in_pre = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "pre" and self.in_pre:
            self.in_pre = False
            self.done = True

    def handle_data(self, data: str) -> None:
        if self.in_pre:
            self.parts.append(data)


def _transform_file_content(content: bytes, source: dict[str, Any]) -> bytes:
    transform = source.get("transform")
    if transform is None:
        return content
    if transform == "html_pre":
        return _extract_html_pre_text(content).encode("utf-8")
    raise ValueError(f"Unsupported DiffOrb data file transform: {transform!r}")


def _extract_html_pre_text(content: bytes) -> str:
    parser = _FirstPreExtractor()
    parser.feed(content.decode("utf-8", errors="replace"))
    text = "".join(parser.parts).strip("\n")
    if not text.startswith("Code"):
        raise DataNotInstalledError("Could not extract an MPC observatory-code table from the downloaded HTML.")
    return f"{text}\n"


def _install_archive_source(source: dict[str, Any], *, force: bool = False) -> tuple[Path, ...]:
    targets = [
        get_writable_data_path(member["path"], create_parent=True)
        for member in source.get("members", [])
    ]
    if targets and not force and all(target.exists() for target in targets):
        return tuple(targets)

    response = requests.get(source["url"], timeout=60)
    response.raise_for_status()

    archive_format = source["archive"]
    if archive_format not in {"tar", "tgz", "tar.gz"}:
        raise ValueError(f"Unsupported DiffOrb data archive format: {archive_format!r}")

    mode = "r:gz" if archive_format in {"tgz", "tar.gz"} else "r:"
    written = []
    with tarfile.open(fileobj=io.BytesIO(response.content), mode=mode) as archive:
        regular_members = {
            member.name: member
            for member in archive.getmembers()
            if member.isfile()
        }
        for member_spec in source.get("members", []):
            member = _find_archive_member(regular_members, member_spec)
            target = get_writable_data_path(member_spec["path"], create_parent=True)
            if target.exists() and not force:
                written.append(target)
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise DataNotInstalledError(f"Archive member {member.name!r} could not be read.")
            target.write_bytes(extracted.read())
            written.append(target)
    return tuple(written)


def _find_archive_member(
        members: dict[str, tarfile.TarInfo],
        member_spec: dict[str, Any]) -> tarfile.TarInfo:
    candidates = [member_spec.get("member"), *member_spec.get("member_candidates", [])]
    for candidate in candidates:
        if candidate and candidate in members:
            return members[candidate]

    target_suffix = Path(member_spec["path"]).as_posix()
    matches = [
        member
        for name, member in members.items()
        if name == target_suffix or name.endswith(f"/{target_suffix}")
    ]
    if len(matches) == 1:
        return matches[0]

    expected = ", ".join(candidate for candidate in candidates if candidate) or target_suffix
    raise DataNotInstalledError(
        f"Archive member for {member_spec['path']!r} was not found. Expected {expected}."
    )
