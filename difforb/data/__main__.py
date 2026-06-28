"""Command-line entry point for DiffOrb data management."""

from __future__ import annotations

import argparse

from difforb.data.manager import (
    dataset_status,
    get_data_dir,
    install_dataset,
    list_datasets,
)


def _print_status(names: list[str]) -> None:
    for name in names:
        status = dataset_status(name)
        marker = "installed" if status.installed else "missing"
        installable = "downloadable" if status.installable else "manual"
        print(f"{name}: {marker} ({installable})")
        for relative_path, state, path in status.files:
            print(f"  {state:9s} {relative_path} -> {path}")


def main(argv: list[str] | None = None) -> int:
    """Run the ``python -m difforb.data`` command."""
    parser = argparse.ArgumentParser(prog="python -m difforb.data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("dir", help="Print the writable DiffOrb data directory.")

    status_parser = subparsers.add_parser("status", help="Show installed data sets.")
    status_parser.add_argument("datasets", nargs="*", help="Data set names. Defaults to all known data sets.")

    install_parser = subparsers.add_parser("install", help="Download one installable data set.")
    install_parser.add_argument("dataset", help="Data set name, or 'all' for all downloadable data sets.")
    install_parser.add_argument("--force", action="store_true", help="Replace existing downloaded files.")

    args = parser.parse_args(argv)

    if args.command == "dir":
        print(get_data_dir())
        return 0

    if args.command == "status":
        names = args.datasets or list(list_datasets())
        _print_status(names)
        return 0

    if args.command == "install":
        names = list_datasets() if args.dataset == "all" else (args.dataset,)
        for name in names:
            if args.dataset == "all" and not dataset_status(name).installable:
                continue
            paths = install_dataset(name, force=args.force)
            for path in paths:
                print(path)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
