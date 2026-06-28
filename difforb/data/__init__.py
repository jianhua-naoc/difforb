"""Optional runtime data management for DiffOrb."""

from difforb.data.manager import (
    DataNotInstalledError,
    dataset_status,
    ensure_data,
    get_data_dir,
    get_data_path,
    get_user_data_dir,
    get_writable_data_path,
    install_dataset,
    list_datasets,
    missing_data_message,
)

__all__ = [
    "DataNotInstalledError",
    "dataset_status",
    "ensure_data",
    "get_data_dir",
    "get_data_path",
    "get_user_data_dir",
    "get_writable_data_path",
    "install_dataset",
    "list_datasets",
    "missing_data_message",
]
