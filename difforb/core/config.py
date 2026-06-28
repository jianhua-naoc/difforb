import jax

from difforb.data import (
    DataNotInstalledError,
    get_data_dir,
    get_data_path,
    get_writable_data_path,
    missing_data_message,
)


DEFAULT_DATA_DIR = get_data_dir()
jax.config.update('jax_enable_x64', True)

__all__ = [
    "DEFAULT_DATA_DIR",
    "DataNotInstalledError",
    "get_data_dir",
    "get_data_path",
    "get_writable_data_path",
    "missing_data_message",
]
