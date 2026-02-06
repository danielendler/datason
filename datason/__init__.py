"""datason — Zero-dependency Python serialization with intelligent type handling."""

from importlib.metadata import version

import datason.plugins  # noqa: F401  # pyright: ignore[reportUnusedImport]

from ._config import (
    DataFrameOrient,
    DateFormat,
    NanHandling,
    SerializationConfig,
    api_config,
    ml_config,
    performance_config,
    strict_config,
)
from ._core import config, dump, dumps
from ._deserialize import load, loads

__version__ = version("datason")

__all__ = [
    "dumps",
    "loads",
    "dump",
    "load",
    "config",
    "SerializationConfig",
    "DateFormat",
    "NanHandling",
    "DataFrameOrient",
    "ml_config",
    "api_config",
    "strict_config",
    "performance_config",
]
