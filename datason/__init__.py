"""JSON serialization for Python APIs, diagnostics, and typed stored data.

The core has zero runtime dependencies. Optional plugins support NumPy,
Pandas, and ML libraries under documented normalization and fidelity contracts.
Type tags are enabled by default; disable them for ordinary JSON consumers.

Quick start::

    import datetime as dt
    import datason

    event = {"observed": dt.datetime(2026, 10, 4, tzinfo=dt.timezone.utc)}
    text = datason.dumps(event)
    assert datason.loads(text) == event

Everyday operations: dumps, loads, dump, load, config.
Configuration enums, SerializationConfig, and preset factories are also exported.
See https://danielendler.github.io/datason/ for installation and policy details.
"""

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
