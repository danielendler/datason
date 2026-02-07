"""datason: Drop-in replacement for json that handles datetime, NumPy, Pandas, and 50+ types.

Zero-dependency Python serialization library. Replace ``json.dumps``
with ``datason.dumps`` to serialize datetime, UUID, Decimal, Path,
NumPy arrays, Pandas DataFrames, PyTorch tensors, TensorFlow tensors,
and scikit-learn models to JSON automatically.

Quick start::

    import datason
    import datetime as dt
    import numpy as np

    data = {"ts": dt.datetime.now(), "scores": np.array([0.9, 0.1])}
    json_str = datason.dumps(data)
    restored = datason.loads(json_str)  # types are reconstructed

API:
    - ``datason.dumps(obj, **config)`` -- serialize to JSON string
    - ``datason.loads(s, **config)`` -- deserialize from JSON string
    - ``datason.dump(obj, fp, **config)`` -- write to file
    - ``datason.load(fp, **config)`` -- read from file
    - ``datason.config(**config)`` -- context manager for temp config
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
