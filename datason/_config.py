"""Serialization configuration for datason.

Provides a SerializationConfig dataclass with sensible defaults
and preset factory functions for common use cases.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DateFormat(Enum):
    """How to serialize datetime objects."""

    ISO = "iso"
    UNIX = "unix"
    UNIX_MS = "unix_ms"
    STRING = "string"


class NanHandling(Enum):
    """How to handle NaN and Infinity in floats."""

    NULL = "null"
    STRING = "string"
    KEEP = "keep"
    DROP = "drop"


class DataFrameOrient(Enum):
    """Pandas DataFrame serialization orientation."""

    RECORDS = "records"
    SPLIT = "split"
    DICT = "dict"
    LIST = "list"
    VALUES = "values"


@dataclass(frozen=True)
class SerializationConfig:
    """Configuration controlling serialization behavior.

    Immutable (frozen) to prevent accidental mutation. Create new
    instances with ``dataclasses.replace()`` or use preset factories:
    ``ml_config()``, ``api_config()``, ``strict_config()``,
    ``performance_config()``.

    Use inline kwargs or the ``datason.config()`` context manager::

        datason.dumps(data, sort_keys=True, nan_handling=NanHandling.STRING)

        with datason.config(date_format=DateFormat.UNIX):
            datason.dumps(data)
    """

    # Type formatting
    date_format: DateFormat = DateFormat.ISO
    dataframe_orient: DataFrameOrient = DataFrameOrient.RECORDS
    nan_handling: NanHandling = NanHandling.NULL
    include_type_hints: bool = True
    sort_keys: bool = False

    # Security limits
    max_depth: int = 50
    max_size: int = 100_000
    max_string_length: int = 1_000_000

    # Behavior
    fallback_to_string: bool = False
    strict: bool = True

    # Deserialization safety
    allow_plugin_deserialization: bool = True

    # Redaction (optional, for security module)
    redact_fields: tuple[str, ...] = field(default_factory=tuple)
    redact_patterns: tuple[str, ...] = field(default_factory=tuple)


def ml_config(**overrides: Any) -> SerializationConfig:
    """Preset for ML workflows: UNIX_MS dates, lenient handling.

    Settings: date_format=UNIX_MS, nan_handling=NULL,
    include_type_hints=True, fallback_to_string=True.

    Example::

        with datason.config(**ml_config().__dict__):
            datason.dumps({"weights": np.array([0.1, 0.9])})
    """
    defaults: dict[str, Any] = {
        "date_format": DateFormat.UNIX_MS,
        "nan_handling": NanHandling.NULL,
        "include_type_hints": True,
        "fallback_to_string": True,
    }
    defaults.update(overrides)
    return SerializationConfig(**defaults)


def api_config(**overrides: Any) -> SerializationConfig:
    """Preset for API responses: ISO dates, sorted keys, no type hints.

    Settings: date_format=ISO, sort_keys=True, nan_handling=NULL,
    include_type_hints=False.

    Example::

        with datason.config(**api_config().__dict__):
            datason.dumps({"created": dt.datetime.now(), "status": "ok"})
    """
    defaults: dict[str, Any] = {
        "date_format": DateFormat.ISO,
        "sort_keys": True,
        "nan_handling": NanHandling.NULL,
        "include_type_hints": False,
    }
    defaults.update(overrides)
    return SerializationConfig(**defaults)


def strict_config(**overrides: Any) -> SerializationConfig:
    """Preset for strict validation: no fallbacks, type hints required.

    Settings: strict=True, fallback_to_string=False,
    include_type_hints=True. Unknown types raise SerializationError.
    """
    defaults: dict[str, Any] = {
        "strict": True,
        "fallback_to_string": False,
        "include_type_hints": True,
    }
    defaults.update(overrides)
    return SerializationConfig(**defaults)


def performance_config(**overrides: Any) -> SerializationConfig:
    """Preset for maximum speed: no type hints, no sorting, keep NaN.

    Settings: include_type_hints=False, sort_keys=False,
    nan_handling=KEEP, fallback_to_string=True. Fastest output,
    but no round-trip reconstruction.
    """
    defaults: dict[str, Any] = {
        "include_type_hints": False,
        "sort_keys": False,
        "nan_handling": NanHandling.KEEP,
        "fallback_to_string": True,
    }
    defaults.update(overrides)
    return SerializationConfig(**defaults)


# Context variable for scoped config
_active_config: ContextVar[SerializationConfig | None] = ContextVar("_active_config", default=None)


def get_active_config() -> SerializationConfig:
    """Return the active config (from context var or default)."""
    return _active_config.get() or SerializationConfig()
