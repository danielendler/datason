"""Core deserialization engine for datason.

Handles recursive traversal of parsed JSON data, dispatching to
plugins for type-annotated dicts. This module must stay under 300 lines.
"""

from __future__ import annotations

import json
from io import IOBase
from typing import Any

from ._config import SerializationConfig, get_active_config
from ._errors import DeserializationError, SecurityError
from ._protocols import DeserializeContext
from ._registry import default_registry
from ._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY


def _deserialize_recursive(data: Any, ctx: DeserializeContext) -> Any:
    """Recursively deserialize parsed JSON data back to Python objects."""
    # Security: depth limit
    if ctx.depth > ctx.config.max_depth:
        raise SecurityError(f"Deserialization depth {ctx.depth} exceeds limit {ctx.config.max_depth}")

    # Primitive types: pass through
    if data is None or isinstance(data, str | int | float | bool):
        return data

    # Dict: check for type metadata, then recurse
    if isinstance(data, dict):
        return _deserialize_dict(data, ctx)

    # List: recurse into elements
    if isinstance(data, list):
        return _deserialize_list(data, ctx)

    return data


def _deserialize_dict(data: dict[str, Any], ctx: DeserializeContext) -> Any:
    """Deserialize a dict, checking for type metadata first."""
    # Check if this dict is a type-annotated value
    if TYPE_METADATA_KEY in data:
        plugin_result = default_registry.find_deserializer(data, ctx)
        if plugin_result is not None:
            _plugin, deserialized = plugin_result
            return deserialized

    # Built-in type hints for core collection types
    type_name = data.get(TYPE_METADATA_KEY)
    if type_name in {"tuple", "set", "frozenset"}:
        raw_value = data.get(VALUE_METADATA_KEY)
        if not isinstance(raw_value, list):
            if ctx.config.strict:
                raise DeserializationError(
                    f"Cannot deserialize {type_name}: expected list in {VALUE_METADATA_KEY}."
                )
            return data
        child = ctx.child()
        items = [_deserialize_recursive(item, child) for item in raw_value]
        if type_name == "tuple":
            return tuple(items)
        if type_name == "set":
            return set(items)
        return frozenset(items)

    if TYPE_METADATA_KEY in data and ctx.config.strict:
        strict_type_name = data.get(TYPE_METADATA_KEY, "<unknown>")
        raise DeserializationError(
            f"No plugin registered to deserialize type "
            f"'{strict_type_name}'. Install the relevant plugin or "
            f"set strict=False in config."
        )

    # Regular dict: recurse into values
    child = ctx.child()
    return {k: _deserialize_recursive(v, child) for k, v in data.items()}


def _deserialize_list(data: list[Any], ctx: DeserializeContext) -> list[Any]:
    """Deserialize a list by recursing into each element."""
    child = ctx.child()
    return [_deserialize_recursive(item, child) for item in data]


# =========================================================================
# Public API
# =========================================================================


def loads(s: str, **kwargs: Any) -> Any:
    """Deserialize a JSON string back to Python objects.

    Drop-in replacement for ``json.loads``. Values serialized with
    ``datason.dumps`` that contain ``__datason_type__`` metadata are
    automatically reconstructed to their original types (datetime,
    UUID, NumPy array, DataFrame, etc.).

    Args:
        s: JSON string to deserialize.
        **kwargs: Override SerializationConfig fields inline
            (strict, fallback_to_string, etc.).

    Returns:
        Deserialized Python object with types reconstructed.

    Raises:
        SecurityError: If depth limit is exceeded.
        DeserializationError: If type metadata is unrecognized and strict=True.

    Examples::

        >>> import datason
        >>> datason.loads('{"name": "Alice", "age": 30}')
        {'name': 'Alice', 'age': 30}

        >>> import datetime as dt
        >>> original = {"ts": dt.datetime(2024, 1, 15)}
        >>> restored = datason.loads(datason.dumps(original))
        >>> isinstance(restored["ts"], dt.datetime)
        True
    """
    _validate_load_kwargs(kwargs)
    config = _resolve_config({k: v for k, v in kwargs.items() if k in _CONFIG_FIELDS})
    ctx = DeserializeContext(config=config)
    parsed = json.loads(s, **_extract_json_loads_kwargs(kwargs))
    return _deserialize_recursive(parsed, ctx)


def load(fp: IOBase, **kwargs: Any) -> Any:
    """Read a JSON file and deserialize back to Python objects.

    Drop-in replacement for ``json.load``.

    Args:
        fp: File-like object with a read() method.
        **kwargs: Override SerializationConfig fields inline.

    Returns:
        Deserialized Python object with types reconstructed.

    Examples::

        >>> import datason, io
        >>> buf = io.StringIO('{"key": "value"}')
        >>> datason.load(buf)
        {'key': 'value'}
    """
    _validate_load_kwargs(kwargs)
    config = _resolve_config({k: v for k, v in kwargs.items() if k in _CONFIG_FIELDS})
    ctx = DeserializeContext(config=config)
    parsed = json.load(fp, **_extract_json_loads_kwargs(kwargs))
    return _deserialize_recursive(parsed, ctx)


_CONFIG_FIELDS = frozenset(SerializationConfig.__dataclass_fields__.keys())
_JSON_LOADS_KWARGS = frozenset({
    "cls",
    "object_hook",
    "parse_float",
    "parse_int",
    "parse_constant",
    "object_pairs_hook",
})


def _validate_load_kwargs(kwargs: dict[str, Any]) -> None:
    """Validate kwargs for loads/load."""
    allowed = _CONFIG_FIELDS | _JSON_LOADS_KWARGS
    for key in kwargs:
        if key not in allowed:
            raise TypeError(f"loads() got an unexpected keyword argument '{key}'")


def _resolve_config(overrides: dict[str, Any]) -> SerializationConfig:
    """Resolve config from kwargs or active context."""
    if overrides:
        base = get_active_config()
        fields = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
        fields.update(overrides)
        return SerializationConfig(**fields)
    return get_active_config()


def _extract_json_loads_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract kwargs intended for json.loads/json.load."""
    return {k: v for k, v in kwargs.items() if k in _JSON_LOADS_KWARGS}
