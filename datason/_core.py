"""Core serialization engine for datason.

Handles recursive traversal of Python objects, dispatching to plugins
for non-JSON-basic types. This module must stay under 300 lines.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from io import IOBase
from typing import Any

from ._config import SerializationConfig, _active_config, get_active_config
from ._errors import SecurityError, SerializationError
from ._protocols import SerializeContext
from ._registry import default_registry
from ._types import JSON_BASIC_TYPES
from .security.redaction import redact_value, should_redact_field

_REDACTED = "[REDACTED]"
_CONFIG_FIELDS = frozenset(SerializationConfig.__dataclass_fields__.keys())
_JSON_DUMPS_KWARGS = frozenset({
    "skipkeys",
    "ensure_ascii",
    "check_circular",
    "allow_nan",
    "cls",
    "indent",
    "separators",
    "default",
    "sort_keys",
})


def _serialize_recursive(obj: Any, ctx: SerializeContext) -> Any:
    """Recursively serialize obj to a JSON-safe representation."""
    # Security: depth limit
    if ctx.depth > ctx.config.max_depth:
        raise SecurityError(f"Serialization depth {ctx.depth} exceeds limit {ctx.config.max_depth}")

    # Security: circular reference detection
    obj_id = id(obj)
    if isinstance(obj, dict | list | tuple | set | frozenset):
        if obj_id in ctx.seen_ids:
            raise SecurityError(f"Circular reference detected for {type(obj).__name__}")
        ctx.seen_ids.add(obj_id)

    try:
        result = _serialize_value(obj, ctx)
    finally:
        # Remove from seen set after processing (allow same object
        # to appear in different branches of the tree)
        ctx.seen_ids.discard(obj_id)

    return result


def _serialize_value(obj: Any, ctx: SerializeContext) -> Any:
    """Serialize a single value, dispatching to plugins if needed."""
    # Fast path: JSON-basic types need no transformation
    if isinstance(obj, JSON_BASIC_TYPES):
        if isinstance(obj, float):
            return _handle_float(obj, ctx)
        return obj

    # Containers: recurse
    if isinstance(obj, dict):
        return _serialize_dict(obj, ctx)
    if isinstance(obj, list):
        return _serialize_sequence(obj, ctx)
    if isinstance(obj, tuple):
        return _serialize_sequence_with_type_hint(obj, "tuple", ctx)
    if isinstance(obj, set):
        return _serialize_sequence_with_type_hint(sorted(obj, key=repr), "set", ctx)
    if isinstance(obj, frozenset):
        return _serialize_sequence_with_type_hint(sorted(obj, key=repr), "frozenset", ctx)

    # Plugin dispatch
    plugin_result = default_registry.find_serializer(obj, ctx)
    if plugin_result is not None:
        _plugin, serialized = plugin_result
        return serialized

    # Fallback
    if ctx.config.fallback_to_string:
        return str(obj)

    raise SerializationError(
        f"Cannot serialize object of type {type(obj).__name__}. "
        f"No plugin registered for this type. "
        f"Set fallback_to_string=True in config to convert to str."
    )


def _handle_float(obj: float, ctx: SerializeContext) -> Any:
    """Handle NaN and Infinity according to config."""
    if math.isnan(obj) or math.isinf(obj):
        match ctx.config.nan_handling.value:
            case "null":
                return None
            case "string":
                if math.isnan(obj):
                    return "NaN"
                if math.isinf(obj):
                    return "Infinity" if obj > 0 else "-Infinity"
                return str(obj)
            case "keep":
                return obj
            case "drop":
                return None
    return obj


def _serialize_dict(obj: dict[str, Any], ctx: SerializeContext) -> dict[str, Any]:
    """Serialize a dict, recursing into values. Applies redaction if configured."""
    if len(obj) > ctx.config.max_size:
        raise SecurityError(f"Dict size {len(obj)} exceeds limit {ctx.config.max_size}")
    child = ctx.child()
    result: dict[str, Any] = {}
    for k, v in obj.items():
        key = str(k)
        if should_redact_field(key, ctx.config.redact_fields):
            result[key] = _REDACTED
        else:
            serialized = _serialize_recursive(v, child)
            result[key] = redact_value(serialized, ctx.config.redact_patterns)
    return result


def _serialize_sequence(obj: Any, ctx: SerializeContext) -> list[Any]:
    """Serialize a list/tuple/set, recursing into elements."""
    if len(obj) > ctx.config.max_size:
        raise SecurityError(f"Sequence size {len(obj)} exceeds limit {ctx.config.max_size}")
    child = ctx.child()
    return [_serialize_recursive(item, child) for item in obj]


def _serialize_sequence_with_type_hint(obj: Any, type_name: str, ctx: SerializeContext) -> Any:
    """Serialize tuple/set/frozenset with optional type hint metadata."""
    serialized = _serialize_sequence(obj, ctx)
    if ctx.config.include_type_hints:
        return {"__datason_type__": type_name, "__datason_value__": serialized}
    return serialized


# =========================================================================
# Public API
# =========================================================================


def dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize any Python object to a JSON string.

    Drop-in replacement for ``json.dumps`` that handles datetime, UUID,
    Decimal, Path, NumPy, Pandas, PyTorch, TensorFlow, and scikit-learn
    objects automatically.

    Args:
        obj: Any Python object to serialize.
        **kwargs: Override SerializationConfig fields inline
            (sort_keys, date_format, nan_handling, include_type_hints,
            fallback_to_string, redact_fields, redact_patterns, etc.).

    Returns:
        JSON string representation of obj.

    Raises:
        SecurityError: If depth/size limits exceeded or circular refs detected.
        SerializationError: If type is unknown and fallback_to_string is False.

    Examples::

        >>> import datason
        >>> datason.dumps({"name": "Alice", "age": 30})
        '{"name": "Alice", "age": 30}'
        >>> datason.dumps({"z": 1, "a": 2}, sort_keys=True)
        '{"a": 2, "z": 1}'
    """
    config_kwargs, json_kwargs = _split_dumps_kwargs(kwargs)
    config = _resolve_config(config_kwargs)
    ctx = SerializeContext(config=config)
    serialized = _serialize_recursive(obj, ctx)
    if "sort_keys" not in json_kwargs:
        json_kwargs["sort_keys"] = config.sort_keys
    if "ensure_ascii" not in json_kwargs:
        json_kwargs["ensure_ascii"] = False
    return json.dumps(serialized, **json_kwargs)


def dump(obj: Any, fp: IOBase, **kwargs: Any) -> None:
    """Serialize any Python object and write JSON to a file.

    Drop-in replacement for ``json.dump``.

    Args:
        obj: Any Python object to serialize.
        fp: File-like object with a write() method.
        **kwargs: Override SerializationConfig fields inline.

    Examples::

        >>> import datason, io
        >>> buf = io.StringIO()
        >>> datason.dump({"key": "value"}, buf)
        >>> buf.getvalue()
        '{"key": "value"}'
    """
    config_kwargs, json_kwargs = _split_dumps_kwargs(kwargs)
    config = _resolve_config(config_kwargs)
    ctx = SerializeContext(config=config)
    serialized = _serialize_recursive(obj, ctx)
    if "sort_keys" not in json_kwargs:
        json_kwargs["sort_keys"] = config.sort_keys
    if "ensure_ascii" not in json_kwargs:
        json_kwargs["ensure_ascii"] = False
    json.dump(serialized, fp, **json_kwargs)


@contextmanager
def config(**kwargs: Any):  # noqa: ANN201
    """Context manager to temporarily set serialization config.

    Thread-safe via ``contextvars.ContextVar``. Each thread/async task
    gets its own config scope.

    Args:
        **kwargs: Any SerializationConfig field to override.

    Yields:
        The active SerializationConfig instance.

    Examples::

        >>> import datason
        >>> with datason.config(sort_keys=True):
        ...     datason.dumps({"z": 1, "a": 2})
        '{"a": 2, "z": 1}'

        >>> from datason import NanHandling
        >>> with datason.config(nan_handling=NanHandling.STRING):
        ...     datason.dumps({"v": float("nan")})
        '{"v": "nan"}'
    """
    new_config = SerializationConfig(**kwargs)
    token = _active_config.set(new_config)
    try:
        yield new_config
    finally:
        _active_config.reset(token)


def _resolve_config(overrides: dict[str, Any]) -> SerializationConfig:
    """Resolve config from: inline kwargs > context var > defaults."""
    if overrides:
        base = get_active_config()
        # Build new config with overrides applied
        fields = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
        fields.update(overrides)
        return SerializationConfig(**fields)
    return get_active_config()


def _split_dumps_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split kwargs into datason config overrides and json.dumps kwargs."""
    config_kwargs: dict[str, Any] = {}
    json_kwargs: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in _CONFIG_FIELDS:
            config_kwargs[key] = value
        elif key in _JSON_DUMPS_KWARGS:
            json_kwargs[key] = value
        else:
            raise TypeError(f"dumps() got an unexpected keyword argument '{key}'")
    return config_kwargs, json_kwargs
