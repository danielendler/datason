"""Shared traversal budgets for JSON representations before reconstruction."""

from __future__ import annotations

from typing import Any

from ._config import SerializationConfig
from ._errors import SecurityError


def check_tree(data: Any, config: SerializationConfig) -> None:
    """Check wire values, including plugin metadata, without executing plugins."""
    stack = [(data, 0)]
    visited = 0
    while stack:
        value, depth = stack.pop()
        visited += 1
        if visited > config.max_nodes:
            raise SecurityError(f"JSON node count exceeds limit {config.max_nodes}")
        if depth > config.max_depth:
            raise SecurityError(f"JSON depth {depth} exceeds limit {config.max_depth}")
        if isinstance(value, str) and len(value) > config.max_string_length:
            raise SecurityError(f"String length exceeds limit {config.max_string_length}")
        if isinstance(value, dict | list):
            if len(value) > config.max_size:
                raise SecurityError(f"Container size {len(value)} exceeds limit {config.max_size}")
            if isinstance(value, dict):
                stack.extend((k, depth) for k in value)
                stack.extend((v, depth + 1) for v in value.values())
            else:
                stack.extend((v, depth + 1) for v in value)


def check_input(s: str | bytes | bytearray, config: SerializationConfig) -> None:
    """Bound encoded input before JSON parsing allocates the object tree."""
    size = len(s.encode("utf-8")) if isinstance(s, str) else len(s)
    if size > config.max_input_bytes:
        raise SecurityError(f"JSON input bytes {size} exceed limit {config.max_input_bytes}")
