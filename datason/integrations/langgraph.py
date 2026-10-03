"""JSON-only implementation of LangGraph's typed serializer protocol."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .._config import SerializationConfig
from .._core import dumps
from .._deserialize import loads
from .._errors import DeserializationError

_FORMAT = "datason-json-v1"


class DatasonSerializer:
    """Persist supported JSON state without automatic application-class hydration.

    Implements ``langgraph.checkpoint.serde.base.SerializerProtocol`` structurally.
    LangGraph is optional and is not imported here. Callbacks and registered
    datason plugins are trusted; this adapter is not an untrusted-code sandbox.
    """

    def __init__(self, config: SerializationConfig | None = None) -> None:
        config = config or SerializationConfig()
        if not config.include_type_hints or config.fallback_to_string or not config.strict:
            raise ValueError("Checkpoint serialization requires type hints, strict loading, and no string fallback")
        if config.redact_fields or config.redact_patterns:
            raise ValueError("Use a separate diagnostic export for redaction; checkpoint state must remain usable")
        self._options = asdict(config)

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return _FORMAT, dumps(obj, **self._options).encode("utf-8")

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        format_name, payload = data
        if format_name != _FORMAT:
            raise DeserializationError(f"Unsupported checkpoint format: {format_name}")
        return loads(payload, **self._options)
