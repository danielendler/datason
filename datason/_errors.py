"""Datason error hierarchy.

Error handling policy:
- SecurityError: Always fatal, never swallowed.
- SerializationError: Fatal by default, configurable fallback to str(obj).
- DeserializationError: Fatal by default, configurable fallback.
- PluginError: Logged via warnings.warn(), falls back to next plugin.
"""

import json


class DatasonError(Exception):
    """Base class for all datason errors."""


class SecurityError(DatasonError):
    """Raised when security limits are exceeded (depth, size, circular refs).

    Always fatal — never catch and ignore this.
    """


class SerializationError(DatasonError):
    """Raised when an object cannot be serialized.

    Fatal by default. With config.fallback_to_string=True, objects are
    converted to str() instead of raising.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.path: str | None = None
        self._path_segments: list[str | int] = []

    def add_path_segment(self, segment: str | int) -> None:
        """Record an ancestor during unwinding without changing a located error."""
        if self.path is None:
            self._path_segments.append(segment)

    def annotate_path(self) -> None:
        """Attach the collected path once, without formatting successful values."""
        if self.path is not None:
            return
        parts = ["$"]
        for segment in reversed(self._path_segments):
            if isinstance(segment, int):
                parts.append(f"[{segment}]")
            elif segment.isidentifier():
                parts.append(f".{segment}")
            else:
                parts.append(f"[{json.dumps(segment, ensure_ascii=False)}]")
        self.path = "".join(parts)
        self.args = (f"{self} (at {self.path})",)


class DeserializationError(DatasonError):
    """Raised when data cannot be deserialized.

    Fatal by default. With config.strict=False, unrecognized type metadata
    is returned as-is instead of raising.
    """


class PluginError(DatasonError):
    """Raised when a plugin fails during serialize/deserialize.

    Non-fatal — the registry logs a warning and tries the next plugin.
    """
