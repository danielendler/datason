"""Datason error hierarchy.

Error handling policy:
- SecurityError: Always fatal, never swallowed.
- SerializationError: Fatal by default, configurable fallback to str(obj).
- DeserializationError: Fatal by default, configurable fallback.
- PluginError: Logged via warnings.warn(), falls back to next plugin.
"""


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


class DeserializationError(DatasonError):
    """Raised when data cannot be deserialized.

    Fatal by default. With config.strict=False, unrecognized type metadata
    is returned as-is instead of raising.
    """


class PluginError(DatasonError):
    """Raised when a plugin fails during serialize/deserialize.

    Non-fatal — the registry logs a warning and tries the next plugin.
    """
