"""Built-in type handler plugins.

Plugins are auto-registered when this package is imported.
Each plugin handles serialization/deserialization for a specific
family of types (datetime, uuid, numpy, etc.).
"""

from .._registry import default_registry
from .datetime import DatetimePlugin
from .decimal import DecimalPlugin
from .path import PathPlugin
from .uuid import UUIDPlugin


def _register_builtins() -> None:
    """Register all built-in plugins with the default registry."""
    for plugin_cls in (DatetimePlugin, UUIDPlugin, DecimalPlugin, PathPlugin):
        default_registry.register(plugin_cls())


_register_builtins()
