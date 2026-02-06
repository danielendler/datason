"""Plugin registry for datason type handlers.

The registry maintains a priority-sorted list of TypePlugin instances.
The core serializer dispatches to plugins via find_serializer/find_deserializer
instead of using if/elif chains.
"""

from __future__ import annotations

import threading
import warnings
from typing import Any

from ._errors import PluginError
from ._protocols import DeserializeContext, SerializeContext, TypePlugin
from ._types import TYPE_METADATA_KEY


class PluginRegistry:
    """Thread-safe registry of TypePlugin instances."""

    def __init__(self) -> None:
        self._plugins: list[TypePlugin] = []
        self._lock = threading.Lock()

    def register(self, plugin: TypePlugin) -> None:
        """Register a plugin, maintaining priority sort order."""
        with self._lock:
            self._plugins.append(plugin)
            self._plugins.sort(key=lambda p: p.priority)

    def find_serializer(self, obj: Any, ctx: SerializeContext) -> tuple[TypePlugin, Any] | None:
        """Find a plugin that can serialize obj, and return (plugin, result).

        Returns None if no plugin can handle the object.
        Plugins that raise PluginError are skipped with a warning.
        """
        with self._lock:
            plugins = list(self._plugins)

        for plugin in plugins:
            try:
                if plugin.can_handle(obj):
                    result = plugin.serialize(obj, ctx)
                    return (plugin, result)
            except PluginError as e:
                warnings.warn(
                    f"Plugin '{plugin.name}' failed on {type(obj).__name__}: {e}",
                    stacklevel=2,
                )
                continue
        return None

    def find_deserializer(self, data: dict[str, Any], ctx: DeserializeContext) -> tuple[TypePlugin, Any] | None:
        """Find a plugin that can deserialize data, and return (plugin, result).

        The data dict must contain TYPE_METADATA_KEY to be considered.
        Returns None if no plugin can handle the data.
        """
        if TYPE_METADATA_KEY not in data:
            return None

        with self._lock:
            plugins = list(self._plugins)

        for plugin in plugins:
            try:
                if plugin.can_deserialize(data):
                    result = plugin.deserialize(data, ctx)
                    return (plugin, result)
            except PluginError as e:
                warnings.warn(
                    f"Plugin '{plugin.name}' failed deserializing type '{data.get(TYPE_METADATA_KEY)}': {e}",
                    stacklevel=2,
                )
                continue
        return None

    @property
    def plugin_count(self) -> int:
        """Number of registered plugins."""
        with self._lock:
            return len(self._plugins)

    def clear(self) -> None:
        """Remove all registered plugins (useful for testing)."""
        with self._lock:
            self._plugins.clear()


# Global default registry
default_registry = PluginRegistry()
