"""Deferred built-in plugins; import targets come only from reviewed code."""

from __future__ import annotations

import importlib
import threading
from typing import Any, cast

from .._protocols import DeserializeContext, SerializeContext, TypePlugin
from .._types import TYPE_METADATA_KEY


def matches_family(obj: Any, roots: tuple[str, ...]) -> bool:
    """Include application subclasses without importing their module names."""
    for base in type(obj).__mro__:
        module = getattr(base, "__module__", None)
        if isinstance(module, str) and module.partition(".")[0] in roots:
            return True
    return False


class LazyPlugin:
    """Keep dispatch priority while constructing an optional plugin once on demand."""

    def __init__(
        self, name: str, priority: int, class_name: str, roots: tuple[str, ...], tags: tuple[str, ...]
    ) -> None:
        self.name = name
        self.priority = priority
        self._class_name = class_name
        self._roots = roots
        self._tags = tags
        self._plugin: TypePlugin | None = None
        self._attempted = False
        self._lock = threading.Lock()

    def _load(self) -> TypePlugin | None:
        if not self._attempted:
            with self._lock:
                if not self._attempted:
                    try:
                        module = importlib.import_module(f"datason.plugins.{self.name}")
                        self._plugin = cast(TypePlugin, getattr(module, self._class_name)())
                        # Keep the descriptor's identity/priority, but bypass its
                        # loading checks once the fully constructed handler exists.
                        self.can_handle = self._plugin.can_handle
                        self.serialize = self._plugin.serialize
                        self.deserialize = self._plugin.deserialize
                    except ImportError:
                        # Match the previous optional registration's unavailable behavior.
                        pass
                    self._attempted = True
        return self._plugin

    def can_handle(self, obj: Any) -> bool:
        if self._plugin is None and not matches_family(obj, self._roots):
            return False
        plugin = self._load()
        return plugin is not None and plugin.can_handle(obj)

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        from .._errors import PluginError

        plugin = self._load()
        if plugin is None:
            raise PluginError(f"Optional plugin '{self.name}' is unavailable")
        return plugin.serialize(obj, ctx)

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        tag = data.get(TYPE_METADATA_KEY)
        if not isinstance(tag, str) or not tag.startswith(self._tags):
            return False
        plugin = self._plugin if self._plugin is not None else self._load()
        return plugin is not None and plugin.can_deserialize(data)

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        from .._errors import PluginError

        plugin = self._load()
        if plugin is None:
            raise PluginError(f"Optional plugin '{self.name}' is unavailable")
        return plugin.deserialize(data, ctx)
