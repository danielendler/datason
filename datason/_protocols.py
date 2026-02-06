"""Protocol definitions for datason's plugin system.

All type handlers implement TypePlugin. The core serialization engine
dispatches to plugins via the registry — it never handles specific
types directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ._config import SerializationConfig


@dataclass
class SerializeContext:
    """State passed through the recursive serialization process."""

    config: SerializationConfig
    depth: int = 0
    seen_ids: set[int] = field(default_factory=set)

    def child(self) -> SerializeContext:
        """Create a child context with incremented depth."""
        return SerializeContext(
            config=self.config,
            depth=self.depth + 1,
            seen_ids=self.seen_ids,
        )


@dataclass
class DeserializeContext:
    """State passed through the recursive deserialization process."""

    config: SerializationConfig
    depth: int = 0

    def child(self) -> DeserializeContext:
        """Create a child context with incremented depth."""
        return DeserializeContext(
            config=self.config,
            depth=self.depth + 1,
        )


@runtime_checkable
class TypePlugin(Protocol):
    """Interface all type handler plugins must implement.

    Priority determines dispatch order (lower = checked first):
    - 0-99: Built-in overrides (reserved)
    - 100-199: Stdlib types (datetime, uuid, decimal, path)
    - 200-299: Data science (numpy, pandas)
    - 300-399: ML frameworks (torch, tensorflow, sklearn)
    - 400+: User-defined plugins
    """

    @property
    def name(self) -> str:
        """Unique name for this plugin."""
        ...

    @property
    def priority(self) -> int:
        """Dispatch priority (lower = checked first)."""
        ...

    def can_handle(self, obj: Any) -> bool:
        """Return True if this plugin can serialize the given object."""
        ...

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        """Convert obj to a JSON-serializable representation."""
        ...

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        """Return True if this plugin can reconstruct from the data dict."""
        ...

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        """Reconstruct the original object from serialized data."""
        ...
