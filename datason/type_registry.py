"""Unified Type Registry for datason.

This module provides a unified approach to type handling where serialization
and deserialization logic are co-located to prevent maintenance issues.

Each type handler contains both serialize() and deserialize() methods,
ensuring they are always updated together.

This solves the "split-brain" architecture problem where serialization and
deserialization logic was scattered across different files, making it easy
to forget to update both when adding new types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class TypeHandler(ABC):
    """Base class for unified type handlers.

    Each type handler is responsible for both serialization and deserialization
    of a specific type, ensuring they stay in sync and preventing the
    split-brain architecture problem.
    """

    @abstractmethod
    def can_handle(self, obj: Any) -> bool:
        """Check if this handler can process the given object.

        Args:
            obj: Object to check

        Returns:
            True if this handler can process the object
        """
        pass

    @abstractmethod
    def serialize(self, obj: Any) -> Dict[str, Any]:
        """Serialize the object to a dict with type metadata.

        Args:
            obj: Object to serialize

        Returns:
            Dict with __datason_type__ and __datason_value__ keys
        """
        pass

    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Deserialize the data back to the original object type.

        Args:
            data: Dict with __datason_type__ and __datason_value__ keys

        Returns:
            Reconstructed object of the original type
        """
        pass

    @property
    @abstractmethod
    def type_name(self) -> str:
        """Return the type name used in metadata.

        This should be a unique identifier for the type that will be
        stored in the __datason_type__ field.

        Returns:
            Unique type identifier string
        """
        pass


class TypeRegistry:
    """Central registry for all type handlers.

    This registry manages all type handlers and provides a unified interface
    for serialization and deserialization. It ensures that each type has
    both serialization and deserialization logic in one place.
    """

    def __init__(self):
        """Initialize the type registry."""
        self._handlers: List[TypeHandler] = []

    def register_handler(self, handler: TypeHandler) -> None:
        """Register a new type handler.

        Args:
            handler: TypeHandler instance to register
        """
        self._handlers.append(handler)

    def find_handler(self, obj: Any) -> Optional[TypeHandler]:
        """Find the appropriate handler for an object.

        Args:
            obj: Object to find handler for

        Returns:
            TypeHandler instance if found, None otherwise
        """
        for handler in self._handlers:
            try:
                if handler.can_handle(obj):
                    return handler
            except Exception:
                # Skip handlers that fail during detection
                continue  # nosec B112
        return None

    def find_handler_by_type_name(self, type_name: str) -> Optional[TypeHandler]:
        """Find handler by type name.

        Args:
            type_name: Type name to search for

        Returns:
            TypeHandler instance if found, None otherwise
        """
        for handler in self._handlers:
            try:
                if handler.type_name == type_name:
                    return handler
            except Exception:
                continue
        return None

    def serialize(self, obj: Any) -> Optional[Dict[str, Any]]:
        """Serialize an object using the appropriate handler.

        Args:
            obj: Object to serialize

        Returns:
            Serialized dict if handler found, None otherwise
        """
        handler = self.find_handler(obj)
        if handler:
            return handler.serialize(obj)
        return None

    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Deserialize data using the appropriate handler.

        Args:
            data: Serialized data dict

        Returns:
            Deserialized object if handler found, original data otherwise
        """
        if not isinstance(data, dict) or "__datason_type__" not in data:
            return data

        type_name = data["__datason_type__"]
        handler = self.find_handler_by_type_name(type_name)

        if handler:
            return handler.deserialize(data)

        # No handler found, return original data
        return data

    def get_registered_types(self) -> List[str]:
        """Get list of all registered type names.

        Returns:
            List of type name strings
        """
        type_names = []
        for handler in self._handlers:
            try:
                type_names.append(handler.type_name)
            except Exception:
                continue
        return type_names

    def clear_handlers(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()


# Global registry instance
_global_registry = TypeRegistry()


def get_type_registry() -> TypeRegistry:
    """Get the global type registry.

    Returns:
        Global TypeRegistry instance
    """
    return _global_registry


def register_type_handler(handler: TypeHandler) -> None:
    """Register a type handler with the global registry.

    Args:
        handler: TypeHandler instance to register
    """
    _global_registry.register_handler(handler)


def serialize_with_registry(obj: Any) -> Optional[Dict[str, Any]]:
    """Serialize using the global registry.

    Args:
        obj: Object to serialize

    Returns:
        Serialized dict if handler found, None otherwise
    """
    return _global_registry.serialize(obj)


def deserialize_with_registry(data: Dict[str, Any]) -> Any:
    """Deserialize using the global registry.

    Args:
        data: Serialized data dict

    Returns:
        Deserialized object if handler found, original data otherwise
    """
    return _global_registry.deserialize(data)
