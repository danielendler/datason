"""Plugin for scikit-learn estimator serialization.

Handles all BaseEstimator subclasses (including Pipeline). Estimators
are serialized via __getstate__() for full fitted-model round-trip.
State dicts are recursively serialized through the core engine so
numpy arrays inside get handled by the numpy plugin.

Security: deserialization only allows importing classes from the
sklearn.* namespace. This prevents arbitrary code execution via
malicious class paths in serialized data.

This module imports sklearn directly — if sklearn is not installed,
the ImportError is caught by plugins/__init__.py and this plugin is
simply not registered.
"""

from __future__ import annotations

import importlib
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from .._errors import PluginError
from .._protocols import DeserializeContext, SerializeContext
from .._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY


class SklearnPlugin:
    """Handles serialization/deserialization of scikit-learn estimators."""

    @property
    def name(self) -> str:
        return "sklearn"

    @property
    def priority(self) -> int:
        return 302

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, BaseEstimator)

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        return _serialize_sklearn(obj, ctx)

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        type_name = data.get(TYPE_METADATA_KEY, "")
        return isinstance(type_name, str) and type_name.startswith("sklearn.")

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        return _deserialize_sklearn(data, ctx)


def _serialize_sklearn(obj: Any, ctx: SerializeContext) -> Any:
    """Serialize an sklearn estimator or pipeline."""
    if isinstance(obj, Pipeline):
        return _serialize_pipeline(obj, ctx)
    return _serialize_estimator(obj, ctx)


def _serialize_estimator(estimator: BaseEstimator, ctx: SerializeContext) -> Any:
    """Serialize a single estimator with its full state."""
    # Import here to avoid circular import at module level
    from .._core import _serialize_recursive  # pyright: ignore[reportPrivateUsage]

    class_path = f"{type(estimator).__module__}.{type(estimator).__qualname__}"
    state = estimator.__getstate__()
    child = ctx.child()

    value = {
        "class": class_path,
        "params": estimator.get_params(deep=False),
        "state": {k: _serialize_recursive(v, child) for k, v in state.items()},
    }
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "sklearn.estimator", VALUE_METADATA_KEY: value}
    return value


def _serialize_pipeline(pipeline: Pipeline, ctx: SerializeContext) -> Any:
    """Serialize a Pipeline with its steps."""
    from .._core import _serialize_recursive  # pyright: ignore[reportPrivateUsage]

    child = ctx.child()
    value = {
        "class": f"{type(pipeline).__module__}.{type(pipeline).__qualname__}",
        "steps": [{"name": name, "estimator": _serialize_recursive(est, child)} for name, est in pipeline.steps],
    }
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "sklearn.Pipeline", VALUE_METADATA_KEY: value}
    return value


def _deserialize_sklearn(data: dict[str, Any], ctx: DeserializeContext) -> Any:
    """Reconstruct an sklearn object from serialized data."""
    type_name = data[TYPE_METADATA_KEY]
    value = data[VALUE_METADATA_KEY]

    match type_name:
        case "sklearn.estimator":
            return _reconstruct_estimator(value, ctx)
        case "sklearn.Pipeline":
            return _reconstruct_pipeline(value, ctx)
        case _:
            raise PluginError(f"Unknown sklearn type: {type_name}")


def _reconstruct_estimator(value: Any, ctx: DeserializeContext) -> BaseEstimator:
    """Reconstruct an estimator from its serialized state."""
    from .._deserialize import _deserialize_recursive  # pyright: ignore[reportPrivateUsage]

    if not isinstance(value, dict):
        raise PluginError(f"Expected dict for estimator, got {type(value).__name__}")

    class_path = value["class"]
    cls = _import_sklearn_class(class_path)
    state = value.get("state", {})

    child = ctx.child()
    deserialized_state = {k: _deserialize_recursive(v, child) for k, v in state.items()}

    instance = object.__new__(cls)
    instance.__setstate__(deserialized_state)
    return instance


def _reconstruct_pipeline(value: Any, ctx: DeserializeContext) -> Pipeline:
    """Reconstruct a Pipeline from its serialized steps."""
    from .._deserialize import _deserialize_recursive  # pyright: ignore[reportPrivateUsage]

    if not isinstance(value, dict):
        raise PluginError(f"Expected dict for Pipeline, got {type(value).__name__}")

    child = ctx.child()
    steps = [(step["name"], _deserialize_recursive(step["estimator"], child)) for step in value["steps"]]
    return Pipeline(steps=steps)


def _import_sklearn_class(class_path: str) -> type:
    """Securely import a class, restricted to sklearn namespace."""
    if not class_path.startswith("sklearn."):
        raise PluginError(f"Security: refusing to import non-sklearn class: {class_path}")
    module_path, _, class_name = class_path.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise PluginError(f"Class not found: {class_path}")
    return cls
