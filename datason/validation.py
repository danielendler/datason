# Optional integration helpers for Pydantic and Marshmallow
from typing import Any, Dict

from .core import serialize

_LAZY_IMPORTS = {
    "BaseModel": None,
    "Schema": None,
}


def _lazy_import_pydantic_base_model():
    """Lazily import pydantic.BaseModel."""
    if _LAZY_IMPORTS["BaseModel"] is None:
        try:
            from pydantic import BaseModel

            _LAZY_IMPORTS["BaseModel"] = BaseModel
        except Exception:
            _LAZY_IMPORTS["BaseModel"] = False
    return _LAZY_IMPORTS["BaseModel"] if _LAZY_IMPORTS["BaseModel"] is not False else None


def _lazy_import_marshmallow_schema():
    """Lazily import marshmallow.Schema."""
    if _LAZY_IMPORTS["Schema"] is None:
        try:
            from marshmallow import Schema

            _LAZY_IMPORTS["Schema"] = Schema
        except Exception:
            _LAZY_IMPORTS["Schema"] = False
    return _LAZY_IMPORTS["Schema"] if _LAZY_IMPORTS["Schema"] is not False else None


def serialize_pydantic(obj: Any) -> Any:
    """Serialize a Pydantic model using datason."""
    BaseModel = _lazy_import_pydantic_base_model()
    if BaseModel is None:
        raise ImportError("Pydantic is required for serialize_pydantic")
    if isinstance(obj, BaseModel):
        try:
            data = obj.model_dump()  # Pydantic v2
        except AttributeError:
            try:
                data = obj.dict()  # Pydantic v1
            except Exception:
                data = obj.__dict__
        return serialize(data)
    return serialize(obj)


def serialize_marshmallow(obj: Any) -> Any:
    """Serialize a Marshmallow schema object or validated data."""
    Schema = _lazy_import_marshmallow_schema()
    if Schema is None:
        raise ImportError("Marshmallow is required for serialize_marshmallow")
    if isinstance(obj, Schema):
        try:
            fields: Dict[str, Any] = {name: field.__class__.__name__ for name, field in obj.fields.items()}
            return serialize(fields)
        except Exception:
            return serialize(obj.__dict__)
    return serialize(obj)


# Attribute access for tests


def __getattr__(name: str) -> Any:
    if name == "BaseModel":
        return _lazy_import_pydantic_base_model()
    if name == "Schema":
        return _lazy_import_marshmallow_schema()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
