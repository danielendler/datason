import builtins
import sys

import datason


def test_core_handles_missing_validation_imports(monkeypatch):
    """serialize should work even if validation imports fail."""
    def raise_import_error(name):
        raise ImportError("no validation")

    monkeypatch.setattr(datason.validation, "__getattr__", raise_import_error, raising=False)
    monkeypatch.setattr(datason.validation, "serialize_pydantic", lambda obj: (_ for _ in ()).throw(AssertionError()), raising=False)
    monkeypatch.setattr(datason.validation, "serialize_marshmallow", lambda obj: (_ for _ in ()).throw(AssertionError()), raising=False)

    result = datason.serialize({"a": 1})
    assert result == {"a": 1}


def test_core_serializes_schema_instance(monkeypatch):
    """serialize should detect marshmallow Schema instances."""
    class DummyField:
        pass

    class DummySchema:
        def __init__(self):
            self.fields = {"x": DummyField(), "y": DummyField()}

    datason.validation._LAZY_IMPORTS["Schema"] = DummySchema
    # ensure __getattr__ returns our dummy class
    monkeypatch.setattr(datason.validation, "_lazy_import_marshmallow_schema", lambda: DummySchema)

    obj = DummySchema()
    result = datason.serialize(obj)
    assert result == {"x": "DummyField", "y": "DummyField"}


def test_serialize_marshmallow_fields_exception(monkeypatch):
    """serialize_marshmallow should fall back to __dict__ on error."""
    class DummySchema:
        pass

    class BadSchema(DummySchema):
        def __init__(self):
            self.value = 123

        @property
        def fields(self):
            raise RuntimeError("boom")

    datason.validation._LAZY_IMPORTS["Schema"] = DummySchema
    monkeypatch.setattr(datason.validation, "_lazy_import_marshmallow_schema", lambda: DummySchema)

    obj = BadSchema()
    result = datason.validation.serialize_marshmallow(obj)
    assert result == {"value": 123}


def test_lazy_import_pydantic_failure(monkeypatch):
    """_lazy_import_pydantic_base_model returns None when import fails."""
    datason.validation._LAZY_IMPORTS["BaseModel"] = None

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pydantic"):
            raise ImportError("no pydantic")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    if "pydantic" in sys.modules:
        monkeypatch.delitem(sys.modules, "pydantic", raising=False)

    result = datason.validation._lazy_import_pydantic_base_model()
    assert result is None
    assert datason.validation._LAZY_IMPORTS["BaseModel"] is False

    monkeypatch.setattr(builtins, "__import__", original_import)


def test_lazy_import_marshmallow_failure(monkeypatch):
    """_lazy_import_marshmallow_schema returns None when import fails."""
    datason.validation._LAZY_IMPORTS["Schema"] = None
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("marshmallow"):
            raise ImportError("no mm")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    if "marshmallow" in sys.modules:
        monkeypatch.delitem(sys.modules, "marshmallow", raising=False)

    result = datason.validation._lazy_import_marshmallow_schema()
    assert result is None
    assert datason.validation._LAZY_IMPORTS["Schema"] is False

    monkeypatch.setattr(builtins, "__import__", original_import)


def test_serialize_pydantic_fallbacks(monkeypatch):
    """serialize_pydantic should fall back to dict and __dict__."""
    class Base:
        pass

    class ModelDict(Base):
        def __init__(self):
            self.a = 1

        def dict(self):
            return {"a": self.a}

    class ModelNoMethods(Base):
        def __init__(self):
            self.b = 2

    datason.validation._LAZY_IMPORTS["BaseModel"] = Base
    monkeypatch.setattr(datason.validation, "_lazy_import_pydantic_base_model", lambda: Base)

    assert datason.validation.serialize_pydantic(ModelDict()) == {"a": 1}
    assert datason.validation.serialize_pydantic(ModelNoMethods()) == {"b": 2}
