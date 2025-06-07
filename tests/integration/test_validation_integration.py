import pytest

import datason


@pytest.mark.features
def test_serialize_pydantic_model() -> None:
    BaseModel = pytest.importorskip("pydantic").BaseModel

    class MyModel(BaseModel):
        a: int
        b: str

    model = MyModel(a=1, b="foo")
    result = datason.serialize(model)
    assert result == {"a": 1, "b": "foo"}

    result2 = datason.serialize_pydantic(model)
    assert result2 == result


@pytest.mark.features
def test_serialize_marshmallow_object() -> None:
    marshmallow = pytest.importorskip("marshmallow")

    class UserSchema(marshmallow.Schema):
        id = marshmallow.fields.Int()
        name = marshmallow.fields.Str()

    schema = UserSchema()
    user = schema.load({"id": 1, "name": "Alice"})

    result = datason.serialize(user)
    assert result == {"id": 1, "name": "Alice"}

    result2 = datason.serialize_marshmallow(user)
    assert result2 == result


@pytest.mark.features
def test_pydantic_helper_import_error(monkeypatch) -> None:
    """serialize_pydantic raises ImportError when pydantic is missing."""
    monkeypatch.setitem(datason.validation._LAZY_IMPORTS, "BaseModel", False)

    with pytest.raises(ImportError):
        datason.validation.serialize_pydantic(object())


def test_marshmallow_helper_import_error(monkeypatch) -> None:
    """serialize_marshmallow raises ImportError when marshmallow is missing."""
    monkeypatch.setitem(datason.validation._LAZY_IMPORTS, "Schema", False)

    with pytest.raises(ImportError):
        datason.validation.serialize_marshmallow(object())


def test_lazy_import_pydantic_cached(monkeypatch) -> None:
    """_lazy_import_pydantic_base_model caches the imported class."""

    import sys
    import types

    dummy = type("DummyModel", (), {})
    module = types.ModuleType("pydantic")
    module.BaseModel = dummy
    monkeypatch.setitem(sys.modules, "pydantic", module)
    datason.validation._LAZY_IMPORTS["BaseModel"] = None

    first = datason.validation._lazy_import_pydantic_base_model()
    assert first is dummy

    del sys.modules["pydantic"]
    second = datason.validation._lazy_import_pydantic_base_model()
    assert second is dummy  # cached


def test_lazy_import_marshmallow_cached(monkeypatch) -> None:
    """_lazy_import_marshmallow_schema caches the imported class."""

    import sys
    import types

    dummy_schema = type("DummySchema", (), {})
    module = types.ModuleType("marshmallow")
    module.Schema = dummy_schema
    monkeypatch.setitem(sys.modules, "marshmallow", module)
    datason.validation._LAZY_IMPORTS["Schema"] = None

    first = datason.validation._lazy_import_marshmallow_schema()
    assert first is dummy_schema

    del sys.modules["marshmallow"]
    second = datason.validation._lazy_import_marshmallow_schema()
    assert second is dummy_schema
