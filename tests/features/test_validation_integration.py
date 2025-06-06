import datason
import pytest


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
    # Simulate missing pydantic
    import importlib

    monkeypatch.setitem(datason.validation._LAZY_IMPORTS, "BaseModel", False)

    with pytest.raises(ImportError):
        datason.validation.serialize_pydantic(object())

