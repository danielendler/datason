import pytest

import datason


def test_pydantic_round_trip() -> None:
    BaseModel = pytest.importorskip("pydantic").BaseModel

    class Model(BaseModel):
        a: int
        b: str

    raw = {"a": 2, "b": "bar"}
    model = Model(**raw)

    serialized = datason.serialize(model)
    deserialized = datason.deserialize(serialized)
    new_model = Model(**deserialized)

    assert new_model == model


def test_marshmallow_round_trip() -> None:
    marshmallow = pytest.importorskip("marshmallow")

    class UserSchema(marshmallow.Schema):
        id = marshmallow.fields.Int()
        name = marshmallow.fields.Str()

    schema = UserSchema()
    raw = {"id": 3, "name": "Bob"}
    user = schema.load(raw)

    serialized = datason.serialize(user)
    deserialized = datason.deserialize(serialized)
    user2 = schema.load(deserialized)

    assert user2 == user
