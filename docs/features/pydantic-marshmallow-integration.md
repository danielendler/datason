# Pydantic & Marshmallow Integration

This document outlines how to integrate Datason with popular schema validation libraries **Pydantic** and **Marshmallow**. The goal is to keep validation and schema generation in their respective libraries while leveraging Datason for serialization and deserialization.

## Background

Many users validate input data with Pydantic or Marshmallow before working with ML/AI workflows. They want an easy way to serialize these validated objects using Datason without losing type fidelity.

## Key Points

- **Validation** remains the responsibility of Pydantic or Marshmallow.
- **Serialization** is handled by Datason.
- Integration helpers are optional and incur no new default dependencies.

## Serializing Pydantic Models

```python
from pydantic import BaseModel
import datason

class MyModel(BaseModel):
    a: int
    b: str

model = MyModel(a=1, b="foo")
json_data = datason.serialize(model)  # Or datason.serialize_pydantic(model)
```

Datason automatically extracts fields from `BaseModel` instances, including nested models. Type information is preserved so the data can be deserialized back to Python primitives or dictionaries.

## Serializing Marshmallow Objects

```python
from marshmallow import Schema, fields
import datason

class UserSchema(Schema):
    id = fields.Int()
    name = fields.Str()

schema = UserSchema()
user = schema.load({"id": 1, "name": "Alice"})
json_data = datason.serialize(user)  # Or datason.serialize_marshmallow(user)
```

The helpers work with the results of `.load()` or `.dump()`, enabling seamless round-tripping through Datason without rewriting validation logic.

## Optional Dependencies

Datason’s core package does not require Pydantic or Marshmallow. If you use these helpers without installing the corresponding library, Datason raises a helpful `ImportError` explaining the missing dependency.

## Documentation & Examples

- [Using Datason with Pydantic](pydantic-marshmallow-integration.md)
- [Using Datason with Marshmallow](pydantic-marshmallow-integration.md)

Include real-world examples in the docs (e.g., FastAPI with Pydantic, Flask with Marshmallow) to demonstrate how Datason plugs into existing validation flows.

## Limitations

- Datason does not perform schema validation.
- Custom Pydantic or Marshmallow fields may require user-defined type handlers for perfect fidelity.
