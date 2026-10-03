# Structured agent data

Datason normalizes dataclass instances to field dictionaries, Pydantic models to
Python-mode field dictionaries using aliases, and Enum members to their values.
It does not import application classes from JSON or run their constructors.
Application plugins with ordinary priority (for example, 400) can override these
fallback normalizers when a stronger type contract is needed.

Install `datason[pydantic]` for the optional Pydantic v2 integration.

For a Pydantic v2 model, validate explicitly after loading:

```python
encoded = datason.dumps(result)
fields = datason.loads(encoded)
restored = Result.model_validate(fields)
```

Dataclass fields and Pydantic output pass through the shared traversal, including
redaction, nested scientific values, string limits, and non-finite-number policies.
Model serializers, property access, and user plugins are trusted Python code.
Pydantic is optional; datason does not install it as a core dependency.

Bytes and bytearrays use validated base64 text. Typed records restore the original
binary type. With `include_type_hints=False`, they become base64 strings; API
consumers must know the field encoding from their schema. Dataclasses and Enum
members are normalized even when type hints are enabled, so they load as fields
and values rather than instances of application classes.

Use `include_type_hints=False` for an ordinary JSON tool response. Check the result
against the tool's schema; serialization alone does not establish schema validity.
Use typed records for internal data that requires supported type reconstruction,
and validate the loaded application state before resuming work.
