# Migrate V1 Type Handler to V2 Plugin

Migrate a type handler from the v1 codebase into a v2 plugin. Argument: the type to migrate (e.g., "numpy", "pandas", "datetime"). $ARGUMENTS

## Workflow

### Step 1: Understand the v1 implementation
Read the relevant sections from these v1 files:
- `datason/core_new.py` - search for the type handling in `_serialize_full_path()`
- `datason/deserializers_new.py` - search for the type in `_deserialize_with_type_metadata()`
- `datason/ml_serializers.py` - if it's an ML type
- `datason/type_handlers.py` - if it has a custom handler
- `datason/converters.py` - if it has conversion logic

Document what the v1 code does: serialization format, metadata keys, edge cases handled.

### Step 2: Create the v2 plugin
Create `datason/plugins/<type>.py` following the TypePlugin Protocol.

Key migration rules:
- Extract ONLY the logic for this specific type (no god functions)
- Keep lazy imports for third-party libraries
- Use `match/case` for type dispatch (Python 3.10+)
- Add proper type annotations
- Handle the case where the library isn't installed
- Keep the same `__datason_type__` / `__datason_value__` metadata format for backward compatibility

### Step 3: Write tests
Create `tests/unit/test_plugin_<type>.py` with:
- Round-trip tests (serialize then deserialize = original)
- Edge cases from v1 tests (search `tests/` for the type name)
- Hypothesis property-based tests
- Snapshot tests for the serialization format
- Test with library not installed (mock ImportError)

### Step 4: Register
Add to `datason/_registry.py` default plugin list.

### Step 5: Verify backward compatibility
Ensure the serialization format matches v1 so old serialized data can still be deserialized.

### Step 6: Run quality checks
Run `/quality-check` to validate.
