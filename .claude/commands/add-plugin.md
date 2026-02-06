# Add New Type Plugin

Guide me through adding a new type handler plugin to datason. I will provide the type name as an argument: $ARGUMENTS

## Workflow

### Step 1: Create the plugin file
Create `datason/plugins/<type_name>.py` following the TypePlugin Protocol:

```python
"""Plugin for <TypeName> serialization."""
from __future__ import annotations
from typing import Any
from .._protocols import TypePlugin, SerializeContext, DeserializeContext

class <TypeName>Plugin:
    """Handles serialization/deserialization of <TypeName> objects."""

    name = "<type_name>"
    priority = <200 for third-party, 100 for stdlib>

    def can_handle(self, obj: Any) -> bool:
        ...

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        ...

    def can_deserialize(self, data: Any) -> bool:
        ...

    def deserialize(self, data: Any, ctx: DeserializeContext) -> Any:
        ...
```

### Step 2: Register the plugin
Add the plugin to the registry in `datason/_registry.py`.

### Step 3: Create tests
Create `tests/unit/test_plugin_<type_name>.py` with:
- Basic serialize/deserialize round-trip test
- Edge cases (None, empty, malformed input)
- Hypothesis property-based test if applicable
- Snapshot test for serialization format

### Step 4: Update documentation
- Add type to supported types list in README
- Add import example to docs

### Step 5: Verify
Run `/quality-check` to ensure everything passes.

## Rules
- Plugin file must be < 500 lines
- No functions > 50 lines
- Third-party imports must be lazy: `try: import X; except ImportError: X = None`
- Must handle the case where the library is not installed
- Must include `__datason_type__` metadata for round-trip fidelity
