# API Integration & Pydantic Compatibility

Datason provides flexible configuration options to ensure seamless integration with modern Python web frameworks like FastAPI and validation libraries like Pydantic.

## The UUID String Problem

When building APIs with FastAPI and Pydantic, a common issue arises with UUID handling:

```python
# Problem: datason auto-converts UUID strings to UUID objects
data = {"id": "ea82f3dd-d770-41b9-9706-69cd3070b4f5"}
result = datason.auto_deserialize(data)
# result = {"id": UUID('ea82f3dd-d770-41b9-9706-69cd3070b4f5')}

# But Pydantic models expect string UUIDs
class SavedGroup(BaseModel):
    id: str  # This fails validation with UUID object!
```

## Solution: API-Compatible Configuration

Datason now provides configuration options to control UUID conversion behavior:

### Quick Solution: Use API Config

```python
from datason.config import get_api_config
import datason

# Use the API preset configuration
api_config = get_api_config()
result = datason.auto_deserialize(data, config=api_config)
# result = {"id": "ea82f3dd-d770-41b9-9706-69cd3070b4f5"}  # Stays as string!
```

### Custom Configuration

```python
from datason.config import SerializationConfig

# Option 1: Set uuid_format to "string"
config = SerializationConfig(uuid_format="string")

# Option 2: Disable UUID parsing entirely
config = SerializationConfig(parse_uuids=False)

# Option 3: Combined approach
config = SerializationConfig(
    uuid_format="string",
    parse_uuids=False,
    date_format=DateFormat.ISO,  # Keep other sensible defaults
    sort_keys=True
)
```

## Real-World Example: FastAPI + Pydantic

```python
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import datason
from datason.config import get_api_config

app = FastAPI()

class SavedGroup(BaseModel):
    id: str  # UUID as string for API compatibility
    name: str
    created_at: datetime  # datetime objects are still converted
    members: List[Dict[str, str]]

@app.post("/groups/")
async def create_group(group_data: dict):
    # Process data with API-compatible configuration
    api_config = get_api_config()
    processed_data = datason.auto_deserialize(group_data, config=api_config)

    # Now this works with Pydantic validation!
    validated_group = SavedGroup(**processed_data)
    return validated_group
```

## Database Integration Example

```python
# Data from database (UUIDs typically stored as strings)
database_result = {
    "id": "ea82f3dd-d770-41b9-9706-69cd3070b4f5",
    "user_id": "12345678-1234-5678-9012-123456789abc",
    "created_at": "2023-01-01T12:00:00Z",
    "metadata": {
        "session_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "tracking_data": {...}
    }
}

# Process for API response
api_config = get_api_config()
api_response = datason.auto_deserialize(database_result, config=api_config)

# All UUIDs remain as strings, but datetimes are still converted
assert isinstance(api_response["id"], str)
assert isinstance(api_response["user_id"], str)
assert isinstance(api_response["metadata"]["session_id"], str)
assert isinstance(api_response["created_at"], datetime)  # Still converted!
```

## Configuration Reference

### UUID-Specific Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `uuid_format` | `str` | `"object"` | `"object"` converts to `uuid.UUID`, `"string"` keeps as `str` |
| `parse_uuids` | `bool` | `True` | Whether to attempt UUID string parsing at all |

### Preset Configurations

| Preset | UUID Behavior | Use Case |
|--------|---------------|----------|
| `get_api_config()` | Strings | FastAPI, Pydantic, web APIs |
| `get_ml_config()` | Objects | ML workflows, internal processing |
| `get_performance_config()` | Objects | High-performance scenarios |

## Migration Guide

### For Existing Applications

If you're already using datason and need API compatibility:

```python
# Before (problematic with Pydantic)
result = datason.auto_deserialize(data)

# After (API-compatible)
from datason.config import get_api_config
api_config = get_api_config()
result = datason.auto_deserialize(data, config=api_config)
```

### For New Applications

Start with the appropriate preset:

```python
# For web APIs
from datason.config import get_api_config
config = get_api_config()

# For ML/data processing
from datason.config import get_ml_config  
config = get_ml_config()

# Use consistently throughout your application
result = datason.auto_deserialize(data, config=config)
```

## Backward Compatibility

All existing code continues to work unchanged:

```python
# This still works exactly as before
result = datason.auto_deserialize(data)
result = datason.deserialize(data, parse_uuids=True)
result = datason.deserialize(data, parse_uuids=False)
```

The new `config` parameter is optional and doesn't break any existing functionality.

## Best Practices

1. **Choose the right preset** for your primary use case
2. **Be consistent** - use the same config throughout your application  
3. **Document your choice** - make it clear which UUID format your API expects
4. **Test edge cases** - ensure your configuration handles all your data patterns
5. **Consider your consumers** - frontend JavaScript typically expects UUID strings

## Related Documentation

- [Configuration System](../configuration/) - Complete configuration reference
- [Type Handling](../advanced-types/) - How datason handles different Python types
- [Performance Guide](../performance/) - Optimization strategies for different configurations
