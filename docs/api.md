# API Reference

datason has 5 public functions:

## datason.dumps

```python
datason.dumps(obj: Any, **kwargs) -> str
```

Serialize any Python object to a JSON string. Drop-in replacement for `json.dumps`.

**Parameters:**

- `obj` -- Any Python object to serialize
- `**kwargs` -- Override any [SerializationConfig](configuration.md) field inline

**Returns:** JSON string

**Raises:**

- `SecurityError` -- depth/size limits exceeded or circular reference detected
- `SerializationError` -- unknown type and `fallback_to_string=False`

**Examples:**

```python
import datason
import datetime as dt
import numpy as np

datason.dumps({"name": "Alice"})
# '{"name": "Alice"}'

datason.dumps({"ts": dt.datetime(2024, 1, 15)})
# '{"ts": {"__datason_type__": "datetime", "__datason_value__": "2024-01-15T00:00:00"}}'

datason.dumps({"arr": np.array([1, 2, 3])})
# '{"arr": {"__datason_type__": "ndarray", "__datason_value__": {"data": [1, 2, 3], ...}}}'

datason.dumps({"z": 1, "a": 2}, sort_keys=True)
# '{"a": 2, "z": 1}'
```

## datason.loads

```python
datason.loads(s: str, **kwargs) -> Any
```

Deserialize a JSON string back to Python objects. Drop-in replacement for `json.loads`. Values with `__datason_type__` metadata are reconstructed to their original types.

**Parameters:**

- `s` -- JSON string to deserialize
- `**kwargs` -- Override any [SerializationConfig](configuration.md) field inline

**Returns:** Deserialized Python object

**Raises:**

- `SecurityError` -- depth limit exceeded
- `DeserializationError` -- unrecognized type metadata and `strict=True`

**Examples:**

```python
import datason

datason.loads('{"name": "Alice"}')
# {'name': 'Alice'}

# Round-trip with type reconstruction
original = {"ts": dt.datetime(2024, 1, 15), "arr": np.array([1, 2, 3])}
restored = datason.loads(datason.dumps(original))
assert isinstance(restored["ts"], dt.datetime)
assert isinstance(restored["arr"], np.ndarray)
```

## datason.dump

```python
datason.dump(obj: Any, fp: IOBase, **kwargs) -> None
```

Serialize and write to a file. Drop-in replacement for `json.dump`.

```python
with open("out.json", "w") as f:
    datason.dump({"ts": dt.datetime.now()}, f)
```

## datason.load

```python
datason.load(fp: IOBase, **kwargs) -> Any
```

Read from a file and deserialize. Drop-in replacement for `json.load`.

```python
with open("out.json") as f:
    data = datason.load(f)
```

## datason.config

```python
@contextmanager
datason.config(**kwargs)
```

Context manager to temporarily set serialization config. Thread-safe via `contextvars.ContextVar`.

```python
with datason.config(sort_keys=True, nan_handling=NanHandling.STRING):
    datason.dumps(data)  # uses these settings

datason.dumps(data)  # back to defaults
```

## Error Types

```python
from datason._errors import (
    DatasonError,          # Base class for all datason errors
    SecurityError,         # Depth/size/circular ref -- always fatal
    SerializationError,    # Unknown type -- fatal unless fallback_to_string=True
    DeserializationError,  # Bad metadata -- fatal unless strict=False
    PluginError,           # Plugin failure -- logs warning, tries next plugin
)
```
