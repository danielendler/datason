# Configuration

## Overview

datason uses a frozen dataclass `SerializationConfig` with sensible defaults. You can override settings three ways:

```python
# 1. Inline kwargs (highest priority)
datason.dumps(data, sort_keys=True)

# 2. Context manager (scoped)
with datason.config(sort_keys=True):
    datason.dumps(data)

# 3. Presets
from datason import ml_config
with datason.config(**ml_config().__dict__):
    datason.dumps(data)
```

## All Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `date_format` | `DateFormat` | `ISO` | Datetime serialization: `ISO`, `UNIX`, `UNIX_MS`, `STRING` |
| `dataframe_orient` | `DataFrameOrient` | `RECORDS` | DataFrame format: `RECORDS`, `SPLIT`, `DICT`, `LIST`, `VALUES` |
| `nan_handling` | `NanHandling` | `NULL` | NaN/Inf handling: `NULL`, `STRING`, `KEEP`, `DROP` |
| `include_type_hints` | `bool` | `True` | Emit `__datason_type__` for round-trip fidelity |
| `sort_keys` | `bool` | `False` | Sort dict keys alphabetically |
| `max_depth` | `int` | `50` | Max nesting depth (security) |
| `max_size` | `int` | `100_000` | Max dict/list items (security) |
| `max_string_length` | `int` | `1_000_000` | Max string length (security) |
| `fallback_to_string` | `bool` | `False` | `str()` unknown types instead of raising |
| `strict` | `bool` | `True` | Raise on unrecognized type metadata in `loads` |
| `allow_plugin_deserialization` | `bool` | `True` | Allow plugin code to run during `loads`/`load` |
| `redact_fields` | `tuple[str, ...]` | `()` | Field names to redact |
| `redact_patterns` | `tuple[str, ...]` | `()` | Regex patterns to redact |

## DateFormat

Controls how `datetime` objects are serialized:

```python
from datason import DateFormat

# ISO 8601 (default)
datason.dumps({"ts": dt}, date_format=DateFormat.ISO)
# "2024-01-15T10:30:00"

# Unix timestamp (seconds)
datason.dumps({"ts": dt}, date_format=DateFormat.UNIX)
# 1705312200.0

# Unix timestamp (milliseconds)
datason.dumps({"ts": dt}, date_format=DateFormat.UNIX_MS)
# 1705312200000.0

# Python str()
datason.dumps({"ts": dt}, date_format=DateFormat.STRING)
# "2024-01-15 10:30:00"
```

## NanHandling

Controls how `float('nan')` and `float('inf')` are serialized:

```python
from datason import NanHandling

datason.dumps({"v": float("nan")}, nan_handling=NanHandling.NULL)    # null
datason.dumps({"v": float("nan")}, nan_handling=NanHandling.STRING)  # "NaN"
datason.dumps({"v": float("nan")}, nan_handling=NanHandling.KEEP)    # NaN (invalid JSON!)
datason.dumps({"v": float("nan")}, nan_handling=NanHandling.DROP)    # null
```

## DataFrameOrient

Controls Pandas DataFrame serialization format:

```python
from datason import DataFrameOrient

# Records (default): [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
datason.dumps(df, dataframe_orient=DataFrameOrient.RECORDS)

# Split: {"columns": ["a","b"], "index": [0,1], "data": [[1,2],[3,4]]}
datason.dumps(df, dataframe_orient=DataFrameOrient.SPLIT)

# Dict: {"a": {"0": 1, "1": 3}, "b": {"0": 2, "1": 4}}
datason.dumps(df, dataframe_orient=DataFrameOrient.DICT)
```

## Presets

Four built-in presets for common workflows:

### ml_config()

For ML pipelines: UNIX_MS timestamps, lenient type handling.

```python
from datason import ml_config

with datason.config(**ml_config().__dict__):
    datason.dumps({"predictions": np.array([0.9, 0.1])})
```

### api_config()

For API responses: ISO dates, sorted keys, no type metadata.

```python
from datason import api_config

with datason.config(**api_config().__dict__):
    datason.dumps({"created": dt.datetime.now(), "status": "ok"})
```

### strict_config()

For validation: unknown types raise errors, type hints required.

### performance_config()

For speed: no type hints, no sorting, keep NaN as-is.
