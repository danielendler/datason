# datason

[![CI](https://github.com/danielendler/datason/actions/workflows/ci.yml/badge.svg)](https://github.com/danielendler/datason/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/danielendler/datason/graph/badge.svg?token=UYL9LvVb8O)](https://codecov.io/gh/danielendler/datason)
[![PyPI version](https://img.shields.io/pypi/v/datason.svg)](https://pypi.org/project/datason/)
[![Python versions](https://img.shields.io/pypi/pyversions/datason.svg)](https://pypi.org/project/datason/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://danielendler.github.io/datason/)

**Drop-in replacement for `json.dumps`/`json.loads` that handles datetime, NumPy, Pandas, PyTorch, and 50+ Python types. Zero dependencies.**

```python
import datason
import datetime as dt
import numpy as np

# Just replace json.dumps with datason.dumps — everything else works
datason.dumps({"ts": dt.datetime.now(), "scores": np.array([0.9, 0.1])})
```

No more `TypeError: Object of type datetime is not JSON serializable`.

## Install

```bash
pip install datason                    # Core (zero dependencies)
pip install datason[numpy]             # + NumPy support
pip install datason[pandas]            # + Pandas support
pip install datason[ml]                # + PyTorch, TensorFlow, scikit-learn, SciPy
pip install datason[all]               # Everything
```

Requires Python 3.10+.

## Quick Start

```python
import datason
import datetime as dt
import uuid
from decimal import Decimal
from pathlib import Path

# Works exactly like json for simple data
datason.dumps({"name": "Alice", "age": 30})
# '{"name": "Alice", "age": 30}'

# But also handles complex types that json.dumps cannot
data = {
    "timestamp": dt.datetime(2024, 6, 15, 10, 30),
    "id": uuid.uuid4(),
    "price": Decimal("19.99"),
    "config_path": Path("/data/models"),
}
json_str = datason.dumps(data)

# And brings them back on deserialization
restored = datason.loads(json_str)
assert isinstance(restored["timestamp"], dt.datetime)
assert isinstance(restored["id"], uuid.UUID)
```

### NumPy + Pandas

```python
import numpy as np
import pandas as pd
import datason

# NumPy arrays serialize with shape and dtype preserved
arr = np.array([[1.0, 2.0], [3.0, 4.0]])
json_str = datason.dumps(arr)
restored = datason.loads(json_str)
assert isinstance(restored, np.ndarray)
assert restored.shape == (2, 2)

# Pandas DataFrames serialize as records by default
df = pd.DataFrame({"name": ["Alice", "Bob"], "score": [95.5, 87.3]})
json_str = datason.dumps(df)
restored = datason.loads(json_str)
assert isinstance(restored, pd.DataFrame)
```

### ML Frameworks

```python
import torch
import datason

# PyTorch tensors
tensor = torch.randn(3, 3)
json_str = datason.dumps({"weights": tensor})
restored = datason.loads(json_str)
assert isinstance(restored["weights"], torch.Tensor)

# Also supports: TensorFlow tensors, scikit-learn models, SciPy sparse matrices
```

## API — 5 Functions

```python
import datason

datason.dumps(obj, **config)    # Serialize to JSON string
datason.loads(s, **config)      # Deserialize from JSON string
datason.dump(obj, fp, **config) # Write to file
datason.load(fp, **config)      # Read from file
datason.config(**config)        # Context manager for temp config
```

That's the entire public API.

## Supported Types

| Category | Types |
|----------|-------|
| **JSON primitives** | `str`, `int`, `float`, `bool`, `None`, `dict`, `list` |
| **Stdlib** | `datetime`, `date`, `time`, `timedelta`, `UUID`, `Decimal`, `complex`, `Path`, `set`, `tuple`, `frozenset` |
| **NumPy** | `ndarray`, `integer`, `floating`, `bool_`, `complexfloating` |
| **Pandas** | `DataFrame`, `Series`, `Timestamp`, `Timedelta` |
| **PyTorch** | `Tensor` |
| **TensorFlow** | `Tensor`, `EagerTensor` |
| **scikit-learn** | All estimators (`LinearRegression`, `RandomForestClassifier`, etc.) |
| **SciPy** | Sparse matrices (`csr`, `csc`, `coo`, etc.) |
| **Polars** | `DataFrame`, `Series` |
| **JAX** | `Array` |
| **Plotly** | `Figure` |

All non-core types are optional — install the relevant extra (`numpy`, `pandas`, `ml`).

## Configuration

```python
import datason
from datason import DateFormat, NanHandling, DataFrameOrient

# Inline overrides
datason.dumps(data, sort_keys=True)
datason.dumps(data, date_format=DateFormat.UNIX)
datason.dumps(data, nan_handling=NanHandling.STRING)
datason.dumps(data, include_type_hints=False)  # Smaller output, no round-trip

# Context manager for scoped config
with datason.config(sort_keys=True, nan_handling=NanHandling.STRING):
    datason.dumps(data)

# Presets for common use cases
from datason import ml_config, api_config, strict_config, performance_config

with datason.config(**ml_config().__dict__):
    datason.dumps(model_output)   # UNIX_MS dates, fallback to string

with datason.config(**api_config().__dict__):
    datason.dumps(response)       # ISO dates, sorted keys, no type hints
```

### Config Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `date_format` | `DateFormat` | `ISO` | How to serialize datetimes: `ISO`, `UNIX`, `UNIX_MS`, `STRING` |
| `dataframe_orient` | `DataFrameOrient` | `RECORDS` | DataFrame format: `RECORDS`, `SPLIT`, `DICT`, `LIST`, `VALUES` |
| `nan_handling` | `NanHandling` | `NULL` | Float NaN/Inf: `NULL`, `STRING`, `KEEP`, `DROP` |
| `include_type_hints` | `bool` | `True` | Emit type metadata for round-trip fidelity |
| `sort_keys` | `bool` | `False` | Sort dict keys in output |
| `max_depth` | `int` | `50` | Max nesting depth (security) |
| `max_size` | `int` | `100_000` | Max dict/list size (security) |
| `fallback_to_string` | `bool` | `False` | `str()` unknown types instead of raising |
| `strict` | `bool` | `True` | Raise on unrecognized type metadata |
| `redact_fields` | `tuple[str, ...]` | `()` | Field names to redact |
| `redact_patterns` | `tuple[str, ...]` | `()` | Regex patterns to redact from strings |

## Security Features

### PII Redaction

```python
# Redact by field name (case-insensitive substring match)
datason.dumps(user_data, redact_fields=("password", "key", "secret", "token"))
# {"username": "alice", "password": "[REDACTED]", "api_key": "[REDACTED]"}

# Redact patterns in string values (built-in: email, ssn, credit_card, phone_us, ipv4)
datason.dumps(data, redact_patterns=("email", "ssn"))
```

### Integrity Verification

```python
from datason.security.integrity import wrap_with_integrity, verify_integrity

# Wrap with hash-based integrity envelope
wrapped = wrap_with_integrity(datason.dumps(data))
is_valid, payload = verify_integrity(wrapped)

# HMAC with secret key
wrapped = wrap_with_integrity(datason.dumps(data), key="secret")
is_valid, payload = verify_integrity(wrapped, key="secret")
```

### Built-in Limits
- **Max depth**: 50 (prevents stack overflow from nested data)
- **Max size**: 100,000 items per dict/list (prevents memory exhaustion)
- **Circular reference detection** (prevents infinite loops)

All limits raise `SecurityError` and are configurable.

## How It Works

datason uses a plugin-based architecture. Every type beyond JSON primitives is handled by a `TypePlugin` registered in a priority-sorted registry:

```
Your object --> dumps() --> Plugin registry --> Type-specific serializer --> JSON
JSON string --> loads() --> Plugin registry --> Type-specific deserializer --> Your object
```

Type metadata is embedded as `{"__datason_type__": "datetime", "__datason_value__": "2024-01-15T10:30:00"}`, enabling lossless round-trips.

### Writing a Custom Plugin

```python
from datason._protocols import TypePlugin, SerializeContext, DeserializeContext
from datason._registry import default_registry
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY

class MoneyPlugin:
    name = "money"
    priority = 400  # 400+ for user plugins

    def can_handle(self, obj):
        return isinstance(obj, Money)

    def serialize(self, obj, ctx):
        return {TYPE_METADATA_KEY: "Money", VALUE_METADATA_KEY: {"amount": str(obj.amount), "currency": obj.currency}}

    def can_deserialize(self, data):
        return data.get(TYPE_METADATA_KEY) == "Money"

    def deserialize(self, data, ctx):
        v = data[VALUE_METADATA_KEY]
        return Money(Decimal(v["amount"]), v["currency"])

default_registry.register(MoneyPlugin())
```

## For AI Agents

datason includes [`llms.txt`](llms.txt) and [`llms-full.txt`](llms-full.txt) for AI agent discoverability. The full reference contains complete API signatures, all config options, and ready-to-use code examples.

## Documentation

Full documentation at **[danielendler.github.io/datason](https://danielendler.github.io/datason/)**.

## License

MIT
