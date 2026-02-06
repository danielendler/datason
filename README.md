# datason

> **v2 rewrite in progress** — Plugin-based architecture, Python 3.10+, zero dependencies.

Zero-dependency Python serialization with intelligent type handling. Drop-in `json` replacement that handles datetime, UUID, Decimal, Path, NumPy, Pandas, and more.

## Install

```bash
pip install datason
```

## Quick Start

```python
import datason

# Drop-in replacement for json.dumps / json.loads
data = {"name": "Alice", "scores": [95.5, 87.3]}
json_str = datason.dumps(data)
restored = datason.loads(json_str)

# Handles complex types automatically
import datetime as dt
import numpy as np
import pandas as pd

result = datason.dumps({
    "timestamp": dt.datetime.now(),
    "array": np.array([1, 2, 3]),
    "df": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
})
```

## API

```python
datason.dumps(obj, **config)    # -> JSON string
datason.loads(s, **config)      # -> Python object
datason.dump(obj, fp, **config) # -> write to file
datason.load(fp, **config)      # -> read from file
datason.config(**config)        # -> context manager
```

## Type Support

| Category | Types |
|----------|-------|
| JSON primitives | `str`, `int`, `float`, `bool`, `None`, `dict`, `list` |
| Stdlib | `datetime`, `date`, `time`, `timedelta`, `UUID`, `Decimal`, `complex`, `Path`, `set`, `tuple`, `frozenset` |
| NumPy | `ndarray`, `integer`, `floating`, `bool_`, `complexfloating` |
| Pandas | `DataFrame`, `Series`, `Timestamp`, `Timedelta` |

NumPy and Pandas are optional — install them separately if needed.

## Configuration

```python
# Inline config
datason.dumps(data, sort_keys=True, include_type_hints=True)

# Context manager
with datason.config(sort_keys=True, nan_handling=NanHandling.STRING):
    datason.dumps(data)

# Presets
from datason._config import ml_config, api_config
with datason.config(**ml_config().__dict__):
    datason.dumps(model_data)
```

## Security Features

### PII Redaction

```python
# Redact sensitive fields by name (case-insensitive substring match)
datason.dumps(user, redact_fields=("password", "key", "secret"))

# Redact patterns in string values (built-in: email, ssn, credit_card, phone_us, ipv4)
datason.dumps(data, redact_patterns=("email", "ssn"))
```

### Integrity Verification

```python
from datason.security.integrity import wrap_with_integrity, verify_integrity

# Hash-based integrity
wrapped = wrap_with_integrity(datason.dumps(data))
is_valid, payload = verify_integrity(wrapped)

# HMAC with secret key
wrapped = wrap_with_integrity(datason.dumps(data), key="secret")
is_valid, payload = verify_integrity(wrapped, key="secret")
```

### Built-in Limits

- **Max depth**: 50 (prevents stack overflow)
- **Max size**: 100,000 items (prevents memory exhaustion)
- **Circular reference detection** (prevents infinite loops)

## Architecture

Plugin-based design where every non-JSON type is handled by a registered `TypePlugin`:

```
datason/
  _core.py          # Serialize engine (dumps/dump)
  _deserialize.py   # Deserialize engine (loads/load)
  _config.py        # SerializationConfig dataclass
  _registry.py      # Plugin dispatch
  plugins/           # datetime, uuid, decimal, path, numpy, pandas
  security/          # redaction, integrity
```

## Status

This branch (`v2`) is a ground-up rewrite with 307 tests and 90%+ coverage. See [CLAUDE.md](CLAUDE.md) for architecture details and [LEARNINGS_AND_STRATEGY.md](LEARNINGS_AND_STRATEGY.md) for the v1 post-mortem.

## License

MIT
