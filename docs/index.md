# datason

**Drop-in replacement for `json.dumps`/`json.loads` that handles datetime, NumPy, Pandas, PyTorch, and 50+ Python types. Zero dependencies.**

```python
import datason
import datetime as dt
import numpy as np

data = {"ts": dt.datetime.now(), "scores": np.array([0.9, 0.1])}
json_str = datason.dumps(data)
restored = datason.loads(json_str)
# restored["ts"] is a datetime, restored["scores"] is a numpy array
```

## Why datason?

Python's `json` module fails on anything beyond primitives:

```python
import json
json.dumps({"ts": datetime.now()})  # TypeError!
json.dumps({"arr": np.array([1,2])})  # TypeError!
```

datason handles all of these types automatically while maintaining the same `dumps`/`loads` API you already know.

## Key Features

- **Zero dependencies** -- only stdlib in core. NumPy, Pandas, ML libs are optional.
- **5-function API** -- `dumps`, `loads`, `dump`, `load`, `config`. That's it.
- **Perfect round-trips** -- types are reconstructed on deserialization.
- **50+ types** -- datetime, UUID, Decimal, Path, NumPy, Pandas, PyTorch, TensorFlow, scikit-learn, SciPy.
- **Security built-in** -- depth limits, size limits, circular reference detection, PII redaction.
- **Plugin architecture** -- extend with custom types in ~20 lines.
- **Thread-safe** -- config scoping via ContextVar, registry with threading.Lock.

## Quick Install

```bash
pip install datason                    # Core (zero dependencies)
pip install datason[numpy]             # + NumPy support
pip install datason[pandas]            # + Pandas support
pip install datason[ml]                # + PyTorch, TensorFlow, scikit-learn, SciPy
pip install datason[all]               # Everything
```

Requires Python 3.10+.

## Next Steps

- [Getting Started](getting-started.md) -- installation and first examples
- [API Reference](api.md) -- complete function signatures
- [Configuration](configuration.md) -- all config options and presets
- [For AI Agents](ai-agents.md) -- llms.txt and machine-readable docs
