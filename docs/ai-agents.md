# For AI Agents

datason is designed to be easily discoverable and usable by AI coding agents (Claude, GPT, Copilot, etc.).

## Machine-Readable Documentation

datason provides [`llms.txt`](https://github.com/danielendler/datason/blob/v2/llms.txt) and [`llms-full.txt`](https://github.com/danielendler/datason/blob/v2/llms-full.txt) following the [llms.txt standard](https://llmstxt.org/).

- **`llms.txt`** -- Short summary with links to API docs, examples, and source files
- **`llms-full.txt`** -- Complete API reference with all function signatures, config options, and ready-to-use code examples in a single file

## Why datason for AI Agents?

AI agents frequently need to serialize complex Python objects when:

- **Generating data pipeline code** -- ML model outputs contain NumPy arrays, timestamps, UUIDs
- **Building API endpoints** -- Response dicts contain datetime objects, Decimal prices
- **Logging and debugging** -- Mixed-type dicts need JSON output for structured logging
- **Persisting state** -- Agent state includes datetime, arrays, paths

With datason, agents can write:

```python
import datason
datason.dumps(any_python_object)
```

Instead of:

```python
import json

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, UUID):
            return str(obj)
        # ... 20 more type checks
        return super().default(obj)

json.dumps(data, cls=CustomEncoder)
```

## Quick Reference for Agents

### Installation

```bash
pip install datason
pip install datason[numpy]   # if using NumPy
pip install datason[pandas]  # if using Pandas
pip install datason[ml]      # if using PyTorch/TF/sklearn
```

### API (5 functions)

```python
import datason

datason.dumps(obj)              # -> JSON string
datason.loads(s)                # -> Python object (types reconstructed)
datason.dump(obj, file)         # -> write JSON to file
datason.load(file)              # -> read JSON from file

with datason.config(sort_keys=True):
    datason.dumps(obj)          # -> sorted JSON string
```

### Common Config Options

```python
datason.dumps(data, sort_keys=True)                # Sort keys
datason.dumps(data, include_type_hints=False)       # No metadata (smaller output)
datason.dumps(data, fallback_to_string=True)        # str() unknown types
datason.dumps(data, redact_fields=("password",))    # PII redaction
```
