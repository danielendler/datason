# Getting Started

## Installation

```bash
pip install datason
```

Optional extras for type support:

```bash
pip install datason[numpy]      # NumPy arrays
pip install datason[pandas]     # Pandas DataFrames/Series
pip install datason[ml]         # PyTorch, TensorFlow, scikit-learn, SciPy
pip install datason[all]        # All of the above
```

## Basic Usage

datason is a drop-in replacement for Python's `json` module:

```python
import datason

# Works just like json.dumps / json.loads
data = {"name": "Alice", "age": 30, "scores": [95.5, 87.3]}
json_str = datason.dumps(data)
restored = datason.loads(json_str)
assert restored == data
```

## Complex Types

The real power is handling types that `json` cannot:

```python
import datason
import datetime as dt
import uuid
from decimal import Decimal
from pathlib import Path

data = {
    "timestamp": dt.datetime(2024, 6, 15, 10, 30),
    "id": uuid.uuid4(),
    "price": Decimal("19.99"),
    "config_path": Path("/data/models"),
}

json_str = datason.dumps(data)
restored = datason.loads(json_str)

assert isinstance(restored["timestamp"], dt.datetime)
assert isinstance(restored["id"], uuid.UUID)
assert isinstance(restored["price"], Decimal)
assert isinstance(restored["config_path"], Path)
```

## NumPy and Pandas

```python
import numpy as np
import pandas as pd
import datason

# NumPy: shape and dtype preserved
arr = np.array([[1.0, 2.0], [3.0, 4.0]])
restored = datason.loads(datason.dumps(arr))
assert isinstance(restored, np.ndarray)
assert restored.shape == (2, 2)

# Pandas: DataFrame columns and dtypes preserved
df = pd.DataFrame({"name": ["Alice", "Bob"], "score": [95.5, 87.3]})
restored = datason.loads(datason.dumps(df))
assert isinstance(restored, pd.DataFrame)
assert list(restored.columns) == ["name", "score"]
```

## File I/O

```python
import datason

# Write to file
with open("data.json", "w") as f:
    datason.dump(data, f)

# Read from file
with open("data.json") as f:
    restored = datason.load(f)
```

## Configuration

```python
import datason
from datason import DateFormat, NanHandling

# Inline overrides
datason.dumps(data, sort_keys=True)
datason.dumps(data, date_format=DateFormat.UNIX)

# Context manager
with datason.config(sort_keys=True, nan_handling=NanHandling.STRING):
    datason.dumps(data)

# Presets
from datason import ml_config
with datason.config(**ml_config().__dict__):
    datason.dumps(model_output)
```

See [Configuration](configuration.md) for all options.
