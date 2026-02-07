"""Basic datason usage: serialize and deserialize Python objects."""

from __future__ import annotations

import io

import datason
from datason._config import NanHandling

# =========================================================================
# 1. Drop-in replacement for json.dumps/json.loads
# =========================================================================

data = {"name": "Alice", "age": 30, "scores": [95.5, 87.3, 91.0]}

# datason.dumps works just like json.dumps
serialized = datason.dumps(data)
print(f"Serialized: {serialized}")

# datason.loads works just like json.loads
restored = datason.loads(serialized)
print(f"Restored:   {restored}")
assert restored == data

# =========================================================================
# 2. Sorted keys for deterministic output
# =========================================================================

with datason.config(sort_keys=True):
    deterministic = datason.dumps({"z": 1, "a": 2, "m": 3})
    print(f"Sorted:     {deterministic}")

# =========================================================================
# 3. NaN/Infinity handling (configurable)
# =========================================================================

data_with_nan = {"value": float("nan"), "inf": float("inf"), "normal": 42.0}

# Default: NaN/Inf become null
default = datason.dumps(data_with_nan)
print(f"NaN→null:   {default}")

# String mode: NaN/Inf become strings
with datason.config(nan_handling=NanHandling.STRING):
    as_string = datason.dumps(data_with_nan)
    print(f"NaN→string: {as_string}")

# =========================================================================
# 4. Sets and tuples
# =========================================================================

# Tuples become lists, sets become sorted lists
mixed = {"tuple": (1, 2, 3), "set": {3, 1, 2}}
print(f"Mixed:      {datason.dumps(mixed)}")

# =========================================================================
# 5. File I/O
# =========================================================================

buffer = io.StringIO()
datason.dump(data, buffer, sort_keys=True)
buffer.seek(0)
from_file = datason.load(buffer)
print(f"From file:  {from_file}")
assert from_file == data

print("\nAll basic examples passed!")
