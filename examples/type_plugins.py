"""Type plugin examples: datetime, UUID, Decimal, Path, NumPy, Pandas."""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from pathlib import Path

import datason

# =========================================================================
# 1. Datetime types (automatic serialization)
# =========================================================================

now = dt.datetime(2024, 6, 15, 10, 30, 0)
event = {
    "timestamp": now,
    "date": now.date(),
    "duration": dt.timedelta(hours=2, minutes=30),
}

serialized = datason.dumps(event)
print(f"Datetime: {serialized}")

# Round-trip: datetime objects are restored when type hints are included
with datason.config(include_type_hints=True):
    serialized = datason.dumps(event)
    restored = datason.loads(serialized)
    print(f"Restored datetime: {restored['timestamp']}")
    assert isinstance(restored["timestamp"], dt.datetime)
    assert isinstance(restored["date"], dt.date)
    assert isinstance(restored["duration"], dt.timedelta)

# =========================================================================
# 2. UUID
# =========================================================================

user = {"id": uuid.uuid4(), "name": "Alice"}
print(f"\nUUID: {datason.dumps(user)}")

# =========================================================================
# 3. Decimal (high-precision numbers)
# =========================================================================

price = {"amount": Decimal("19.99"), "tax": Decimal("0.0825")}
print(f"Decimal: {datason.dumps(price)}")

# =========================================================================
# 4. Complex numbers
# =========================================================================

signal = {"impedance": complex(3.5, 2.1)}
print(f"Complex: {datason.dumps(signal)}")

# =========================================================================
# 5. Pathlib paths
# =========================================================================

config = {"data_dir": Path("/data/models"), "output": Path("./results")}
print(f"Paths: {datason.dumps(config)}")

# =========================================================================
# 6. NumPy arrays and scalars
# =========================================================================

try:
    import numpy as np

    array = np.array([1.0, 2.0, 3.0])
    matrix = np.random.default_rng(42).random((3, 3))

    print(f"\nNumPy 1D: {datason.dumps(array)}")
    print(f"NumPy 2D: {datason.dumps(matrix)}")

    # Round-trip with type hints preserves dtype and shape
    with datason.config(include_type_hints=True):
        s = datason.dumps(matrix)
        restored = datason.loads(s)
        assert isinstance(restored, np.ndarray)
        assert restored.shape == (3, 3)
        print(f"Restored ndarray shape: {restored.shape}")

except ImportError:
    print("\nNumPy not installed, skipping numpy examples.")

# =========================================================================
# 7. Pandas DataFrames and Series
# =========================================================================

try:
    import pandas as pd

    df = pd.DataFrame({"name": ["Alice", "Bob"], "score": [95.5, 87.3]})
    print(f"\nDataFrame: {datason.dumps(df)}")

    # Different orientations
    from datason._config import DataFrameOrient

    with datason.config(dataframe_orient=DataFrameOrient.SPLIT):
        print(f"Split orient: {datason.dumps(df)}")

    # Timestamps
    ts = pd.Timestamp("2024-01-15 10:30:00")
    print(f"Timestamp: {datason.dumps(ts)}")

    # Round-trip
    with datason.config(include_type_hints=True):
        s = datason.dumps(df)
        restored = datason.loads(s)
        assert isinstance(restored, pd.DataFrame)
        print(f"Restored DataFrame shape: {restored.shape}")

except ImportError:
    print("\nPandas not installed, skipping pandas examples.")

print("\nAll type plugin examples passed!")
