import datason
import pytest
import datetime as dt
from decimal import Decimal
import numpy as np
from datason import NanHandling, DateFormat

def test_extreme_dates():
    # Test min/max dates
    data = {
        "min": dt.datetime.min,
        "max": dt.datetime.max,
        "epoch": dt.datetime(1970, 1, 1)
    }
    json_str = datason.dumps(data)
    restored = datason.loads(json_str)
    assert restored["min"] == dt.datetime.min
    assert restored["max"] == dt.datetime.max

def test_huge_decimal():
    # Huge decimal that might cause issues with standard float conversion
    huge = Decimal("1" + "0" * 100 + ".123456789")
    json_str = datason.dumps(huge)
    restored = datason.loads(json_str)
    assert restored == huge
    assert isinstance(restored, Decimal)

def test_nan_inf_handling():
    data = {"nan": float('nan'), "inf": float('inf'), "ninf": float('-inf')}
    
    # NULL handling (default)
    res_null = datason.loads(datason.dumps(data, nan_handling=NanHandling.NULL))
    assert res_null["nan"] is None
    assert res_null["inf"] is None
    
    # STRING handling
    res_str = datason.loads(datason.dumps(data, nan_handling=NanHandling.STRING))
    assert res_str["nan"] == "NaN"
    assert res_str["inf"] == "Infinity"
    
    # DROP handling
    res_drop = datason.loads(datason.dumps(data, nan_handling=NanHandling.DROP))
    assert "nan" not in res_drop
    assert "inf" not in res_drop

def test_complex_nested_structure():
    data = {
        "a": [1, 2, {"b": dt.datetime.now()}],
        "c": (1, 2, 3), # Tuple
        "d": {Decimal("1.1"), Decimal("2.2")}, # Set
        "e": np.array([1, 2, 3]),
        "f": frozenset([1, 2])
    }
    json_str = datason.dumps(data)
    restored = datason.loads(json_str)
    
    assert restored["a"][2]["b"] == data["a"][2]["b"]
    assert restored["c"] == data["c"]
    assert isinstance(restored["c"], tuple)
    assert restored["d"] == data["d"]
    assert isinstance(restored["d"], set)
    assert np.array_equal(restored["e"], data["e"])
    assert restored["f"] == data["f"]
    assert isinstance(restored["f"], frozenset)

def test_empty_containers():
    data = {
        "list": [],
        "dict": {},
        "set": set(),
        "tuple": (),
        "ndarray": np.array([])
    }
    json_str = datason.dumps(data)
    restored = datason.loads(json_str)
    assert restored == data

def test_unix_ms_date_format():
    now = dt.datetime(2024, 1, 1, 12, 0, 0)
    # Expected unix ms: 1704110400000
    json_str = datason.dumps(now, date_format=DateFormat.UNIX_MS)
    # When using UNIX_MS, does it round-trip back to datetime?
    restored = datason.loads(json_str)
    assert restored == now
    assert isinstance(restored, dt.datetime)

if __name__ == "__main__":
    pytest.main([__file__])
