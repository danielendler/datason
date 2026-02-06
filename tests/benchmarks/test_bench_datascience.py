"""Data science benchmarks for datason v2.

Benchmarks for NumPy and Pandas serialization plugins.
Measures overhead of plugin dispatch + type conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import datason

# =========================================================================
# Fixtures: NumPy data
# =========================================================================


def _make_small_array() -> np.ndarray:
    """10-element 1D array."""
    return np.arange(10, dtype=np.float64)


def _make_medium_array() -> np.ndarray:
    """1000-element 1D array."""
    return np.random.default_rng(42).random(1000)


def _make_2d_array() -> np.ndarray:
    """100x10 2D array."""
    return np.random.default_rng(42).random((100, 10))


def _make_numpy_dict() -> dict:
    """Dict mixing numpy scalars and arrays."""
    return {
        "int_val": np.int64(42),
        "float_val": np.float64(3.14),
        "bool_val": np.bool_(True),
        "small_array": np.arange(5),
        "label": "test",
    }


# =========================================================================
# Fixtures: Pandas data
# =========================================================================


def _make_small_dataframe() -> pd.DataFrame:
    """10-row DataFrame with 3 columns."""
    return pd.DataFrame(
        {
            "id": range(10),
            "name": [f"item_{i}" for i in range(10)],
            "value": np.random.default_rng(42).random(10),
        }
    )


def _make_medium_dataframe() -> pd.DataFrame:
    """1000-row DataFrame with 5 columns."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "id": range(1000),
            "name": [f"item_{i}" for i in range(1000)],
            "value": rng.random(1000),
            "score": rng.integers(0, 100, 1000),
            "active": rng.choice([True, False], 1000),
        }
    )


def _make_series() -> pd.Series:
    """100-element named Series."""
    return pd.Series(
        np.random.default_rng(42).random(100),
        name="measurements",
    )


def _make_timestamps() -> list[pd.Timestamp]:
    """100 Pandas Timestamps."""
    return [pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i) for i in range(100)]


# =========================================================================
# NumPy serialization benchmarks
# =========================================================================


def test_bench_numpy_small_array(benchmark) -> None:
    """Benchmark: serialize a 10-element ndarray."""
    arr = _make_small_array()
    result = benchmark(datason.dumps, arr)
    assert isinstance(result, str)


def test_bench_numpy_medium_array(benchmark) -> None:
    """Benchmark: serialize a 1000-element ndarray."""
    arr = _make_medium_array()
    result = benchmark(datason.dumps, arr)
    assert isinstance(result, str)


def test_bench_numpy_2d_array(benchmark) -> None:
    """Benchmark: serialize a 100x10 ndarray."""
    arr = _make_2d_array()
    result = benchmark(datason.dumps, arr)
    assert isinstance(result, str)


def test_bench_numpy_mixed_dict(benchmark) -> None:
    """Benchmark: serialize dict with mixed numpy types."""
    data = _make_numpy_dict()
    result = benchmark(datason.dumps, data)
    assert isinstance(result, str)


# =========================================================================
# Pandas serialization benchmarks
# =========================================================================


def test_bench_pandas_small_dataframe(benchmark) -> None:
    """Benchmark: serialize a 10-row DataFrame."""
    df = _make_small_dataframe()
    result = benchmark(datason.dumps, df)
    assert isinstance(result, str)


def test_bench_pandas_medium_dataframe(benchmark) -> None:
    """Benchmark: serialize a 1000-row DataFrame."""
    df = _make_medium_dataframe()
    result = benchmark(datason.dumps, df)
    assert isinstance(result, str)


def test_bench_pandas_series(benchmark) -> None:
    """Benchmark: serialize a 100-element Series."""
    series = _make_series()
    result = benchmark(datason.dumps, series)
    assert isinstance(result, str)


def test_bench_pandas_timestamps(benchmark) -> None:
    """Benchmark: serialize 100 Timestamps in a list."""
    timestamps = _make_timestamps()
    result = benchmark(datason.dumps, timestamps)
    assert isinstance(result, str)


# =========================================================================
# Deserialization benchmarks
# =========================================================================


def test_bench_numpy_loads_medium_array(benchmark) -> None:
    """Benchmark: deserialize a 1000-element ndarray (with type hints)."""
    arr = _make_medium_array()
    serialized = datason.dumps(arr, include_type_hints=True)
    result = benchmark(datason.loads, serialized)
    assert isinstance(result, np.ndarray)


def test_bench_pandas_loads_medium_dataframe(benchmark) -> None:
    """Benchmark: deserialize a 1000-row DataFrame (with type hints)."""
    df = _make_medium_dataframe()
    serialized = datason.dumps(df, include_type_hints=True)
    result = benchmark(datason.loads, serialized)
    assert isinstance(result, pd.DataFrame)


# =========================================================================
# Round-trip benchmarks
# =========================================================================


def test_bench_numpy_round_trip_2d(benchmark) -> None:
    """Benchmark: full round-trip for a 100x10 ndarray (with type hints)."""
    arr = _make_2d_array()

    def round_trip() -> object:
        s = datason.dumps(arr, include_type_hints=True)
        return datason.loads(s)

    result = benchmark(round_trip)
    assert isinstance(result, np.ndarray)


def test_bench_pandas_round_trip_dataframe(benchmark) -> None:
    """Benchmark: full round-trip for a 10-row DataFrame (with type hints)."""
    df = _make_small_dataframe()

    def round_trip() -> object:
        s = datason.dumps(df, include_type_hints=True)
        return datason.loads(s)

    result = benchmark(round_trip)
    assert isinstance(result, pd.DataFrame)
