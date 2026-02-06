"""Core performance benchmarks for datason v2.

Baseline benchmarks run BEFORE optimization. Targets from CLAUDE.md:
- Simple data: < 1ms
- Complex data: < 5ms
- Throughput: > 250K items/sec
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest

import datason

# =========================================================================
# Fixtures: test data
# =========================================================================


@pytest.fixture()
def simple_dict() -> dict:
    """10-key flat dict with basic JSON types."""
    return {
        "name": "Alice",
        "age": 30,
        "score": 95.5,
        "active": True,
        "email": "alice@example.com",
        "city": "Zurich",
        "count": 42,
        "ratio": 0.618,
        "verified": False,
        "notes": None,
    }


@pytest.fixture()
def nested_dict() -> dict:
    """3 levels deep, ~100 total keys."""
    return {
        f"group_{i}": {
            f"item_{j}": {
                "value": j * 1.1,
                "label": f"label_{i}_{j}",
                "active": j % 2 == 0,
            }
            for j in range(10)
        }
        for i in range(3)
    }


@pytest.fixture()
def list_of_dicts() -> list:
    """1000 small dicts."""
    return [{"id": i, "name": f"item_{i}", "value": i * 0.5, "ok": True} for i in range(1000)]


@pytest.fixture()
def datetime_objects() -> list:
    """500 objects with datetime fields."""
    base = dt.datetime(2024, 1, 1, 12, 0, 0)
    return [
        {
            "id": i,
            "created": base + dt.timedelta(hours=i),
            "date": (base + dt.timedelta(days=i)).date(),
            "duration": dt.timedelta(seconds=i * 10),
        }
        for i in range(500)
    ]


@pytest.fixture()
def uuid_objects() -> list:
    """500 objects with UUID fields."""
    return [{"id": uuid.UUID(int=i), "name": f"entity_{i}", "score": i * 0.1} for i in range(500)]


# =========================================================================
# Serialization benchmarks
# =========================================================================


def test_bench_simple_dict(benchmark, simple_dict: dict) -> None:
    """Benchmark: serialize a 10-key flat dict."""
    result = benchmark(datason.dumps, simple_dict)
    assert isinstance(result, str)


def test_bench_nested_dict(benchmark, nested_dict: dict) -> None:
    """Benchmark: serialize a 3-level nested dict (~100 keys)."""
    result = benchmark(datason.dumps, nested_dict)
    assert isinstance(result, str)


def test_bench_list_of_dicts(benchmark, list_of_dicts: list) -> None:
    """Benchmark: serialize 1000 small dicts."""
    result = benchmark(datason.dumps, list_of_dicts)
    assert isinstance(result, str)


def test_bench_with_datetimes(benchmark, datetime_objects: list) -> None:
    """Benchmark: serialize 500 objects with datetime/date/timedelta fields."""
    result = benchmark(datason.dumps, datetime_objects)
    assert isinstance(result, str)


def test_bench_with_uuids(benchmark, uuid_objects: list) -> None:
    """Benchmark: serialize 500 objects with UUID fields."""
    result = benchmark(datason.dumps, uuid_objects)
    assert isinstance(result, str)


# =========================================================================
# Deserialization benchmarks
# =========================================================================


def test_bench_loads_simple(benchmark, simple_dict: dict) -> None:
    """Benchmark: deserialize a simple dict (no type metadata)."""
    serialized = datason.dumps(simple_dict)
    result = benchmark(datason.loads, serialized)
    assert isinstance(result, dict)


def test_bench_loads_list_of_dicts(benchmark, list_of_dicts: list) -> None:
    """Benchmark: deserialize 1000 small dicts."""
    serialized = datason.dumps(list_of_dicts)
    result = benchmark(datason.loads, serialized)
    assert isinstance(result, list)


def test_bench_loads_datetimes(benchmark, datetime_objects: list) -> None:
    """Benchmark: deserialize 500 objects with datetime metadata."""
    serialized = datason.dumps(datetime_objects)
    result = benchmark(datason.loads, serialized)
    assert isinstance(result, list)


def test_bench_loads_uuids(benchmark, uuid_objects: list) -> None:
    """Benchmark: deserialize 500 objects with UUID metadata."""
    serialized = datason.dumps(uuid_objects)
    result = benchmark(datason.loads, serialized)
    assert isinstance(result, list)


# =========================================================================
# Round-trip benchmarks
# =========================================================================


def test_bench_round_trip_simple(benchmark, simple_dict: dict) -> None:
    """Benchmark: full serialize → deserialize round-trip (simple)."""

    def round_trip() -> dict:
        s = datason.dumps(simple_dict)
        return datason.loads(s)

    result = benchmark(round_trip)
    assert result == simple_dict


def test_bench_round_trip_nested(benchmark, nested_dict: dict) -> None:
    """Benchmark: full round-trip (nested dict)."""

    def round_trip() -> dict:
        s = datason.dumps(nested_dict)
        return datason.loads(s)

    result = benchmark(round_trip)
    assert result == nested_dict


def test_bench_round_trip_datetimes(benchmark, datetime_objects: list) -> None:
    """Benchmark: full round-trip (500 datetime objects)."""

    def round_trip() -> list:
        s = datason.dumps(datetime_objects)
        return datason.loads(s)

    result = benchmark(round_trip)
    assert len(result) == 500


# =========================================================================
# Throughput benchmark
# =========================================================================


def test_bench_throughput_small_items(benchmark) -> None:
    """Benchmark: throughput for many small items (target: >250K items/sec)."""
    items = [{"x": i} for i in range(1000)]

    def serialize_all() -> list[str]:
        return [datason.dumps(item) for item in items]

    results = benchmark(serialize_all)
    assert len(results) == 1000
