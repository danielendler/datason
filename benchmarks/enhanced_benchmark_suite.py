#!/usr/bin/env python3
"""Enhanced datason Benchmark Suite

Comprehensive performance testing for datason's new configuration system
and advanced type handling features.

This extends the existing benchmark_real_performance.py with:
- Configuration system performance comparison
- Advanced type handling benchmarks
- Pandas DataFrame orientation benchmarks
- NaN handling strategy performance
- Type coercion strategy comparison
- Custom serializer performance

Usage:
    python enhanced_benchmark_suite.py
"""

import decimal
import enum
import functools
import json
import time
import uuid
from collections import namedtuple
from datetime import datetime
from pathlib import Path

# Optional dependencies
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

import datason
from datason.config import (
    DataFrameOrient,
    DateFormat,
    NanHandling,
    SerializationConfig,
    TypeCoercion,
    get_api_config,
    get_ml_config,
    get_performance_config,
    get_strict_config,
)


class Status(enum.Enum):
    """Test enum for benchmarking type handlers."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


Point = namedtuple("Point", ["x", "y"])


def time_operation(func, *args, **kwargs):
    """Time a single operation."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return end - start, result


def benchmark_operation(func, data, iterations=5):
    """Benchmark an operation multiple times and return statistics."""
    times = []

    # Warm-up run
    time_operation(func, data)

    # Actual benchmark runs
    for _ in range(iterations):
        elapsed, _ = time_operation(func, data)
        times.append(elapsed)

    mean_time = sum(times) / len(times)
    variance = sum((t - mean_time) ** 2 for t in times) / len(times)
    stdev = variance**0.5

    return {
        "mean": mean_time,
        "stdev": stdev,
        "min": min(times),
        "max": max(times),
        "operations_per_sec": 1.0 / mean_time if mean_time > 0 else 0,
    }


def create_advanced_test_data():
    """Create test data with advanced Python types."""
    return {
        "decimals": [decimal.Decimal(f"{i}.{i}") for i in range(100)],
        "uuids": [uuid.uuid4() for _ in range(100)],
        "complex_nums": [complex(i, i + 1) for i in range(100)],
        "paths": [Path(f"/data/file_{i}.txt") for i in range(50)],
        "enums": [Status.ACTIVE, Status.PENDING, Status.INACTIVE] * 50,
        "mixed": {
            "decimal": decimal.Decimal("123.456"),
            "path": Path("/home/user"),
            "status": Status.ACTIVE,
            "uuid": uuid.uuid4(),
            "complex": 1 + 2j,
        },
    }


def create_pandas_test_data():
    """Create test data with pandas objects."""
    if not HAS_PANDAS:
        return {"empty": True}

    return {
        "large_df": pd.DataFrame(
            {
                "values": np.random.random(5000) if HAS_NUMPY else range(5000),
                "categories": [f"cat_{i % 20}" for i in range(5000)],
                "timestamps": pd.date_range("2023-01-01", periods=5000, freq="h"),
            }
        ),
        "mixed_types_df": pd.DataFrame(
            {
                "uuids": [uuid.uuid4() for _ in range(200)],
                "decimals": [decimal.Decimal(f"{i}.99") for i in range(200)],
                "complex_nums": [complex(i, i + 1) for i in range(200)],
            }
        ),
        "nan_df": pd.DataFrame(
            {
                "values": [1, 2, float("nan"), 4, None, 6, float("inf")] * 50,  # 350 items
                "strings": ["a", "b", None, "d", "e", None, "f"] * 50,  # 350 items to match
            }
        ),
    }


def benchmark_configuration_presets():
    """Benchmark different configuration presets."""
    print("\n=== Configuration Presets Performance ===")

    # Create test data
    advanced_data = create_advanced_test_data()
    pandas_data = create_pandas_test_data()

    configs = [
        (get_ml_config(), "ML Config"),
        (get_api_config(), "API Config"),
        (get_strict_config(), "Strict Config"),
        (get_performance_config(), "Performance Config"),
        (None, "Default (No Config)"),
    ]

    test_datasets = [
        (advanced_data, "Advanced Types"),
        (pandas_data, "Pandas Data") if HAS_PANDAS else ({"empty": True}, "Pandas Data (unavailable)"),
    ]

    for data, data_name in test_datasets:
        print(f"\n--- {data_name} ---")
        results = {}

        for config, config_name in configs:
            if config is None:

                def serialize_func(d):
                    return datason.serialize(d)
            else:
                serialize_func = functools.partial(datason.serialize, config=config)

            stats = benchmark_operation(serialize_func, data)
            results[config_name] = stats

            print(
                f"{config_name:20s}: {stats['mean'] * 1000:6.2f}ms ± {stats['stdev'] * 1000:5.2f}ms "
                f"({stats['operations_per_sec']:8.0f} ops/sec)"
            )

    return results


def benchmark_date_formats():
    """Benchmark different date format options."""
    print("\n=== Date Format Performance ===")

    # Create data with many datetime objects
    data = {
        "timestamps": [datetime.now() for _ in range(1000)],
        "single_date": datetime.now(),
        "nested": {"dates": [datetime.now() for _ in range(500)]},
    }

    formats = [
        (DateFormat.ISO, "ISO Format"),
        (DateFormat.UNIX, "Unix Timestamp"),
        (DateFormat.UNIX_MS, "Unix Milliseconds"),
        (DateFormat.STRING, "String Format"),
        (DateFormat.CUSTOM, "Custom Format (%Y-%m-%d)"),
    ]

    results = {}

    for date_format, format_name in formats:
        if date_format == DateFormat.CUSTOM:
            config = SerializationConfig(date_format=date_format, custom_date_format="%Y-%m-%d")
        else:
            config = SerializationConfig(date_format=date_format)

        serialize_func = functools.partial(datason.serialize, config=config)
        stats = benchmark_operation(serialize_func, data)
        results[format_name] = stats

        print(f"{format_name:20s}: {stats['mean'] * 1000:6.2f}ms ± {stats['stdev'] * 1000:5.2f}ms")

    return results


def benchmark_nan_handling():
    """Benchmark NaN handling strategies."""
    print("\n=== NaN Handling Performance ===")

    # Create data with many NaN values
    nan_data = {
        "values": [1, 2, float("nan"), 4, None, float("inf")] * 500,
        "nested": {"more_nans": [float("nan"), None, 42] * 300},
    }

    strategies = [
        (NanHandling.NULL, "Convert to NULL"),
        (NanHandling.STRING, "Convert to String"),
        (NanHandling.KEEP, "Keep Original"),
        (NanHandling.DROP, "Drop Values"),
    ]

    results = {}

    for strategy, strategy_name in strategies:
        config = SerializationConfig(nan_handling=strategy)
        serialize_func = functools.partial(datason.serialize, config=config)
        stats = benchmark_operation(serialize_func, nan_data)
        results[strategy_name] = stats

        print(f"{strategy_name:20s}: {stats['mean'] * 1000:6.2f}ms ± {stats['stdev'] * 1000:5.2f}ms")

    return results


def benchmark_type_coercion():
    """Benchmark type coercion strategies."""
    print("\n=== Type Coercion Performance ===")

    # Create data with many advanced types
    type_data = {
        "decimals": [decimal.Decimal(f"{i}.{i}") for i in range(200)],
        "uuids": [uuid.uuid4() for _ in range(200)],
        "complex_nums": [complex(i, i + 1) for i in range(200)],
        "paths": [Path(f"/data/file_{i}.txt") for i in range(100)],
        "enums": [Status.ACTIVE, Status.PENDING] * 100,
    }

    strategies = [
        (TypeCoercion.STRICT, "Strict (Preserve All)"),
        (TypeCoercion.SAFE, "Safe (Default)"),
        (TypeCoercion.AGGRESSIVE, "Aggressive (Simplify)"),
    ]

    results = {}

    for strategy, strategy_name in strategies:
        config = SerializationConfig(type_coercion=strategy)
        serialize_func = functools.partial(datason.serialize, config=config)
        stats = benchmark_operation(serialize_func, type_data)
        results[strategy_name] = stats

        print(f"{strategy_name:20s}: {stats['mean'] * 1000:6.2f}ms ± {stats['stdev'] * 1000:5.2f}ms")

    return results


def benchmark_dataframe_orientations():
    """Benchmark pandas DataFrame orientations."""
    if not HAS_PANDAS:
        print("\n=== DataFrame Orientations ===")
        print("Pandas not available - skipping DataFrame benchmarks")
        return {}

    print("\n=== DataFrame Orientations Performance ===")

    # Create various DataFrame sizes
    dataframes = {
        "small": pd.DataFrame(
            {
                "A": range(100),
                "B": [f"item_{i}" for i in range(100)],
                "C": pd.date_range("2024-01-01", periods=100),
            }
        ),
        "medium": pd.DataFrame(
            {
                "values": np.random.random(1000) if HAS_NUMPY else range(1000),
                "categories": [f"cat_{i % 10}" for i in range(1000)],
                "ids": [uuid.uuid4() for _ in range(1000)],
            }
        ),
        "large": pd.DataFrame(
            {
                "data": np.random.random(5000) if HAS_NUMPY else range(5000),
                "groups": [f"group_{i % 50}" for i in range(5000)],
            }
        ),
    }

    orientations = [
        (DataFrameOrient.RECORDS, "Records"),
        (DataFrameOrient.SPLIT, "Split"),
        (DataFrameOrient.INDEX, "Index"),
        (DataFrameOrient.COLUMNS, "Columns"),
        (DataFrameOrient.VALUES, "Values"),
        (DataFrameOrient.TABLE, "Table"),
    ]

    results = {}

    for df_name, df in dataframes.items():
        print(f"\n--- {df_name.title()} DataFrame ({len(df)} rows) ---")
        df_results = {}

        for orient, orient_name in orientations:
            config = SerializationConfig(dataframe_orient=orient)
            test_data = {"dataframe": df}

            serialize_func = functools.partial(datason.serialize, config=config)
            stats = benchmark_operation(serialize_func, test_data)

            df_results[orient_name] = stats
            print(f"{orient_name:12s}: {stats['mean'] * 1000:6.2f}ms ± {stats['stdev'] * 1000:5.2f}ms")

        results[df_name] = df_results

    return results


def benchmark_custom_serializers():
    """Benchmark custom serializer performance."""
    print("\n=== Custom Serializers Performance ===")

    # Define custom types and serializers
    class CustomObject:
        def __init__(self, obj_id, data):
            self.id = obj_id
            self.data = data

    def fast_serializer(obj):
        return {"id": obj.id, "type": "fast"}

    def detailed_serializer(obj):
        return {
            "id": obj.id,
            "data": obj.data,
            "type": "detailed",
            "size": len(str(obj.data)),
        }

    # Create test data
    test_data = {"objects": [CustomObject(i, f"data_{i}" * 10) for i in range(1000)]}

    configs = [
        (None, "No Custom Serializer"),
        ({CustomObject: fast_serializer}, "Fast Custom Serializer"),
        ({CustomObject: detailed_serializer}, "Detailed Custom Serializer"),
    ]

    results = {}

    for custom_serializers, config_name in configs:
        if custom_serializers is None:

            def serialize_func(d):
                return datason.serialize(d)
        else:
            config = SerializationConfig(custom_serializers=custom_serializers)
            serialize_func = functools.partial(datason.serialize, config=config)

        stats = benchmark_operation(serialize_func, test_data)
        results[config_name] = stats

        print(f"{config_name:25s}: {stats['mean'] * 1000:6.2f}ms ± {stats['stdev'] * 1000:5.2f}ms")

    return results


def benchmark_memory_usage():
    """Analyze memory usage of different configurations."""
    print("\n=== Memory Usage Analysis ===")

    # Create substantial test data
    test_data = {
        "large_list": list(range(10000)),
        "dataframes": [
            pd.DataFrame({"values": range(1000), "text": [f"item_{i}" for i in range(1000)]}) for _ in range(5)
        ]
        if HAS_PANDAS
        else [],
        "mixed_objects": {
            "decimals": [decimal.Decimal(f"{i}.{i}") for i in range(500)],
            "paths": [Path(f"/path/to/file_{i}.txt") for i in range(500)],
            "uuids": [uuid.uuid4() for _ in range(500)],
        },
    }

    configs = [
        (get_performance_config(), "Performance Config"),
        (get_strict_config(), "Strict Config"),
        (SerializationConfig(nan_handling=NanHandling.DROP), "NaN Drop Config"),
    ]

    for config, config_name in configs:
        result = datason.serialize(test_data, config=config)
        json_str = json.dumps(result)
        size_kb = len(json_str.encode("utf-8")) / 1024

        print(f"{config_name:20s}: ~{size_kb:.1f}KB serialized size")


def run_comprehensive_benchmarks():
    """Run all benchmark tests."""
    print("datason Enhanced Benchmark Suite")
    print("=" * 50)
    print(f"Python version: {datason.__version__}")
    print(f"NumPy available: {HAS_NUMPY}")
    print(f"Pandas available: {HAS_PANDAS}")

    all_results = {}

    # Run all benchmarks
    all_results["config_presets"] = benchmark_configuration_presets()
    all_results["date_formats"] = benchmark_date_formats()
    all_results["nan_handling"] = benchmark_nan_handling()
    all_results["type_coercion"] = benchmark_type_coercion()
    all_results["dataframe_orientations"] = benchmark_dataframe_orientations()
    all_results["custom_serializers"] = benchmark_custom_serializers()

    # Memory usage analysis
    benchmark_memory_usage()

    # Summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)

    # Find fastest configuration for DataFrames
    df_results = all_results.get("config_presets", {})
    if df_results:
        fastest_config = None
        fastest_time = float("inf")
        for config_name, stats in df_results.items():
            if isinstance(stats, dict) and "mean" in stats and stats["mean"] < fastest_time:
                fastest_time = stats["mean"]
                fastest_config = config_name

        if fastest_config:
            print(f"Fastest Config: {fastest_config} ({fastest_time * 1000:.2f}ms)")

    # Find fastest type coercion
    coercion_results = all_results.get("type_coercion", {})
    if coercion_results:
        fastest_coercion = min(coercion_results.items(), key=lambda x: x[1]["mean"])
        print(f"Fastest Type Coercion: {fastest_coercion[0]} ({fastest_coercion[1]['mean'] * 1000:.2f}ms)")

    print("\nConfiguration Recommendations:")
    print("- For ML pipelines: Use get_ml_config() (optimized for numeric data)")
    print("- For APIs: Use get_api_config() (human-readable, consistent)")
    print("- For performance: Use get_performance_config() (minimal overhead)")
    print("- For debugging: Use get_strict_config() (maximum information)")

    return all_results


if __name__ == "__main__":
    results = run_comprehensive_benchmarks()
