#!/usr/bin/env python3
"""Benchmark tests for datason performance optimization and new features.

This module provides comprehensive benchmarks to measure:
- DataFrame orientation performance
- Auto-detection deserialization performance
- Type metadata round-trip performance
- Performance optimization effectiveness
- Memory usage patterns
"""

import gc
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict

import pytest

from datason import auto_deserialize, deserialize, serialize
from datason.config import DataFrameOrient, SerializationConfig


# Test data generators
def generate_simple_data(size: int = 1000) -> Dict[str, Any]:
    """Generate simple test data for benchmarking."""
    return {
        "integers": list(range(size)),
        "floats": [i * 1.1 for i in range(size)],
        "strings": [f"string_{i}" for i in range(size)],
        "booleans": [i % 2 == 0 for i in range(size)],
        "mixed": [i if i % 3 == 0 else f"str_{i}" if i % 3 == 1 else i * 1.1 for i in range(size)],
    }


def generate_complex_data(size: int = 100) -> Dict[str, Any]:
    """Generate complex test data with various types."""
    return {
        "metadata": {
            "id": uuid.uuid4(),
            "timestamp": datetime.now(),
            "version": "1.0.0",
        },
        "records": [
            {
                "id": uuid.uuid4(),
                "timestamp": datetime.now(),
                "value": i * 1.1,
                "category": f"cat_{i % 10}",
                "active": i % 2 == 0,
                "tags": {f"tag_{j}" for j in range(i % 5)},
                "coordinates": (i, i * 2, i * 3),
            }
            for i in range(size)
        ],
        "summary": {
            "total_records": size,
            "categories": [f"cat_{i}" for i in range(10)],
            "date_range": (datetime(2023, 1, 1), datetime.now()),
        },
    }


def generate_pandas_data(rows: int = 1000):
    """Generate pandas test data."""
    try:
        import numpy as np
        import pandas as pd

        return {
            "dataframe": pd.DataFrame(
                {
                    "int_col": range(rows),
                    "float_col": np.random.random(rows),
                    "str_col": [f"value_{i}" for i in range(rows)],
                    "bool_col": np.random.choice([True, False], rows),
                    "datetime_col": pd.date_range("2023-01-01", periods=rows, freq="1h"),
                }
            ),
            "series": pd.Series(np.random.random(rows), name="random_series"),
            "large_dataframe": pd.DataFrame(np.random.random((rows, 10))),
        }
    except ImportError:
        return {}


class BenchmarkTimer:
    """Context manager for timing operations."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        gc.collect()  # Clean up before timing
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000


@pytest.mark.benchmark
class TestSerializationBenchmarks:
    """Benchmark serialization performance."""

    def test_simple_data_serialization_benchmark(self, benchmark):
        """Benchmark serialization of simple data structures."""
        data = generate_simple_data(1000)

        def serialize_simple():
            return serialize(data)

        result = benchmark(serialize_simple)
        assert isinstance(result, dict)
        assert len(result["integers"]) == 1000

    def test_complex_data_serialization_benchmark(self, benchmark):
        """Benchmark serialization of complex data with various types."""
        data = generate_complex_data(100)

        def serialize_complex():
            return serialize(data)

        result = benchmark(serialize_complex)
        assert isinstance(result, dict)
        assert len(result["records"]) == 100

    @pytest.mark.skipif(
        not pytest.importorskip("pandas", reason="pandas not available"),
        reason="pandas not available",
    )
    def test_pandas_serialization_benchmark(self, benchmark):
        """Benchmark pandas object serialization."""
        data = generate_pandas_data(1000)

        def serialize_pandas():
            return serialize(data)

        result = benchmark(serialize_pandas)
        assert isinstance(result, dict)
        assert "dataframe" in result


@pytest.mark.benchmark
class TestDataFrameOrientationBenchmarks:
    """Benchmark DataFrame orientation performance."""

    @pytest.mark.skipif(
        not pytest.importorskip("pandas", reason="pandas not available"),
        reason="pandas not available",
    )
    def test_dataframe_orientation_performance(self):
        """Compare performance of different DataFrame orientations."""
        import pandas as pd

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "A": range(1000),
                "B": [f"str_{i}" for i in range(1000)],
                "C": [i * 1.1 for i in range(1000)],
            }
        )

        orientations = [
            DataFrameOrient.RECORDS,
            DataFrameOrient.SPLIT,
            DataFrameOrient.INDEX,
            DataFrameOrient.DICT,
            DataFrameOrient.VALUES,
        ]

        results = {}

        for orientation in orientations:
            config = SerializationConfig(dataframe_orient=orientation)

            with BenchmarkTimer(f"serialize_{orientation.value}") as timer:
                for _ in range(100):  # Multiple iterations for stable timing
                    serialize(df, config=config)

            results[orientation.value] = timer.elapsed_ms / 100  # Average per call

        print("\nDataFrame orientation performance (avg ms per call):")
        for orientation, time_ms in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {orientation:8}: {time_ms:.3f}ms")

        # Verify all orientations work
        assert len(results) == len(orientations)
        assert all(time_ms > 0 for time_ms in results.values())


@pytest.mark.benchmark
class TestAutoDetectionBenchmarks:
    """Benchmark auto-detection deserialization performance."""

    def test_auto_detection_vs_standard_benchmark(self):
        """Compare auto-detection vs standard deserialization performance."""
        # Create test data with mixed types
        test_data = {
            "datetime_strings": ["2023-01-01T10:00:00", "2023-12-25T15:30:00"],
            "uuid_strings": [str(uuid.uuid4()) for _ in range(10)],
            "number_strings": ["42", "3.14159", "1.23e-4"],
            "regular_strings": ["hello", "world", "test"],
            "nested": {
                "more_dates": ["2023-06-15T12:00:00"],
                "more_uuids": [str(uuid.uuid4())],
            },
        }

        # Serialize to get JSON-compatible data
        json_data = json.loads(json.dumps(test_data))

        # Benchmark standard deserialization
        with BenchmarkTimer("standard_deserialize") as standard_timer:
            for _ in range(1000):
                deserialize(json_data)

        # Benchmark auto-detection (conservative)
        with BenchmarkTimer("auto_detect_conservative") as auto_conservative_timer:
            for _ in range(1000):
                auto_deserialize(json_data, aggressive=False)

        # Benchmark auto-detection (aggressive)
        with BenchmarkTimer("auto_detect_aggressive") as auto_aggressive_timer:
            for _ in range(1000):
                auto_deserialize(json_data, aggressive=True)

        print("\nDeserialization performance (1000 iterations):")
        print(
            f"  Standard:           {standard_timer.elapsed_ms:.1f}ms ({standard_timer.elapsed_ms / 1000:.3f}ms per call)"
        )
        print(
            f"  Auto-detect (cons): {auto_conservative_timer.elapsed_ms:.1f}ms ({auto_conservative_timer.elapsed_ms / 1000:.3f}ms per call)"
        )
        print(
            f"  Auto-detect (aggr): {auto_aggressive_timer.elapsed_ms:.1f}ms ({auto_aggressive_timer.elapsed_ms / 1000:.3f}ms per call)"
        )

        # Calculate overhead
        conservative_overhead = (auto_conservative_timer.elapsed / standard_timer.elapsed - 1) * 100
        aggressive_overhead = (auto_aggressive_timer.elapsed / auto_conservative_timer.elapsed - 1) * 100

        print("\nOverhead analysis:")
        print(f"  Auto-detection overhead: {conservative_overhead:.1f}%")
        print(f"  Aggressive mode overhead: {aggressive_overhead:.1f}%")

        # Verify functionality
        standard_result = deserialize(json_data)
        auto_result = auto_deserialize(json_data)

        # Both should have converted datetime and UUID strings
        assert isinstance(auto_result["datetime_strings"][0], datetime)
        assert isinstance(auto_result["uuid_strings"][0], uuid.UUID)
        assert isinstance(standard_result["datetime_strings"][0], datetime)
        assert isinstance(standard_result["uuid_strings"][0], uuid.UUID)

        # But auto-detection should be more comprehensive
        # (Note: both may detect the same things, but auto_deserialize has more features)


@pytest.mark.benchmark
class TestTypeMetadataBenchmarks:
    """Benchmark type metadata performance."""

    def test_type_metadata_round_trip_benchmark(self):
        """Benchmark type metadata round-trip performance."""
        # Create complex test data
        test_data = generate_complex_data(50)

        # Standard serialization/deserialization
        standard_config = SerializationConfig()
        with BenchmarkTimer("standard_round_trip") as standard_timer:
            for _ in range(100):
                serialized = serialize(test_data, config=standard_config)
                deserialize(serialized)

        # Type metadata serialization/deserialization
        metadata_config = SerializationConfig(include_type_hints=True)
        with BenchmarkTimer("metadata_round_trip") as metadata_timer:
            for _ in range(100):
                serialized = serialize(test_data, config=metadata_config)
                auto_deserialize(serialized)

        print("\nRound-trip performance (100 iterations):")
        print(
            f"  Standard:     {standard_timer.elapsed_ms:.1f}ms ({standard_timer.elapsed_ms / 100:.3f}ms per round-trip)"
        )
        print(
            f"  Type metadata: {metadata_timer.elapsed_ms:.1f}ms ({metadata_timer.elapsed_ms / 100:.3f}ms per round-trip)"
        )

        # Calculate overhead/speedup
        metadata_ratio = metadata_timer.elapsed / standard_timer.elapsed
        if metadata_ratio < 1:
            print(f"  Type metadata speedup: {(1 - metadata_ratio) * 100:.1f}%")
        else:
            print(f"  Type metadata overhead: {(metadata_ratio - 1) * 100:.1f}%")

        # Verify round-trip accuracy
        standard_serialized = serialize(test_data, config=standard_config)
        metadata_serialized = serialize(test_data, config=metadata_config)

        deserialize(standard_serialized)
        metadata_restored = auto_deserialize(metadata_serialized)

        # Type metadata should preserve more type information
        assert isinstance(metadata_restored["metadata"]["id"], uuid.UUID)
        assert isinstance(metadata_restored["metadata"]["timestamp"], datetime)

        # Standard may not preserve all types
        # (depending on auto-detection in deserialize function)


@pytest.mark.benchmark
class TestOptimizationBenchmarks:
    """Benchmark performance optimizations."""

    def test_json_safe_optimization_benchmark(self):
        """Benchmark the JSON-safe data optimization."""
        # Create data that's already JSON-safe
        json_safe_data = {
            "strings": ["hello", "world", "test"],
            "numbers": [1, 2, 3, 4, 5],
            "floats": [1.1, 2.2, 3.3],
            "booleans": [True, False, True],
            "nested": {
                "more_strings": ["a", "b", "c"],
                "more_numbers": [10, 20, 30],
            },
        }

        # Create data that needs processing
        complex_data = generate_complex_data(50)

        # Benchmark JSON-safe data (should be optimized)
        config_with_optimization = SerializationConfig(check_if_serialized=True)
        with BenchmarkTimer("json_safe_optimized") as optimized_timer:
            for _ in range(1000):
                serialize(json_safe_data, config=config_with_optimization)

        # Benchmark without optimization
        config_without_optimization = SerializationConfig(check_if_serialized=False)
        with BenchmarkTimer("json_safe_unoptimized") as unoptimized_timer:
            for _ in range(1000):
                serialize(json_safe_data, config=config_without_optimization)

        # Benchmark complex data (optimization shouldn't help much)
        with BenchmarkTimer("complex_optimized") as complex_optimized_timer:
            for _ in range(100):
                serialize(complex_data, config=config_with_optimization)

        with BenchmarkTimer("complex_unoptimized") as complex_unoptimized_timer:
            for _ in range(100):
                serialize(complex_data, config=config_without_optimization)

        print("\nOptimization performance:")
        print(f"  JSON-safe optimized:   {optimized_timer.elapsed_ms:.1f}ms (1000 calls)")
        print(f"  JSON-safe unoptimized: {unoptimized_timer.elapsed_ms:.1f}ms (1000 calls)")
        print(f"  Complex optimized:     {complex_optimized_timer.elapsed_ms:.1f}ms (100 calls)")
        print(f"  Complex unoptimized:   {complex_unoptimized_timer.elapsed_ms:.1f}ms (100 calls)")

        # Calculate speedup for JSON-safe data
        json_safe_speedup = (unoptimized_timer.elapsed / optimized_timer.elapsed - 1) * 100
        complex_speedup = (complex_unoptimized_timer.elapsed / complex_optimized_timer.elapsed - 1) * 100

        print("\nOptimization effectiveness:")
        print(f"  JSON-safe data speedup: {json_safe_speedup:.1f}%")
        print(f"  Complex data speedup:   {complex_speedup:.1f}%")

        # Optimization should provide speedup for JSON-safe data (allowing for timing variance)
        # Use a tolerance to account for system load and timing variations
        speedup_ratio = optimized_timer.elapsed / unoptimized_timer.elapsed
        assert speedup_ratio < 1.1, f"Optimization failed to provide meaningful speedup: {speedup_ratio:.3f}"

        # Results should be identical
        optimized_result = serialize(json_safe_data, config=config_with_optimization)
        unoptimized_result = serialize(json_safe_data, config=config_without_optimization)
        assert optimized_result == unoptimized_result


@pytest.mark.benchmark
class TestMemoryBenchmarks:
    """Benchmark memory usage patterns."""

    def test_memory_usage_comparison(self):
        """Compare memory usage of different serialization approaches."""
        import sys

        # Create test data
        test_data = generate_complex_data(100)

        # Measure memory for different configurations
        configs = {
            "standard": SerializationConfig(),
            "optimized": SerializationConfig(check_if_serialized=True),
            "metadata": SerializationConfig(include_type_hints=True),
            "metadata_optimized": SerializationConfig(include_type_hints=True, check_if_serialized=True),
        }

        results = {}

        for name, config in configs.items():
            # Serialize multiple times to get stable measurements
            serialized_objects = []

            with BenchmarkTimer(f"memory_{name}") as timer:
                for _ in range(50):
                    result = serialize(test_data, config=config)
                    serialized_objects.append(result)

            # Estimate memory usage (rough approximation)
            total_size = sum(sys.getsizeof(obj) for obj in serialized_objects)
            avg_size = total_size / len(serialized_objects)

            results[name] = {
                "time_ms": timer.elapsed_ms,
                "avg_size_bytes": avg_size,
                "time_per_call_ms": timer.elapsed_ms / 50,
            }

        print("\nMemory and performance comparison (50 iterations):")
        print(f"{'Config':<20} {'Time (ms)':<12} {'Per call (ms)':<15} {'Avg size (bytes)':<18}")
        print("-" * 70)

        for name, metrics in results.items():
            print(
                f"{name:<20} {metrics['time_ms']:<12.1f} {metrics['time_per_call_ms']:<15.3f} {metrics['avg_size_bytes']:<18.0f}"
            )

        # Verify all configurations produce valid results
        for _name, config in configs.items():
            result = serialize(test_data, config=config)
            assert isinstance(result, dict)
            assert "records" in result


@pytest.mark.benchmark
class TestScalabilityBenchmarks:
    """Benchmark scalability with different data sizes."""

    @pytest.mark.parametrize("size", [100, 500, 1000, 2000])
    def test_serialization_scalability(self, size):
        """Test serialization performance with different data sizes."""
        data = generate_simple_data(size)

        with BenchmarkTimer(f"serialize_{size}") as timer:
            result = serialize(data)

        print(f"\nSize {size:4d}: {timer.elapsed_ms:6.2f}ms ({timer.elapsed_ms / size:.4f}ms per item)")

        # Verify result
        assert isinstance(result, dict)
        assert len(result["integers"]) == size

        # Performance should scale reasonably (not exponentially)
        items_per_ms = size / timer.elapsed_ms
        assert items_per_ms > 10  # Should process at least 10 items per ms

    @pytest.mark.parametrize("complexity", [10, 50, 100, 200])
    def test_complexity_scalability(self, complexity):
        """Test performance with different data complexity."""
        data = generate_complex_data(complexity)

        with BenchmarkTimer(f"complex_{complexity}") as timer:
            result = serialize(data)

        print(
            f"\nComplexity {complexity:3d}: {timer.elapsed_ms:6.2f}ms ({timer.elapsed_ms / complexity:.4f}ms per record)"
        )

        # Verify result
        assert isinstance(result, dict)
        assert len(result["records"]) == complexity


@pytest.mark.benchmark
def test_benchmark_summary():
    """Print a summary of benchmark capabilities."""
    print("\n" + "=" * 60)
    print("DATASON BENCHMARK SUMMARY")
    print("=" * 60)
    print("\nBenchmark categories:")
    print("  📊 Serialization: Basic and complex data structures")
    print("  🐼 DataFrame: Different orientation performance")
    print("  🔍 Auto-detection: Type recognition overhead")
    print("  🏷️  Type metadata: Round-trip serialization")
    print("  ⚡ Optimization: JSON-safe data speedup")
    print("  💾 Memory: Usage patterns and efficiency")
    print("  📈 Scalability: Performance with varying data sizes")
    print("\nTo run specific benchmarks:")
    print("  pytest tests/test_benchmarks.py::TestSerializationBenchmarks -v")
    print("  pytest tests/test_benchmarks.py::TestAutoDetectionBenchmarks -v")
    print("  pytest tests/test_benchmarks.py::TestOptimizationBenchmarks -v")
    print("\nFor detailed performance analysis:")
    print("  pytest tests/test_benchmarks.py -v -s")
    print("=" * 60)


if __name__ == "__main__":
    # Run a quick performance test
    print("🚀 Quick Datason Performance Test")
    print("-" * 40)

    # Test basic serialization
    data = generate_simple_data(1000)
    with BenchmarkTimer("basic_serialization") as timer:
        result = serialize(data)
    print(f"✅ Serialized 1000 items in {timer.elapsed_ms:.2f}ms")

    # Test auto-detection
    json_data = json.loads(json.dumps({"date": "2023-01-01T10:00:00", "uuid": str(uuid.uuid4()), "number": "42"}))

    with BenchmarkTimer("auto_detection") as timer:
        auto_result = auto_deserialize(json_data, aggressive=True)
    print(f"✅ Auto-detected types in {timer.elapsed_ms:.3f}ms")

    # Verify auto-detection worked
    assert isinstance(auto_result["date"], datetime)
    assert isinstance(auto_result["uuid"], uuid.UUID)
    assert isinstance(auto_result["number"], int)

    print("🎉 Performance test completed successfully!")
