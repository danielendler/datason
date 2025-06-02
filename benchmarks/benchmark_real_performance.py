#!/usr/bin/env python3
"""datason Official Performance Benchmark Suite

This script provides the authoritative performance measurements for datason.
It measures real-world performance across different data types and use cases
to provide transparent, reproducible benchmarks.

Key measurements:
- Serialization speed vs standard JSON
- Memory usage with large datasets
- Performance with ML/AI data types
- Throughput for high-volume operations
- Round-trip (serialize + deserialize) timing

Methodology:
- Uses statistical analysis (multiple runs, mean, std dev)
- Tests both datason advantages and limitations
- Compares against standard library and common alternatives
- Include both datason strengths and weaknesses
- Provides context for when to use datason vs alternatives

Usage:
    python benchmark_real_performance.py

This will run comprehensive benchmarks and output detailed results.
All timings are measured on the current system configuration.

USAGE:
    python benchmark_real_performance.py

WHAT IT MEASURES:
    - Simple data performance vs standard JSON
    - Complex data performance vs alternatives (pickle)
    - High-throughput scenarios (large datasets, NumPy, Pandas)
    - Round-trip performance (serialize + JSON + deserialize)
    - Statistical analysis with multiple iterations

METHODOLOGY:
    - 5 iterations per test for statistical reliability
    - Uses time.perf_counter() for high-precision timing
    - Tests with realistic data structures
    - Reports mean ± standard deviation
    - Fair comparisons (like-for-like where possible)

ENVIRONMENT:
    - Requires: NumPy, Pandas, PyTorch (for ML object testing)
    - Tested on: Python 3.13.3, macOS
    - Results may vary by platform and Python version

OUTPUT:
    - Console output with detailed measurements
    - Summary section for documentation use
    - Real numbers replace estimates in docs

BENCHMARKING BEST PRACTICES:
    - Run on dedicated machine (minimal background processes)
    - Multiple runs to verify consistency
    - Representative data sizes and structures
    - Include both datason strengths and weaknesses

This script is referenced in:
    - README.md performance section
    - docs/AI_USAGE_GUIDE.md benchmarks
    - docs/FEATURE_MATRIX.md performance data
    - docs/BENCHMARKING.md methodology
    - CONTRIBUTING.md performance standards
"""

import json
import pickle  # nosec B403 - Safe usage for benchmarking only, no untrusted data
import statistics
import time
import uuid
from datetime import datetime

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

import datason as ds


def time_operation(func, *args, **kwargs):
    """Time a single operation and return result and duration."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def benchmark_operation(func, data, iterations=5):
    """Run benchmark multiple times and return statistics."""
    times = []
    for _ in range(iterations):
        _, duration = time_operation(func, data)
        times.append(duration)

    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def create_test_data():
    """Create various test datasets."""
    datasets = {}

    # Simple data (JSON-compatible)
    datasets["simple"] = {
        "users": [{"id": i, "name": f"user_{i}", "active": True, "score": i * 1.5} for i in range(1000)],
        "metadata": {"total": 1000, "generated": "2024-01-01", "version": "1.0"},
    }

    # Complex data (datason advantages)
    datasets["complex"] = {
        "sessions": [
            {
                "id": uuid.uuid4(),
                "start_time": datetime.now(),
                "user_data": {
                    "preferences": [f"pref_{j}" for j in range(5)],
                    "last_login": datetime.now(),
                },
            }
            for i in range(500)
        ],
        "system": {"startup_time": datetime.now(), "config_id": uuid.uuid4()},
    }

    # Large nested data
    datasets["large_nested"] = {}
    for i in range(100):
        datasets["large_nested"][f"group_{i}"] = {
            "items": [
                {
                    "id": uuid.uuid4(),
                    "timestamp": datetime.now(),
                    "data": list(range(20)),
                }
                for j in range(50)
            ]
        }

    if HAS_NUMPY:
        datasets["numpy"] = {
            "arrays": {
                "small": np.random.random(100),
                "medium": np.random.random(1000),
                "large": np.random.random(10000),
                "matrix": np.random.random((100, 100)),
                "int_array": np.arange(1000),
                "bool_array": np.random.choice([True, False], 1000),
            }
        }

    if HAS_PANDAS:
        datasets["pandas"] = {
            "dataframes": {
                "small": pd.DataFrame(
                    {
                        "A": np.random.random(100) if HAS_NUMPY else list(range(100)),
                        "B": pd.date_range("2023-01-01", periods=100),
                        "C": [f"item_{i}" for i in range(100)],
                    }
                ),
                "large": pd.DataFrame(
                    {
                        "values": np.random.random(5000) if HAS_NUMPY else list(range(5000)),
                        "timestamps": pd.date_range("2023-01-01", periods=5000),
                        "categories": [f"cat_{i % 10}" for i in range(5000)],
                    }
                ),
            },
            "series": pd.Series(np.random.random(1000) if HAS_NUMPY else list(range(1000))),
        }

    return datasets


def run_comparison_benchmarks():
    """Compare datason with standard JSON on compatible data."""
    print("🏁 Running Performance Benchmarks...")
    print("=" * 60)

    datasets = create_test_data()
    results = {}

    # Test simple data (both libraries can handle)
    simple_data = datasets["simple"]

    print("\n📊 Simple Data Performance (1000 users)")
    print("-" * 40)

    # Standard JSON
    json_stats = benchmark_operation(json.dumps, simple_data)
    print(f"Standard JSON:     {json_stats['mean'] * 1000:.2f}ms ± {json_stats['stdev'] * 1000:.2f}ms")

    # datason
    sp_stats = benchmark_operation(ds.serialize, simple_data)
    print(f"datason:         {sp_stats['mean'] * 1000:.2f}ms ± {sp_stats['stdev'] * 1000:.2f}ms")

    # Ratio
    ratio = sp_stats["mean"] / json_stats["mean"]
    print(f"datason/JSON:    {ratio:.2f}x")

    results["simple"] = {"json": json_stats, "datason": sp_stats, "ratio": ratio}

    # Test complex data (only datason can handle)
    complex_data = datasets["complex"]

    print("\n🧩 Complex Data Performance (500 sessions with UUIDs/datetimes)")
    print("-" * 55)

    sp_complex_stats = benchmark_operation(ds.serialize, complex_data)
    print(f"datason:         {sp_complex_stats['mean'] * 1000:.2f}ms ± {sp_complex_stats['stdev'] * 1000:.2f}ms")

    # Try to serialize with pickle for comparison
    pickle_stats = benchmark_operation(pickle.dumps, complex_data)
    print(f"Pickle:            {pickle_stats['mean'] * 1000:.2f}ms ± {pickle_stats['stdev'] * 1000:.2f}ms")

    pickle_ratio = sp_complex_stats["mean"] / pickle_stats["mean"]
    print(f"datason/Pickle:  {pickle_ratio:.2f}x")

    results["complex"] = {
        "datason": sp_complex_stats,
        "pickle": pickle_stats,
        "pickle_ratio": pickle_ratio,
    }

    # Test large nested data
    large_data = datasets["large_nested"]

    print("\n📈 Large Nested Data Performance (100 groups × 50 items)")
    print("-" * 50)

    sp_large_stats = benchmark_operation(ds.serialize, large_data)
    print(f"datason:         {sp_large_stats['mean'] * 1000:.2f}ms ± {sp_large_stats['stdev'] * 1000:.2f}ms")

    # Calculate throughput
    item_count = 100 * 50  # 5000 items
    throughput = item_count / sp_large_stats["mean"]
    print(f"Throughput:        {throughput:.0f} items/second")

    results["large"] = {"datason": sp_large_stats, "throughput": throughput}

    # Test numpy data if available
    if HAS_NUMPY:
        numpy_data = datasets["numpy"]

        print("\n🔢 NumPy Data Performance")
        print("-" * 30)

        sp_numpy_stats = benchmark_operation(ds.serialize, numpy_data)
        print(f"datason:         {sp_numpy_stats['mean'] * 1000:.2f}ms ± {sp_numpy_stats['stdev'] * 1000:.2f}ms")

        # Calculate data size
        total_elements = 100 + 1000 + 10000 + (100 * 100) + 1000 + 1000  # ~122K elements
        elements_per_sec = total_elements / sp_numpy_stats["mean"]
        print(f"Elements/sec:      {elements_per_sec:.0f}")

        results["numpy"] = {
            "datason": sp_numpy_stats,
            "elements_per_sec": elements_per_sec,
        }

    # Test pandas data if available
    if HAS_PANDAS:
        pandas_data = datasets["pandas"]

        print("\n🐼 Pandas Data Performance")
        print("-" * 30)

        sp_pandas_stats = benchmark_operation(ds.serialize, pandas_data)
        print(f"datason:         {sp_pandas_stats['mean'] * 1000:.2f}ms ± {sp_pandas_stats['stdev'] * 1000:.2f}ms")

        # DataFrame size
        df_rows = len(pandas_data["dataframes"]["small"]) + len(pandas_data["dataframes"]["large"])
        rows_per_sec = df_rows / sp_pandas_stats["mean"]
        print(f"DataFrame rows/sec: {rows_per_sec:.0f}")

        results["pandas"] = {"datason": sp_pandas_stats, "rows_per_sec": rows_per_sec}

    return results


def test_round_trip_performance():
    """Test serialization + deserialization performance."""
    print("\n🔄 Round-trip Performance (Serialize + Deserialize)")
    print("-" * 50)

    test_data = {
        "users": [
            {
                "id": uuid.uuid4(),
                "created": datetime.now(),
                "preferences": [f"pref_{j}" for j in range(3)],
            }
            for i in range(200)
        ]
    }

    def round_trip_test(data):
        """Test full round-trip performance."""
        serialized = ds.serialize(data)
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)
        return ds.deserialize(parsed)

    rt_stats = benchmark_operation(round_trip_test, test_data)
    print(f"Round-trip:        {rt_stats['mean'] * 1000:.2f}ms ± {rt_stats['stdev'] * 1000:.2f}ms")

    # Test just serialization
    ser_stats = benchmark_operation(ds.serialize, test_data)
    print(f"Serialize only:    {ser_stats['mean'] * 1000:.2f}ms ± {ser_stats['stdev'] * 1000:.2f}ms")

    # Test just deserialization
    serialized = ds.serialize(test_data)
    json_str = json.dumps(serialized)
    parsed = json.loads(json_str)

    deser_stats = benchmark_operation(ds.deserialize, parsed)
    print(f"Deserialize only:  {deser_stats['mean'] * 1000:.2f}ms ± {deser_stats['stdev'] * 1000:.2f}ms")

    print(f"Total check:       {(ser_stats['mean'] + deser_stats['mean']) * 1000:.2f}ms")

    return {"round_trip": rt_stats, "serialize": ser_stats, "deserialize": deser_stats}


def generate_performance_summary(results, round_trip_results):
    """Generate a summary for documentation."""
    print("\n" + "=" * 60)
    print("📋 PERFORMANCE SUMMARY FOR DOCUMENTATION")
    print("=" * 60)

    print("\n**Simple Data (JSON-compatible):**")
    simple = results["simple"]
    print(f"- datason: {simple['datason']['mean'] * 1000:.1f}ms")
    print(f"- Standard JSON: {simple['json']['mean'] * 1000:.1f}ms")
    print(f"- Overhead: {simple['ratio']:.1f}x")

    print("\n**Complex Data (UUIDs, datetimes):**")
    complex_data = results["complex"]
    print(f"- datason: {complex_data['datason']['mean'] * 1000:.1f}ms")
    print(f"- Pickle: {complex_data['pickle']['mean'] * 1000:.1f}ms")
    print(f"- vs Pickle: {complex_data['pickle_ratio']:.1f}x")

    print("\n**Throughput:**")
    large = results["large"]
    print(f"- Large datasets: {large['throughput']:.0f} items/second")

    if "numpy" in results:
        numpy_data = results["numpy"]
        print(f"- NumPy arrays: {numpy_data['elements_per_sec']:.0f} elements/second")

    if "pandas" in results:
        pandas_data = results["pandas"]
        print(f"- Pandas DataFrames: {pandas_data['rows_per_sec']:.0f} rows/second")

    print("\n**Round-trip Performance:**")
    print(f"- Serialize + JSON + Deserialize: {round_trip_results['round_trip']['mean'] * 1000:.1f}ms")
    print(f"- Serialize only: {round_trip_results['serialize']['mean'] * 1000:.1f}ms")
    print(f"- Deserialize only: {round_trip_results['deserialize']['mean'] * 1000:.1f}ms")


if __name__ == "__main__":
    print("🚀 datason Real Performance Benchmarks")
    print(f"Python {'.'.join(map(str, __import__('sys').version_info[:3]))}")
    print(f"NumPy: {'✅' if HAS_NUMPY else '❌'}")
    print(f"Pandas: {'✅' if HAS_PANDAS else '❌'}")

    # Run benchmarks
    results = run_comparison_benchmarks()
    round_trip_results = test_round_trip_performance()

    # Generate summary
    generate_performance_summary(results, round_trip_results)

    print("\n✅ Benchmarks complete! Use these real numbers in documentation.")
