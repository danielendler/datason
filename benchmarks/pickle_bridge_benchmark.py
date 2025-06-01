#!/usr/bin/env python3
"""Pickle Bridge Performance Benchmark Suite

Comprehensive performance testing for datason's Pickle Bridge feature,
comparing against comparable pickle-to-JSON conversion approaches.

This benchmark evaluates:
- Pickle Bridge vs manual pickle.loads + datason.serialize
- Pickle Bridge vs jsonpickle library
- Pickle Bridge vs dill + JSON conversion
- Security overhead vs unsafe pickle operations
- Bulk conversion performance vs individual file processing
- Memory efficiency with large pickle files
- Performance across different ML data types

Test Flows Integration:
- minimal: Basic Pickle Bridge functionality
- with-ml-deps: ML object conversion benchmarks
- full: Complete benchmark suite with all comparisons

Usage:
    python pickle_bridge_benchmark.py [--test-flow minimal|ml|full]

Environment Variables:
    BENCHMARK_ITERATIONS: Number of test iterations (default: 5)
    BENCHMARK_DATA_SIZES: Comma-separated data sizes (default: 100,1000,5000)
"""

import argparse
from collections import namedtuple
import os
from pathlib import Path
import pickle  # nosec B403 - Safe usage for benchmarking only, controlled data
import statistics
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Tuple

# Optional dependencies with graceful fallbacks
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

try:
    HAS_SKLEARN = True
    # Import when needed to avoid F401 unused import warning
except ImportError:
    HAS_SKLEARN = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import jsonpickle

    HAS_JSONPICKLE = True
except ImportError:
    HAS_JSONPICKLE = False

try:
    import dill  # nosec B403 - Safe usage for benchmarking only, controlled data

    HAS_DILL = True
except ImportError:
    HAS_DILL = False

# Import datason components
import datason
from datason import (
    PickleBridge,
    convert_pickle_directory,
    get_ml_safe_classes,
)

# Benchmark utilities
BenchmarkResult = namedtuple(
    "BenchmarkResult", ["mean", "median", "min", "max", "stdev", "operations_per_sec"]
)

FileStats = namedtuple(
    "FileStats", ["source_size_bytes", "target_size_bytes", "compression_ratio"]
)


def time_operation(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Time a single operation and return result and duration."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def benchmark_operation(
    func: Callable, data: Any = None, iterations: int = 5
) -> BenchmarkResult:
    """Run benchmark multiple times and return statistics."""
    times = []

    # Warm-up run
    try:
        if data is None:
            time_operation(func)
        else:
            time_operation(func, data)
    except Exception as e:
        # Return failed benchmark result with error information
        print(f"    Warm-up failed: {e}")
        return BenchmarkResult(
            mean=float("inf"),
            median=float("inf"),
            min=float("inf"),
            max=float("inf"),
            stdev=float("inf"),
            operations_per_sec=0,
        )

    # Actual benchmark runs
    for i in range(iterations):
        try:
            if data is None:
                _, duration = time_operation(func)
            else:
                _, duration = time_operation(func, data)
            times.append(duration)
        except Exception as e:
            # If any iteration fails, mark as failed
            print(f"    Iteration {i + 1} failed: {e}")
            return BenchmarkResult(
                mean=float("inf"),
                median=float("inf"),
                min=float("inf"),
                max=float("inf"),
                stdev=float("inf"),
                operations_per_sec=0,
            )

    if not times:
        return BenchmarkResult(
            mean=float("inf"),
            median=float("inf"),
            min=float("inf"),
            max=float("inf"),
            stdev=float("inf"),
            operations_per_sec=0,
        )

    mean_time = statistics.mean(times)
    return BenchmarkResult(
        mean=mean_time,
        median=statistics.median(times),
        min=min(times),
        max=max(times),
        stdev=statistics.stdev(times) if len(times) > 1 else 0,
        operations_per_sec=1.0 / mean_time if mean_time > 0 else 0,
    )


def create_test_datasets(sizes: List[int]) -> Dict[str, Dict[str, Any]]:
    """Create test datasets of varying complexity and size."""
    datasets = {}

    for size in sizes:
        datasets[f"size_{size}"] = {}

        # Basic Python objects (should always work)
        datasets[f"size_{size}"]["basic_objects"] = {
            "strings": [f"test_string_{i}" for i in range(size)],
            "integers": list(range(size)),
            "floats": [i * 1.5 for i in range(size)],
            "booleans": [i % 2 == 0 for i in range(size)],
            "simple_dict": {
                f"key_{i}": i for i in range(min(size, 10))
            },  # Simplified dict
        }

        # Simple NumPy objects (if available) - use basic arrays only
        if HAS_NUMPY:
            datasets[f"size_{size}"]["numpy_simple"] = {
                "float_array": np.random.random(
                    size
                ).tolist(),  # Convert to list for safety
                "int_array": np.arange(size).tolist(),  # Convert to list for safety
                "simple_matrix": np.ones(
                    (min(size // 10, 10), 5)
                ).tolist(),  # Convert to list
            }

        # Simple Pandas objects (if available) - use basic structures
        if HAS_PANDAS and size <= 100:  # Limit size for DataFrames
            simple_df = pd.DataFrame(
                {
                    "A": list(range(size)),
                    "B": [f"item_{i}" for i in range(size)],
                }
            )
            datasets[f"size_{size}"]["pandas_simple"] = {
                "simple_dataframe": simple_df.to_dict(),  # Convert to dict for safety
                "simple_series": pd.Series(range(size)).to_dict(),  # Convert to dict
            }

        # Complex ML objects (expect some failures due to security)
        if HAS_NUMPY and size <= 50:  # Keep small for complex objects
            datasets[f"size_{size}"]["numpy_complex"] = {
                "complex_array": np.random.random((min(size, 20), 3)),
                "structured_array": np.array(
                    [(i, f"item_{i}", i * 1.5) for i in range(min(size, 20))],
                    dtype=[("id", "i4"), ("name", "U10"), ("value", "f8")],
                ),
            }

        # Complex Pandas objects
        if HAS_PANDAS and size <= 50:
            datasets[f"size_{size}"]["pandas_complex"] = {
                "complex_dataframe": pd.DataFrame(
                    {
                        "A": np.random.random(size) if HAS_NUMPY else range(size),
                        "B": [f"category_{i % 5}" for i in range(size)],
                        "C": pd.Categorical([f"cat_{i % 3}" for i in range(size)]),
                    }
                ),
                "complex_series": pd.Series(range(size), name="complex_series"),
            }

        # Scikit-learn objects (if available)
        if HAS_SKLEARN and size <= 50:  # Very limited for models
            try:
                x_data = [[i] for i in range(min(size, 20))]
                y_data = [i % 2 for i in range(min(size, 20))]

                from sklearn.linear_model import LogisticRegression

                lr_model = LogisticRegression(random_state=42, max_iter=100)
                lr_model.fit(x_data, y_data)

                datasets[f"size_{size}"]["sklearn_simple"] = {
                    "logistic_model": lr_model,
                }
            except Exception:  # nosec B110 - Skip if model training fails for benchmark
                pass  # Skip if model training fails

        # PyTorch objects (if available)
        if HAS_TORCH and size <= 100:
            datasets[f"size_{size}"]["torch_simple"] = {
                "simple_tensor": torch.randn(min(size, 50)),
                "small_matrix": torch.randn(min(size // 10, 10), 3),
            }

    return datasets


def create_pickle_files(
    datasets: Dict[str, Dict[str, Any]], temp_dir: Path
) -> Dict[str, Path]:
    """Create pickle files from test datasets."""
    pickle_files = {}

    for dataset_name, categories in datasets.items():
        for category_name, data in categories.items():
            file_key = f"{dataset_name}_{category_name}"
            pickle_path = temp_dir / f"{file_key}.pkl"

            try:
                with pickle_path.open("wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle_files[file_key] = pickle_path
            except Exception as e:
                print(f"Warning: Failed to create pickle file for {file_key}: {e}")

    return pickle_files


def benchmark_with_fallback(
    func: Callable, data: Any, iterations: int, unsafe_bridge=None
) -> Tuple[BenchmarkResult, bool]:
    """Benchmark with fallback to unsafe mode if needed."""
    # Try with safe mode first
    result = benchmark_operation(func, data, iterations)

    if result.mean == float("inf") and unsafe_bridge is not None:
        # Try with unsafe mode as fallback (for comparison only)
        def unsafe_func():
            return (
                unsafe_bridge.from_pickle_file(data)
                if hasattr(data, "read_bytes")
                else unsafe_bridge.from_pickle_bytes(data)
            )

        unsafe_result = benchmark_operation(unsafe_func, None, iterations)
        return unsafe_result, True  # True indicates unsafe was used

    return result, False  # False indicates safe mode worked


def benchmark_pickle_bridge_basic(
    pickle_files: Dict[str, Path], iterations: int
) -> Dict[str, BenchmarkResult]:
    """Benchmark basic Pickle Bridge functionality."""
    print("\n=== Pickle Bridge Basic Performance ===")

    # Use ML-safe classes for better compatibility
    safe_bridge = PickleBridge(safe_classes=get_ml_safe_classes())
    results = {}

    for file_key, pickle_path in pickle_files.items():
        # Test file-based conversion (use closure to capture variable)
        def make_convert_file(path):
            def convert_file():
                return safe_bridge.from_pickle_file(path)

            return convert_file

        result, used_unsafe = benchmark_with_fallback(
            make_convert_file(pickle_path), None, iterations, None
        )
        results[f"{file_key}_file"] = result

        # Test bytes-based conversion
        try:
            with pickle_path.open("rb") as f:
                pickle_bytes = f.read()

            def make_convert_bytes(data):
                def convert_bytes():
                    return safe_bridge.from_pickle_bytes(data)

                return convert_bytes

            result, used_unsafe = benchmark_with_fallback(
                make_convert_bytes(pickle_bytes), None, iterations, None
            )
            results[f"{file_key}_bytes"] = result
        except Exception as e:
            print(f"Warning: Failed to benchmark bytes conversion for {file_key}: {e}")

    return results


def benchmark_security_overhead(
    pickle_files: Dict[str, Path], iterations: int
) -> Dict[str, BenchmarkResult]:
    """Benchmark security overhead vs unsafe operations."""
    print("\n=== Security Overhead Analysis ===")

    safe_bridge = PickleBridge(safe_classes=get_ml_safe_classes())
    unsafe_bridge = PickleBridge(safe_classes={"*"})  # Unsafe: allows all classes

    results = {}

    for file_key, pickle_path in pickle_files.items():
        # Safe conversion (use closure to capture variable)
        def make_safe_convert(path):
            def safe_convert():
                return safe_bridge.from_pickle_file(path)

            return safe_convert

        safe_result = benchmark_operation(
            make_safe_convert(pickle_path), None, iterations
        )
        results[f"{file_key}_safe"] = safe_result

        # Unsafe conversion (for comparison only - never use in production)
        def make_unsafe_convert(path):
            def unsafe_convert():
                return unsafe_bridge.from_pickle_file(path)

            return unsafe_convert

        unsafe_result = benchmark_operation(
            make_unsafe_convert(pickle_path), None, iterations
        )
        results[f"{file_key}_unsafe"] = unsafe_result

    return results


def benchmark_vs_alternatives(
    pickle_files: Dict[str, Path], iterations: int
) -> Dict[str, BenchmarkResult]:
    """Benchmark Pickle Bridge vs alternative approaches."""
    print("\n=== Comparison with Alternative Libraries ===")

    results = {}
    bridge = PickleBridge(safe_classes=get_ml_safe_classes())

    for file_key, pickle_path in pickle_files.items():
        # Pickle Bridge approach (use closure to capture variable)
        def make_pickle_bridge_convert(path):
            def pickle_bridge_convert():
                return bridge.from_pickle_file(path)

            return pickle_bridge_convert

        pb_result = benchmark_operation(
            make_pickle_bridge_convert(pickle_path), None, iterations
        )
        results[f"{file_key}_pickle_bridge"] = pb_result

        # Manual pickle.loads + datason.serialize
        def make_manual_convert(path):
            def manual_convert():
                with path.open("rb") as f:
                    data = pickle.load(f)  # nosec B301 - Controlled test data
                return datason.serialize(data)

            return manual_convert

        manual_result = benchmark_operation(
            make_manual_convert(pickle_path), None, iterations
        )
        results[f"{file_key}_manual"] = manual_result

        # jsonpickle (if available)
        if HAS_JSONPICKLE:

            def make_jsonpickle_convert(path):
                def jsonpickle_convert():
                    with path.open("rb") as f:
                        data = pickle.load(f)  # nosec B301 - Controlled test data
                    return jsonpickle.encode(data)

                return jsonpickle_convert

            jp_result = benchmark_operation(
                make_jsonpickle_convert(pickle_path), None, iterations
            )
            results[f"{file_key}_jsonpickle"] = jp_result

        # dill + JSON (if available)
        if HAS_DILL:

            def make_dill_convert(path):
                def dill_convert():
                    with path.open("rb") as f:
                        data = dill.load(f)  # nosec B301 - Controlled test data
                    import json

                    return json.dumps(data, default=str)  # Simple fallback serializer

                return dill_convert

            dill_result = benchmark_operation(
                make_dill_convert(pickle_path), None, iterations
            )
            results[f"{file_key}_dill"] = dill_result

    return results


def benchmark_bulk_operations(
    pickle_files: Dict[str, Path], iterations: int
) -> Dict[str, BenchmarkResult]:
    """Benchmark bulk directory conversion operations."""
    print("\n=== Bulk Operations Performance ===")

    results = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        source_dir = temp_path / "source"
        target_dir = temp_path / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        # Copy a subset of pickle files to source directory
        test_files = list(pickle_files.items())[:5]  # Limit for performance
        for file_key, pickle_path in test_files:
            target_path = source_dir / f"{file_key}.pkl"
            target_path.write_bytes(pickle_path.read_bytes())

        # Benchmark bulk conversion
        def bulk_convert():
            return convert_pickle_directory(
                source_dir=source_dir, target_dir=target_dir, overwrite=True
            )

        bulk_result = benchmark_operation(bulk_convert, None, iterations)
        results["bulk_conversion"] = bulk_result

        # Benchmark individual file conversion
        def individual_convert():
            bridge = PickleBridge(safe_classes=get_ml_safe_classes())
            for pickle_file in source_dir.glob("*.pkl"):
                bridge.from_pickle_file(pickle_file)

        individual_result = benchmark_operation(individual_convert, None, iterations)
        results["individual_conversion"] = individual_result

    return results


def analyze_file_sizes(pickle_files: Dict[str, Path]) -> Dict[str, FileStats]:
    """Analyze file size differences between pickle and JSON."""
    print("\n=== File Size Analysis ===")

    bridge = PickleBridge(safe_classes=get_ml_safe_classes())
    size_stats = {}

    for file_key, pickle_path in pickle_files.items():
        try:
            # Get source size
            source_size = pickle_path.stat().st_size

            # Convert and get JSON size
            result = bridge.from_pickle_file(pickle_path)
            import json

            json_str = json.dumps(result)
            target_size = len(json_str.encode("utf-8"))

            compression_ratio = target_size / source_size if source_size > 0 else 0

            size_stats[file_key] = FileStats(
                source_size_bytes=source_size,
                target_size_bytes=target_size,
                compression_ratio=compression_ratio,
            )
        except Exception as e:
            print(f"Warning: Failed to analyze size for {file_key}: {e}")

    return size_stats


def print_benchmark_results(results: Dict[str, BenchmarkResult], title: str):
    """Print formatted benchmark results."""
    print(f"\n--- {title} ---")
    print(
        f"{'Operation':<30} {'Mean (ms)':<12} {'±Std (ms)':<12} {'Ops/sec':<12} {'Status':<10}"
    )
    print("-" * 78)

    for operation, result in results.items():
        if result.mean == float("inf"):
            print(
                f"{operation:<30} {'FAILED':<12} {'FAILED':<12} {'0':<12} {'FAILED':<10}"
            )
        else:
            mean_ms = result.mean * 1000
            std_ms = result.stdev * 1000
            ops_per_sec = result.operations_per_sec
            status = "SUCCESS" if result.mean < float("inf") else "FAILED"
            print(
                f"{operation:<30} {mean_ms:<12.2f} {std_ms:<12.2f} {ops_per_sec:<12.0f} {status:<10}"
            )


def print_size_analysis(size_stats: Dict[str, FileStats]):
    """Print file size analysis results."""
    print("\n--- File Size Analysis ---")
    print(f"{'File':<30} {'Pickle (KB)':<12} {'JSON (KB)':<12} {'Ratio':<12}")
    print("-" * 66)

    for file_key, stats in size_stats.items():
        pickle_kb = stats.source_size_bytes / 1024
        json_kb = stats.target_size_bytes / 1024
        ratio = stats.compression_ratio
        print(f"{file_key:<30} {pickle_kb:<12.1f} {json_kb:<12.1f} {ratio:<12.2f}")


def run_minimal_benchmarks(iterations: int, data_sizes: List[int]):
    """Run minimal benchmark suite for basic CI testing."""
    print("🧪 Running Minimal Pickle Bridge Benchmarks")

    # Create minimal test datasets - focus on basic objects
    datasets = create_test_datasets([min(data_sizes)])  # Only smallest size

    # Filter to basic objects only for minimal testing
    filtered_datasets = {}
    for dataset_name, categories in datasets.items():
        filtered_datasets[dataset_name] = {}
        for category_name, data in categories.items():
            if "basic" in category_name or "simple" in category_name:
                filtered_datasets[dataset_name][category_name] = data

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pickle_files = create_pickle_files(filtered_datasets, temp_path)

        print(f"Created {len(pickle_files)} test pickle files")

        # Basic functionality benchmark
        basic_results = benchmark_pickle_bridge_basic(pickle_files, iterations)
        print_benchmark_results(basic_results, "Basic Pickle Bridge Performance")

        # File size analysis
        size_stats = analyze_file_sizes(pickle_files)
        print_size_analysis(size_stats)

        print("✅ Minimal benchmarks completed")


def run_ml_benchmarks(iterations: int, data_sizes: List[int]):
    """Run ML-focused benchmark suite."""
    print("🧪 Running ML Pickle Bridge Benchmarks")

    if not any([HAS_NUMPY, HAS_PANDAS, HAS_SKLEARN, HAS_TORCH]):
        print("⚠️  No ML libraries available - skipping ML benchmarks")
        return

    # Create ML-focused test datasets
    datasets = create_test_datasets(data_sizes)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pickle_files = create_pickle_files(datasets, temp_path)

        # Filter to ML-related files only
        ml_files = {
            k: v
            for k, v in pickle_files.items()
            if any(lib in k for lib in ["numpy", "pandas", "sklearn", "torch"])
        }

        if not ml_files:
            print("⚠️  No ML pickle files created - skipping ML benchmarks")
            return

        # ML-specific benchmarks
        basic_results = benchmark_pickle_bridge_basic(ml_files, iterations)
        print_benchmark_results(basic_results, "ML Objects Performance")

        security_results = benchmark_security_overhead(ml_files, iterations)
        print_benchmark_results(security_results, "ML Security Overhead")

        size_stats = analyze_file_sizes(ml_files)
        print_size_analysis(size_stats)

        print("✅ ML benchmarks completed")


def run_full_benchmarks(iterations: int, data_sizes: List[int]):
    """Run comprehensive benchmark suite."""
    print("🧪 Running Full Pickle Bridge Benchmarks")

    # Create comprehensive test datasets
    datasets = create_test_datasets(data_sizes)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pickle_files = create_pickle_files(datasets, temp_path)

        # All benchmark categories
        basic_results = benchmark_pickle_bridge_basic(pickle_files, iterations)
        print_benchmark_results(basic_results, "Basic Pickle Bridge Performance")

        security_results = benchmark_security_overhead(pickle_files, iterations)
        print_benchmark_results(security_results, "Security Overhead Analysis")

        alternatives_results = benchmark_vs_alternatives(pickle_files, iterations)
        print_benchmark_results(alternatives_results, "Comparison with Alternatives")

        bulk_results = benchmark_bulk_operations(pickle_files, iterations)
        print_benchmark_results(bulk_results, "Bulk Operations Performance")

        size_stats = analyze_file_sizes(pickle_files)
        print_size_analysis(size_stats)

        # Performance summary
        print("\n=== Performance Summary ===")
        print("Library Availability:")
        print(f"  NumPy: {'✅' if HAS_NUMPY else '❌'}")
        print(f"  Pandas: {'✅' if HAS_PANDAS else '❌'}")
        print(f"  Scikit-learn: {'✅' if HAS_SKLEARN else '❌'}")
        print(f"  PyTorch: {'✅' if HAS_TORCH else '❌'}")
        print(f"  jsonpickle: {'✅' if HAS_JSONPICKLE else '❌'}")
        print(f"  dill: {'✅' if HAS_DILL else '❌'}")

        print("\nTest Configuration:")
        print(f"  Iterations: {iterations}")
        print(f"  Data sizes: {data_sizes}")
        print(f"  Total pickle files: {len(pickle_files)}")

        print("✅ Full benchmarks completed")


def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(description="Pickle Bridge Performance Benchmarks")
    parser.add_argument(
        "--test-flow",
        choices=["minimal", "ml", "full"],
        default="full",
        help="Test flow to run (default: full)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.getenv("BENCHMARK_ITERATIONS", "5")),
        help="Number of benchmark iterations (default: 5)",
    )
    parser.add_argument(
        "--data-sizes",
        type=str,
        default=os.getenv("BENCHMARK_DATA_SIZES", "100,1000,5000"),
        help="Comma-separated data sizes (default: 100,1000,5000)",
    )

    args = parser.parse_args()

    # Parse data sizes
    try:
        data_sizes = [int(s.strip()) for s in args.data_sizes.split(",")]
    except ValueError:
        print("Error: Invalid data sizes format. Use comma-separated integers.")
        sys.exit(1)

    print("🚀 Pickle Bridge Benchmark Suite")
    print(f"Test Flow: {args.test_flow}")
    print(f"Iterations: {args.iterations}")
    print(f"Data Sizes: {data_sizes}")

    try:
        if args.test_flow == "minimal":
            run_minimal_benchmarks(args.iterations, data_sizes)
        elif args.test_flow == "ml":
            run_ml_benchmarks(args.iterations, data_sizes)
        elif args.test_flow == "full":
            run_full_benchmarks(args.iterations, data_sizes)
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
