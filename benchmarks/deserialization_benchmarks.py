#!/usr/bin/env python3
"""
Comprehensive Deserialization Performance Benchmarks

This script benchmarks different deserialization approaches to establish
baselines before optimizing the hot path and to track performance over time.

Usage:
    python benchmarks/deserialization_benchmarks.py
    python benchmarks/deserialization_benchmarks.py --quick  # For faster testing
"""

import argparse
import json
import time
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict
from functools import partial

# Test imports - graceful fallbacks
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

# Import datason
import datason
from datason.config import SerializationConfig
from datason.deserializers import auto_deserialize, deserialize, deserialize_fast


class DeserializationBenchmark:
    """Comprehensive deserialization benchmark suite."""

    def __init__(self, iterations: int = 1000, warmup: int = 100):
        self.iterations = iterations
        self.warmup = warmup
        self.results = {}

    def create_test_data(self) -> Dict[str, Any]:
        """Create test datasets for benchmarking."""
        datasets = {}

        # 1. Basic types - simple JSON data
        datasets["basic_types"] = {
            "strings": [f"item_{i}" for i in range(100)],
            "numbers": list(range(100)),
            "floats": [i * 0.1 for i in range(100)],
            "booleans": [i % 2 == 0 for i in range(100)],
            "nested": {"level1": {"level2": {"data": [1, 2, 3, 4, 5] * 10}}},
        }

        # 2. Date/UUID heavy data (string processing intensive)
        datasets["datetime_uuid_heavy"] = {
            "timestamps": ["2023-01-01T10:00:00", "2023-01-01T11:00:00", "2023-01-01T12:00:00"]
            * 50,  # 150 datetime strings
            "ids": ["12345678-1234-5678-9012-123456789abc", "87654321-4321-8765-2109-cba987654321"]
            * 50,  # 100 UUID strings
            "mixed": [
                {
                    "id": "12345678-1234-5678-9012-123456789abc",
                    "created": "2023-01-01T10:00:00",
                    "name": f"item_{i}",
                    "value": i * 1.5,
                }
                for i in range(25)
            ],
        }

        # 3. Complex types (with type metadata)
        complex_data = {
            "decimals": [Decimal("123.456"), Decimal("789.012")],
            "paths": [Path("/tmp/test1.txt"), Path("/tmp/test2.txt")],  # nosec B108
            "complex_nums": [complex(1, 2), complex(3, 4)],
            "sets": [{1, 2, 3}, {4, 5, 6}],
            "tuples": [(1, 2, 3), (4, 5, 6)],
        }

        config_with_hints = SerializationConfig(include_type_hints=True)
        datasets["complex_types_serialized"] = datason.serialize(complex_data, config_with_hints)
        datasets["complex_types_no_hints"] = datason.serialize(complex_data)

        # 4. Large nested structure
        datasets["large_nested"] = {
            "users": [
                {
                    "id": i,
                    "profile": {
                        "name": f"User {i}",
                        "email": f"user{i}@example.com",
                        "preferences": {
                            "theme": "dark" if i % 2 else "light",
                            "notifications": [f"type_{j}" for j in range(3)],
                        },
                    },
                    "data": {"metrics": [i * j for j in range(10)], "timestamps": ["2023-01-01T10:00:00"] * 5},
                }
                for i in range(50)
            ]
        }

        # 5. ML-like data (if available)
        if HAS_NUMPY and HAS_PANDAS:
            ml_data = {
                "features": np.random.rand(100, 10).tolist(),
                "labels": [i % 2 for i in range(100)],
                "metadata": {
                    "created": datetime.now(),
                    "model_id": uuid.uuid4(),
                    "config": {"learning_rate": 0.01, "epochs": 100},
                },
            }
            datasets["ml_data_serialized"] = datason.serialize(ml_data)

        return datasets

    def benchmark_function(self, func, data, name: str) -> Dict[str, float]:
        """Benchmark a function with warmup and multiple iterations."""
        # Warmup
        for _ in range(self.warmup):
            func(data)

        # Actual benchmark
        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            func(data)
            end = time.perf_counter()
            times.append(end - start)

        return {
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "std": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
        }

    def benchmark_deserialize_functions(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark different deserialization functions."""
        results = {}

        functions = [
            ("deserialize_old", deserialize),
            ("deserialize_fast", deserialize_fast),
            ("auto_deserialize_conservative", lambda x: auto_deserialize(x, aggressive=False)),
            ("auto_deserialize_aggressive", lambda x: auto_deserialize(x, aggressive=True)),
        ]

        for dataset_name, data in datasets.items():
            print(f"\n📊 Benchmarking dataset: {dataset_name}")
            results[dataset_name] = {}

            for func_name, func in functions:
                try:
                    # Special handling for functions that need config
                    if func_name == "deserialize_fast" and "types_serialized" in dataset_name:
                        # For type metadata tests, use config with explicit capture
                        config = SerializationConfig(include_type_hints=True)
                        current_func = func  # Capture function reference
                        current_config = config  # Capture config reference
                        
                        def test_func(x, f=current_func, c=current_config):
                            return f(x, c)
                    else:
                        test_func = func

                    benchmark_results = self.benchmark_function(test_func, data, func_name)
                    results[dataset_name][func_name] = benchmark_results

                    mean_ms = benchmark_results["mean"] * 1000
                    std_ms = benchmark_results["std"] * 1000
                    print(f"  {func_name:30s}: {mean_ms:6.2f}ms ± {std_ms:5.2f}ms")

                except Exception as e:
                    print(f"  {func_name:30s}: ERROR - {e}")
                    results[dataset_name][func_name] = {"error": str(e)}

        return results

    def benchmark_hot_path_effectiveness(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark hot path vs full path effectiveness."""
        print("\n🔥 Hot Path Effectiveness Analysis")
        print("=" * 50)

        results = {}

        # Test with basic types (should hit hot path)
        basic_data = datasets["basic_types"]
        print("\n📈 Basic types (should use hot path):")

        old_results = self.benchmark_function(deserialize, basic_data, "old_deserialize")
        fast_results = self.benchmark_function(deserialize_fast, basic_data, "fast_deserialize")

        speedup = old_results["mean"] / fast_results["mean"]
        print(f"  Old deserialize:  {old_results['mean'] * 1000:.2f}ms")
        print(f"  Fast deserialize: {fast_results['mean'] * 1000:.2f}ms")
        print(f"  Speedup:          {speedup:.2f}x")

        results["basic_types"] = {"old": old_results, "fast": fast_results, "speedup": speedup}

        # Test with complex types (should use full path)
        if "complex_types_serialized" in datasets:
            complex_data = datasets["complex_types_serialized"]
            print("\n📈 Complex types (should use full path):")

            config = SerializationConfig(include_type_hints=True)
            old_results = self.benchmark_function(deserialize, complex_data, "old_deserialize")
            fast_results = self.benchmark_function(
                lambda x: deserialize_fast(x, config), complex_data, "fast_deserialize"
            )

            speedup = old_results["mean"] / fast_results["mean"]
            print(f"  Old deserialize:  {old_results['mean'] * 1000:.2f}ms")
            print(f"  Fast deserialize: {fast_results['mean'] * 1000:.2f}ms")
            print(f"  Speedup:          {speedup:.2f}x")

            results["complex_types"] = {"old": old_results, "fast": fast_results, "speedup": speedup}

        return results

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print("🚀 Starting Comprehensive Deserialization Benchmarks")
        print("=" * 60)
        print(f"Iterations: {self.iterations}, Warmup: {self.warmup}")

        # Create test datasets
        print("\n📝 Creating test datasets...")
        datasets = self.create_test_data()
        print(f"Created {len(datasets)} datasets")

        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "iterations": self.iterations,
                "warmup": self.warmup,
                "datason_version": getattr(datason, "__version__", "unknown"),
                "datasets": list(datasets.keys()),
            },
            "function_comparison": self.benchmark_deserialize_functions(datasets),
            "hot_path_analysis": self.benchmark_hot_path_effectiveness(datasets),
        }

        return results

    def generate_report(self, results: Dict[str, Any]) -> None:
        """Generate a human-readable report."""
        print("\n" + "=" * 60)
        print("📊 DESERIALIZATION BENCHMARK REPORT")
        print("=" * 60)

        # Overall summary
        print("\n📈 Benchmark Summary:")
        print(f"   Timestamp: {results['metadata']['timestamp']}")
        print(f"   Iterations: {results['metadata']['iterations']}")
        print(f"   Datasets tested: {len(results['metadata']['datasets'])}")

        # Function comparison summary
        print("\n🏁 Performance Comparison (deserialize_fast vs deserialize_old):")

        function_results = results["function_comparison"]
        speedups = []

        for dataset_name, dataset_results in function_results.items():
            if "deserialize_old" in dataset_results and "deserialize_fast" in dataset_results:
                old_time = dataset_results["deserialize_old"]["mean"]
                fast_time = dataset_results["deserialize_fast"]["mean"]
                speedup = old_time / fast_time
                speedups.append(speedup)

                print(f"   {dataset_name:25s}: {speedup:5.2f}x speedup")

        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            print(f"\n   Average speedup: {avg_speedup:.2f}x")
            print(f"   Best speedup:    {max(speedups):.2f}x")
            print(f"   Worst speedup:   {min(speedups):.2f}x")

        # Hot path analysis
        if "hot_path_analysis" in results:
            print("\n🔥 Hot Path Analysis:")
            hot_path_results = results["hot_path_analysis"]

            for test_type, test_results in hot_path_results.items():
                speedup = test_results.get("speedup", 0)
                print(f"   {test_type:20s}: {speedup:.2f}x speedup")

        # Recommendations
        print("\n💡 Performance Recommendations:")

        if speedups:
            if avg_speedup > 1.5:
                print("   ✅ deserialize_fast shows significant improvement!")
                print("   ✅ Consider making it the default deserialization function")
            elif avg_speedup > 1.1:
                print("   ⚠️  deserialize_fast shows modest improvement")
                print("   🔧 Consider optimizing the hot path further")
            else:
                print("   ❌ deserialize_fast not consistently faster")
                print("   🔧 Hot path optimization needed")

        print("\n🎯 Next Steps:")
        print("   1. Optimize hot path for commonly accessed data types")
        print("   2. Reduce function call overhead in tight loops")
        print("   3. Add more type-specific fast paths")
        print("   4. Consider caching for repeated type detection")


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Run deserialization benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (fewer iterations)")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup iterations")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")

    args = parser.parse_args()

    # Adjust iterations for quick mode
    if args.quick:
        iterations = 100
        warmup = 10
    else:
        iterations = args.iterations
        warmup = args.warmup

    # Run benchmark
    benchmark = DeserializationBenchmark(iterations=iterations, warmup=warmup)
    results = benchmark.run_comprehensive_benchmark()

    # Generate report
    benchmark.generate_report(results)

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {args.output}")

    # Also save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmarks/results/deserialization_benchmark_{timestamp}.json"
    Path(filename).parent.mkdir(exist_ok=True)

    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 Results saved to: {filename}")


if __name__ == "__main__":
    main()
