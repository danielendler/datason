#!/usr/bin/env python3
"""
CI Performance Tracker for datason
==================================

This script provides a stable, CI-friendly performance testing suite designed for:
1. Consistent, reproducible performance measurements
2. Historical data tracking
3. Regression detection
4. Progress measurement over incremental improvements

The benchmarks are designed to be stable (minimal variance) and representative
of real-world usage patterns.
"""

import json
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from statistics import mean, stdev
from typing import Any, Dict, Optional

import datason


class PerformanceTracker:
    """Tracks performance metrics with historical context."""

    def __init__(self, results_dir: str = "benchmarks/results"):
        self.results_dir = results_dir
        self.metadata = self._gather_metadata()
        os.makedirs(results_dir, exist_ok=True)

    def _gather_metadata(self) -> Dict[str, Any]:
        """Gather consistent environment metadata."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "datason_version": getattr(datason, "__version__", "unknown"),
            "git_commit": os.environ.get("GITHUB_SHA", "unknown"),
            "git_ref": os.environ.get("GITHUB_REF", "unknown"),
            "ci_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "runner": os.environ.get("RUNNER_OS", "local"),
        }

    def benchmark_function(self, func, iterations: int = 10, warmup: int = 2) -> Dict[str, float]:
        """Run a function multiple times and return statistics."""
        # Warmup runs to stabilize performance
        for _ in range(warmup):
            func()

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append(end - start)

        return {
            "mean": mean(times),
            "min": min(times),
            "max": max(times),
            "std": stdev(times) if len(times) > 1 else 0.0,
            "median": sorted(times)[len(times) // 2],
            "iterations": iterations,
        }


class StableBenchmarkSuite:
    """Stable benchmark suite designed for CI tracking."""

    def __init__(self):
        self.tracker = PerformanceTracker()

    def create_stable_test_data(self) -> Dict[str, Any]:
        """Create deterministic test data for consistent benchmarking."""
        # Use fixed seed/deterministic data to ensure consistency across runs
        return {
            "api_response": {
                "status": "success",
                "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                "request_id": "12345678-1234-5678-9012-123456789012",  # Fixed UUID
                "data": [
                    {
                        "id": i,
                        "name": f"Item {i:03d}",
                        "price": Decimal(f"{19.99 + i * 0.50:.2f}"),
                        "created_at": datetime(2024, 1, 1, 12, min(i, 59), 0, tzinfo=timezone.utc),
                        "tags": [f"tag{j}" for j in range(i % 3 + 1)],
                        "active": i % 2 == 0,
                    }
                    for i in range(100)  # Fixed size for consistency
                ],
            },
            "config_data": {
                "app_name": "TestApp",
                "version": "1.0.0",
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "timeout": 30.0,
                },
                "created": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                "feature_flags": [f"feature_{i}" for i in range(10)],
            },
            "simple_types": {
                "string": "hello world",
                "integer": 42,
                "float": 3.14159,
                "boolean": True,
                "null": None,
                "list": [1, 2, 3, 4, 5],
                "nested": {"a": {"b": {"c": "deep"}}},
            },
        }

    def benchmark_serialization(self) -> Dict[str, Any]:
        """Benchmark serialization performance."""
        test_data = self.create_stable_test_data()
        results = {}

        for name, data in test_data.items():
            print(f"📊 Benchmarking serialization: {name}")

            # Standard datason
            def serialize_standard(data=data):
                return datason.serialize(data)

            # Performance config
            perf_config = datason.get_performance_config()

            def serialize_perf(data=data, config=perf_config):
                return datason.serialize(data, config=config)

            results[name] = {
                "standard": self.tracker.benchmark_function(serialize_standard),
                "performance_config": self.tracker.benchmark_function(serialize_perf),
            }

            # Calculate improvement
            std_time = results[name]["standard"]["mean"]
            perf_time = results[name]["performance_config"]["mean"]
            improvement = ((std_time - perf_time) / std_time) * 100
            results[name]["performance_improvement_pct"] = improvement

            print(f"  Standard: {std_time * 1000:.2f}ms")
            print(f"  Performance: {perf_time * 1000:.2f}ms")
            print(f"  Improvement: {improvement:.1f}%")

        return results

    def benchmark_deserialization(self) -> Dict[str, Any]:
        """Benchmark deserialization performance."""
        test_data = self.create_stable_test_data()
        results = {}

        for name, data in test_data.items():
            print(f"📊 Benchmarking deserialization: {name}")

            # Serialize data first
            serialized = datason.serialize(data)

            # Standard deserialization
            def deserialize_standard(serialized=serialized):
                return datason.deserialize(serialized)

            results[name] = {
                "standard": self.tracker.benchmark_function(deserialize_standard),
            }

            print(f"  Standard: {results[name]['standard']['mean'] * 1000:.2f}ms")

        return results

    def benchmark_type_detection_overhead(self) -> Dict[str, Any]:
        """Benchmark specific type detection patterns."""
        results = {}

        test_cases = {
            "simple_dict": {"a": 1, "b": "hello", "c": True},
            "datetime_objects": [datetime.now(timezone.utc) for _ in range(50)],
            "uuid_objects": [str(uuid.UUID(int=i)) for i in range(50)],
            "decimal_objects": [Decimal(f"{i}.99") for i in range(50)],
            "mixed_list": [
                {"id": i, "uuid": str(uuid.UUID(int=i)), "timestamp": datetime.now(timezone.utc)} for i in range(25)
            ],
        }

        # Benchmark each test case
        for name, data in test_cases.items():

            def serialize_typed(data=data):
                return datason.serialize(data)

            results[name] = self.tracker.benchmark_function(serialize_typed)
            print(f"  Time: {results[name]['mean'] * 1000:.2f}ms")

        return results

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("🚀 Starting CI Performance Benchmark Suite")
        print("=" * 60)

        results = {
            "metadata": self.tracker.metadata,
            "benchmarks": {
                "serialization": self.benchmark_serialization(),
                "deserialization": self.benchmark_deserialization(),
                "type_detection": self.benchmark_type_detection_overhead(),
            },
        }

        # Save results
        filename = f"performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.tracker.results_dir, filename)

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {filepath}")

        # Also save as latest.json for easy CI access
        latest_path = os.path.join(self.tracker.results_dir, "latest.json")
        with open(latest_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return results


class PerformanceRegression:
    """Detect performance regressions by comparing with baseline."""

    def __init__(self, results_dir: str = "benchmarks/results"):
        self.results_dir = results_dir

    def load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline performance data."""
        baseline_path = os.path.join(self.results_dir, "baseline.json")
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                return json.load(f)
        return None

    def save_baseline(self, results: Dict[str, Any]) -> None:
        """Save current results as new baseline."""
        baseline_path = os.path.join(self.results_dir, "baseline.json")
        with open(baseline_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📋 Saved new baseline: {baseline_path}")

    def compare_with_baseline(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with baseline and detect regressions."""
        baseline = self.load_baseline()
        if not baseline:
            print("⚠️  No baseline found. Saving current results as baseline.")
            self.save_baseline(current_results)
            return {"status": "baseline_created"}

        comparison = {
            "status": "compared",
            "baseline_metadata": baseline.get("metadata", {}),
            "current_metadata": current_results.get("metadata", {}),
            "performance_changes": {},
            "regressions": [],
            "improvements": [],
        }

        # Compare each benchmark
        for category in ["serialization", "deserialization", "type_detection"]:
            if category not in baseline.get("benchmarks", {}):
                continue

            baseline_bench = baseline["benchmarks"][category]
            current_bench = current_results["benchmarks"][category]

            for test_name in baseline_bench:
                if test_name not in current_bench:
                    continue

                # Compare mean times
                if category in ["serialization", "deserialization"]:
                    # These have nested structure (standard/performance_config)
                    for config in ["standard", "performance_config"]:
                        if config in baseline_bench[test_name] and config in current_bench[test_name]:
                            baseline_time = baseline_bench[test_name][config]["mean"]
                            current_time = current_bench[test_name][config]["mean"]
                            change_pct = ((current_time - baseline_time) / baseline_time) * 100

                            key = f"{category}.{test_name}.{config}"
                            comparison["performance_changes"][key] = {
                                "baseline_ms": baseline_time * 1000,
                                "current_ms": current_time * 1000,
                                "change_pct": change_pct,
                            }

                            # Detect significant changes (>5% threshold)
                            if change_pct > 5:
                                comparison["regressions"].append(
                                    {
                                        "test": key,
                                        "change_pct": change_pct,
                                        "baseline_ms": baseline_time * 1000,
                                        "current_ms": current_time * 1000,
                                    }
                                )
                            elif change_pct < -5:
                                comparison["improvements"].append(
                                    {
                                        "test": key,
                                        "change_pct": change_pct,
                                        "baseline_ms": baseline_time * 1000,
                                        "current_ms": current_time * 1000,
                                    }
                                )
                else:
                    # Type detection has flat structure
                    baseline_time = baseline_bench[test_name]["mean"]
                    current_time = current_bench[test_name]["mean"]
                    change_pct = ((current_time - baseline_time) / baseline_time) * 100

                    key = f"{category}.{test_name}"
                    comparison["performance_changes"][key] = {
                        "baseline_ms": baseline_time * 1000,
                        "current_ms": current_time * 1000,
                        "change_pct": change_pct,
                    }

                    if change_pct > 5:
                        comparison["regressions"].append(
                            {
                                "test": key,
                                "change_pct": change_pct,
                                "baseline_ms": baseline_time * 1000,
                                "current_ms": current_time * 1000,
                            }
                        )
                    elif change_pct < -5:
                        comparison["improvements"].append(
                            {
                                "test": key,
                                "change_pct": change_pct,
                                "baseline_ms": baseline_time * 1000,
                                "current_ms": current_time * 1000,
                            }
                        )

        return comparison


def main():
    """Main function for CI execution."""
    print("🎯 Datason CI Performance Tracker")
    print("=" * 50)

    # Run benchmarks
    suite = StableBenchmarkSuite()
    results = suite.run_all_benchmarks()

    # Check for regressions
    regression_checker = PerformanceRegression()
    comparison = regression_checker.compare_with_baseline(results)

    # Print summary
    print("\n📊 Performance Summary:")
    print("-" * 30)

    if comparison["status"] == "baseline_created":
        print("✅ Baseline created successfully")
        return 0

    print(f"✅ Compared with baseline from {comparison['baseline_metadata'].get('timestamp', 'unknown')}")

    if comparison["regressions"]:
        print(f"\n🔴 Performance Regressions ({len(comparison['regressions'])}):")
        for reg in comparison["regressions"]:
            print(
                f"  {reg['test']}: {reg['change_pct']:+.1f}% ({reg['current_ms']:.2f}ms vs {reg['baseline_ms']:.2f}ms)"
            )

    if comparison["improvements"]:
        print(f"\n🟢 Performance Improvements ({len(comparison['improvements'])}):")
        for imp in comparison["improvements"]:
            print(
                f"  {imp['test']}: {imp['change_pct']:+.1f}% ({imp['current_ms']:.2f}ms vs {imp['baseline_ms']:.2f}ms)"
            )

    if not comparison["regressions"] and not comparison["improvements"]:
        print("🟡 No significant performance changes detected")

    # Save comparison results
    comparison_path = os.path.join("benchmarks/results", "latest_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    # Return appropriate exit code
    if comparison["regressions"]:
        print(f"\n❌ Found {len(comparison['regressions'])} performance regressions")
        return 1
    else:
        print("\n✅ No performance regressions detected")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
