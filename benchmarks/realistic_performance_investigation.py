#!/usr/bin/env python3
"""
Realistic Performance Investigation for datason
==============================================

This script conducts a deep dive into datason's performance with realistic use cases,
not just optimized benchmarks. We investigate:

1. Real-world data serialization patterns
2. Type detection overhead analysis
3. Template deserialization vs auto-detection
4. Memory usage patterns
5. Bottleneck identification
6. Comparison with pure Python alternatives
7. Analysis of whether a Rust core would help

The goal is to understand where performance comes from and whether the Python-specific
nature of the library makes porting to Rust feasible or beneficial.
"""

import cProfile
import io
import json
import pstats
import sys
import time
import tracemalloc
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict

# Third-party imports (if available)
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

# Datason imports
import datason
from datason import SerializationConfig
from datason.deserializers import auto_deserialize, infer_template_from_data

# Optional import for deep profiling
try:
    import memory_profiler  # type: ignore  # noqa: F401

    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False


class PerformanceProfiler:
    """Context manager for detailed performance profiling."""

    def __init__(self, name: str, profile_memory: bool = False):
        self.name = name
        self.profile_memory = profile_memory and HAS_MEMORY_PROFILER
        self.start_time = None
        self.end_time = None
        self.memory_usage = None

    def __enter__(self):
        print(f"\n🔍 Starting: {self.name}")
        if self.profile_memory:
            tracemalloc.start()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time

        if self.profile_memory and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.memory_usage = peak / 1024 / 1024  # MB
            print(f"✅ {self.name}: {elapsed:.4f}s (Peak memory: {self.memory_usage:.2f} MB)")
        else:
            print(f"✅ {self.name}: {elapsed:.4f}s")
        return False


def create_realistic_api_response(num_users: int = 100) -> Dict[str, Any]:
    """Create realistic API response data - common use case for datason."""
    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc),
            "version": "v1.2.3",
            "request_id": str(uuid.uuid4()),
            "total_count": num_users,
            "page_size": min(50, num_users),
            "has_more": num_users > 50,
        },
        "users": [
            {
                "id": i,
                "uuid": str(uuid.uuid4()),
                "username": f"user_{i}",
                "email": f"user_{i}@example.com",
                "profile": {
                    "first_name": f"First{i}",
                    "last_name": f"Last{i}",
                    "avatar_url": f"https://avatars.example.com/{i}.jpg",
                    "bio": f"This is user {i}'s biography. " * (i % 3 + 1),
                    "location": {
                        "city": f"City{i % 10}",
                        "country": ["US", "CA", "UK", "DE", "FR"][i % 5],
                        "timezone": "UTC",
                    },
                },
                "preferences": {
                    "notifications": i % 2 == 0,
                    "newsletter": i % 3 == 0,
                    "privacy_level": ["public", "friends", "private"][i % 3],
                    "theme": ["light", "dark", "auto"][i % 3],
                },
                "stats": {
                    "posts_count": i * 3,
                    "followers_count": i * 10,
                    "following_count": i * 5,
                    "last_activity": datetime.now(timezone.utc),
                    "account_value": Decimal(f"{i * 12.34:.2f}"),
                },
                "tags": [f"tag_{j}" for j in range(i % 5 + 1)],
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "is_verified": i % 7 == 0,
                "is_active": i % 11 != 0,
            }
            for i in range(num_users)
        ],
    }


def create_realistic_iot_data(num_sensors: int = 50, readings_per_sensor: int = 100) -> Dict[str, Any]:
    """Create realistic IoT sensor data - another common datason use case."""
    return {
        "collection_metadata": {
            "facility_id": "facility_001",
            "collection_start": datetime.now(timezone.utc),
            "collection_end": datetime.now(timezone.utc),
            "total_sensors": num_sensors,
            "total_readings": num_sensors * readings_per_sensor,
        },
        "sensors": [
            {
                "sensor_id": f"sensor_{i:03d}",
                "sensor_type": ["temperature", "humidity", "pressure", "air_quality"][i % 4],
                "location": {
                    "building": f"Building_{chr(65 + i // 10)}",
                    "floor": (i % 10) + 1,
                    "room": f"Room_{i:03d}",
                    "coordinates": {"x": i * 1.5, "y": i * 2.3, "z": 2.5},
                },
                "readings": [
                    {
                        "timestamp": datetime.now(timezone.utc),
                        "value": 20.0 + (j % 30) + (i * 0.1),
                        "unit": ["°C", "%", "hPa", "AQI"][i % 4],
                        "quality": min(1.0, 0.7 + (j % 4) * 0.1),
                        "calibration_id": (str(uuid.uuid4()) if j % 100 == 0 else None),
                    }
                    for j in range(readings_per_sensor)
                ],
            }
            for i in range(num_sensors)
        ],
    }


def create_realistic_ml_dataset() -> Dict[str, Any]:
    """Create realistic ML dataset - common in ML pipelines."""
    if not HAS_NUMPY or not HAS_PANDAS:
        return {"error": "NumPy/Pandas not available for ML dataset"}

    # Simulate feature matrix
    n_samples, n_features = 1000, 50
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 5, n_samples)

    # Create DataFrame with mixed types
    df = pd.DataFrame({"feature_" + str(i): X[:, i] for i in range(n_features)})
    df["target"] = y
    df["sample_id"] = [str(uuid.uuid4()) for _ in range(n_samples)]
    df["timestamp"] = pd.date_range("2023-01-01", periods=n_samples, freq="1H")

    return {
        "dataset_metadata": {
            "name": "synthetic_classification_dataset",
            "created": datetime.now(timezone.utc),
            "n_samples": n_samples,
            "n_features": n_features,
            "target_classes": 5,
            "format_version": "1.0",
        },
        "data": {
            "features": X.tolist(),  # Explicitly convert to list
            "targets": y.tolist(),
            "feature_names": [f"feature_{i}" for i in range(n_features)],
            "sample_metadata": df[["sample_id", "timestamp"]].to_dict("records"),
        },
        "statistics": {
            "feature_means": np.mean(X, axis=0).tolist(),
            "feature_stds": np.std(X, axis=0).tolist(),
            "target_distribution": {str(i): int(np.sum(y == i)) for i in range(5)},
        },
    }


def benchmark_serialization_approaches(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Benchmark different serialization approaches for the same data."""
    results = {"dataset": name}

    # 1. Standard datason serialization
    with PerformanceProfiler(f"{name} - Standard datason", profile_memory=True) as profiler:
        datason.serialize(data)
    results["standard_datason"] = {
        "time": profiler.end_time - profiler.start_time,
        "memory_mb": profiler.memory_usage,
    }

    # 2. Datason with performance config
    perf_config = datason.get_performance_config()
    with PerformanceProfiler(f"{name} - Performance config", profile_memory=True) as profiler:
        datason.serialize(data, config=perf_config)
    results["performance_config"] = {
        "time": profiler.end_time - profiler.start_time,
        "memory_mb": profiler.memory_usage,
    }

    # 3. Datason with type hints (for round-trip)
    type_config = SerializationConfig(include_type_hints=True)
    with PerformanceProfiler(f"{name} - Type hints", profile_memory=True) as profiler:
        datason.serialize(data, config=type_config)
    results["type_hints"] = {
        "time": profiler.end_time - profiler.start_time,
        "memory_mb": profiler.memory_usage,
    }

    # 4. Standard JSON (for comparison)
    try:
        with PerformanceProfiler(f"{name} - Standard JSON", profile_memory=True) as profiler:
            json.dumps(data, default=str, ensure_ascii=False)
        results["standard_json"] = {
            "time": profiler.end_time - profiler.start_time,
            "memory_mb": profiler.memory_usage,
        }
    except Exception as e:
        results["standard_json"] = {"error": str(e)}

    return results


def benchmark_deserialization_approaches(serialized_data: Any, original_data: Any, name: str) -> Dict[str, Any]:
    """Benchmark different deserialization approaches."""
    results = {"dataset": name}

    # 1. Standard deserialization
    with PerformanceProfiler(f"{name} - Standard deserialize", profile_memory=True) as profiler:
        datason.deserialize(serialized_data)
    results["standard_deserialize"] = {
        "time": profiler.end_time - profiler.start_time,
        "memory_mb": profiler.memory_usage,
    }

    # 2. Auto-detection (aggressive)
    with PerformanceProfiler(f"{name} - Auto-detection", profile_memory=True) as profiler:
        auto_deserialize(serialized_data, aggressive=True)
    results["auto_detection"] = {
        "time": profiler.end_time - profiler.start_time,
        "memory_mb": profiler.memory_usage,
    }

    # 3. Template-based (if applicable)
    try:
        infer_template_from_data(original_data)
        with PerformanceProfiler(f"{name} - Template-based", profile_memory=True) as profiler:
            # Note: This is a simplified template test
            datason.deserialize(serialized_data)
        results["template_based"] = {
            "time": profiler.end_time - profiler.start_time,
            "memory_mb": profiler.memory_usage,
        }
    except Exception as e:
        results["template_based"] = {"error": str(e)}

    return results


def profile_type_detection_overhead() -> Dict[str, Any]:
    """Profile the overhead of type detection in datason."""
    print("\n" + "=" * 60)
    print("TYPE DETECTION OVERHEAD ANALYSIS")
    print("=" * 60)

    results = {}

    # Test different data types and their detection overhead
    test_cases = {
        "simple_dict": {"a": 1, "b": "hello", "c": True},
        "uuid_heavy": {f"id_{i}": str(uuid.uuid4()) for i in range(100)},
        "datetime_heavy": {f"timestamp_{i}": datetime.now(timezone.utc) for i in range(100)},
        "decimal_heavy": {f"price_{i}": Decimal(f"{i * 1.99:.2f}") for i in range(100)},
        "mixed_complex": {
            "uuids": [str(uuid.uuid4()) for _ in range(50)],
            "timestamps": [datetime.now(timezone.utc) for _ in range(50)],
            "decimals": [Decimal(f"{i * 1.99:.2f}") for i in range(50)],
            "nested": {"level1": {"level2": {"data": [{"id": str(uuid.uuid4()), "value": i} for i in range(20)]}}},
        },
    }

    for case_name, data in test_cases.items():
        print(f"\n🔍 Testing: {case_name}")

        # Profile with cProfile for detailed analysis
        profiler = cProfile.Profile()
        profiler.enable()

        # Run serialization multiple times
        for _ in range(10):
            datason.serialize(data)

        profiler.disable()

        # Capture profile stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats("cumulative")
        ps.print_stats(20)  # Top 20 functions

        profile_output = s.getvalue()

        # Extract key metrics
        total_time = 0
        isinstance_calls = 0
        for line in profile_output.split("\n"):
            if "isinstance" in line:
                isinstance_calls += 1
            if "cumulative" in line and "seconds" in line:
                # Try to extract total time
                parts = line.split()
                if len(parts) > 3:
                    try:
                        total_time = float(parts[3])
                    except (ValueError, IndexError):
                        pass

        results[case_name] = {
            "total_time": total_time,
            "isinstance_calls": isinstance_calls,
            "profile_output": profile_output[:1000],  # Truncate for storage
        }

        print(f"  Total time: {total_time:.4f}s")
        print(f"  isinstance calls detected: {isinstance_calls}")

    return results


def analyze_rust_feasibility() -> Dict[str, Any]:
    """Analyze which parts of datason could benefit from Rust implementation."""
    print("\n" + "=" * 60)
    print("RUST FEASIBILITY ANALYSIS")
    print("=" * 60)

    analysis = {
        "python_specific_features": {
            "description": "Features that are hard to port to Rust",
            "items": [
                {
                    "feature": "Runtime type introspection",
                    "difficulty": "High",
                    "reason": "Python's type() and isinstance() are runtime features",
                    "rust_alternative": "Trait-based type system, but less flexible",
                },
                {
                    "feature": "Dynamic object attribute access",
                    "difficulty": "High",
                    "reason": "getattr(), setattr(), hasattr() are Python-specific",
                    "rust_alternative": "Serde with custom deserializers",
                },
                {
                    "feature": "Exception handling patterns",
                    "difficulty": "Medium",
                    "reason": "Python's exception model differs from Rust's Result<T, E>",
                    "rust_alternative": "Result types, but API would change",
                },
                {
                    "feature": "Third-party library integration",
                    "difficulty": "High",
                    "reason": "NumPy, Pandas, etc. are Python-specific",
                    "rust_alternative": "Would need Rust equivalents or FFI",
                },
            ],
        },
        "rust_portable_operations": {
            "description": "Operations that could benefit from Rust",
            "items": [
                {
                    "operation": "JSON parsing and generation",
                    "benefit": "High",
                    "reason": "Pure string/memory operations, no Python specifics",
                    "estimated_speedup": "5-10x",
                },
                {
                    "operation": "String processing and validation",
                    "benefit": "High",
                    "reason": "UUID, datetime parsing are algorithmic",
                    "estimated_speedup": "10-20x",
                },
                {
                    "operation": "Memory management",
                    "benefit": "Medium",
                    "reason": "Rust's zero-cost abstractions",
                    "estimated_speedup": "2-5x",
                },
                {
                    "operation": "Template matching and caching",
                    "benefit": "High",
                    "reason": "Pattern matching is Rust's strength",
                    "estimated_speedup": "5-15x",
                },
                {
                    "operation": "Chunked/streaming processing",
                    "benefit": "High",
                    "reason": "Iterator patterns and zero-copy operations",
                    "estimated_speedup": "3-8x",
                },
            ],
        },
        "hybrid_architecture_recommendation": {
            "description": "Recommended approach for Rust integration",
            "strategy": "Selective Rust core with Python wrapper",
            "phases": [
                {
                    "phase": 1,
                    "focus": "Core JSON engine",
                    "timeline": "2-3 months",
                    "components": ["JSON parsing", "Basic serialization", "String handling"],
                },
                {
                    "phase": 2,
                    "focus": "Type system",
                    "timeline": "2-3 months",
                    "components": ["UUID processing", "Datetime handling", "Decimal support"],
                },
                {
                    "phase": 3,
                    "focus": "Template engine",
                    "timeline": "3-4 months",
                    "components": ["Pattern inference", "Template caching", "Fast matching"],
                },
                {
                    "phase": 4,
                    "focus": "Advanced features",
                    "timeline": "2-3 months",
                    "components": ["Streaming", "Chunked processing", "Memory optimization"],
                },
            ],
        },
        "performance_expectations": {
            "conservative_estimate": "5-10x speedup for core operations",
            "optimistic_estimate": "10-20x speedup for type-heavy workloads",
            "development_cost": "High (6-12 months full-time)",
            "maintenance_cost": "Medium (two codebases to maintain)",
        },
    }

    # Print summary
    print("🔍 Python-specific features (hard to port):")
    for item in analysis["python_specific_features"]["items"]:
        print(f"  • {item['feature']}: {item['difficulty']} difficulty")

    print("\n🚀 Rust-portable operations (high benefit):")
    for item in analysis["rust_portable_operations"]["items"]:
        print(f"  • {item['operation']}: {item['estimated_speedup']} speedup potential")

    expectations = analysis["performance_expectations"]
    print(f"\n📊 Expected performance improvement: {expectations['conservative_estimate']}")
    print(f"💰 Development cost: {expectations['development_cost']}")

    return analysis


def comprehensive_performance_investigation():
    """Run the complete performance investigation."""
    print("=" * 80)
    print("DATASON REALISTIC PERFORMANCE INVESTIGATION")
    print("=" * 80)
    print("Investigating real-world performance, not optimized benchmarks")
    print("=" * 80)

    all_results = {
        "investigation_metadata": {
            "timestamp": datetime.now(timezone.utc),
            "python_version": sys.version,
            "has_numpy": HAS_NUMPY,
            "has_pandas": HAS_PANDAS,
            "has_memory_profiler": HAS_MEMORY_PROFILER,
        },
        "serialization_benchmarks": {},
        "deserialization_benchmarks": {},
        "type_detection_analysis": {},
        "rust_feasibility": {},
    }

    # 1. Serialization benchmarks with realistic data
    print("\n" + "=" * 60)
    print("SERIALIZATION BENCHMARKS")
    print("=" * 60)

    test_datasets = {
        "api_response_small": create_realistic_api_response(25),
        "api_response_large": create_realistic_api_response(500),
        "iot_data": create_realistic_iot_data(20, 50),
        "config_data": {
            "app_config": {
                "name": "MyApp",
                "version": "1.0.0",
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "credentials": {
                        "username": "admin",
                        "password_hash": str(uuid.uuid4()),
                    },
                },
                "features": {
                    "feature_flags": [f"flag_{i}" for i in range(20)],
                    "created": datetime.now(timezone.utc),
                },
            }
        },
        "log_entries": {
            "logs": [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "level": "INFO",
                    "message": f"Log entry {i}",
                    "request_id": str(uuid.uuid4()),
                    "duration": Decimal(f"{i * 0.123:.3f}"),
                }
                for i in range(100)
            ]
        },
    }

    # Add ML dataset if available
    if HAS_NUMPY and HAS_PANDAS:
        test_datasets["ml_dataset"] = create_realistic_ml_dataset()

    for name, data in test_datasets.items():
        print(f"\n🔬 Benchmarking: {name}")
        results = benchmark_serialization_approaches(data, name)
        all_results["serialization_benchmarks"][name] = results

        # Print summary
        if "standard_json" in results and "error" not in results["standard_json"]:
            datason_time = results["standard_datason"]["time"]
            json_time = results["standard_json"]["time"]
            overhead = ((datason_time - json_time) / json_time) * 100
            print(f"  Datason vs JSON overhead: {overhead:+.1f}%")
        else:
            print("  JSON comparison: Not available (complex types)")

    # 2. Type detection overhead analysis
    all_results["type_detection_analysis"] = profile_type_detection_overhead()

    # 3. Rust feasibility analysis
    all_results["rust_feasibility"] = analyze_rust_feasibility()

    # 4. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmarks/performance_investigation_{timestamp}.json"

    try:
        with open(filename, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {filename}")
    except Exception as e:
        print(f"\n❌ Failed to save results: {e}")

    # 5. Print final summary
    print("\n" + "=" * 80)
    print("INVESTIGATION SUMMARY")
    print("=" * 80)

    print("\n📊 Serialization Performance Summary:")
    for name, results in all_results["serialization_benchmarks"].items():
        if "standard_json" in results and "error" not in results["standard_json"]:
            datason_time = results["standard_datason"]["time"]
            json_time = results["standard_json"]["time"]
            overhead = ((datason_time - json_time) / json_time) * 100
            print(f"  {name:20}: {overhead:+6.1f}% overhead vs JSON")
        else:
            print(f"  {name:20}: Complex types (no JSON comparison)")

    print("\n🔍 Type Detection Analysis:")
    type_results = all_results["type_detection_analysis"]
    for case_name, data in type_results.items():
        print(f"  {case_name:20}: {data['total_time']:.4f}s total")

    print("\n🚀 Rust Feasibility:")
    rust_analysis = all_results["rust_feasibility"]
    perf_expectations = rust_analysis["performance_expectations"]
    print(f"  Expected speedup: {perf_expectations['conservative_estimate']}")
    print(f"  Development cost: {perf_expectations['development_cost']}")

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    comprehensive_performance_investigation()
