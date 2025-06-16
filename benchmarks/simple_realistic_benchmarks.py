#!/usr/bin/env python3
"""
Simple Realistic Benchmarks for datason
=======================================

This script tests realistic, everyday use cases for datason without requiring
optional dependencies. It focuses on:

1. API response processing
2. Configuration file handling
3. Log data serialization
4. Simple ML data structures
5. Comparison with standard JSON

The goal is to understand real-world performance without optimized/tailored scenarios.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List

import datason


def create_api_response_data(size: str = "medium") -> Dict[str, Any]:
    """Create realistic API response data."""
    sizes = {"small": 25, "medium": 100, "large": 500, "xl": 1000}

    num_items = sizes.get(size, 100)

    return {
        "status": "success",
        "timestamp": datetime.now(timezone.utc),
        "request_id": str(uuid.uuid4()),
        "pagination": {
            "page": 1,
            "per_page": 50,
            "total": num_items,
            "has_next": num_items > 50,
        },
        "data": [
            {
                "id": i,
                "name": f"Item {i}",
                "description": f"This is item number {i} with some description text.",
                "price": Decimal(f"{19.99 + i * 0.50:.2f}"),
                "category": ["electronics", "books", "clothing", "home"][i % 4],
                "tags": [f"tag{j}" for j in range(i % 3 + 1)],
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "metadata": {
                    "views": i * 10,
                    "rating": min(5.0, 1.0 + (i % 50) * 0.1),
                    "in_stock": i % 7 != 0,
                    "featured": i % 13 == 0,
                },
            }
            for i in range(num_items)
        ],
    }


def create_config_data() -> Dict[str, Any]:
    """Create realistic configuration data."""
    return {
        "app_name": "MyApplication",
        "version": "1.2.3",
        "debug": False,
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "myapp_db",
            "pool_size": 10,
            "timeout": 30.0,
        },
        "cache": {"enabled": True, "ttl": 3600, "max_memory": "512MB"},
        "features": {
            "new_ui": True,
            "analytics": False,
            "beta_features": ["feature_a", "feature_b"],
        },
        "created": datetime.now(timezone.utc),
        "config_id": str(uuid.uuid4()),
    }


def create_log_data() -> List[Dict[str, Any]]:
    """Create realistic log entries."""
    return [
        {
            "timestamp": datetime.now(timezone.utc),
            "level": ["INFO", "WARNING", "ERROR", "DEBUG"][i % 4],
            "message": f"Log message {i}: Something happened in the system",
            "module": f"module_{i % 5}",
            "user_id": str(uuid.uuid4()) if i % 3 == 0 else None,
            "session_id": str(uuid.uuid4()),
            "duration_ms": i * 12.5,
            "metadata": {
                "ip_address": f"192.168.1.{i % 255}",
                "user_agent": "Mozilla/5.0 (compatible; App/1.0)",
                "endpoint": f"/api/v1/endpoint_{i % 10}",
            },
        }
        for i in range(200)
    ]


def create_simple_ml_data() -> Dict[str, Any]:
    """Create simple ML-like data without NumPy/Pandas."""
    return {
        "model_info": {
            "name": "simple_classifier",
            "version": "1.0",
            "created": datetime.now(timezone.utc),
            "model_id": str(uuid.uuid4()),
        },
        "training_data": [
            {
                "features": [i * 0.1, i * 0.2, i * 0.3],
                "label": i % 2,
                "sample_id": str(uuid.uuid4()),
            }
            for i in range(1000)
        ],
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "dropout": 0.2,
        },
        "metrics": {
            "accuracy": 0.87,
            "precision": 0.85,
            "recall": 0.89,
            "f1_score": 0.87,
        },
    }


def benchmark_function(func, iterations: int = 5) -> Dict[str, Any]:
    """Benchmark a function with multiple iterations."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        func()  # Just time the operation
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "total_time": sum(times),
        "iterations": iterations,
    }


def compare_with_standard_json(data: Any, name: str) -> Dict[str, Any]:
    """Compare datason performance with standard JSON."""

    print(f"\n🔬 Benchmarking: {name}")

    # Datason serialization
    def datason_serialize():
        return datason.serialize(data)

    datason_results = benchmark_function(datason_serialize, iterations=10)

    # Try standard JSON (might fail with complex types)
    try:

        def json_serialize():
            return json.dumps(data, default=str, ensure_ascii=False)

        json_results = benchmark_function(json_serialize, iterations=10)
        json_error = None
    except Exception as e:
        json_results = None
        json_error = str(e)

    # Datason with performance config
    perf_config = datason.get_performance_config()

    def perf_serialize():
        return datason.serialize(data, config=perf_config)

    perf_results = benchmark_function(perf_serialize, iterations=10)

    results = {
        "dataset": name,
        "datason_standard": datason_results,
        "datason_performance": perf_results,
        "json_standard": json_results,
        "json_error": json_error,
    }

    # Print results
    print(
        f"  Datason (standard):    {datason_results['mean_time'] * 1000:.2f}ms ± "
        f"{(datason_results['max_time'] - datason_results['min_time']) * 1000:.2f}ms"
    )
    print(
        f"  Datason (performance): {perf_results['mean_time'] * 1000:.2f}ms ± "
        f"{(perf_results['max_time'] - perf_results['min_time']) * 1000:.2f}ms"
    )

    if json_results:
        print(
            f"  Standard JSON:         {json_results['mean_time'] * 1000:.2f}ms ± "
            f"{(json_results['max_time'] - json_results['min_time']) * 1000:.2f}ms"
        )
        overhead = ((datason_results["mean_time"] - json_results["mean_time"]) / json_results["mean_time"]) * 100
        print(f"  Datason overhead:      {overhead:+.1f}%")
    else:
        print(f"  Standard JSON:         FAILED ({json_error})")

    return results


def test_deserialization_performance() -> None:
    """Test deserialization performance with realistic data."""
    print("\n" + "=" * 60)
    print("DESERIALIZATION PERFORMANCE TEST")
    print("=" * 60)

    # Create test data
    api_data = create_api_response_data("medium")

    # Serialize with different approaches
    standard_serialized = datason.serialize(api_data)

    # Test standard deserialization
    print("\n🔍 Testing standard deserialization...")

    def deserialize_func():
        return datason.deserialize(standard_serialized)

    deserialize_results = benchmark_function(deserialize_func, iterations=10)
    print(f"  Standard deserialize: {deserialize_results['mean_time'] * 1000:.2f}ms")

    # Test auto-detection
    print("\n🔍 Testing auto-detection...")
    from datason.deserializers_new import auto_deserialize

    def auto_func():
        return auto_deserialize(standard_serialized, aggressive=True)

    auto_results = benchmark_function(auto_func, iterations=10)
    print(f"  Auto-detection: {auto_results['mean_time'] * 1000:.2f}ms")

    overhead = ((auto_results["mean_time"] - deserialize_results["mean_time"]) / deserialize_results["mean_time"]) * 100
    print(f"  Auto-detection overhead: {overhead:+.1f}%")


def test_template_performance() -> None:
    """Test template-based deserialization performance."""
    print("\n" + "=" * 60)
    print("TEMPLATE DESERIALIZATION TEST")
    print("=" * 60)

    from datason.deserializers_new import TemplateDeserializer, infer_template_from_data

    # Create test data
    items = [
        {
            "id": i,
            "name": f"Item {i}",
            "price": i * 1.99,
            "created": datetime.now(timezone.utc),
            "active": i % 2 == 0,
        }
        for i in range(100)
    ]

    # Infer template from small sample
    print("🔍 Inferring template...")
    template_start = time.perf_counter()
    template = infer_template_from_data(items[:5])
    template_time = time.perf_counter() - template_start
    print(f"  Template inference: {template_time * 1000:.2f}ms")

    # Test template deserialization
    print("\n🔍 Testing template deserialization...")
    deserializer = TemplateDeserializer(template)

    def template_func():
        return [deserializer.deserialize(item) for item in items]

    template_results = benchmark_function(template_func, iterations=5)
    print(f"  Template deserialize: {template_results['mean_time'] * 1000:.2f}ms")
    print(f"  Per-item average: {template_results['mean_time'] * 1000 / 100:.3f}ms")

    # Compare with standard deserialization
    def standard_func():
        return [datason.deserialize(item) for item in items]

    standard_results = benchmark_function(standard_func, iterations=5)
    print(f"  Standard deserialize: {standard_results['mean_time'] * 1000:.2f}ms")

    speedup = standard_results["mean_time"] / template_results["mean_time"]
    print(f"  Template speedup: {speedup:.2f}x")


def main():
    """Run all realistic benchmarks."""
    print("=" * 80)
    print("DATASON REALISTIC PERFORMANCE BENCHMARKS")
    print("=" * 80)
    print("Testing real-world use cases, not optimized scenarios")
    print("=" * 80)

    # Test datasets
    test_cases = [
        ("API Response (Small)", create_api_response_data("small")),
        ("API Response (Large)", create_api_response_data("large")),
        ("Configuration Data", create_config_data()),
        ("Log Entries", create_log_data()),
        ("Simple ML Data", create_simple_ml_data()),
    ]

    all_results = []

    # Run serialization benchmarks
    print("\n" + "=" * 60)
    print("SERIALIZATION BENCHMARKS")
    print("=" * 60)

    for name, data in test_cases:
        result = compare_with_standard_json(data, name)
        all_results.append(result)

    # Run deserialization tests
    test_deserialization_performance()

    # Run template tests
    test_template_performance()

    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print("\n📊 Serialization Performance vs Standard JSON:")
    for result in all_results:
        name = result["dataset"]
        if result["json_standard"]:
            datason_time = result["datason_standard"]["mean_time"]
            json_time = result["json_standard"]["mean_time"]
            overhead = ((datason_time - json_time) / json_time) * 100
            print(f"  {name:25}: {overhead:+6.1f}% overhead")
        else:
            print(f"  {name:25}: Complex types (no JSON comparison)")

    print("\n🎯 Key Findings:")
    print("  • Datason adds significant overhead vs pure JSON")
    print("  • Performance varies greatly by data complexity")
    print("  • Template deserialization provides meaningful speedups")
    print("  • Type detection is a major performance factor")

    print("\n💡 Recommendations:")
    print("  • Use performance config for production workloads")
    print("  • Consider templates for repetitive data structures")
    print("  • Profile your specific use case for optimization")
    print("  • Evaluate if datason's features justify the overhead")

    print("\n" + "=" * 80)
    print("BENCHMARKS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
