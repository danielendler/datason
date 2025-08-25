#!/usr/bin/env python3
"""
DataSON Python 3.13 Optimization Demo

This script demonstrates how DataSON can leverage Python 3.13's new features
while maintaining backward compatibility with older Python versions.
"""

import sys
import time
from typing import Any, Dict, List

try:
    # Try importing optimized features
    from datason.optimizations import (
        JITOptimizedPatternMatcher,
        ParallelRedactionProcessor,
        VersionAwareOptimizations,
        get_optimization_info,
    )

    optimizations_available = True
except ImportError:
    optimizations_available = False

import datason
from datason.redaction import RedactionEngine


def create_test_datasets() -> Dict[str, List[Dict[str, Any]]]:
    """Create test datasets of varying sizes to demonstrate optimizations."""

    # Small dataset - traditional processing
    small_dataset = [
        {
            "user_id": i,
            "username": f"user_{i}",
            "email": f"user_{i}@example.com",
            "password": f"secret_{i}",
            "profile": {"api_key": f"key_{i}", "settings": {"theme": "dark", "notifications": True}},
        }
        for i in range(10)
    ]

    # Medium dataset - benefits from caching optimizations
    medium_dataset = [
        {
            "user_id": i,
            "username": f"user_{i}",
            "email": f"user_{i}@company.com",
            "password": f"password_{i}",
            "session_token": f"token_{i}",
            "api_credentials": {
                "api_key": f"api_key_{i}",
                "secret_key": f"secret_key_{i}",
                "refresh_token": f"refresh_{i}",
            },
            "metadata": {
                "created_at": f"2024-01-{(i % 30) + 1:02d}",
                "last_login": f"2024-02-{(i % 28) + 1:02d}",
                "ip_address": f"192.168.1.{i % 255}",
                "user_agent": f"Browser/{i % 10}.0",
            },
        }
        for i in range(100)
    ]

    # Large dataset - benefits from parallel processing
    large_dataset = [
        {
            "record_id": i,
            "user_data": {
                "username": f"user_{i}",
                "email": f"user_{i}@bigcorp.com",
                "password": f"secure_pass_{i}",
                "ssn": f"{100 + i:03d}-{20 + (i % 80):02d}-{1000 + i:04d}",
            },
            "financial": {
                "account_number": f"ACC{i:06d}",
                "routing_number": f"RTN{i:06d}",
                "credit_card": f"4532-{i:04d}-{i * 2:04d}-{i * 3:04d}",
            },
            "system": {
                "api_key": f"system_api_key_{i}",
                "service_token": f"service_token_{i}",
                "internal_id": f"internal_{i}",
                "debug_info": {"trace_id": f"trace_{i}", "session_id": f"session_{i}", "request_id": f"request_{i}"},
            },
            "analytics": {
                "events": [
                    {"type": "login", "timestamp": f"2024-{j:02d}-15", "ip": f"10.0.{j}.{i}"}
                    for j in range(1, 13)  # 12 months of data
                ]
            },
        }
        for i in range(1000)
    ]

    return {"small": small_dataset, "medium": medium_dataset, "large": large_dataset}


def benchmark_redaction_performance():
    """Benchmark DataSON redaction performance across different Python versions."""
    print("🐍 DataSON Python Version Performance Analysis")
    print("=" * 80)

    # Display Python version and available optimizations
    if optimizations_available:
        opt_info = get_optimization_info()
        print(f"Python Version: {opt_info['python_version']}")
        print(f"Optimization Level: {opt_info['optimization_level']}")
        print(f"JIT Available: {opt_info['jit_available']}")
        print(f"Free Threading: {opt_info['free_threading_available']}")
        print(f"Expected Performance Boost: {opt_info['expected_performance_boost']}")
    else:
        print(f"Python Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        print("Optimizations: Not Available (using standard implementation)")

    print("=" * 80)

    # Create redaction engines
    standard_engine = RedactionEngine(
        redact_fields=["password", "*.api_key", "*.secret_key", "ssn", "credit_card"],
        redact_patterns=[
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
            r"\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card pattern
        ],
        redaction_replacement="[REDACTED]",
    )

    # Create test datasets
    datasets = create_test_datasets()

    # Benchmark each dataset size
    for dataset_name, dataset in datasets.items():
        print(f"\n📊 Benchmarking {dataset_name.upper()} dataset ({len(dataset)} objects)")
        print("-" * 60)

        # Benchmark standard serialization
        start_time = time.perf_counter()
        for _ in range(3):  # Multiple runs for averaging
            _standard_results = [datason.serialize(obj) for obj in dataset]
        standard_time = (time.perf_counter() - start_time) / 3

        # Benchmark secure serialization
        start_time = time.perf_counter()
        for _ in range(3):
            _secure_results = [datason.dump_secure(obj) for obj in dataset]
        secure_time = (time.perf_counter() - start_time) / 3

        # Benchmark manual redaction processing
        start_time = time.perf_counter()
        for _ in range(3):
            _manual_results = [standard_engine.process_object(obj) for obj in dataset]
        manual_time = (time.perf_counter() - start_time) / 3

        # Calculate overhead and display results
        overhead = secure_time / standard_time if standard_time > 0 else 0
        manual_overhead = manual_time / standard_time if standard_time > 0 else 0

        print(f"Standard Serialization:  {standard_time * 1000:8.2f} ms")
        print(f"Secure Serialization:    {secure_time * 1000:8.2f} ms ({overhead:.2f}x overhead)")
        print(f"Manual Redaction:        {manual_time * 1000:8.2f} ms ({manual_overhead:.2f}x overhead)")

        # Show per-object timing for context
        if len(dataset) > 0:
            per_obj_standard = (standard_time / len(dataset)) * 1000000  # microseconds
            per_obj_secure = (secure_time / len(dataset)) * 1000000
            print(f"Per-object Standard:     {per_obj_standard:8.1f} μs")
            print(f"Per-object Secure:       {per_obj_secure:8.1f} μs")


def demonstrate_version_specific_features():
    """Demonstrate version-specific optimizations."""
    print(f"\n🚀 Python {sys.version_info.major}.{sys.version_info.minor} Specific Features")
    print("=" * 80)

    if not optimizations_available:
        print("⚠️  Optimization module not available - using standard implementation")
        return

    # Demonstrate version-aware optimizations
    optimizer = VersionAwareOptimizations()
    print(f"Optimization Level: {optimizer.optimization_level}")
    print(f"Cache Size: {optimizer.cache_size}")
    print(f"Parallel Threshold: {optimizer.parallel_threshold}")

    # Demonstrate pattern matching optimization
    patterns = [("password", None), ("*.api_key", None), ("user.email", None)]

    if sys.version_info >= (3, 13):
        print("\n🔥 Using JIT-Optimized Pattern Matching")
        matcher = JITOptimizedPatternMatcher(patterns)
        test_fields = ["user.password", "config.api_key", "user.email", "public.data"]

        start_time = time.perf_counter()
        for _ in range(10000):  # Many iterations to see JIT benefit
            _results = [matcher.match_field_jit_friendly(field) for field in test_fields]
        jit_time = time.perf_counter() - start_time

        print(f"JIT Pattern Matching (10k iterations): {jit_time * 1000:.2f} ms")
    else:
        print("\n🔧 Using Standard Pattern Matching")
        matcher = JITOptimizedPatternMatcher(patterns)
        test_fields = ["user.password", "config.api_key", "user.email", "public.data"]

        start_time = time.perf_counter()
        for _ in range(10000):
            _results = [matcher.match_field_fallback(field) for field in test_fields]
        standard_time = time.perf_counter() - start_time

        print(f"Standard Pattern Matching (10k iterations): {standard_time * 1000:.2f} ms")

    # Demonstrate parallel processing
    if sys.version_info >= (3, 13) and optimizations_available:
        print("\n⚡ Testing Parallel Processing")
        processor = ParallelRedactionProcessor()

        # Create test data for parallel processing
        test_objects = [{"data": f"value_{i}", "secret": f"secret_{i}"} for i in range(500)]

        def simple_redactor(obj):
            return {k: "[REDACTED]" if "secret" in k else v for k, v in obj.items()}

        # Sequential processing
        start_time = time.perf_counter()
        _sequential_results = [simple_redactor(obj) for obj in test_objects]
        sequential_time = time.perf_counter() - start_time

        # Parallel processing
        start_time = time.perf_counter()
        _parallel_results = processor.process_objects_parallel(simple_redactor, test_objects)
        parallel_time = time.perf_counter() - start_time

        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        print(f"Sequential Processing: {sequential_time * 1000:.2f} ms")
        print(f"Parallel Processing:   {parallel_time * 1000:.2f} ms ({speedup:.2f}x speedup)")


def show_compatibility_matrix():
    """Show what features are available across Python versions."""
    print("\n📋 DataSON Feature Compatibility Matrix")
    print("=" * 80)

    features = {
        "Basic Redaction": "✅ Python 3.8+",
        "Regex Pre-compilation": "✅ Python 3.8+",
        "LRU Field Caching": "✅ Python 3.8+",
        "Early Exit Optimization": "✅ Python 3.8+",
        "Advanced Caching": "✅ Python 3.12+",
        "JIT-Optimized Patterns": "🚀 Python 3.13+",
        "Free-Threading Parallel": "🚀 Python 3.13+",
        "Large Dict Parallel": "🚀 Python 3.13+",
    }

    print(f"{'Feature':<25} {'Availability':<20}")
    print("-" * 45)
    for feature, availability in features.items():
        print(f"{feature:<25} {availability:<20}")

    print(f"\nCurrent Environment: Python {sys.version_info.major}.{sys.version_info.minor}")

    if sys.version_info >= (3, 13):
        print("🎉 All optimizations available!")
    elif sys.version_info >= (3, 12):
        print("✨ Advanced optimizations available!")
    else:
        print("⚙️  Basic optimizations available")


def main():
    """Main demonstration function."""
    print("🔧 DataSON Python 3.13 Optimization Demonstration")
    print("=" * 80)
    print(f"Running on Python {sys.version}")
    print()

    try:
        # Run performance benchmarks
        benchmark_redaction_performance()

        # Demonstrate version-specific features
        demonstrate_version_specific_features()

        # Show compatibility information
        show_compatibility_matrix()

        print(f"\n{'=' * 80}")
        print("✨ Demonstration complete!")
        print("\n💡 Key Takeaways:")
        if sys.version_info >= (3, 13):
            print("   - Python 3.13's JIT compilation can provide 20-40% speedups")
            print("   - Free-threading enables parallel processing of large datasets")
            print("   - Advanced caching strategies reduce memory pressure")
        elif sys.version_info >= (3, 12):
            print("   - Python 3.12 provides solid performance improvements")
            print("   - Optimized dictionary and string operations benefit DataSON")
            print("   - Consider upgrading to Python 3.13 for maximum performance")
        else:
            print("   - Current Python version provides baseline performance")
            print("   - Consider upgrading to Python 3.12+ for better performance")
            print("   - All security features work correctly across versions")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
