#!/usr/bin/env python3
"""Quick test of the new deserialize_fast function."""

import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Tuple

from datason import serialize
from datason.config import SerializationConfig
from datason.deserializers import deserialize_fast


def test_basic_types() -> List[Tuple[str, bool]]:
    """Test basic type round-trips."""
    print("🧪 Testing basic types...")

    # Test with type hints enabled for round-trip
    config = SerializationConfig(include_type_hints=True)

    test_cases = [
        ("datetime", datetime.now()),
        ("UUID", uuid.uuid4()),
        ("Decimal", Decimal("123.456")),
        ("Path", Path("/tmp/test.txt")),  # nosec B108
        ("complex", complex(1, 2)),
        ("set", {1, 2, 3}),
        ("tuple", (1, 2, 3)),
    ]

    results = []
    for name, original in test_cases:
        try:
            # Serialize with type hints
            serialized = serialize(original, config)
            print(f"  {name}: {type(serialized)} = {serialized}")

            # Deserialize with new function
            deserialized = deserialize_fast(serialized, config)
            print(f"    -> {type(deserialized)} = {deserialized}")

            # Check if round-trip worked
            success = isinstance(deserialized, type(original)) and deserialized == original

            results.append((name, success))
            print("    ✅ SUCCESS" if success else "    ❌ FAILED")

        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            results.append((name, False))

        print()

    return results


def test_performance_comparison() -> None:
    """Quick performance comparison."""
    print("🚀 Performance comparison...")

    import time

    from datason.deserializers import deserialize

    # Test data
    data = {
        "numbers": list(range(1000)),
        "strings": [f"item_{i}" for i in range(100)],
        "nested": {"level1": {"level2": {"data": [1, 2, 3, 4, 5] * 20}}},
    }

    iterations = 1000

    # Test old deserialize
    start = time.perf_counter()
    for _ in range(iterations):
        result1 = deserialize(data)
    old_time = time.perf_counter() - start

    # Test new deserialize_fast
    start = time.perf_counter()
    for _ in range(iterations):
        result2 = deserialize_fast(data)
    new_time = time.perf_counter() - start

    print(f"  Old deserialize: {old_time:.4f}s ({iterations} iterations)")
    print(f"  New deserialize_fast: {new_time:.4f}s ({iterations} iterations)")
    print(f"  Speedup: {old_time / new_time:.2f}x")

    # Verify results are the same
    print(f"  Results equal: {result1 == result2}")


if __name__ == "__main__":
    print("Testing new deserialize_fast implementation...")
    print("=" * 50)

    # Test basic types
    results = test_basic_types()

    # Summary
    print("\n📊 Summary:")
    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"  {passed}/{total} tests passed ({passed / total * 100:.1f}%)")

    # Performance test
    print("\n" + "=" * 50)
    test_performance_comparison()
