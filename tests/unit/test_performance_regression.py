"""Performance regression tests for optimization validation.

These tests ensure that key optimizations continue to provide expected performance
improvements without relying on the complex profiling system.
"""

import time

import pytest

import datason


def time_operation(func, *args, **kwargs):
    """Time a function call and return duration in milliseconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start) * 1000, result


class TestPerformanceRegression:
    """Test performance characteristics of key optimizations."""

    def test_homogeneous_array_optimization(self):
        """Test that homogeneous arrays are processed efficiently."""
        # This should trigger array optimization in core_new.py
        homogeneous_data = [{"id": i, "name": f"user_{i}", "active": True} for i in range(100)]

        duration_ms, result = time_operation(datason.save_string, homogeneous_data)
        loaded_data = datason.load_basic(result)

        # Ensure correctness
        assert loaded_data == homogeneous_data

        # Performance expectation: 100 simple objects should serialize in <5ms
        assert duration_ms < 5.0, f"Homogeneous array took {duration_ms:.2f}ms, expected <5ms"

    def test_nested_structure_optimization(self):
        """Test that nested structures with homogeneous patterns are optimized."""
        # This should trigger nested structure optimization
        nested_data = {
            "users": [{"id": i, "name": f"user_{i}", "role": "member"} for i in range(50)],
            "metadata": {"total": 50, "status": "active", "version": "1.0"},
            "config": {"max_users": 1000, "debug": False},
        }

        duration_ms, result = time_operation(datason.save_string, nested_data)
        loaded_data = datason.load_basic(result)

        # Ensure correctness
        assert loaded_data == nested_data

        # Performance expectation: Nested structure should serialize in <3ms
        assert duration_ms < 3.0, f"Nested structure took {duration_ms:.2f}ms, expected <3ms"

    def test_string_interning_benefit(self):
        """Test that repeated strings benefit from interning optimization."""
        # This should trigger string interning for repeated values
        data_with_repeated_strings = [
            {
                "id": i,
                "status": "active",  # Repeated string
                "type": "user",  # Repeated string
                "region": "us-east-1" if i < 25 else "us-west-2",  # Semi-repeated
                "unique_field": f"unique_value_{i}",
            }
            for i in range(50)
        ]

        duration_ms, result = time_operation(datason.save_string, data_with_repeated_strings)
        loaded_data = datason.load_basic(result)

        # Ensure correctness
        assert loaded_data == data_with_repeated_strings

        # Performance expectation: String interning should keep time reasonable
        assert duration_ms < 4.0, f"String interning case took {duration_ms:.2f}ms, expected <4ms"

    def test_basic_api_performance_baseline(self):
        """Test basic API performance as a regression baseline."""
        simple_data = {
            "message": "Hello, world!",
            "timestamp": "2025-01-15T10:30:00Z",
            "user_id": 12345,
            "metadata": {"version": 1, "source": "test"},
        }

        # Test serialization
        ser_duration_ms, json_str = time_operation(datason.save_string, simple_data)

        # Test deserialization
        deser_duration_ms, loaded_data = time_operation(datason.load_basic, json_str)

        # Ensure correctness
        assert loaded_data == simple_data

        # Performance expectations for basic operations
        assert ser_duration_ms < 1.0, f"Simple serialization took {ser_duration_ms:.2f}ms, expected <1ms"
        assert deser_duration_ms < 1.0, f"Simple deserialization took {deser_duration_ms:.2f}ms, expected <1ms"

    def test_round_trip_performance_consistency(self):
        """Test that round-trip operations maintain consistent performance."""
        test_data = {
            "large_array": [{"index": i, "value": f"item_{i}"} for i in range(200)],
            "nested": {
                "level1": {
                    "level2": {
                        "data": [1, 2, 3, 4, 5] * 20  # 100 integers
                    }
                }
            },
        }

        # Perform multiple round-trips to test consistency
        durations = []
        for _ in range(5):
            start = time.perf_counter()
            json_str = datason.save_string(test_data)
            loaded = datason.load_basic(json_str)
            end = time.perf_counter()

            assert loaded == test_data  # Correctness check
            durations.append((end - start) * 1000)

        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)

        # Performance expectations
        assert avg_duration < 10.0, f"Average round-trip took {avg_duration:.2f}ms, expected <10ms"

        # Consistency check: max shouldn't be more than 3x min (excluding startup costs)
        if min_duration > 0.1:  # Only check if operations are measurable
            ratio = max_duration / min_duration
            assert ratio < 3.0, (
                f"Performance inconsistent: {ratio:.1f}x variance (min: {min_duration:.2f}ms, max: {max_duration:.2f}ms)"
            )

    @pytest.mark.parametrize("size", [10, 50, 100])
    def test_scaling_performance(self, size):
        """Test that performance scales reasonably with data size."""
        data = [{"id": i, "value": f"data_{i}"} for i in range(size)]

        duration_ms, result = time_operation(datason.save_string, data)
        loaded = datason.load_basic(result)

        assert loaded == data

        # Performance should scale roughly linearly (with some overhead)
        # Allow 0.1ms per item plus 2ms base overhead
        expected_max = (size * 0.1) + 2.0
        assert duration_ms < expected_max, f"Size {size} took {duration_ms:.2f}ms, expected <{expected_max:.1f}ms"


if __name__ == "__main__":
    # Allow running tests directly for quick validation
    test = TestPerformanceRegression()

    print("Running performance regression tests...")

    test.test_basic_api_performance_baseline()
    print("✓ Basic API performance baseline")

    test.test_homogeneous_array_optimization()
    print("✓ Homogeneous array optimization")

    test.test_nested_structure_optimization()
    print("✓ Nested structure optimization")

    test.test_string_interning_benefit()
    print("✓ String interning benefit")

    test.test_round_trip_performance_consistency()
    print("✓ Round-trip performance consistency")

    for size in [10, 50, 100]:
        test.test_scaling_performance(size)
        print(f"✓ Scaling performance for size {size}")

    print("\nAll performance regression tests passed!")
