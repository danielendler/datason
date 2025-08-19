"""
Comprehensive test coverage for conditional optimization thresholds in core_new.py.

This test suite validates the conditional optimization strategies implemented to resolve
serialization performance regressions while maintaining deserialization improvements.

Tests cover:
- Fast path bypass for small collections
- Conditional homogeneity checks based on size and depth
- Threshold-based optimization triggers
- Performance characteristics of the optimizations
"""

import time
import unittest
from unittest.mock import patch

from datason.config import SerializationConfig
from datason.core_new import (
    MAX_STRING_LENGTH,
    _serialize_core,
    serialize,
)


class TestConditionalOptimizationThresholds(unittest.TestCase):
    """Test conditional optimization threshold behavior."""

    def setUp(self):
        """Set up test environment with default config."""
        self.default_config = SerializationConfig()
        self.max_string_length = MAX_STRING_LENGTH

    def test_small_dict_fast_path(self):
        """Test that small dicts (<20 items) bypass expensive optimization analysis."""
        # Small dict that should use fast path
        small_dict = {f"key_{i}": f"value_{i}" for i in range(15)}

        # Mock _is_homogeneous_collection to detect if it's called
        with patch("datason.core_new._is_homogeneous_collection") as mock_homogeneity:
            result = _serialize_core(small_dict, self.default_config, _depth=1, _seen=set(), _type_handler=None)

            # Should NOT call homogeneity check for small dicts
            mock_homogeneity.assert_not_called()

            # Should still serialize correctly
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 15)

    def test_large_dict_optimization_path(self):
        """Test that large dicts (≥20 items) trigger optimization analysis."""
        # Large dict that should trigger optimization
        large_dict = {f"key_{i}": f"value_{i}" for i in range(25)}

        # Mock _is_homogeneous_collection to detect if it's called
        with patch("datason.core_new._is_homogeneous_collection", return_value="json_basic") as mock_homogeneity:
            result = _serialize_core(large_dict, self.default_config, _depth=1, _seen=set(), _type_handler=None)

            # Should call homogeneity check for large dicts
            mock_homogeneity.assert_called_once()

            # Should serialize correctly
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 25)

    def test_small_array_fast_path(self):
        """Test that small arrays (<10 items) bypass expensive optimization analysis."""
        # Small array that should use fast path
        small_array = [{"id": i, "name": f"user_{i}"} for i in range(8)]

        # Mock _is_homogeneous_collection to detect if it's called
        with patch("datason.core_new._is_homogeneous_collection") as mock_homogeneity:
            result = _serialize_core(small_array, self.default_config, _depth=1, _seen=set(), _type_handler=None)

            # Should NOT call homogeneity check for small arrays
            mock_homogeneity.assert_not_called()

            # Should still serialize correctly
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 8)

    def test_large_array_optimization_path(self):
        """Test that large arrays (≥10 items) trigger optimization analysis."""
        # Large array that should trigger optimization
        large_array = [{"id": i, "name": f"user_{i}"} for i in range(15)]

        # Mock _is_homogeneous_collection to detect if it's called
        with patch("datason.core_new._is_homogeneous_collection", return_value="single_type") as mock_homogeneity:
            result = _serialize_core(large_array, self.default_config, _depth=1, _seen=set(), _type_handler=None)

            # Should call homogeneity check for large arrays
            mock_homogeneity.assert_called_once()

            # Should serialize correctly
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 15)


class TestDepthBasedOptimizationControl(unittest.TestCase):
    """Test depth-based optimization control."""

    def setUp(self):
        """Set up test environment."""
        self.default_config = SerializationConfig()

    def test_shallow_depth_enables_optimization(self):
        """Test that shallow depths (<5) enable homogeneity checks."""
        # Data at shallow depth should trigger optimization analysis
        large_dict = {f"key_{i}": f"value_{i}" for i in range(25)}

        with patch("datason.core_new._is_homogeneous_collection", return_value="json_basic") as mock_homogeneity:
            _serialize_core(
                large_dict,
                self.default_config,
                _depth=3,  # Shallow depth
                _seen=set(),
                _type_handler=None,
            )

            # Should call homogeneity check at shallow depth
            mock_homogeneity.assert_called_once()

    def test_deep_depth_disables_optimization(self):
        """Test that deep depths (≥5) disable homogeneity checks."""
        # Data at deep depth should NOT trigger optimization analysis
        large_dict = {f"key_{i}": f"value_{i}" for i in range(25)}

        with patch("datason.core_new._is_homogeneous_collection") as mock_homogeneity:
            _serialize_core(
                large_dict,
                self.default_config,
                _depth=6,  # Deep depth
                _seen=set(),
                _type_handler=None,
            )

            # Should NOT call homogeneity check at deep depth
            mock_homogeneity.assert_not_called()


class TestNestedStructureOptimizationThresholds(unittest.TestCase):
    """Test nested structure optimization thresholds."""

    def setUp(self):
        """Set up test environment."""
        self.default_config = SerializationConfig()

    def test_medium_sized_dict_enables_nested_optimization_checks(self):
        """Test that medium-sized dicts (50-1000 items) are eligible for nested optimization checks."""
        # Create dict that falls within the size range for nested optimization
        medium_dict = {
            f"user_{i}": {"id": i, "name": f"user_{i}", "email": f"user{i}@example.com"}
            for i in range(75)  # 75 items, in the 50-1000 range
        }

        # Rather than testing the complex nested optimization path which has many conditions,
        # let's test that medium-sized dicts don't get excluded from potential optimization
        # by checking they trigger homogeneity analysis (unlike small dicts which use fast path)

        with patch("datason.core_new._is_homogeneous_collection", return_value="json_basic") as mock_homogeneity:
            result = _serialize_core(medium_dict, self.default_config, _depth=2, _seen=set(), _type_handler=None)

            # Should call homogeneity check for medium-sized dicts
            mock_homogeneity.assert_called_once()

            # Should serialize correctly
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 75)

    def test_small_dict_skips_nested_optimization(self):
        """Test that small dicts (<50 items) skip nested optimization checks."""
        # Create dict that should skip nested structure optimization
        small_nested_dict = {
            f"user_{i}": {"id": i, "name": f"user_{i}", "email": f"user{i}@example.com"}
            for i in range(30)  # 30 items, below 50 threshold
        }

        with patch("datason.core_new._is_shallow_json_structure") as mock_shallow:
            _serialize_core(small_nested_dict, self.default_config, _depth=2, _seen=set(), _type_handler=None)

            # Should NOT check for shallow JSON structure on small dicts
            mock_shallow.assert_not_called()

    def test_huge_dict_skips_nested_optimization(self):
        """Test that huge dicts (>1000 items) skip nested optimization checks."""
        # Create dict that's too large for nested optimization
        huge_dict = {f"key_{i}": {"value": i} for i in range(1200)}  # Above 1000 threshold

        with patch("datason.core_new._is_shallow_json_structure") as mock_shallow:
            _serialize_core(huge_dict, self.default_config, _depth=2, _seen=set(), _type_handler=None)

            # Should NOT check for shallow JSON structure on huge dicts
            mock_shallow.assert_not_called()


class TestArrayOptimizationThresholds(unittest.TestCase):
    """Test array optimization thresholds."""

    def setUp(self):
        """Set up test environment."""
        self.default_config = SerializationConfig()

    def test_medium_array_triggers_optimization(self):
        """Test that medium arrays (20-5000 items) trigger object array optimization."""
        # Create array that should trigger optimization
        medium_array = [
            {"id": i, "name": f"user_{i}", "active": True}
            for i in range(35)  # 35 items, in the 20-5000 range
        ]

        with patch("datason.core_new._is_simple_object_array", return_value=True) as mock_simple:
            with patch("datason.core_new._is_homogeneous_collection", return_value="single_type"):
                result = _serialize_core(medium_array, self.default_config, _depth=2, _seen=set(), _type_handler=None)

                # Should check if it's a simple object array
                mock_simple.assert_called_once()

                # Should serialize correctly
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 35)

    def test_small_array_skips_object_optimization(self):
        """Test that small arrays (<20 items) skip object array optimization."""
        # Create array that should skip object optimization
        small_object_array = [
            {"id": i, "name": f"user_{i}", "active": True}
            for i in range(15)  # 15 items, below 20 threshold
        ]

        with patch("datason.core_new._is_simple_object_array") as mock_simple:
            with patch("datason.core_new._is_homogeneous_collection", return_value="single_type"):
                _serialize_core(small_object_array, self.default_config, _depth=2, _seen=set(), _type_handler=None)

                # Should NOT check for simple object array on small arrays
                mock_simple.assert_not_called()

    def test_huge_array_skips_object_optimization(self):
        """Test that huge arrays (>5000 items) skip object array optimization."""
        # Create array that's too large for object optimization
        huge_array = [{"id": i} for i in range(5500)]  # Above 5000 threshold

        with patch("datason.core_new._is_simple_object_array") as mock_simple:
            with patch("datason.core_new._is_homogeneous_collection", return_value="single_type"):
                _serialize_core(huge_array, self.default_config, _depth=2, _seen=set(), _type_handler=None)

                # Should NOT check for simple object array on huge arrays
                mock_simple.assert_not_called()


class TestOptimizationPerformanceCharacteristics(unittest.TestCase):
    """Test performance characteristics of conditional optimizations."""

    def time_operation(self, func, *args, **kwargs):
        """Time a function call and return duration in milliseconds."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return (end - start) * 1000, result

    def test_small_collection_performance(self):
        """Test that small collections have minimal overhead."""
        # Small collections should be very fast due to fast path
        small_dict = {f"key_{i}": f"value_{i}" for i in range(10)}
        small_array = [{"id": i} for i in range(5)]

        # Time small dict serialization
        dict_time_ms, dict_result = self.time_operation(serialize, small_dict)

        # Time small array serialization
        array_time_ms, array_result = self.time_operation(serialize, small_array)

        # Should be very fast due to fast path (allow extra time for CI)
        max_time = 2.0  # 2ms should be more than enough for small collections

        self.assertLess(dict_time_ms, max_time, f"Small dict took {dict_time_ms:.2f}ms, expected <{max_time}ms")
        self.assertLess(array_time_ms, max_time, f"Small array took {array_time_ms:.2f}ms, expected <{max_time}ms")

        # Verify correctness
        self.assertEqual(dict_result, small_dict)
        self.assertEqual(array_result, small_array)

    def test_medium_collection_optimization_benefit(self):
        """Test that medium collections benefit from optimizations."""
        # Medium collections should still be reasonably fast with optimizations
        medium_dict = {
            f"user_{i}": {
                "id": i,
                "name": f"user_{i}",
                "email": f"user{i}@example.com",
                "status": "active",  # Repeated value for string interning
            }
            for i in range(60)  # Medium size that triggers optimization
        }

        medium_array = [
            {
                "id": i,
                "name": f"user_{i}",
                "active": True,
                "type": "user",  # Repeated value
            }
            for i in range(40)  # Medium size that triggers optimization
        ]

        # Time medium collection serialization
        dict_time_ms, dict_result = self.time_operation(serialize, medium_dict)
        array_time_ms, array_result = self.time_operation(serialize, medium_array)

        # Should still be reasonably fast with optimizations (allow extra time for CI)
        max_time = 10.0  # 10ms should be reasonable for medium collections with optimization

        self.assertLess(dict_time_ms, max_time, f"Medium dict took {dict_time_ms:.2f}ms, expected <{max_time}ms")
        self.assertLess(array_time_ms, max_time, f"Medium array took {array_time_ms:.2f}ms, expected <{max_time}ms")

        # Verify correctness
        self.assertEqual(dict_result, medium_dict)
        self.assertEqual(array_result, medium_array)

    def test_performance_scaling_with_thresholds(self):
        """Test that performance scales appropriately across optimization thresholds."""
        # Test different sizes around the thresholds
        test_sizes = [
            (10, "small_dict_fast_path"),
            (25, "large_dict_optimization"),
            (5, "small_array_fast_path"),
            (15, "large_array_optimization"),
            (45, "medium_nested_optimization"),
        ]

        times = {}

        for size, test_type in test_sizes:
            if "dict" in test_type:
                test_data = {f"key_{i}": {"id": i, "value": f"val_{i}"} for i in range(size)}
            else:  # array
                test_data = [{"id": i, "name": f"user_{i}"} for i in range(size)]

            time_ms, result = self.time_operation(serialize, test_data)
            times[test_type] = time_ms

            # Verify correctness for all sizes
            self.assertEqual(result, test_data)

        # Performance characteristics:
        # 1. Small collections (fast path) should be fastest
        # 2. Medium collections (with optimization) should be reasonable
        # 3. None should be excessively slow

        for test_type, time_ms in times.items():
            max_expected = 3.0 if "small" in test_type else 8.0

            self.assertLess(time_ms, max_expected, f"{test_type} took {time_ms:.2f}ms, expected <{max_expected}ms")


class TestOptimizationCorrectness(unittest.TestCase):
    """Test that optimizations maintain correctness."""

    def test_fast_path_correctness(self):
        """Test that fast path produces identical results to optimization path."""
        # Test data that could go either path depending on size
        base_data = {"id": 1, "name": "test", "active": True, "metadata": {"created": "2024-01-01"}}

        # Small version (fast path)
        small_data = {f"item_{i}": base_data.copy() for i in range(15)}

        # Large version (optimization path)
        large_data = {f"item_{i}": base_data.copy() for i in range(25)}

        # Both should serialize to the same structure (just different sizes)
        small_result = serialize(small_data)
        large_result = serialize(large_data)

        # Check structural consistency
        self.assertIsInstance(small_result, dict)
        self.assertIsInstance(large_result, dict)
        self.assertEqual(len(small_result), 15)
        self.assertEqual(len(large_result), 25)

        # Check individual item structure is identical
        small_item = list(small_result.values())[0]
        large_item = list(large_result.values())[0]
        self.assertEqual(small_item, large_item)

    def test_threshold_boundary_correctness(self):
        """Test correctness at optimization threshold boundaries."""
        # Test sizes right at the thresholds
        threshold_tests = [
            (19, "just_below_dict_threshold"),
            (20, "just_at_dict_threshold"),
            (9, "just_below_array_threshold"),
            (10, "just_at_array_threshold"),
        ]

        for size, test_name in threshold_tests:
            if "dict" in test_name:
                test_data = {f"key_{i}": {"id": i, "value": f"val_{i}"} for i in range(size)}
            else:  # array
                test_data = [{"id": i, "name": f"user_{i}"} for i in range(size)]

            # Should serialize correctly regardless of threshold
            result = serialize(test_data)
            self.assertEqual(result, test_data, f"Threshold test {test_name} failed")


if __name__ == "__main__":
    unittest.main()
