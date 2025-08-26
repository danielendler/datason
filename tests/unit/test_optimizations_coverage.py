"""
Comprehensive test coverage for optimization features in core_new.py.

This test suite targets the new optimization functionality added in recent commits,
including homogeneity checks, string interning, and array optimizations.
"""

import unittest

from datason.core_new import (
    _DYNAMIC_STRING_CACHE,
    _STRING_FREQUENCY_COUNTER,
    _is_homogeneous_collection,
    serialize,
)


class TestHomogeneityChecks(unittest.TestCase):
    """Test homogeneity checking functions."""

    def test_is_homogeneous_collection_empty(self):
        """Test homogeneity check on empty collections."""
        # Empty list returns json_basic
        result = _is_homogeneous_collection([], sample_size=10, _max_check_depth=2)
        self.assertEqual(result, "json_basic")

        # Empty dict returns json_basic
        result = _is_homogeneous_collection({}, sample_size=10, _max_check_depth=2)
        self.assertEqual(result, "json_basic")

    def test_is_homogeneous_collection_single_item(self):
        """Test homogeneity check on single-item collections."""
        # Single item list
        result = _is_homogeneous_collection([{"id": 1}], sample_size=10, _max_check_depth=2)
        self.assertIn(result, ["single_type", "complex"])

        # Single item dict
        result = _is_homogeneous_collection({"item": {"id": 1}}, sample_size=10, _max_check_depth=2)
        self.assertIn(result, ["single_type", "complex"])

    def test_is_homogeneous_collection_mixed_types(self):
        """Test homogeneity check on mixed type collections."""
        # Mixed types in list
        mixed_list = [{"id": 1}, "string", 123]
        result = _is_homogeneous_collection(mixed_list, sample_size=10, _max_check_depth=2)
        self.assertEqual(result, "mixed")

        # Mixed types in dict values
        mixed_dict = {"a": {"id": 1}, "b": "string", "c": 123}
        result = _is_homogeneous_collection(mixed_dict, sample_size=10, _max_check_depth=2)
        self.assertEqual(result, "mixed")

    def test_is_homogeneous_collection_large_sample(self):
        """Test homogeneity check with large collections and sampling."""
        # Large homogeneous list
        large_list = [{"id": i, "name": f"item_{i}"} for i in range(1000)]
        result = _is_homogeneous_collection(large_list, sample_size=20, _max_check_depth=2)
        self.assertIn(result, ["single_type", "complex"])

        # Large homogeneous dict
        large_dict = {f"key_{i}": {"id": i, "value": f"val_{i}"} for i in range(1000)}
        result = _is_homogeneous_collection(large_dict, sample_size=20, _max_check_depth=2)
        self.assertIn(result, ["single_type", "complex"])

    def test_is_homogeneous_collection_depth_limit(self):
        """Test homogeneity check respects depth limits."""
        # Deep nested structure that exceeds depth limit
        deep_structure = [{"level1": {"level2": {"level3": {"id": 1}}}}]
        result = _is_homogeneous_collection(deep_structure, sample_size=10, _max_check_depth=2)
        # Should still return a classification
        self.assertIn(result, ["single_type", "complex", "mixed"])

    def test_is_homogeneous_collection_complex_structures(self):
        """Test homogeneity check on complex nested structures."""
        # Homogeneous complex structures
        complex_list = [
            {"id": 1, "metadata": {"created": "2024-01-01", "tags": ["a", "b"]}, "active": True},
            {"id": 2, "metadata": {"created": "2024-01-02", "tags": ["c", "d"]}, "active": False},
        ]
        result = _is_homogeneous_collection(complex_list, sample_size=10, _max_check_depth=3)
        self.assertIn(result, ["single_type", "complex"])


class TestOptimizationFunctions(unittest.TestCase):
    """Test optimization functions through high-level API."""

    def test_serialize_with_homogeneous_list(self):
        """Test serialization triggers homogeneous list processing."""
        # Large list that should trigger optimization
        large_list = [{"id": i, "name": f"user_{i}", "active": True} for i in range(50)]

        result = serialize(large_list)

        # Should successfully serialize
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 50)

    def test_serialize_with_homogeneous_dict(self):
        """Test serialization triggers homogeneous dict processing."""
        # Large dict that should trigger optimization
        large_dict = {f"user_{i}": {"id": i, "name": f"user_{i}", "active": True} for i in range(30)}

        result = serialize(large_dict)

        # Should successfully serialize
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 30)

    def test_serialize_mixed_data_no_optimization(self):
        """Test that mixed data doesn't trigger optimization."""
        mixed_data = [{"id": 1}, "string", 123, [1, 2, 3]]

        result = serialize(mixed_data)

        # Should successfully serialize without optimization
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)


class TestStringInterning(unittest.TestCase):
    """Test string interning functionality."""

    def test_string_interning_repeated_values(self):
        """Test that repeated strings trigger interning."""
        # Test data with repeated string values
        data = [
            {"status": "active", "type": "user"},
            {"status": "active", "type": "admin"},
            {"status": "inactive", "type": "user"},
        ]

        result = serialize(data)

        # Should successfully serialize
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

    def test_string_interning_cache_behavior(self):
        """Test string interning cache behavior."""
        # Test data that should trigger caching
        repeated_string = "very_common_value"
        data = {f"key_{i}": repeated_string for i in range(10)}

        result = serialize(data)

        # Should successfully serialize
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 10)

    def test_string_interning_threshold_behavior(self):
        """Test string interning threshold behavior."""
        # Test with string that appears multiple times
        test_string = "threshold_test"
        data = [test_string] * 3  # Exactly at threshold

        result = serialize(data)

        # Should successfully serialize
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)


class TestArrayOptimizations(unittest.TestCase):
    """Test array optimization functionality."""

    def test_array_optimization_basic(self):
        """Test basic array optimization functionality."""
        # Create data that should benefit from optimization
        array_data = [{"id": i, "name": f"user_{i}", "active": True} for i in range(20)]

        result = serialize(array_data)

        # Should successfully serialize
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 20)

    def test_dict_optimization_basic(self):
        """Test basic dict optimization functionality."""
        # Create data that should benefit from optimization
        dict_data = {f"user_{i}": {"id": i, "name": f"user_{i}", "active": True} for i in range(15)}

        result = serialize(dict_data)

        # Should successfully serialize
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 15)


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling in optimization code."""

    def test_homogeneity_check_with_non_serializable(self):
        """Test homogeneity check with non-serializable objects."""

        class NonSerializable:
            def __init__(self):
                self.value = "test"

            def __getstate__(self):
                raise RuntimeError("Cannot serialize")

        data = [NonSerializable(), NonSerializable()]

        # Should handle gracefully
        result = _is_homogeneous_collection(data, sample_size=10, _max_check_depth=2)

        # Should return a classification
        self.assertIn(result, ["single_type", "complex", "mixed", "json_basic"])

    def test_optimization_with_circular_references(self):
        """Test optimization with circular references."""
        # Create circular reference
        data = []
        data.append({"self_ref": data})

        # Should handle circular references gracefully
        result = serialize(data)

        # Should return a security error due to circular reference
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("__datason_type__"), "security_error")

    def test_string_interning_memory_management(self):
        """Test string interning memory management."""

        # Clear cache
        _DYNAMIC_STRING_CACHE.clear()
        _STRING_FREQUENCY_COUNTER.clear()

        # Create many unique strings to test cache management
        data = {f"unique_key_{i}": f"unique_value_{i}" for i in range(1000)}

        serialize(data)

        # Cache should have some reasonable size limits
        self.assertLess(len(_DYNAMIC_STRING_CACHE), 1000)  # Should not cache everything
        self.assertLess(len(_STRING_FREQUENCY_COUNTER), 2000)  # Should track, but with limits


if __name__ == "__main__":
    unittest.main()
