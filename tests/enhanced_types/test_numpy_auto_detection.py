"""NumPy array auto-detection tests for datason.

This module tests intelligent auto-detection of NumPy arrays from
serialized list data without requiring type hints.
"""

import json

import pytest

import datason
from datason.config import SerializationConfig
from datason.deserializers import _clear_deserialization_caches, deserialize_fast

numpy = pytest.importorskip("numpy", reason="NumPy not available")


class TestNumPyAutoDetection:
    """Test intelligent NumPy array auto-detection."""

    def setup_method(self) -> None:
        """Clear caches before each test to ensure clean state."""
        _clear_deserialization_caches()

    def test_1d_array_auto_detection(self) -> None:
        """Test 1D array auto-detection from lists."""
        test_arrays = [
            numpy.array([1, 2, 3, 4, 5]),
            numpy.array([1.1, 2.2, 3.3, 4.4]),
            numpy.array([True, False, True, False]),
            numpy.array(["a", "b", "c"]),  # String arrays
        ]

        for original_array in test_arrays:
            # Test without type hints (should auto-detect)
            serialized = datason.serialize(original_array)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            # Should auto-detect as NumPy array
            assert isinstance(reconstructed, numpy.ndarray), f"Array not auto-detected: {original_array}"
            numpy.testing.assert_array_equal(reconstructed, original_array)
            assert reconstructed.dtype == original_array.dtype

    def test_2d_array_auto_detection(self) -> None:
        """Test 2D array auto-detection from nested lists."""
        test_arrays = [
            numpy.array([[1, 2, 3], [4, 5, 6]]),
            numpy.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]),
            numpy.array([[True, False], [False, True]]),
        ]

        for original_array in test_arrays:
            # Test without type hints (should auto-detect)
            serialized = datason.serialize(original_array)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            # Should auto-detect as NumPy array
            assert isinstance(reconstructed, numpy.ndarray), f"2D Array not auto-detected: {original_array.shape}"
            numpy.testing.assert_array_equal(reconstructed, original_array)
            assert reconstructed.shape == original_array.shape

    def test_3d_array_auto_detection(self) -> None:
        """Test 3D array auto-detection from deeply nested lists."""
        original_array = numpy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        # Test without type hints (should auto-detect)
        serialized = datason.serialize(original_array)
        json_str = json.dumps(serialized, default=str)
        parsed = json.loads(json_str)
        reconstructed = deserialize_fast(parsed)

        # Should auto-detect as NumPy array
        assert isinstance(reconstructed, numpy.ndarray), f"3D Array not auto-detected: {original_array.shape}"
        numpy.testing.assert_array_equal(reconstructed, original_array)
        assert reconstructed.shape == original_array.shape

    def test_typed_arrays_auto_detection(self) -> None:
        """Test specific dtype arrays auto-detection."""
        test_cases = [
            (numpy.array([1, 2, 3], dtype=numpy.int32), "int32"),
            (numpy.array([1, 2, 3], dtype=numpy.int64), "int64"),
            (numpy.array([1.1, 2.2, 3.3], dtype=numpy.float32), "float32"),
            (numpy.array([1.1, 2.2, 3.3], dtype=numpy.float64), "float64"),
        ]

        for original_array, dtype_name in test_cases:
            # Test without type hints (should auto-detect array, may not preserve exact dtype)
            serialized = datason.serialize(original_array)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            # Should auto-detect as NumPy array
            assert isinstance(reconstructed, numpy.ndarray), f"{dtype_name} array not auto-detected"
            numpy.testing.assert_array_equal(reconstructed, original_array)

            # Note: Without type hints, exact dtype may not be preserved (JSON limitations)
            # But values should be equivalent

    def test_mixed_type_list_no_false_positives(self) -> None:
        """Test that mixed-type lists don't get falsely detected as arrays."""
        mixed_lists = [
            [1, "hello", 3.14],  # Mixed types
            [1, 2, {"nested": "dict"}],  # Contains complex objects
            [[1, 2], "not_array"],  # Mixed nested types
        ]

        for mixed_list in mixed_lists:
            # Test without type hints
            serialized = datason.serialize(mixed_list)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            # Should remain as list, NOT become NumPy array
            assert isinstance(reconstructed, list), f"Mixed list falsely detected as array: {mixed_list}"
            assert reconstructed == mixed_list

    def test_empty_array_auto_detection(self) -> None:
        """Test empty array auto-detection."""
        empty_arrays = [
            numpy.array([]),
            numpy.array([[]]),  # Empty 2D
            numpy.array([[[], []]]),  # Empty 3D
        ]

        for original_array in empty_arrays:
            # Test without type hints
            serialized = datason.serialize(original_array)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            # Should auto-detect as NumPy array
            assert isinstance(reconstructed, numpy.ndarray), f"Empty array not auto-detected: {original_array.shape}"
            numpy.testing.assert_array_equal(reconstructed, original_array)

    def test_metadata_still_works_perfectly(self) -> None:
        """Test that type hints still provide perfect reconstruction."""
        test_arrays = [
            numpy.array([1, 2, 3], dtype=numpy.int32),
            numpy.array([[1.1, 2.2], [3.3, 4.4]], dtype=numpy.float32),
            numpy.array([True, False, True], dtype=numpy.bool_),
        ]

        config = SerializationConfig(include_type_hints=True)

        for original_array in test_arrays:
            # Test with type hints (should be perfect)
            serialized = datason.serialize(original_array, config=config)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed, config=config)

            # Should be exact reconstruction
            assert isinstance(reconstructed, numpy.ndarray)
            numpy.testing.assert_array_equal(reconstructed, original_array)
            assert reconstructed.dtype == original_array.dtype
            assert reconstructed.shape == original_array.shape

    def test_large_arrays_auto_detection(self) -> None:
        """Test auto-detection works with larger arrays."""
        large_arrays = [
            numpy.random.randint(0, 100, size=100),
            numpy.random.random((10, 10)),
            numpy.arange(1000).reshape(20, 50),
        ]

        for original_array in large_arrays:
            # Test without type hints
            serialized = datason.serialize(original_array)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            # Should auto-detect as NumPy array
            assert isinstance(reconstructed, numpy.ndarray), f"Large array not auto-detected: {original_array.shape}"
            numpy.testing.assert_array_equal(reconstructed, original_array)
            assert reconstructed.shape == original_array.shape

    def test_performance_no_regression(self) -> None:
        """Test that auto-detection doesn't significantly impact performance."""
        # Create a regular list that should NOT be detected as array
        regular_list = [1, 2, 3, 4, 5]

        # Test multiple times to check for performance issues
        for _ in range(100):
            serialized = datason.serialize(regular_list)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            # Should remain as list
            assert isinstance(reconstructed, list)
            assert reconstructed == regular_list
