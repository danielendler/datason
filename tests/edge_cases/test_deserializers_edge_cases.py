"""
Deserializers Module Edge Cases and Coverage Boosters

This file contains ALL tests specifically designed to cover edge cases and error paths
in the datason.deserializers module. Consolidated from multiple scattered files to avoid duplication.

This includes:
- Type auto-detection edge cases (NumPy, Pandas)
- Deserializer error handling
- Template deserialization edge cases
- Import failure scenarios
- Performance edge cases
"""

import json
import unittest
from unittest.mock import patch

import datason
from datason.deserializers_new import deserialize_fast

# Optional imports with fallbacks
try:
    import numpy

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    numpy = None

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


class TestNumPyAutoDetection(unittest.TestCase):
    """Test intelligent NumPy array auto-detection."""

    def setUp(self):
        """Clear caches before each test."""
        if HAS_NUMPY:
            datason.clear_caches()

    @unittest.skipUnless(HAS_NUMPY, "NumPy not available")
    def test_1d_array_auto_detection(self):
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
            # Enable auto-detection for this test
            from datason.config import SerializationConfig

            config = SerializationConfig(auto_detect_types=True)
            reconstructed = deserialize_fast(parsed, config=config)

            # Should auto-detect as NumPy array
            self.assertIsInstance(reconstructed, numpy.ndarray)
            numpy.testing.assert_array_equal(reconstructed, original_array)

    @unittest.skipUnless(HAS_NUMPY, "NumPy not available")
    def test_2d_array_auto_detection(self):
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
            # Enable auto-detection for this test
            from datason.config import SerializationConfig

            config = SerializationConfig(auto_detect_types=True)
            reconstructed = deserialize_fast(parsed, config=config)

            # Should auto-detect as NumPy array
            self.assertIsInstance(reconstructed, numpy.ndarray)
            numpy.testing.assert_array_equal(reconstructed, original_array)
            self.assertEqual(reconstructed.shape, original_array.shape)

    @unittest.skipUnless(HAS_NUMPY, "NumPy not available")
    def test_mixed_type_list_no_false_positives(self):
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
            self.assertIsInstance(reconstructed, list)
            self.assertEqual(reconstructed, mixed_list)

    @unittest.skipUnless(HAS_NUMPY, "NumPy not available")
    def test_empty_array_auto_detection(self):
        """Test empty array auto-detection."""
        empty_arrays = [
            numpy.array([]),
            numpy.array([[]]),  # Empty 2D
        ]

        for original_array in empty_arrays:
            # Test without type hints
            serialized = datason.serialize(original_array)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            # Enable auto-detection for this test
            from datason.config import SerializationConfig

            config = SerializationConfig(auto_detect_types=True)
            reconstructed = deserialize_fast(parsed, config=config)

            # Should auto-detect as NumPy array
            self.assertIsInstance(reconstructed, numpy.ndarray)
            numpy.testing.assert_array_equal(reconstructed, original_array)


class TestPandasAutoDetection(unittest.TestCase):
    """Test intelligent Pandas auto-detection."""

    def setUp(self):
        """Clear caches before each test."""
        if HAS_PANDAS:
            datason.clear_caches()

    @unittest.skipUnless(HAS_PANDAS, "Pandas not available")
    def test_dataframe_auto_detection(self):
        """Test DataFrame auto-detection from dict of lists."""
        # Create simple DataFrame
        original_df = pd.DataFrame({"A": [1, 2, 3], "B": [4.1, 5.2, 6.3], "C": ["x", "y", "z"]})

        # Test without type hints (should auto-detect)
        serialized = datason.serialize(original_df)
        json_str = json.dumps(serialized, default=str)
        parsed = json.loads(json_str)
        # Enable auto-detection for this test
        from datason.config import SerializationConfig

        config = SerializationConfig(auto_detect_types=True)
        reconstructed = deserialize_fast(parsed, config=config)

        # Should auto-detect as DataFrame
        self.assertIsInstance(reconstructed, pd.DataFrame)
        pd.testing.assert_frame_equal(reconstructed, original_df)

    @unittest.skipUnless(HAS_PANDAS, "Pandas not available")
    def test_series_auto_detection(self):
        """Test Series auto-detection from lists."""
        test_series = [pd.Series([1, 2, 3, 4, 5]), pd.Series([1.1, 2.2, 3.3, 4.4]), pd.Series(["a", "b", "c", "d"])]

        for original_series in test_series:
            # Test without type hints (should auto-detect)
            serialized = datason.serialize(original_series)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            # Enable auto-detection for this test
            from datason.config import SerializationConfig

            config = SerializationConfig(auto_detect_types=True)
            reconstructed = deserialize_fast(parsed, config=config)

            # Should auto-detect as Series
            self.assertIsInstance(reconstructed, pd.Series)

    @unittest.skipUnless(HAS_PANDAS, "Pandas not available")
    def test_dict_no_false_dataframe_detection(self):
        """Test that regular dicts don't get falsely detected as DataFrames."""
        test_dicts = [
            {"A": [1, 2], "B": "not_list"},  # Mixed value types
            {"single": "value"},  # Single non-list value
            {"A": [1, 2], "B": [3]},  # Different length lists
        ]

        for test_dict in test_dicts:
            # Test without type hints
            serialized = datason.serialize(test_dict)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            # Should remain as dict, NOT become DataFrame
            self.assertIsInstance(reconstructed, dict)
            self.assertEqual(reconstructed, test_dict)


class TestDeserializerErrorHandling(unittest.TestCase):
    """Test deserializer error handling edge cases."""

    def test_invalid_json_structure(self):
        """Test handling of invalid JSON structures."""
        invalid_structures = [
            None,
            "",
            123,  # Not a dict/list
            {"incomplete": True},  # Missing expected fields
        ]

        for invalid_data in invalid_structures:
            # Should handle gracefully, not crash
            result = deserialize_fast(invalid_data)
            # Should return the input unchanged if can't deserialize
            self.assertEqual(result, invalid_data)

    def test_missing_type_hints(self):
        """Test deserialization when type hints are missing."""
        # Data that looks like it could be typed but has no hints
        data_without_hints = {"values": [1, 2, 3, 4, 5], "nested": {"inner": [1.1, 2.2, 3.3]}}

        result = deserialize_fast(data_without_hints)
        # Should remain as basic Python types
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["values"], list)
        self.assertIsInstance(result["nested"], dict)

    def test_corrupted_type_metadata(self):
        """Test handling of corrupted type metadata."""
        corrupted_data = {"__type__": "InvalidType", "__module__": "nonexistent.module", "data": [1, 2, 3]}

        # Should handle gracefully and fall back
        result = deserialize_fast(corrupted_data)
        self.assertIsInstance(result, (dict, list))

    def test_import_error_during_deserialization(self):
        """Test import errors during deserialization."""
        # Mock an ImportError for a deserializer
        fake_data = {"__type__": "SomeClass", "__module__": "fake.module", "data": "test"}

        # Should handle import error gracefully
        result = deserialize_fast(fake_data)
        # Should fall back to dict or handle appropriately
        self.assertIsInstance(result, (dict, str))


class TestTemplateDeserializationEdgeCases(unittest.TestCase):
    """Test template deserialization edge cases."""

    def test_template_with_missing_fields(self):
        """Test template deserialization with missing required fields."""
        # Template expects certain fields but they're missing
        template_data = {
            "name": "test_template",
            "fields": ["field1", "field2"],
            # Missing "field2"
            "field1": "value1",
        }

        # Should handle gracefully
        result = deserialize_fast(template_data)
        self.assertIsInstance(result, dict)

    def test_template_with_extra_fields(self):
        """Test template deserialization with extra unexpected fields."""
        template_data = {
            "expected_field": "value",
            "unexpected_field": "should_be_handled",
            "another_extra": {"nested": "data"},
        }

        # Should handle extra fields gracefully
        result = deserialize_fast(template_data)
        self.assertIsInstance(result, dict)
        self.assertIn("expected_field", result)

    def test_deeply_nested_template(self):
        """Test deeply nested template structures."""
        deep_template = {}
        current = deep_template

        # Create deep nesting
        for i in range(20):
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]
        current["final_value"] = "deep"

        # Should handle deep nesting
        result = deserialize_fast(deep_template)
        self.assertIsInstance(result, dict)


class TestLazyImportsAndHotPath(unittest.TestCase):
    """Test lazy import behavior and hot path optimizations."""

    def test_numpy_lazy_import_fallback(self):
        """Test behavior when NumPy import fails during deserialization."""
        # Mock numpy import failure
        with patch.dict("sys.modules", {"numpy": None}):
            # Data that would normally trigger numpy deserialization
            numpy_like_data = {"__type__": "ndarray", "__module__": "numpy", "data": [[1, 2, 3], [4, 5, 6]]}

            # Should handle gracefully when numpy not available
            result = deserialize_fast(numpy_like_data)
            # Should fall back to list or dict
            self.assertIsInstance(result, (list, dict))

    def test_pandas_lazy_import_fallback(self):
        """Test behavior when Pandas import fails during deserialization."""
        # Mock pandas import failure
        with patch.dict("sys.modules", {"pandas": None}):
            # Data that would normally trigger pandas deserialization
            pandas_like_data = {
                "__type__": "DataFrame",
                "__module__": "pandas",
                "data": {"A": [1, 2, 3], "B": [4, 5, 6]},
            }

            # Should handle gracefully when pandas not available
            result = deserialize_fast(pandas_like_data)
            # Should fall back to dict
            self.assertIsInstance(result, dict)

    def test_hot_path_simple_types(self):
        """Test that simple types use optimized hot paths."""
        simple_data = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        result = deserialize_fast(simple_data)

        # Should be identical (hot path optimization)
        self.assertEqual(result, simple_data)
        self.assertIsInstance(result["string"], str)
        self.assertIsInstance(result["number"], int)
        self.assertIsInstance(result["float"], float)
        self.assertIsInstance(result["boolean"], bool)
        self.assertIsNone(result["null"])

    def test_performance_no_regression_simple_objects(self):
        """Test that simple deserialization remains fast."""
        import time

        simple_data = {"key": "value", "number": 123, "list": [1, 2, 3]}

        # Test multiple deserializations
        start_time = time.time()
        for _ in range(1000):
            result = deserialize_fast(simple_data)
            self.assertEqual(result, simple_data)
        end_time = time.time()

        # Should complete quickly
        self.assertLess(end_time - start_time, 1.0)  # Should be very fast


class TestBasicTypeEnhancements(unittest.TestCase):
    """Test basic type enhancement edge cases."""

    def test_enhanced_dict_deserialization(self):
        """Test enhanced dictionary deserialization."""
        # Dict with mixed types that could benefit from enhancements
        enhanced_dict = {
            "regular_key": "regular_value",
            "numeric_values": [1, 2, 3, 4, 5],
            "nested": {"deeper": {"deep_list": [10, 20, 30]}},
        }

        result = deserialize_fast(enhanced_dict)
        self.assertEqual(result, enhanced_dict)
        self.assertIsInstance(result, dict)

    def test_enhanced_list_deserialization(self):
        """Test enhanced list deserialization."""
        # List with mixed types
        enhanced_list = [1, "string", 3.14, True, None, {"nested": "dict"}, [1, 2, 3]]

        result = deserialize_fast(enhanced_list)
        self.assertEqual(result, enhanced_list)
        self.assertIsInstance(result, list)

    def test_type_coercion_edge_cases(self):
        """Test edge cases in type coercion."""
        # Data that might cause type coercion issues
        coercion_data = {
            "string_numbers": ["1", "2", "3"],
            "mixed_numbers": [1, "2", 3.0],
            "boolean_strings": ["true", "false", "True", "False"],
        }

        result = deserialize_fast(coercion_data)
        self.assertIsInstance(result, dict)
        # Should preserve original types (no unwanted coercion)
        self.assertIsInstance(result["string_numbers"][0], str)
        self.assertIsInstance(result["mixed_numbers"][1], str)
        self.assertIsInstance(result["boolean_strings"][0], str)
