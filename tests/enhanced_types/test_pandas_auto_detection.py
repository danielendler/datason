"""Pandas DataFrame auto-detection tests for datason.

This module tests intelligent auto-detection of pandas DataFrames and Series
from serialized data without requiring type hints.
"""

import json

import pytest

import datason
from datason.config import SerializationConfig
from datason.deserializers import deserialize_fast

pandas = pytest.importorskip("pandas", reason="Pandas not available")


class TestPandasAutoDetection:
    """Test intelligent pandas DataFrame and Series auto-detection."""

    def setup_method(self) -> None:
        """Clear caches before each test to ensure clean state."""
        datason.clear_caches()

    def test_dataframe_from_records_auto_detection(self) -> None:
        """Test DataFrame auto-detection from list of records."""
        # Create test DataFrames
        test_dataframes = [
            pandas.DataFrame(
                [
                    {"name": "Alice", "age": 25, "city": "NYC"},
                    {"name": "Bob", "age": 30, "city": "LA"},
                    {"name": "Charlie", "age": 35, "city": "Chicago"},
                ]
            ),
            pandas.DataFrame(
                [{"x": 1, "y": 2.5, "z": True}, {"x": 2, "y": 3.7, "z": False}, {"x": 3, "y": 1.2, "z": True}]
            ),
            pandas.DataFrame(
                [
                    {"product": "A", "sales": 100, "profit": 20.5},
                    {"product": "B", "sales": 150, "profit": 35.2},
                    {"product": "C", "sales": 200, "profit": 45.8},
                ]
            ),
        ]

        for original_df in test_dataframes:
            # Test without type hints (should auto-detect)
            serialized = datason.serialize(original_df)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            # Should auto-detect as DataFrame
            assert isinstance(reconstructed, pandas.DataFrame), "DataFrame not auto-detected"
            pandas.testing.assert_frame_equal(reconstructed, original_df)

    def test_series_auto_detection(self) -> None:
        """Test Series auto-detection from appropriate data structures."""
        # Create test Series
        test_series = [
            pandas.Series([1, 2, 3, 4, 5], name="numbers"),
            pandas.Series(["a", "b", "c", "d"], name="letters"),
            pandas.Series([1.1, 2.2, 3.3], name="floats"),
            pandas.Series([True, False, True, False], name="bools"),
        ]

        for original_series in test_series:
            # Test without type hints (should auto-detect)
            serialized = datason.serialize(original_series)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            # Should auto-detect as Series
            assert isinstance(reconstructed, pandas.Series), f"Series not auto-detected: {original_series.name}"
            pandas.testing.assert_series_equal(reconstructed, original_series)

    def test_empty_dataframe_auto_detection(self) -> None:
        """Test empty DataFrame auto-detection."""
        empty_dataframes = [
            pandas.DataFrame(),
            pandas.DataFrame(columns=["a", "b", "c"]),
            pandas.DataFrame({"x": [], "y": [], "z": []}),
        ]

        for original_df in empty_dataframes:
            # Test without type hints
            serialized = datason.serialize(original_df)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            # Should auto-detect as DataFrame
            assert isinstance(reconstructed, pandas.DataFrame), "Empty DataFrame not auto-detected"
            pandas.testing.assert_frame_equal(reconstructed, original_df)

    def test_no_false_positives_for_regular_lists(self) -> None:
        """Test that regular lists don't get falsely detected as DataFrames."""
        regular_data = [
            [1, 2, 3, 4, 5],  # Simple list
            [{"a": 1}, {"b": 2}],  # Different keys - not DataFrame
            [{"a": 1, "b": 2}, {"a": 3}],  # Inconsistent schema
            [{"nested": {"deep": "value"}}],  # Complex nested objects
            [],  # Empty list
        ]

        for data in regular_data:
            # Test without type hints
            serialized = datason.serialize(data)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            # Should remain as list or dict, NOT become DataFrame
            assert not isinstance(reconstructed, pandas.DataFrame), f"False positive DataFrame detection: {data}"

    def test_metadata_still_works_perfectly(self) -> None:
        """Test that type hints still provide perfect reconstruction."""
        test_dataframes = [
            pandas.DataFrame({"a": [1, 2, 3], "b": [4.5, 5.6, 6.7]}),
            pandas.DataFrame({"x": ["hello", "world"], "y": [True, False]}),
        ]

        config = SerializationConfig(include_type_hints=True)

        for original_df in test_dataframes:
            # Test with type hints (should be perfect)
            serialized = datason.serialize(original_df, config=config)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed, config=config)

            # Should be exact reconstruction
            assert isinstance(reconstructed, pandas.DataFrame)
            pandas.testing.assert_frame_equal(reconstructed, original_df)

    def test_large_dataframes_auto_detection(self) -> None:
        """Test auto-detection works with larger DataFrames."""
        # Create larger test DataFrames
        large_data = []
        for i in range(100):
            large_data.append({"id": i, "value": i * 2.5, "category": f"cat_{i % 5}", "active": i % 2 == 0})

        original_df = pandas.DataFrame(large_data)

        # Test without type hints
        serialized = datason.serialize(original_df)
        json_str = json.dumps(serialized, default=str)
        parsed = json.loads(json_str)
        reconstructed = deserialize_fast(parsed)

        # Should auto-detect as DataFrame
        assert isinstance(reconstructed, pandas.DataFrame), "Large DataFrame not auto-detected"
        pandas.testing.assert_frame_equal(reconstructed, original_df)

    def test_mixed_types_dataframe_auto_detection(self) -> None:
        """Test auto-detection with mixed data types."""
        original_df = pandas.DataFrame(
            [
                {"str_col": "hello", "int_col": 42, "float_col": 3.14, "bool_col": True},
                {"str_col": "world", "int_col": 24, "float_col": 2.71, "bool_col": False},
                {"str_col": "test", "int_col": 99, "float_col": 1.41, "bool_col": True},
            ]
        )

        # Test without type hints
        serialized = datason.serialize(original_df)
        json_str = json.dumps(serialized, default=str)
        parsed = json.loads(json_str)
        reconstructed = deserialize_fast(parsed)

        # Should auto-detect as DataFrame
        assert isinstance(reconstructed, pandas.DataFrame), "Mixed types DataFrame not auto-detected"
        pandas.testing.assert_frame_equal(reconstructed, original_df)

    def test_performance_no_regression(self) -> None:
        """Test that auto-detection doesn't significantly impact performance."""
        # Create regular data that should NOT be detected as DataFrame
        regular_data = [
            {"key1": "value1", "key2": "value2"},  # Single dict
            [1, 2, 3, 4, 5],  # Simple list
            {"nested": {"data": "structure"}},  # Nested dict
        ]

        # Test multiple times to check for performance issues
        for data in regular_data:
            for _ in range(10):
                serialized = datason.serialize(data)
                json_str = json.dumps(serialized, default=str)
                parsed = json.loads(json_str)
                reconstructed = deserialize_fast(parsed)

                # Should remain as original type
                assert not isinstance(reconstructed, pandas.DataFrame)
                assert not isinstance(reconstructed, pandas.Series)
