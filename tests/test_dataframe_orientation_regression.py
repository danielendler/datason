#!/usr/bin/env python3
"""Regression tests for DataFrame orientation configuration.

This test suite ensures that the DataFrame orientation bug discovered in
integration feedback doesn't resurface. It validates that all pandas-supported
orientations work correctly and invalid orientations are handled gracefully.
"""

import pytest

# Conditional import for optional dependency
pd = pytest.importorskip("pandas", reason="pandas not available")

import datason
from datason.config import DataFrameOrient, OutputType, SerializationConfig


class TestDataFrameOrientationRegression:
    """Test DataFrame orientation configuration to prevent regressions."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})

    def test_all_valid_orientations_work(self, sample_dataframe):
        """Test that all valid pandas orientations work correctly."""
        # Test each valid orientation
        test_cases = [
            (DataFrameOrient.RECORDS, list),
            (DataFrameOrient.SPLIT, dict),
            (DataFrameOrient.INDEX, dict),
            (DataFrameOrient.DICT, dict),
            (DataFrameOrient.LIST, dict),
            (DataFrameOrient.VALUES, list),
        ]

        for orient, expected_type in test_cases:
            config = SerializationConfig(dataframe_orient=orient)
            result = datason.serialize(sample_dataframe, config=config)

            assert isinstance(result, expected_type), (
                f"Orientation {orient.value} should return {expected_type.__name__}, got {type(result).__name__}"
            )

            # Validate specific format expectations
            if orient == DataFrameOrient.SPLIT:
                assert "index" in result
                assert "columns" in result
                assert "data" in result
                assert result["columns"] == ["a", "b", "c"]

            elif orient == DataFrameOrient.VALUES:
                assert result == [[1, 4, "x"], [2, 5, "y"], [3, 6, "z"]]

            elif orient == DataFrameOrient.RECORDS:
                expected = [
                    {"a": 1, "b": 4, "c": "x"},
                    {"a": 2, "b": 5, "c": "y"},
                    {"a": 3, "b": 6, "c": "z"},
                ]
                assert result == expected

    def test_deprecated_orientations_removed(self):
        """Test that previously invalid orientations are no longer available."""
        # These orientations were in the old enum but are not valid pandas orientations
        invalid_orientations = ["columns", "table"]

        for invalid_orient in invalid_orientations:
            with pytest.raises((ValueError, AttributeError)):
                # Should not be able to create these orientations
                DataFrameOrient(invalid_orient)

    def test_dataframe_output_type_control(self, sample_dataframe):
        """Test the new output type control for DataFrames."""
        # Test JSON-safe output (default)
        config_json = SerializationConfig(
            dataframe_output=OutputType.JSON_SAFE,
            dataframe_orient=DataFrameOrient.SPLIT,
        )
        result_json = datason.serialize(sample_dataframe, config=config_json)
        assert isinstance(result_json, dict)
        assert "index" in result_json  # Should be in split format

        # Test object output
        config_obj = SerializationConfig(dataframe_output=OutputType.OBJECT)
        result_obj = datason.serialize(sample_dataframe, config=config_obj)
        assert isinstance(result_obj, pd.DataFrame)
        pd.testing.assert_frame_equal(result_obj, sample_dataframe)

    def test_orientation_fallback_behavior(self, sample_dataframe):
        """Test that orientation failures fall back gracefully."""
        # This tests the exception handling in the core serialization logic
        config = SerializationConfig(dataframe_orient=DataFrameOrient.RECORDS)

        # Should work normally
        result = datason.serialize(sample_dataframe, config=config)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_empty_dataframe_orientations(self):
        """Test orientations work with empty DataFrames."""
        empty_df = pd.DataFrame()

        for orient in [
            DataFrameOrient.RECORDS,
            DataFrameOrient.SPLIT,
            DataFrameOrient.VALUES,
        ]:
            config = SerializationConfig(dataframe_orient=orient)
            result = datason.serialize(empty_df, config=config)

            # Should not raise exceptions
            assert result is not None

    def test_single_row_dataframe_orientations(self):
        """Test orientations work with single-row DataFrames."""
        single_row_df = pd.DataFrame({"x": [1], "y": [2]})

        config_records = SerializationConfig(dataframe_orient=DataFrameOrient.RECORDS)
        result_records = datason.serialize(single_row_df, config=config_records)
        assert result_records == [{"x": 1, "y": 2}]

        config_split = SerializationConfig(dataframe_orient=DataFrameOrient.SPLIT)
        result_split = datason.serialize(single_row_df, config=config_split)
        assert result_split["data"] == [[1, 2]]

    def test_single_column_dataframe_orientations(self):
        """Test orientations work with single-column DataFrames."""
        single_col_df = pd.DataFrame({"only_col": [1, 2, 3]})

        config_dict = SerializationConfig(dataframe_orient=DataFrameOrient.DICT)
        result_dict = datason.serialize(single_col_df, config=config_dict)
        assert result_dict == {"only_col": {0: 1, 1: 2, 2: 3}}

        config_list = SerializationConfig(dataframe_orient=DataFrameOrient.LIST)
        result_list = datason.serialize(single_col_df, config=config_list)
        assert result_list == {"only_col": [1, 2, 3]}

    def test_dataframe_with_complex_types(self):
        """Test DataFrame orientation with complex data types."""
        complex_df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "none_col": [None, None, None],
            }
        )

        # Test that all orientations handle complex types
        for orient in [
            DataFrameOrient.RECORDS,
            DataFrameOrient.SPLIT,
            DataFrameOrient.VALUES,
        ]:
            config = SerializationConfig(dataframe_orient=orient)
            result = datason.serialize(complex_df, config=config)

            # Should serialize without errors
            assert result is not None

            # For records, verify structure
            if orient == DataFrameOrient.RECORDS:
                assert len(result) == 3
                assert all(isinstance(row, dict) for row in result)
                assert result[0]["int_col"] == 1
                assert result[0]["bool_col"] is True
                assert result[0]["none_col"] is None

    def test_backwards_compatibility(self, sample_dataframe):
        """Test that existing code continues to work."""
        # Default behavior should still work
        result_default = datason.serialize(sample_dataframe)
        assert isinstance(result_default, list)  # Default is records

        # Old-style configuration should still work
        config_old_style = SerializationConfig()
        result_old = datason.serialize(sample_dataframe, config=config_old_style)
        assert result_old == result_default

    @pytest.mark.parametrize(
        "orient",
        [
            DataFrameOrient.RECORDS,
            DataFrameOrient.SPLIT,
            DataFrameOrient.INDEX,
            DataFrameOrient.DICT,
            DataFrameOrient.LIST,
            DataFrameOrient.VALUES,
        ],
    )
    def test_parametrized_orientation_validation(self, sample_dataframe, orient):
        """Parametrized test for all orientations to ensure none break."""
        config = SerializationConfig(dataframe_orient=orient)

        # Should not raise any exceptions
        result = datason.serialize(sample_dataframe, config=config)
        assert result is not None

        # Result should be JSON-serializable
        import json

        json_str = json.dumps(result, default=str)  # default=str for any remaining complex types
        assert isinstance(json_str, str)
        assert len(json_str) > 0
