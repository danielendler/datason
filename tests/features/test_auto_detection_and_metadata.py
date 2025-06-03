#!/usr/bin/env python3
"""Tests for auto-detection and metadata capabilities.

This module tests the automatic detection of data types and the inclusion
of metadata for roundtrip serialization/deserialization.
"""

import json
import uuid
from datetime import datetime

import pytest

import datason
from datason.config import SerializationConfig

# Conditional imports for optional dependencies
pd = pytest.importorskip("pandas", reason="pandas not available")
np = pytest.importorskip("numpy", reason="numpy not available")


class TestAutoDetectionDeserialization:
    """Test the new auto-detection deserialization feature."""

    def test_auto_deserialize_basic_types(self):
        """Test auto-detection for basic types."""
        # Should detect datetime strings
        result = datason.auto_deserialize("2023-01-01T12:00:00")
        assert isinstance(result, datetime)
        assert result.year == 2023

        # Should detect UUID strings
        uuid_str = "12345678-1234-5678-9012-123456789abc"
        result = datason.auto_deserialize(uuid_str)
        assert isinstance(result, uuid.UUID)
        assert str(result) == uuid_str

        # Basic types should remain unchanged
        assert datason.auto_deserialize(42) == 42
        assert datason.auto_deserialize("hello") == "hello"
        assert datason.auto_deserialize(True) is True

    def test_auto_deserialize_aggressive_mode(self):
        """Test aggressive auto-detection mode."""
        # Should detect numbers in aggressive mode
        result = datason.auto_deserialize("123", aggressive=True)
        assert result == 123
        assert isinstance(result, int)

        result = datason.auto_deserialize("123.45", aggressive=True)
        assert result == 123.45
        assert isinstance(result, float)

        # Should detect booleans in aggressive mode
        assert datason.auto_deserialize("true", aggressive=True) is True
        assert datason.auto_deserialize("false", aggressive=True) is False

        # Scientific notation
        result = datason.auto_deserialize("1.23e-4", aggressive=True)
        assert isinstance(result, float)
        assert abs(result - 1.23e-4) < 1e-10

    def test_auto_deserialize_dataframe_detection(self):
        """Test aggressive DataFrame detection."""
        # Create data that looks like a DataFrame
        data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]}

        result = datason.auto_deserialize(data, aggressive=True)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b", "c"]
        assert len(result) == 3

    def test_auto_deserialize_split_format_detection(self):
        """Test detection of pandas split format."""
        split_data = {
            "index": [0, 1, 2],
            "columns": ["a", "b", "c"],
            "data": [[1, 4, "x"], [2, 5, "y"], [3, 6, "z"]],
        }

        result = datason.auto_deserialize(split_data)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b", "c"]
        assert len(result) == 3

    def test_auto_deserialize_series_detection(self):
        """Test aggressive Series detection."""
        # Numeric data that could be a Series
        data = [1, 2, 3, 4, 5]

        result = datason.auto_deserialize(data, aggressive=True)
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_auto_deserialize_nested_structures(self):
        """Test auto-detection in nested structures."""
        data = {
            "timestamp": "2023-01-01T12:00:00",
            "user_id": "12345678-1234-5678-9012-123456789abc",
            "data": {
                "numbers": [
                    "1",
                    "2",
                    "3",
                ],  # Should be detected as numbers in aggressive mode
                "flags": ["true", "false"],  # Should be detected as booleans
            },
        }

        result = datason.auto_deserialize(data, aggressive=True)

        # Check top-level conversions
        assert isinstance(result["timestamp"], datetime)
        assert isinstance(result["user_id"], uuid.UUID)

        # Check nested conversions (convert Series to list for comparison)
        numbers = result["data"]["numbers"]
        if isinstance(numbers, pd.Series):
            numbers = numbers.tolist()
        assert numbers == [1, 2, 3]

        flags = result["data"]["flags"]
        if isinstance(flags, pd.Series):
            flags = flags.tolist()
        assert flags == [True, False]


class TestTypeMetadataSupport:
    """Test the new type metadata support for round-trip serialization."""

    def test_serialize_with_type_hints_datetime(self):
        """Test datetime serialization with type hints."""
        dt = datetime(2023, 1, 1, 12, 0, 0)
        config = SerializationConfig(include_type_hints=True)

        result = datason.serialize(dt, config=config)

        assert isinstance(result, dict)
        assert result["__datason_type__"] == "datetime"
        assert result["__datason_value__"] == dt.isoformat()

    def test_serialize_with_type_hints_uuid(self):
        """Test UUID serialization with type hints."""
        test_uuid = uuid.uuid4()
        config = SerializationConfig(include_type_hints=True)

        result = datason.serialize(test_uuid, config=config)

        assert isinstance(result, dict)
        assert result["__datason_type__"] == "uuid.UUID"
        assert result["__datason_value__"] == str(test_uuid)

    def test_serialize_with_type_hints_dataframe(self):
        """Test DataFrame serialization with type hints."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        config = SerializationConfig(include_type_hints=True)

        result = datason.serialize(df, config=config)

        assert isinstance(result, dict)
        assert result["__datason_type__"] == "pandas.DataFrame"
        # The value should be the serialized DataFrame
        assert isinstance(result["__datason_value__"], list)

    def test_serialize_with_type_hints_series(self):
        """Test Series serialization with type hints."""
        series = pd.Series([1, 2, 3, 4])
        config = SerializationConfig(include_type_hints=True)

        result = datason.serialize(series, config=config)

        assert isinstance(result, dict)
        assert result["__datason_type__"] == "pandas.Series"
        assert isinstance(result["__datason_value__"], dict)

    def test_serialize_with_type_hints_set(self):
        """Test set serialization with type hints."""
        test_set = {1, 2, 3}
        config = SerializationConfig(include_type_hints=True)

        result = datason.serialize(test_set, config=config)

        assert isinstance(result, dict)
        assert result["__datason_type__"] == "set"
        assert isinstance(result["__datason_value__"], list)
        assert set(result["__datason_value__"]) == test_set

    def test_serialize_with_type_hints_tuple(self):
        """Test tuple serialization with type hints."""
        test_tuple = (1, 2, 3)
        config = SerializationConfig(include_type_hints=True)

        result = datason.serialize(test_tuple, config=config)

        assert isinstance(result, dict)
        assert result["__datason_type__"] == "tuple"
        assert result["__datason_value__"] == [1, 2, 3]

    def test_serialize_with_type_hints_numpy_array(self):
        """Test numpy array serialization with type hints."""
        arr = np.array([1, 2, 3, 4])
        config = SerializationConfig(include_type_hints=True)

        result = datason.serialize(arr, config=config)

        assert isinstance(result, dict)
        assert result["__datason_type__"] == "numpy.ndarray"
        assert result["__datason_value__"] == [1, 2, 3, 4]


class TestRoundTripSerialization:
    """Test perfect round-trip serialization using type metadata."""

    def test_round_trip_datetime(self):
        """Test perfect round-trip for datetime objects."""
        original = datetime(2023, 1, 1, 12, 30, 45, 123456)
        config = SerializationConfig(include_type_hints=True)

        # Serialize with type hints
        serialized = datason.serialize(original, config=config)

        # Deserialize (should automatically detect metadata)
        restored = datason.deserialize(serialized)

        assert isinstance(restored, datetime)
        assert restored == original

    def test_round_trip_uuid(self):
        """Test perfect round-trip for UUID objects."""
        original = uuid.uuid4()
        config = SerializationConfig(include_type_hints=True)

        serialized = datason.serialize(original, config=config)
        restored = datason.deserialize(serialized)

        assert isinstance(restored, uuid.UUID)
        assert restored == original

    def test_round_trip_dataframe(self):
        """Test perfect round-trip for DataFrame objects."""
        original = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
            }
        )
        config = SerializationConfig(include_type_hints=True)

        serialized = datason.serialize(original, config=config)
        restored = datason.deserialize(serialized)

        assert isinstance(restored, pd.DataFrame)
        pd.testing.assert_frame_equal(restored, original)

    def test_round_trip_series(self):
        """Test perfect round-trip for Series objects."""
        original = pd.Series([1, 2, 3, 4], name="test_series")
        config = SerializationConfig(include_type_hints=True)

        serialized = datason.serialize(original, config=config)
        restored = datason.deserialize(serialized)

        assert isinstance(restored, pd.Series)
        pd.testing.assert_series_equal(restored, original)

    def test_round_trip_set(self):
        """Test perfect round-trip for set objects."""
        original = {1, 2, 3, "hello", "world"}
        config = SerializationConfig(include_type_hints=True)

        serialized = datason.serialize(original, config=config)
        restored = datason.deserialize(serialized)

        assert isinstance(restored, set)
        assert restored == original

    def test_round_trip_tuple(self):
        """Test perfect round-trip for tuple objects."""
        original = (1, "hello", 3.14, True)
        config = SerializationConfig(include_type_hints=True)

        serialized = datason.serialize(original, config=config)
        restored = datason.deserialize(serialized)

        assert isinstance(restored, tuple)
        assert restored == original

    def test_round_trip_numpy_array(self):
        """Test perfect round-trip for numpy arrays."""
        original = np.array([1, 2, 3, 4, 5])
        config = SerializationConfig(include_type_hints=True)

        serialized = datason.serialize(original, config=config)
        restored = datason.deserialize(serialized)

        assert isinstance(restored, np.ndarray)
        np.testing.assert_array_equal(restored, original)

    def test_round_trip_complex_nested_structure(self) -> None:
        """Test round-trip for complex nested data with multiple types."""
        original = {
            "timestamp": datetime(2023, 1, 1, 12, 0, 0),
            "user_id": uuid.uuid4(),
            "data": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            "tags": {"python", "datason", "serialization"},
            "coordinates": (1.23, 4.56),
            "metrics": np.array([0.1, 0.2, 0.3]),
        }

        config = SerializationConfig(include_type_hints=True)

        serialized = datason.serialize(original, config=config)
        restored = datason.deserialize(serialized)

        # Check all types are preserved
        assert isinstance(restored["timestamp"], datetime)
        assert isinstance(restored["user_id"], uuid.UUID)
        assert isinstance(restored["data"], pd.DataFrame)
        assert isinstance(restored["tags"], set)
        assert isinstance(restored["coordinates"], tuple)
        assert isinstance(restored["metrics"], np.ndarray)

        # Check values are preserved
        assert restored["timestamp"] == original["timestamp"]
        assert restored["user_id"] == original["user_id"]
        pd.testing.assert_frame_equal(restored["data"], original["data"])
        assert restored["tags"] == original["tags"]
        assert restored["coordinates"] == original["coordinates"]
        np.testing.assert_array_equal(restored["metrics"], original["metrics"])


class TestIntegrationScenarios:
    """Test scenarios based on the integration feedback."""

    def test_user_integration_scenario_with_auto_detection(self):
        """Test the exact scenario from the integration feedback."""
        # Simulate user's financial modeling data
        data = {
            "portfolio": {
                "assets": [
                    {"symbol": "AAPL", "shares": "100", "price": "150.25"},
                    {"symbol": "GOOGL", "shares": "50", "price": "2500.50"},
                ],
                "last_updated": "2023-12-01T15:30:00Z",
            },
            "user_id": "12345678-1234-5678-9012-123456789abc",
        }

        # With auto-detection, should intelligently convert types
        result = datason.auto_deserialize(data, aggressive=True)

        # Check conversions
        assert isinstance(result["user_id"], uuid.UUID)
        assert isinstance(result["portfolio"]["last_updated"], datetime)

        # Shares and prices should be converted to numbers
        assets = result["portfolio"]["assets"]
        assert isinstance(assets[0]["shares"], int)
        assert assets[0]["shares"] == 100
        assert isinstance(assets[0]["price"], float)
        assert abs(assets[0]["price"] - 150.25) < 0.001

    def test_round_trip_production_workflow(self) -> None:
        """Test a complete round-trip workflow for production scenarios."""
        # Use fixed values to avoid flakiness between test runs
        fixed_datetime = datetime(2023, 12, 1, 15, 30, 45, 123456)
        fixed_uuid = uuid.UUID("12345678-1234-5678-9012-123456789abc")

        # Original data with various types (similar to user's use case)
        original_data = {
            "analysis_date": fixed_datetime,
            "request_id": fixed_uuid,
            "results": pd.DataFrame({"metric": ["roi", "volatility", "sharpe"], "value": [0.15, 0.25, 1.2]}),
            "parameters": {
                "lookback_days": 252,
                "confidence_level": 0.95,
                "risk_free_rate": 0.02,
            },
            "tags": {"quarterly", "equity", "us-market"},
        }

        # Serialize with type metadata for perfect round-trip
        config = SerializationConfig(include_type_hints=True)
        serialized = datason.serialize(original_data, config=config)

        # Convert to JSON and back (simulating storage/transmission)
        json_str = json.dumps(serialized, default=str)
        parsed_back = json.loads(json_str)

        # Deserialize back to original types
        restored_data = datason.deserialize(parsed_back)

        # Verify perfect restoration
        assert isinstance(restored_data["analysis_date"], datetime)
        assert isinstance(restored_data["request_id"], uuid.UUID)
        assert isinstance(restored_data["results"], pd.DataFrame)
        assert isinstance(restored_data["tags"], set)

        # Values should be preserved
        assert restored_data["analysis_date"] == original_data["analysis_date"]
        assert restored_data["request_id"] == original_data["request_id"]
        pd.testing.assert_frame_equal(restored_data["results"], original_data["results"])
        assert restored_data["tags"] == original_data["tags"]
        assert restored_data["parameters"] == original_data["parameters"]
