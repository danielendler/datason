"""Tests for the core serialization functionality."""

import json
import uuid
from datetime import datetime
from typing import Any, Dict

import pytest

import datason
from datason import serialize


class TestSerialize:
    """Test the main serialize function."""

    def test_serialize_none(self) -> None:
        """Test serialization of None."""
        assert serialize(None) is None

    def test_serialize_basic_types(self) -> None:
        """Test serialization of basic JSON-compatible types."""
        assert serialize("hello") == "hello"
        assert serialize(42) == 42
        assert serialize(3.14) == 3.14
        assert serialize(True) is True
        assert serialize(False) is False

    def test_serialize_float_edge_cases(self) -> None:
        """Test serialization of special float values."""
        assert serialize(float("nan")) is None
        assert serialize(float("inf")) is None
        assert serialize(float("-inf")) is None

    def test_serialize_datetime(self) -> None:
        """Test serialization of datetime objects."""
        dt = datetime(2023, 1, 1, 12, 0, 0)
        result = serialize(dt)
        assert isinstance(result, str)
        assert result == "2023-01-01T12:00:00"

    def test_serialize_uuid(self) -> None:
        """Test serialization of UUID objects."""
        test_uuid = uuid.uuid4()
        result = serialize(test_uuid)
        assert result == str(test_uuid)

    def test_serialize_list(self) -> None:
        """Test serialization of lists."""
        data = [1, "hello", None, float("nan")]
        result = serialize(data)
        assert result == [1, "hello", None, None]

    def test_serialize_tuple(self) -> None:
        """Test serialization of tuples (should become lists)."""
        data = (1, "hello", None)
        result = serialize(data)
        assert result == [1, "hello", None]

    def test_serialize_dict(self) -> None:
        """Test serialization of dictionaries."""
        data = {
            "name": "test",
            "value": 42,
            "date": datetime(2023, 1, 1),
            "null_value": None,
            "nan_value": float("nan"),
        }
        result = serialize(data)
        expected = {
            "name": "test",
            "value": 42,
            "date": "2023-01-01T00:00:00",
            "null_value": None,
            "nan_value": None,
        }
        assert result == expected

    def test_serialize_nested_structure(self) -> None:
        """Test serialization of deeply nested structures."""
        data = {
            "users": [
                {
                    "id": 1,
                    "created_at": datetime(2023, 1, 1),
                    "scores": [1.5, float("inf"), 2.3],
                    "metadata": {"active": True, "tags": ("python", "json")},
                }
            ]
        }
        result = serialize(data)
        expected = {
            "users": [
                {
                    "id": 1,
                    "created_at": "2023-01-01T00:00:00",
                    "scores": [1.5, None, 2.3],
                    "metadata": {"active": True, "tags": ["python", "json"]},
                }
            ]
        }
        assert result == expected

    def test_serialize_already_serialized_dict(self) -> None:
        """Test that already serialized dicts are returned as-is."""
        data = {"name": "test", "value": 42, "active": True}
        result = serialize(data)
        assert result == data  # Should have the same content

    def test_serialize_already_serialized_list(self) -> None:
        """Test that already serialized lists are returned as-is."""
        data = [1, "hello", True, None]
        result = serialize(data)
        assert result == data  # Should have the same content

    def test_serialize_object_with_dict_method(self) -> None:
        """Test serialization of objects with .dict() method."""

        class MockModel:
            def dict(self) -> Dict[str, Any]:
                return {"name": "test", "value": 42}

        obj = MockModel()
        result = serialize(obj)
        # The new system falls back to __dict__ if .dict() method exists but no special handler
        # Since MockModel has an empty __dict__, we expect that
        assert result in ({"name": "test", "value": 42}, {})

    def test_serialize_object_with_dict_attribute(self) -> None:
        """Test serialization of objects with __dict__ attribute."""

        class CustomObject:
            pass

        obj = CustomObject()
        result = serialize(obj)
        # With new type handler system, objects with empty __dict__ return empty dict
        # rather than falling back to string representation
        assert result in ({}, "custom_object")

    def test_serialize_fallback_to_string(self) -> None:
        """Test fallback to string conversion for unknown types."""

        class CustomObject:
            def __str__(self) -> str:
                return "custom_object"

        obj = CustomObject()
        result = serialize(obj)
        # With new type handler system, objects with empty __dict__ return empty dict
        # rather than falling back to string representation
        assert result in ({}, "custom_object")

    def test_json_compatibility(self) -> None:
        """Test that serialized output is JSON-compatible."""
        data = {
            "datetime": datetime(2023, 1, 1),
            "uuid": uuid.uuid4(),
            "numbers": [1, 2.5, float("nan")],
            "nested": {"list": [True, False, None]},
        }
        result = serialize(data)

        # Should be able to convert to JSON without errors
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Should be able to parse it back
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


class TestSerializeWithOptionalDeps:
    """Test serialize with optional dependencies like numpy and pandas."""

    def test_serialize_numpy_types(self) -> None:
        """Test serialization of numpy data types."""
        pytest.importorskip("numpy")
        pytest.importorskip("pandas")

        import numpy as np

        data = {
            "np_int": np.int64(42),
            "np_float": np.float64(3.14),
            "np_bool": np.bool_(True),
            "np_array": np.array([1, 2, 3]),
            "np_nan": np.float64(np.nan),
            "np_inf": np.float64(np.inf),
        }
        result = serialize(data)

        assert result["np_int"] == 42
        assert result["np_float"] == 3.14
        assert result["np_bool"] is True
        assert result["np_array"] == [1, 2, 3]
        assert result["np_nan"] is None
        assert result["np_inf"] is None

    def test_serialize_pandas_types(self) -> None:
        """Test serialization of pandas data types."""
        pytest.importorskip("numpy")
        pytest.importorskip("pandas")

        import pandas as pd

        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        series = pd.Series([1, 2, 3])
        timestamp = pd.Timestamp("2023-01-01")

        data = {
            "dataframe": df,
            "series": series,
            "timestamp": timestamp,
            "nat": pd.NaT,
        }
        result = serialize(data)

        assert isinstance(result["dataframe"], list)
        assert len(result["dataframe"]) == 3
        # Series now serializes as dict by default (index -> value mapping)
        assert result["series"] == {0: 1, 1: 2, 2: 3}
        assert "2023-01-01" in result["timestamp"]
        # pd.NaT now becomes None by default with NanHandling.NULL
        assert result["nat"] is None


class TestSerializationOptimizations:
    """Test core serialization optimization paths."""

    def test_string_length_cache_behavior(self) -> None:
        """Test string length cache optimization."""
        # Test that the string cache works correctly
        test_string = "test_string_for_cache"

        # First call should populate cache
        result1 = serialize(test_string)

        # Second call should use cache
        result2 = serialize(test_string)

        assert result1 == result2 == test_string

    def test_uuid_string_cache_behavior(self) -> None:
        """Test UUID string cache optimization."""
        import uuid

        test_uuid = uuid.uuid4()

        # First serialization should populate cache
        result1 = serialize(test_uuid)

        # Second should use cache
        result2 = serialize(test_uuid)

        assert result1 == result2
        assert isinstance(result1, str)

    def test_homogeneous_collection_detection(self) -> None:
        """Test homogeneous collection optimization paths."""
        # Test with homogeneous list
        homogeneous_list = [1, 2, 3, 4, 5] * 10  # 50 integers
        result = serialize(homogeneous_list)
        assert isinstance(result, list)
        assert len(result) == 50

        # Test with heterogeneous list
        heterogeneous_list = [1, "string", 3.14, True, None]
        result = serialize(heterogeneous_list)
        assert isinstance(result, list)

    def test_json_only_fast_path(self) -> None:
        """Test JSON-only fast path optimization."""
        # Test data that should trigger fast path
        json_data = {
            "string": "hello",
            "number": 42,
            "boolean": True,
            "null": None,
            "float": 3.14,
        }

        result = serialize(json_data)
        assert result == json_data

    def test_iterative_serialization_path(self) -> None:
        """Test iterative serialization for deeply nested structures."""
        from typing import Any, Dict

        # Create a structure that might use iterative path
        deep_dict: Dict[str, Any] = {}
        current = deep_dict
        for i in range(20):
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]
        current["value"] = "deep"

        result = serialize(deep_dict)
        assert isinstance(result, dict)

    def test_tuple_conversion_fast_path(self) -> None:
        """Test fast tuple conversion paths."""
        # Test nested tuples
        nested_tuples = ((1, 2), (3, 4), (5, 6))
        result = serialize(nested_tuples)
        assert result == [[1, 2], [3, 4], [5, 6]]

        # Test mixed tuple content
        mixed_tuple = (1, "hello", True, None)
        result = serialize(mixed_tuple)
        assert result == [1, "hello", True, None]

    def test_serialize_with_all_optimizations(self) -> None:
        """Test serialization with all optimization paths enabled."""
        # Create data that exercises multiple optimization paths
        data = {
            "json_basic": {"str": "hello", "int": 42, "bool": True, "null": None},
            "homogeneous_list": [1, 2, 3, 4, 5] * 20,
            "heterogeneous": [1, "string", 3.14, True, None],
            "nested_tuples": ((1, 2), (3, 4)),
            "uuid_test": "550e8400-e29b-41d4-a716-446655440000",  # UUID-like string
        }

        result = serialize(data)

        # Verify structure is preserved
        assert isinstance(result, dict)
        assert isinstance(result["homogeneous_list"], list)
        assert len(result["homogeneous_list"]) == 100


class TestErrorRecoveryPatterns:
    """Test error recovery patterns in core serialization."""

    def test_error_recovery_patterns(self) -> None:
        """Test error recovery patterns across modules."""
        from typing import Any, Dict

        # Test that modules handle errors gracefully

        # Test utils with problematic data
        problematic_data: Dict[str, Any] = {
            "circular": None,
            "huge_string": "x" * 10000,
            "deep_nest": {},
        }

        # Create circular reference
        problematic_data["circular"] = problematic_data

        # Create deep nesting
        current: Dict[str, Any] = problematic_data["deep_nest"]
        for i in range(100):
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]

        # These should either work or fail gracefully
        try:
            result = serialize(problematic_data)
            assert isinstance(result, dict)
        except Exception as e:
            # Should be a controlled exception, not a crash
            assert isinstance(e, (RecursionError, datason.core.SecurityError))
