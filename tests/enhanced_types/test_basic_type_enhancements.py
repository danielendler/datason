"""Enhanced basic type support tests for datason.

This module tests improvements to basic type round-trip support,
focusing on auto-detection and intelligent fallbacks.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest

import datason
from datason.config import SerializationConfig
from datason.deserializers import _clear_deserialization_caches, deserialize_fast


class TestBasicTypeEnhancements:
    """Test enhanced basic type support."""

    def setup_method(self):
        """Clear caches before each test to ensure clean state."""
        _clear_deserialization_caches()

    def test_uuid_auto_detection_comprehensive(self):
        """Test UUID auto-detection with various UUID formats."""
        test_uuids = [
            uuid.UUID("12345678-1234-5678-9012-123456789abc"),  # Fixed UUID
            uuid.uuid4(),  # Random UUID
            uuid.UUID("00000000-0000-0000-0000-000000000000"),  # All zeros
            uuid.UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"),  # All F's
        ]

        for test_uuid in test_uuids:
            # Test without type hints (should auto-detect)
            serialized = datason.serialize(test_uuid)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            assert isinstance(reconstructed, uuid.UUID), f"UUID not detected: {test_uuid}"
            assert reconstructed == test_uuid, f"UUID value mismatch: {test_uuid}"

    def test_datetime_auto_detection_comprehensive(self):
        """Test datetime auto-detection with various formats."""
        test_datetimes = [
            datetime(2023, 1, 1, 12, 0, 0),
            datetime(2023, 12, 31, 23, 59, 59, 999999),
            datetime.now(),
        ]

        for test_dt in test_datetimes:
            # Test without type hints (should auto-detect)
            serialized = datason.serialize(test_dt)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            assert isinstance(reconstructed, datetime), f"Datetime not detected: {test_dt}"
            # Compare with microsecond precision (JSON may lose some precision)
            assert abs((reconstructed - test_dt).total_seconds()) < 1, f"Datetime value mismatch: {test_dt}"

    def test_path_auto_detection_comprehensive(self):
        """Test Path auto-detection with various path formats."""
        test_paths = [
            Path("/home/test.txt"),
            Path("/tmp/data.json"),
            Path("./relative/path.py"),
            Path("../parent/file.csv"),
        ]

        for test_path in test_paths:
            # Test without type hints (should auto-detect)
            serialized = datason.serialize(test_path)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            assert isinstance(reconstructed, Path), f"Path not detected: {test_path}"
            assert reconstructed == test_path, f"Path value mismatch: {test_path}"

    def test_set_requires_type_hints(self):
        """Test that sets require type hints for proper reconstruction."""
        test_set = {"python", "datason", "ml"}

        # Without type hints: set → list
        serialized = datason.serialize(test_set)
        json_str = json.dumps(serialized, default=str)
        parsed = json.loads(json_str)
        reconstructed = deserialize_fast(parsed)

        assert isinstance(reconstructed, list), "Set should become list without type hints"
        assert set(reconstructed) == test_set, "Set values should be preserved"

        # With type hints: perfect round-trip
        config = SerializationConfig(include_type_hints=True)
        serialized_with_hints = datason.serialize(test_set, config=config)
        json_str_with_hints = json.dumps(serialized_with_hints, default=str)
        parsed_with_hints = json.loads(json_str_with_hints)
        reconstructed_with_hints = deserialize_fast(parsed_with_hints, config=config)

        assert isinstance(reconstructed_with_hints, set), "Set should be preserved with type hints"
        assert reconstructed_with_hints == test_set, "Set should round-trip perfectly"

    def test_tuple_requires_type_hints(self):
        """Test that tuples require type hints for proper reconstruction."""
        test_tuple = (1, "hello", 3.14)

        # Without type hints: tuple → list
        serialized = datason.serialize(test_tuple)
        json_str = json.dumps(serialized, default=str)
        parsed = json.loads(json_str)
        reconstructed = deserialize_fast(parsed)

        assert isinstance(reconstructed, list), "Tuple should become list without type hints"
        assert list(test_tuple) == reconstructed, "Tuple values should be preserved"

        # With type hints: perfect round-trip
        config = SerializationConfig(include_type_hints=True)
        serialized_with_hints = datason.serialize(test_tuple, config=config)
        json_str_with_hints = json.dumps(serialized_with_hints, default=str)
        parsed_with_hints = json.loads(json_str_with_hints)
        reconstructed_with_hints = deserialize_fast(parsed_with_hints, config=config)

        assert isinstance(reconstructed_with_hints, tuple), "Tuple should be preserved with type hints"
        assert reconstructed_with_hints == test_tuple, "Tuple should round-trip perfectly"

    def test_nested_structure_intelligent_handling(self):
        """Test intelligent handling of nested structures."""
        nested = {
            "timestamp": datetime.now(),
            "id": uuid.uuid4(),
            "data": [1, 2, {"nested": "value"}],
            "tags": {"python", "datason"},  # This set will become a list
            "coords": (1.0, 2.0),  # This tuple will become a list
        }

        # Test without type hints
        serialized = datason.serialize(nested)
        json_str = json.dumps(serialized, default=str)
        parsed = json.loads(json_str)
        reconstructed = deserialize_fast(parsed)

        # Verify auto-detected types
        assert isinstance(reconstructed["timestamp"], datetime)
        assert isinstance(reconstructed["id"], uuid.UUID)
        assert isinstance(reconstructed["data"], list)
        assert isinstance(reconstructed["tags"], list)  # set → list
        assert isinstance(reconstructed["coords"], list)  # tuple → list

        # Verify values (accounting for type conversions)
        assert reconstructed["timestamp"] == nested["timestamp"]
        assert reconstructed["id"] == nested["id"]
        assert reconstructed["data"] == nested["data"]
        assert set(reconstructed["tags"]) == nested["tags"]
        assert tuple(reconstructed["coords"]) == nested["coords"]

    def test_complex_and_decimal_always_preserved(self):
        """Test that complex numbers and Decimals are always preserved."""
        from decimal import Decimal

        test_cases = [
            complex(1, 2),
            complex(0, 0),
            complex(3.14, -2.71),
            Decimal("123.456"),
            Decimal("0"),
            Decimal("1E-10"),
        ]

        for test_value in test_cases:
            # Test without type hints (should still work)
            serialized = datason.serialize(test_value)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            assert type(reconstructed) is type(test_value), f"Type not preserved: {test_value}"
            assert reconstructed == test_value, f"Value not preserved: {test_value}"


@pytest.mark.skipif(not pytest.importorskip("pandas", reason="pandas not available"))
class TestPandasTypeDetection:
    """Test enhanced pandas type detection and reconstruction."""

    def setup_method(self):
        """Clear caches before each test to ensure clean state."""
        _clear_deserialization_caches()

    def test_dataframe_auto_detection_improvements(self):
        """Test improved DataFrame auto-detection capabilities."""
        import pandas as pd

        # Simple DataFrame
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Test without type hints (should detect as records)
        serialized = datason.serialize(df)
        json_str = json.dumps(serialized, default=str)
        parsed = json.loads(json_str)
        reconstructed = deserialize_fast(parsed)

        # Without type hints, DataFrame should become a list of records
        assert isinstance(reconstructed, list), "DataFrame should become list without type hints"
        assert len(reconstructed) == len(df), "Record count should match"

        # With type hints: perfect round-trip
        config = SerializationConfig(include_type_hints=True)
        serialized_with_hints = datason.serialize(df, config=config)
        json_str_with_hints = json.dumps(serialized_with_hints, default=str)
        parsed_with_hints = json.loads(json_str_with_hints)
        reconstructed_with_hints = deserialize_fast(parsed_with_hints, config=config)

        assert isinstance(reconstructed_with_hints, pd.DataFrame), "DataFrame should be preserved with type hints"
        pd.testing.assert_frame_equal(reconstructed_with_hints, df)


@pytest.mark.skipif(not pytest.importorskip("numpy", reason="numpy not available"))
class TestNumPyTypeDetection:
    """Test enhanced NumPy type detection and reconstruction."""

    def setup_method(self):
        """Clear caches before each test to ensure clean state."""
        _clear_deserialization_caches()

    def test_numpy_array_auto_detection_improvements(self):
        """Test improved NumPy array auto-detection capabilities."""
        import numpy as np

        test_arrays = [
            np.array([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            np.array([1.1, 2.2, 3.3]),
        ]

        for arr in test_arrays:
            # Test without type hints (should become list)
            serialized = datason.serialize(arr)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            assert isinstance(reconstructed, list), "Array should become list without type hints"
            assert np.array(reconstructed).tolist() == arr.tolist(), "Array values should be preserved"

            # With type hints: perfect round-trip
            config = SerializationConfig(include_type_hints=True)
            serialized_with_hints = datason.serialize(arr, config=config)
            json_str_with_hints = json.dumps(serialized_with_hints, default=str)
            parsed_with_hints = json.loads(json_str_with_hints)
            reconstructed_with_hints = deserialize_fast(parsed_with_hints, config=config)

            assert isinstance(reconstructed_with_hints, np.ndarray), "Array should be preserved with type hints"
            np.testing.assert_array_equal(reconstructed_with_hints, arr)

    def test_numpy_scalar_auto_detection(self):
        """Test NumPy scalar auto-detection."""
        import numpy as np

        test_scalars = [
            np.int32(42),
            np.float64(3.14),
            np.bool_(True),
        ]

        for scalar in test_scalars:
            # Test without type hints (should become Python equivalent)
            serialized = datason.serialize(scalar)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)
            reconstructed = deserialize_fast(parsed)

            # Should become equivalent Python type
            if isinstance(scalar, np.bool_):
                assert isinstance(reconstructed, bool)
            elif isinstance(scalar, np.integer):
                assert isinstance(reconstructed, int)
            elif isinstance(scalar, np.floating):
                assert isinstance(reconstructed, float)

            assert reconstructed == scalar, "Scalar value should be preserved"

            # With type hints: exact type preservation
            config = SerializationConfig(include_type_hints=True)
            serialized_with_hints = datason.serialize(scalar, config=config)
            json_str_with_hints = json.dumps(serialized_with_hints, default=str)
            parsed_with_hints = json.loads(json_str_with_hints)
            reconstructed_with_hints = deserialize_fast(parsed_with_hints, config=config)

            assert type(reconstructed_with_hints) is type(scalar), "Exact type should be preserved with type hints"
            assert reconstructed_with_hints == scalar, "Scalar should round-trip perfectly"
