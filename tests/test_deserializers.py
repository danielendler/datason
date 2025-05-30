"""
Tests for SerialPy deserialization functionality.

This module tests the bidirectional serialization capabilities by testing
the deserialize functions and round-trip serialization/deserialization.
"""

from datetime import datetime
import json
import uuid

import pytest

import serialpy as sp


class TestDeserialize:
    """Test the main deserialize function."""

    def test_deserialize_none(self) -> None:
        """Test deserialization of None."""
        assert sp.deserialize(None) is None

    def test_deserialize_basic_types(self) -> None:
        """Test deserialization of basic JSON types."""
        assert sp.deserialize("hello") == "hello"
        assert sp.deserialize(42) == 42
        assert sp.deserialize(3.14) == 3.14
        assert sp.deserialize(True) is True
        assert sp.deserialize(False) is False

    def test_deserialize_datetime_strings(self) -> None:
        """Test deserialization of datetime strings."""
        # ISO format
        result = sp.deserialize("2023-01-01T12:00:00")
        assert isinstance(result, datetime)
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 12

        # With timezone
        result = sp.deserialize("2023-01-01T12:00:00Z")
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

    def test_deserialize_uuid_strings(self) -> None:
        """Test deserialization of UUID strings."""
        uuid_str = "12345678-1234-5678-9012-123456789abc"
        result = sp.deserialize(uuid_str)
        assert isinstance(result, uuid.UUID)
        assert str(result) == uuid_str

    def test_deserialize_with_parsing_disabled(self) -> None:
        """Test deserialization with parsing options disabled."""
        # Disable datetime parsing
        result = sp.deserialize("2023-01-01T12:00:00", parse_dates=False)
        assert isinstance(result, str)

        # Disable UUID parsing
        uuid_str = "12345678-1234-5678-9012-123456789abc"
        result = sp.deserialize(uuid_str, parse_uuids=False)
        assert isinstance(result, str)

    def test_deserialize_lists(self) -> None:
        """Test deserialization of lists."""
        data = [
            "2023-01-01T12:00:00",
            "12345678-1234-5678-9012-123456789abc",
            "plain string",
            42,
            None,
        ]
        result = sp.deserialize(data)

        assert isinstance(result[0], datetime)
        assert isinstance(result[1], uuid.UUID)
        assert result[2] == "plain string"
        assert result[3] == 42
        assert result[4] is None

    def test_deserialize_dicts(self) -> None:
        """Test deserialization of dictionaries."""
        data = {
            "created_at": "2023-01-01T12:00:00",
            "id": "12345678-1234-5678-9012-123456789abc",
            "name": "test",
            "value": 42,
            "active": True,
        }
        result = sp.deserialize(data)

        assert isinstance(result["created_at"], datetime)
        assert isinstance(result["id"], uuid.UUID)
        assert result["name"] == "test"
        assert result["value"] == 42
        assert result["active"] is True

    def test_deserialize_nested_structures(self) -> None:
        """Test deserialization of nested data structures."""
        data = {
            "user": {
                "id": "12345678-1234-5678-9012-123456789abc",
                "created_at": "2023-01-01T12:00:00",
                "profile": {
                    "name": "Test User",
                    "timestamps": ["2023-01-01T10:00:00", "2023-01-02T11:00:00"],
                },
            },
            "metadata": {"processed_at": "2023-01-01T15:30:00"},
        }
        result = sp.deserialize(data)

        assert isinstance(result["user"]["id"], uuid.UUID)
        assert isinstance(result["user"]["created_at"], datetime)
        assert isinstance(result["user"]["profile"]["timestamps"][0], datetime)
        assert isinstance(result["user"]["profile"]["timestamps"][1], datetime)
        assert isinstance(result["metadata"]["processed_at"], datetime)


class TestRoundTripSerialization:
    """Test round-trip serialization and deserialization."""

    def test_basic_round_trip(self) -> None:
        """Test basic round-trip serialization."""
        original = {
            "datetime": datetime(2023, 1, 1, 12, 0, 0),
            "uuid": uuid.uuid4(),
            "string": "hello",
            "number": 42,
            "boolean": True,
            "null": None,
        }

        # Serialize then deserialize
        serialized = sp.serialize(original)
        deserialized = sp.deserialize(serialized)

        assert isinstance(deserialized["datetime"], datetime)
        assert isinstance(deserialized["uuid"], uuid.UUID)
        assert deserialized["string"] == original["string"]
        assert deserialized["number"] == original["number"]
        assert deserialized["boolean"] == original["boolean"]
        assert deserialized["null"] == original["null"]

    def test_complex_round_trip(self) -> None:
        """Test round-trip with complex nested structures."""
        original = {
            "users": [
                {
                    "id": uuid.uuid4(),
                    "created_at": datetime(2023, 1, 1, 10, 0, 0),
                    "metadata": {
                        "last_login": datetime(2023, 1, 15, 14, 30, 0),
                        "session_id": uuid.uuid4(),
                    },
                },
                {
                    "id": uuid.uuid4(),
                    "created_at": datetime(2023, 2, 1, 11, 0, 0),
                    "metadata": {
                        "last_login": datetime(2023, 2, 10, 16, 45, 0),
                        "session_id": uuid.uuid4(),
                    },
                },
            ],
            "processed_at": datetime(2023, 3, 1, 9, 0, 0),
        }

        # Serialize then deserialize
        serialized = sp.serialize(original)
        deserialized = sp.deserialize(serialized)

        # Check structure is preserved
        assert len(deserialized["users"]) == 2

        # Check types are restored
        for i, user in enumerate(deserialized["users"]):
            assert isinstance(user["id"], uuid.UUID)
            assert isinstance(user["created_at"], datetime)
            assert isinstance(user["metadata"]["last_login"], datetime)
            assert isinstance(user["metadata"]["session_id"], uuid.UUID)

        assert isinstance(deserialized["processed_at"], datetime)

    def test_json_round_trip(self) -> None:
        """Test round-trip through actual JSON serialization."""
        original = {
            "datetime": datetime(2023, 1, 1, 12, 0, 0),
            "uuid": uuid.uuid4(),
            "data": [1, 2, 3],
        }

        # Serialize with SerialPy
        serialized = sp.serialize(original)

        # Convert to JSON string and back
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)

        # Deserialize with SerialPy
        deserialized = sp.deserialize(parsed)

        assert isinstance(deserialized["datetime"], datetime)
        assert isinstance(deserialized["uuid"], uuid.UUID)
        assert deserialized["data"] == [1, 2, 3]


class TestSafeDeserialize:
    """Test safe deserialization functionality."""

    def test_safe_deserialize_valid_json(self) -> None:
        """Test safe deserialization with valid JSON."""
        json_str = '{"date": "2023-01-01T12:00:00", "value": 42}'
        result = sp.safe_deserialize(json_str)

        assert isinstance(result["date"], datetime)
        assert result["value"] == 42

    def test_safe_deserialize_invalid_json(self) -> None:
        """Test safe deserialization with invalid JSON."""
        invalid_json = '{"invalid": json}'
        result = sp.safe_deserialize(invalid_json)

        # Should return the original string
        assert result == invalid_json

    def test_safe_deserialize_with_options(self) -> None:
        """Test safe deserialization with parsing options."""
        json_str = '{"date": "2023-01-01T12:00:00"}'
        result = sp.safe_deserialize(json_str, parse_dates=False)

        assert isinstance(result["date"], str)


class TestUtilityFunctions:
    """Test utility parsing functions."""

    def test_parse_datetime_string(self) -> None:
        """Test datetime string parsing."""
        # Valid datetime strings
        assert sp.parse_datetime_string("2023-01-01T12:00:00") is not None
        assert sp.parse_datetime_string("2023-01-01T12:00:00Z") is not None

        # Invalid datetime strings
        assert sp.parse_datetime_string("not a date") is None
        assert sp.parse_datetime_string("2023") is None
        assert sp.parse_datetime_string("") is None

    def test_parse_uuid_string(self) -> None:
        """Test UUID string parsing."""
        # Valid UUID strings
        valid_uuid = "12345678-1234-5678-9012-123456789abc"
        result = sp.parse_uuid_string(valid_uuid)
        assert isinstance(result, uuid.UUID)
        assert str(result) == valid_uuid

        # Invalid UUID strings
        assert sp.parse_uuid_string("not a uuid") is None
        assert sp.parse_uuid_string("12345678-1234-5678-9012") is None
        assert sp.parse_uuid_string("") is None


class TestDatetimeEdgeCases:
    """Test edge cases for datetime parsing."""

    def test_various_datetime_formats(self) -> None:
        """Test parsing various datetime formats."""
        formats = [
            "2023-01-01T12:00:00",
            "2023-01-01T12:00:00.123456",
            "2023-01-01T12:00:00Z",
            "2023-01-01T12:00:00+00:00",
            "2023-01-01T12:00:00-05:00",
        ]

        for fmt in formats:
            result = sp.deserialize(fmt)
            assert isinstance(result, datetime), f"Failed to parse: {fmt}"

    def test_non_datetime_strings(self) -> None:
        """Test that non-datetime strings remain as strings."""
        non_dates = [
            "hello world",
            "123-456-789",
            "2023",
            "random text",
            "email@example.com",
        ]

        for text in non_dates:
            result = sp.deserialize(text)
            assert result == text, f"Should remain string: {text}"


class TestOptionalDependencies:
    """Test deserialization with optional dependencies."""

    def test_deserialize_with_pandas(self) -> None:
        """Test deserialization when pandas is available."""
        pytest.importorskip("pandas")

        # This should work even with pandas available
        result = sp.deserialize("2023-01-01T12:00:00")
        assert isinstance(result, datetime)

    def test_deserialize_to_pandas(self) -> None:
        """Test pandas-specific deserialization."""
        pytest.importorskip("pandas")

        data = {"timestamp": "2023-01-01T12:00:00", "values": [1, 2, 3]}

        result = sp.deserialize_to_pandas(data)
        assert isinstance(result["timestamp"], datetime)
        assert result["values"] == [1, 2, 3]
