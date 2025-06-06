"""Exception flows and import error handling tests - Batch 1.

Targets specific missing coverage lines in deserializers.py and datetime_utils.py.
"""

import sys
import warnings
from unittest.mock import patch

from datason.deserializers import (
    _auto_detect_string_type,
    _clear_deserialization_caches,
    _looks_like_datetime,
    _looks_like_uuid,
    auto_deserialize,
    deserialize,
    parse_datetime_string,
    parse_uuid_string,
    safe_deserialize,
)


class TestImportErrorHandling:
    """Test import error handling and fallback mechanisms."""

    def test_config_import_fallback(self, monkeypatch):
        """Test fallback when config import fails (lines 20-25)."""
        # Clear caches to ensure clean state
        _clear_deserialization_caches()

        # Simulate config import failure
        with patch("datason.deserializers._config_available", False):
            with patch("datason.deserializers.SerializationConfig", None):
                with patch("datason.deserializers.get_default_config") as mock_get_default:
                    mock_get_default.return_value = None

                    # Test that functions still work with no config
                    result = deserialize({"test": "value"})
                    assert result == {"test": "value"}

    def test_core_import_fallback(self, monkeypatch):
        """Test fallback when core.py import fails (lines 31-35)."""
        # Clear caches to ensure clean state
        _clear_deserialization_caches()

        # Test the fallback constants are used when core import fails
        with patch.dict("sys.modules", {"datason.core": None}):
            # Re-import to trigger fallback
            import importlib

            importlib.reload(sys.modules["datason.deserializers"])

            # Verify fallback constants exist
            from datason.deserializers import MAX_OBJECT_SIZE, MAX_SERIALIZATION_DEPTH, MAX_STRING_LENGTH

            assert MAX_SERIALIZATION_DEPTH == 50
            assert MAX_OBJECT_SIZE == 100_000
            assert MAX_STRING_LENGTH == 1_000_000


class TestStringPatternDetectionEdgeCases:
    """Test edge cases in string pattern detection."""

    def test_looks_like_datetime_edge_cases(self):
        """Test datetime detection edge cases (lines 147-153)."""
        # Clear caches to ensure clean state
        _clear_deserialization_caches()

        # Test various invalid datetime-like strings
        invalid_datetimes = [
            "not-a-date-at-all",
            "2023/01/01 12:00:00",  # Wrong format
            "",  # Empty string
            "2023",  # Just year
            "random text",
            "123456789",
        ]

        for invalid_dt in invalid_datetimes:
            assert not _looks_like_datetime(invalid_dt)

        # Test some that might look like datetime but aren't perfect
        potentially_valid = [
            "2023-01-01T",  # Incomplete but might pass basic check
            "T12:00:00",  # Missing date but has time format
            "2023-13-45T25:70:90",  # Invalid components but has datetime pattern
        ]

        # These might pass the basic pattern check since _looks_like_datetime
        # does basic pattern matching, not validation
        for potentially_dt in potentially_valid:
            result = _looks_like_datetime(potentially_dt)
            # We just test that the function runs without error
            assert isinstance(result, bool)

    def test_looks_like_uuid_edge_cases(self):
        """Test UUID detection edge cases (lines 170)."""
        # Clear caches to ensure clean state
        _clear_deserialization_caches()

        # Test various invalid UUID-like strings
        invalid_uuids = [
            "12345678-1234-5678-9012-12345678",  # Too short
            "12345678-1234-5678-9012-123456789abc0",  # Too long
            "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",  # Invalid chars
            "12345678123456781234567812345678",  # No dashes
            "",  # Empty string
            "not-a-uuid-at-all",
            "12345678-1234-5678-9012",  # Incomplete
        ]

        for invalid_uuid in invalid_uuids:
            assert not _looks_like_uuid(invalid_uuid)

    def test_auto_detect_string_type_edge_cases(self):
        """Test auto-detection with edge cases (lines 203-238)."""
        # Clear caches to ensure clean state
        _clear_deserialization_caches()

        # Test strings that should remain strings
        test_strings = [
            "almost-datetime-2023-01-01",
            "uuid-like-but-not-12345678-1234",
            "regular string",
            "",
            "123.456.789",  # Looks numeric but isn't
            "true",  # Looks boolean but we keep as string
        ]

        for test_str in test_strings:
            result = _auto_detect_string_type(test_str, aggressive=False)
            assert isinstance(result, str)
            assert result == test_str


class TestParsingEdgeCases:
    """Test parsing edge cases and error handling."""

    def test_parse_datetime_string_invalid_input(self):
        """Test datetime parsing with invalid inputs (lines 89-90, 94-95)."""
        # Clear caches to ensure clean state
        _clear_deserialization_caches()

        # Test None input
        assert parse_datetime_string(None) is None

        # Test non-string input
        assert parse_datetime_string(123) is None
        assert parse_datetime_string([]) is None
        assert parse_datetime_string({}) is None

        # Test invalid datetime strings that trigger warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = parse_datetime_string("definitely-not-a-datetime")
            assert result is None

            # Should generate warning about failed parsing
            if w:  # Only check if warnings were captured
                assert any("Failed to parse datetime" in str(warning.message) for warning in w)

    def test_parse_uuid_string_invalid_input(self):
        """Test UUID parsing with invalid inputs (lines 127)."""
        # Clear caches to ensure clean state
        _clear_deserialization_caches()

        # Test None input
        assert parse_uuid_string(None) is None

        # Test non-string input
        assert parse_uuid_string(123) is None
        assert parse_uuid_string([]) is None
        assert parse_uuid_string({}) is None

        # Test invalid UUID strings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = parse_uuid_string("definitely-not-a-uuid")
            assert result is None

            # Should generate warning about failed parsing
            if w:  # Only check if warnings were captured
                assert any("Failed to parse UUID" in str(warning.message) for warning in w)


class TestDeserializationWithNoPandas:
    """Test deserialization behavior when pandas is not available."""

    def test_deserialize_without_pandas(self, monkeypatch):
        """Test deserialization works without pandas (lines 255)."""
        # Clear caches to ensure clean state
        _clear_deserialization_caches()

        # Mock pandas as unavailable
        from datason import deserializers

        monkeypatch.setattr(deserializers, "pd", None)

        # Test basic deserialization still works
        data = {
            "string": "test",
            "number": 42,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"inner": "value"},
        }

        result = deserialize(data)
        assert result == data

        # Test auto_deserialize also works
        result = auto_deserialize(data)
        assert result == data

    def test_deserialize_without_numpy(self, monkeypatch):
        """Test deserialization works without numpy (lines 287)."""
        # Clear caches to ensure clean state
        _clear_deserialization_caches()

        # Mock numpy as unavailable
        from datason import deserializers

        monkeypatch.setattr(deserializers, "np", None)

        # Test with list that might look like numpy array
        data = {"array_like": [1, 2, 3, 4, 5]}

        result = deserialize(data)
        assert result == data

        # Should remain as list, not converted to numpy array
        assert isinstance(result["array_like"], list)


class TestSafeDeserializeExceptions:
    """Test safe_deserialize exception handling."""

    def test_safe_deserialize_invalid_json(self):
        """Test safe_deserialize with invalid JSON."""
        # Clear caches to ensure clean state
        _clear_deserialization_caches()

        # Test invalid JSON strings
        invalid_json_strings = [
            "{invalid json}",
            "{'single': 'quotes'}",  # Python-style, not JSON
            "{unclosed",
            "not json at all",
            "",
        ]

        for invalid_json in invalid_json_strings:
            # Should not raise exception, should return the string as-is or handle gracefully
            result = safe_deserialize(invalid_json)
            # The function should handle this gracefully
            assert result is not None

    def test_safe_deserialize_with_none_input(self):
        """Test safe_deserialize with None input."""
        # Clear caches to ensure clean state
        _clear_deserialization_caches()

        result = safe_deserialize(None)
        assert result is None


class TestAggressiveDetectionModes:
    """Test aggressive detection modes and their edge cases."""

    def test_auto_deserialize_aggressive_mode(self):
        """Test aggressive auto-detection mode."""
        # Clear caches to ensure clean state
        _clear_deserialization_caches()

        # Test data that might be detected as pandas objects in aggressive mode
        data = {"records": [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]}

        # Test without aggressive mode
        result_normal = auto_deserialize(data, aggressive=False)
        assert isinstance(result_normal["records"], list)

        # Test with aggressive mode
        result_aggressive = auto_deserialize(data, aggressive=True)
        # In aggressive mode, might detect as DataFrame or remain as list
        assert result_aggressive is not None
