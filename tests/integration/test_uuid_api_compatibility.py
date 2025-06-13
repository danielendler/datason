"""
Test UUID handling for API compatibility (FastAPI/Pydantic integration).

This module tests the new UUID configuration options that address the feedback
from the financial model team regarding FastAPI + Pydantic integration.
"""

import uuid

import pytest

import datason
from datason.config import SerializationConfig, get_api_config


class TestUUIDAPICompatibility:
    """Test UUID handling configuration for API compatibility."""

    def test_default_uuid_behavior(self):
        """Test that default behavior converts UUIDs to objects."""
        uuid_string = "12345678-1234-5678-9012-123456789abc"

        # Default behavior should convert to UUID object
        result = datason.auto_deserialize(uuid_string)
        assert isinstance(result, uuid.UUID)
        assert str(result) == uuid_string

    def test_api_config_keeps_uuids_as_strings(self):
        """Test that API config keeps UUIDs as strings."""
        uuid_string = "12345678-1234-5678-9012-123456789abc"
        api_config = get_api_config()

        # API config should keep UUIDs as strings
        result = datason.auto_deserialize(uuid_string, config=api_config)
        assert isinstance(result, str)
        assert result == uuid_string

    def test_custom_config_uuid_format_string(self):
        """Test custom config with uuid_format='string'."""
        uuid_string = "12345678-1234-5678-9012-123456789abc"
        config = SerializationConfig(uuid_format="string")

        result = datason.auto_deserialize(uuid_string, config=config)
        assert isinstance(result, str)
        assert result == uuid_string

    def test_custom_config_uuid_format_object(self):
        """Test custom config with uuid_format='object'."""
        uuid_string = "12345678-1234-5678-9012-123456789abc"
        config = SerializationConfig(uuid_format="object")

        result = datason.auto_deserialize(uuid_string, config=config)
        assert isinstance(result, uuid.UUID)
        assert str(result) == uuid_string

    def test_custom_config_parse_uuids_false(self):
        """Test custom config with parse_uuids=False."""
        uuid_string = "12345678-1234-5678-9012-123456789abc"
        config = SerializationConfig(parse_uuids=False)

        result = datason.auto_deserialize(uuid_string, config=config)
        assert isinstance(result, str)
        assert result == uuid_string

    def test_complex_data_structure_with_api_config(self):
        """Test complex data structures with API-compatible UUID handling."""
        data = {
            "user_id": "12345678-1234-5678-9012-123456789abc",
            "session_id": "87654321-4321-8765-2109-cba987654321",
            "metadata": {"tracking_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "created_at": "2023-01-01T12:00:00"},
            "items": [
                {"id": "11111111-2222-3333-4444-555555555555", "name": "item1"},
                {"id": "66666666-7777-8888-9999-aaaaaaaaaaaa", "name": "item2"},
            ],
        }

        api_config = get_api_config()
        result = datason.auto_deserialize(data, config=api_config)

        # All UUIDs should remain as strings
        assert isinstance(result["user_id"], str)
        assert isinstance(result["session_id"], str)
        assert isinstance(result["metadata"]["tracking_id"], str)
        assert isinstance(result["items"][0]["id"], str)
        assert isinstance(result["items"][1]["id"], str)

        # But datetime should still be converted (different from UUID handling)
        from datetime import datetime

        assert isinstance(result["metadata"]["created_at"], datetime)

    def test_mixed_uuid_and_other_types(self):
        """Test that UUID config doesn't affect other type conversions."""
        data = {
            "uuid": "12345678-1234-5678-9012-123456789abc",
            "datetime": "2023-01-01T12:00:00",
            "number": "42",
            "boolean": "true",
            "regular_string": "hello world",
        }

        api_config = get_api_config()
        result = datason.auto_deserialize(data, aggressive=True, config=api_config)

        # UUID should stay as string
        assert isinstance(result["uuid"], str)
        assert result["uuid"] == "12345678-1234-5678-9012-123456789abc"

        # Other types should still be converted appropriately
        from datetime import datetime

        assert isinstance(result["datetime"], datetime)

        # Note: aggressive mode is needed for number/boolean conversion
        assert isinstance(result["number"], int)
        assert result["number"] == 42
        assert isinstance(result["boolean"], bool)
        assert result["boolean"] is True
        assert isinstance(result["regular_string"], str)

    def test_pydantic_compatible_example(self):
        """Test data format that would work with Pydantic models."""
        # Simulate data coming from a database where UUIDs are stored as strings
        database_data = {
            "id": "ea82f3dd-d770-41b9-9706-69cd3070b4f5",
            "name": "Test Group",
            "created_at": "2023-01-01T12:00:00Z",
            "members": [
                {"id": "11111111-2222-3333-4444-555555555555", "email": "user1@example.com"},
                {"id": "22222222-3333-4444-5555-666666666666", "email": "user2@example.com"},
            ],
        }

        # Process with API config (Pydantic-compatible)
        api_config = get_api_config()
        result = datason.auto_deserialize(database_data, config=api_config)

        # This would now work with a Pydantic model expecting string UUIDs:
        # class SavedGroup(BaseModel):
        #     id: str  # This works because UUID stays as string
        #     name: str
        #     created_at: datetime  # This works because datetime is still converted
        #     members: List[Dict[str, str]]

        assert isinstance(result["id"], str)
        assert result["id"] == "ea82f3dd-d770-41b9-9706-69cd3070b4f5"

        for member in result["members"]:
            assert isinstance(member["id"], str)

        # Datetime should still be converted for proper type handling
        from datetime import datetime

        assert isinstance(result["created_at"], datetime)


class TestBackwardCompatibility:
    """Test that existing behavior is preserved when no config is provided."""

    def test_auto_deserialize_without_config_still_works(self):
        """Test that auto_deserialize without config parameter still works as before."""
        uuid_string = "12345678-1234-5678-9012-123456789abc"

        # Should work exactly as before
        result = datason.auto_deserialize(uuid_string)
        assert isinstance(result, uuid.UUID)
        assert str(result) == uuid_string

    def test_regular_deserialize_still_works(self):
        """Test that regular deserialize function still works as before."""
        uuid_string = "12345678-1234-5678-9012-123456789abc"

        # With UUID parsing enabled (default)
        result = datason.deserialize(uuid_string, parse_uuids=True)
        assert isinstance(result, uuid.UUID)

        # With UUID parsing disabled
        result = datason.deserialize(uuid_string, parse_uuids=False)
        assert isinstance(result, str)


class TestConfigurationPresets:
    """Test that configuration presets work correctly."""

    def test_ml_config_preserves_uuid_objects(self):
        """Test that ML config keeps UUIDs as objects for ML workflows."""
        from datason.config import get_ml_config

        uuid_string = "12345678-1234-5678-9012-123456789abc"
        ml_config = get_ml_config()

        # ML config should convert UUIDs to objects (default behavior)
        result = datason.auto_deserialize(uuid_string, config=ml_config)
        assert isinstance(result, uuid.UUID)

    def test_api_config_documentation_example(self):
        """Test the exact example from the API config documentation."""
        # This is the pattern described in the issue
        data = {"id": "ea82f3dd-d770-41b9-9706-69cd3070b4f5"}

        # Without API config - would convert to UUID (problematic for Pydantic)
        result_default = datason.auto_deserialize(data)
        assert isinstance(result_default["id"], uuid.UUID)

        # With API config - keeps as string (Pydantic-compatible)
        api_config = get_api_config()
        result_api = datason.auto_deserialize(data, config=api_config)
        assert isinstance(result_api["id"], str)
        assert result_api["id"] == "ea82f3dd-d770-41b9-9706-69cd3070b4f5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
