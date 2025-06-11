import uuid

import datason
from datason.config import SerializationConfig


class TestUUIDEdgeCases:
    """Edge case tests for UUID handling configuration."""

    def test_uuid_format_invalid_case_defaults_to_object(self) -> None:
        """Invalid uuid_format values should fall back to object parsing."""
        uuid_string = "12345678-1234-5678-9012-123456789abc"
        config = SerializationConfig(uuid_format="String")

        result = datason.auto_deserialize(uuid_string, config=config)
        assert isinstance(result, uuid.UUID)
        assert str(result) == uuid_string

    def test_nested_uuid_default_config_parses(self) -> None:
        """Default config should parse nested UUID strings."""
        data = {"user": {"id": "12345678-1234-5678-9012-123456789abc"}}

        result = datason.auto_deserialize(data)
        assert isinstance(result["user"]["id"], uuid.UUID)
        assert str(result["user"]["id"]) == "12345678-1234-5678-9012-123456789abc"
