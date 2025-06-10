"""Tests for error handling and security limits."""

import pytest

import datason
from datason import SecurityError
from datason.config import SerializationConfig


class TestErrorHandling:
    """Test error handling and security features."""

    def test_custom_serializer_failure(self) -> None:
        """Test graceful failure when a custom serializer raises an exception."""

        def failing_serializer(_obj: object) -> None:  # Use _obj to indicate unused
            raise ValueError("Intentional serialization failure")

        config = SerializationConfig(custom_serializers={int: failing_serializer})
        with pytest.raises(ValueError, match="Intentional failure"):
            datason.serialize(123, config=config)

    def test_security_limits(self) -> None:
        """Test that security limits (e.g., max depth) are enforced."""
        # Test max depth
        nested_data = [1, [2, [3, [4, [5]]]]]
        config_depth_3 = SerializationConfig(max_depth=3)

        with pytest.raises(SecurityError, match="Maximum serialization depth"):
            datason.serialize(nested_data, config=config_depth_3)

        # Test no error if within limits
        config_depth_5 = SerializationConfig(max_depth=5)
        datason.serialize(nested_data, config=config_depth_5)

    def test_large_object_limits(self) -> None:
        """Test that limits on large objects (e.g., strings) are enforced."""
        config = SerializationConfig(max_string_length=10)

        # Test string length limit
        with pytest.raises(SecurityError, match="String length"):
            datason.serialize("a" * 11, config=config)

        # Test no error if within limits
        datason.serialize("a" * 10, config=config)
