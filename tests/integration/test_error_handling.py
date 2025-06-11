"""Tests for error handling and security limits."""

import pytest

import datason
from datason import SecurityError
from datason.config import SerializationConfig


class TestErrorHandling:
    """Test error handling and security features."""

    def test_basic_serialization(self) -> None:
        """Test that basic serialization works."""
        config = SerializationConfig()
        result = datason.serialize(123, config=config)
        assert result == 123

    def test_security_limits(self) -> None:
        """Test that security limits (e.g., max depth) are enforced."""
        # Save and restore global config to prevent test pollution
        original_config = datason.get_default_config()
        try:
            # Ensure clean slate
            datason.reset_default_config()

            # Test max depth
            nested_data = [1, [2, [3, [4, [5]]]]]
            config_depth_3 = SerializationConfig(max_depth=3)

            with pytest.raises(SecurityError, match="Maximum serialization depth"):
                datason.serialize(nested_data, config=config_depth_3)

            # Test no error if within limits
            config_depth_5 = SerializationConfig(max_depth=5)
            datason.serialize(nested_data, config=config_depth_5)
        finally:
            # Always restore original config
            datason.set_default_config(original_config)

    def test_large_object_limits(self) -> None:
        """Test that limits on large objects (e.g., strings) are handled by truncation."""
        config = SerializationConfig(max_string_length=10)

        # Test string length limit - truncates and adds [TRUNCATED] marker
        result = datason.serialize("a" * 15, config=config)
        assert "[TRUNCATED]" in result

        # Test no truncation if within limits
        result = datason.serialize("a" * 10, config=config)
        assert result == "a" * 10
