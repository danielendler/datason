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
        # Use reset_default_config for clean isolation
        datason.reset_default_config()

        try:
            # Test max depth - should raise SecurityError
            nested_data = [1, [2, [3, [4, [5]]]]]
            config_depth_3 = SerializationConfig(max_depth=3)

            # Test that deep nesting raises SecurityError
            security_error_raised = False
            try:
                datason.serialize(nested_data, config=config_depth_3)
            except SecurityError as e:
                security_error_raised = True
                assert "Maximum serialization depth" in str(e)
            except Exception as e:
                pytest.fail(f"Expected SecurityError but got {type(e).__name__}: {e}")

            # Should have raised SecurityError
            assert security_error_raised, "Expected SecurityError to be raised for deep nesting"

            # Test no error if within limits
            config_depth_5 = SerializationConfig(max_depth=5)
            result = datason.serialize(nested_data, config=config_depth_5)
            assert result is not None  # Should succeed
        finally:
            # Always restore defaults to avoid test pollution
            datason.reset_default_config()

    def test_large_object_limits(self) -> None:
        """Test that limits on large objects (e.g., strings) are handled by truncation."""
        config = SerializationConfig(max_string_length=10)

        # Test string length limit - truncates and adds [TRUNCATED] marker
        result = datason.serialize("a" * 15, config=config)
        assert "[TRUNCATED]" in result

        # Test no truncation if within limits
        result = datason.serialize("a" * 10, config=config)
        assert result == "a" * 10
