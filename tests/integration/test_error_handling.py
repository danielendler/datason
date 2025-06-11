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
        # Clear caches and reset global state for clean isolation
        datason.clear_caches()
        datason.reset_default_config()

        # Import garbage collection for thorough cleanup
        import gc

        gc.collect()

        try:
            # Test max depth with explicit config - create deeper nesting to ensure limit is hit
            nested_data = {"level1": {"level2": {"level3": {"level4": {"level5": {"level6": "too_deep"}}}}}}
            config_depth_3 = SerializationConfig(max_depth=3)

            # Test that deep nesting raises security-related exception
            # Based on backup tests, this might be SecurityError or RecursionError
            security_exception_raised = False
            exception_message = ""
            try:
                result = datason.serialize(nested_data, config=config_depth_3)
                # If no exception, something is wrong
                pytest.fail(f"Expected security exception but serialization succeeded: {result}")
            except (SecurityError, RecursionError) as e:
                security_exception_raised = True
                exception_message = str(e)
                # Should mention depth in some way
                assert any(
                    word in exception_message.lower() for word in ["depth", "maximum", "limit", "security", "recursion"]
                )
            except Exception as e:
                # Log unexpected exception type for debugging
                pytest.fail(f"Expected SecurityError or RecursionError but got {type(e).__name__}: {e}")

            # Should have raised some form of security/depth exception
            assert security_exception_raised, (
                f"Expected security exception to be raised for deep nesting. Got message: {exception_message}"
            )

            # Test no error if within limits
            config_depth_10 = SerializationConfig(max_depth=10)
            result = datason.serialize(nested_data, config=config_depth_10)
            assert result is not None  # Should succeed with higher limit

        finally:
            # Always restore defaults to avoid test pollution
            datason.clear_caches()
            datason.reset_default_config()
            gc.collect()

    def test_large_object_limits(self) -> None:
        """Test that limits on large objects (e.g., strings) are handled by truncation."""
        config = SerializationConfig(max_string_length=10)

        # Test string length limit - truncates and adds [TRUNCATED] marker
        result = datason.serialize("a" * 15, config=config)
        assert "[TRUNCATED]" in result

        # Test no truncation if within limits
        result = datason.serialize("a" * 10, config=config)
        assert result == "a" * 10
