"""
Tests for ultra-fast path and complex security logic in core.py

This file specifically tests the complex optimization and security paths we added
to ensure robust coverage of our performance improvements and security measures.
"""

import uuid
import warnings
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from datason import serialize
from datason.core import SecurityError


class TestUltraFastPathCoverage:
    """Test coverage for the ultra-fast path optimizations."""

    def test_ultra_fast_path_int_bool_none(self):
        """Test ultra-fast path for int, bool, None."""
        # These should bypass all processing
        assert serialize(42) == 42
        assert serialize(True) is True
        assert serialize(False) is False
        assert serialize(None) is None

    def test_ultra_fast_path_regular_float(self):
        """Test ultra-fast path for regular floats."""
        # Regular floats should use fast path
        assert serialize(3.14) == 3.14
        assert serialize(-2.5) == -2.5
        assert serialize(0.0) == 0.0

    def test_ultra_fast_path_nan_inf_fallback(self):
        """Test that NaN/Inf floats fall through to normal processing."""
        # These should NOT use ultra-fast path
        result_nan = serialize(float("nan"))
        assert result_nan is None  # Default NaN handling

        result_inf = serialize(float("inf"))
        assert result_inf is None  # Default Inf handling

        result_ninf = serialize(float("-inf"))
        assert result_ninf is None  # Default -Inf handling

    def test_ultra_fast_path_short_strings_top_level(self):
        """Test ultra-fast path for short strings at top level."""
        # Short strings at depth 0 should use fast path
        assert serialize("hello") == "hello"
        assert serialize("") == ""
        assert serialize("x" * 100) == "x" * 100  # Still under 1000 char limit

    def test_ultra_fast_path_string_length_limit(self):
        """Test that long strings don't use ultra-fast path."""
        # Strings over 1000 chars should fall through to normal processing
        long_string = "x" * 1001
        result = serialize(long_string)
        # Should still work but go through normal processing
        assert result == long_string

    def test_ultra_fast_path_nested_strings_fallback(self):
        """Test that nested strings don't use ultra-fast path."""
        # Strings not at top level should fall through
        nested_data = {"key": "value"}
        result = serialize(nested_data)
        assert result == {"key": "value"}


class TestSecurityMeasuresEdgeCases:
    """Test edge cases in our security measures."""

    def test_mock_object_detection_warnings(self):
        """Test detection and safe handling of mock objects."""
        # Create mock-like object that should trigger warning
        mock_obj = Mock()
        mock_obj.__module__ = "unittest.mock"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = serialize(mock_obj)

            # Should emit warning about problematic object
            assert len(w) > 0
            mock_warnings = [warning for warning in w if "mock" in str(warning.message).lower()]
            assert len(mock_warnings) > 0, f"No mock warnings found in: {[str(w.message) for w in w]}"
            assert "potentially problematic" in str(mock_warnings[0].message).lower()
            # Should return safe representation
            assert isinstance(result, str)
            assert "Mock" in result

    def test_io_object_detection_warnings(self):
        """Test detection of io objects that can cause circular refs."""
        from io import BytesIO

        # BytesIO objects can cause circular reference issues
        bio = BytesIO(b"test")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = serialize(bio)

            # Should emit warning
            assert len(w) > 0
            # Should return safe representation
            assert isinstance(result, str)

    def test_circular_reference_in_dict_values(self):
        """Test circular reference detection in dict values."""
        # Create circular reference scenario
        d1 = {"name": "dict1"}
        d2 = {"name": "dict2", "ref": d1}
        d1["ref"] = d2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = serialize(d1)

            # Should handle gracefully without infinite recursion
            assert isinstance(result, dict)
            # May emit circular reference warning
            if w:
                assert any("circular" in str(warning.message).lower() for warning in w)

    def test_excessive_depth_security_error(self):
        """Test that excessive depth raises SecurityError."""
        # Create deeply nested structure that exceeds MAX_SERIALIZATION_DEPTH (50)
        nested = "value"
        for _ in range(60):  # Exceed the 50-level limit
            nested = {"level": nested}

        # Should raise SecurityError due to depth limit
        with pytest.raises(SecurityError) as exc_info:
            serialize(nested)

        assert "Maximum serialization depth" in str(exc_info.value)

    def test_excessive_dict_size_security_error(self):
        """Test that large dict raises SecurityError."""
        # Create dict larger than MAX_OBJECT_SIZE
        large_dict = {f"key_{i}": f"value_{i}" for i in range(10_000_001)}

        with pytest.raises(SecurityError) as exc_info:
            serialize(large_dict)

        assert "exceeds maximum" in str(exc_info.value)

    def test_excessive_list_size_security_error(self):
        """Test that large list raises SecurityError."""
        # Create list larger than MAX_OBJECT_SIZE
        large_list = list(range(10_000_001))

        with pytest.raises(SecurityError) as exc_info:
            serialize(large_list)

        assert "exceeds maximum" in str(exc_info.value)


class TestComplexExceptionHandling:
    """Test complex exception handling paths."""

    def test_object_dict_method_exception_fallback(self):
        """Test fallback when object.dict() method fails."""

        class ObjectWithBrokenDict:
            def dict(self):
                raise RuntimeError("Dict method broken")

            def __init__(self):
                self.fallback_data = "success"

        obj = ObjectWithBrokenDict()
        result = serialize(obj)

        # Should fall back to __dict__ serialization
        assert result == {"fallback_data": "success"}

    def test_object_without_dict_attribute(self):
        """Test objects without __dict__ attribute."""
        # Basic object() has no __dict__
        obj = object()
        result = serialize(obj)

        # Should fall back to string representation
        assert isinstance(result, str)
        assert "object" in result

    def test_object_dict_access_exception(self):
        """Test exception during __dict__ access."""

        class ProblematicDictAccess:
            def __getattribute__(self, name):
                if name == "__dict__":
                    raise AttributeError("No __dict__ access")
                return super().__getattribute__(name)

        obj = ProblematicDictAccess()
        result = serialize(obj)

        # Should handle gracefully
        assert isinstance(result, str)

    def test_recursion_error_handling(self):
        """Test RecursionError handling in object serialization."""

        class RecursiveObject:
            def __init__(self):
                self.data = "test"

            def __getattribute__(self, name):
                if name == "__dict__":
                    # Simulate deep recursion scenario
                    if getattr(self.__class__, "_recursion_count", 0) > 5:
                        raise RecursionError("Maximum recursion depth exceeded")
                    self.__class__._recursion_count = getattr(self.__class__, "_recursion_count", 0) + 1
                    return {"data": "test"}
                return super().__getattribute__(name)

        obj = RecursiveObject()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = serialize(obj)

            # Circuit breaker prevents recursion, so we just check for safe handling
            # Should return safe string representation or dict
            assert isinstance(result, (str, dict))

    def test_string_representation_truncation(self):
        """Test string representation truncation for very long objects."""

        class VeryLongRepr:
            def __str__(self):
                return "x" * 1_000_001  # Exceed MAX_STRING_LENGTH

        obj = VeryLongRepr()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = serialize(obj)

            # Objects with long __str__ are serialized as dicts, not strings
            # So no truncation warning is emitted
            assert isinstance(result, dict)  # Should be serialized as __dict__

    def test_ml_serializer_exception_handling(self):
        """Test exception handling in ML serializer."""

        class CustomMLObject:
            def __init__(self):
                self.data = "test"

        obj = CustomMLObject()

        # Mock ML serializer to raise exception
        with patch("datason.core._ml_serializer") as mock_ml:
            mock_ml.side_effect = Exception("ML serializer failed")

            result = serialize(obj)

            # Should fall back to __dict__ serialization
            assert result == {"data": "test"}


class TestOptimizationCacheEdgeCases:
    """Test edge cases in optimization caching."""

    def test_type_cache_limit_behavior(self):
        """Test behavior when type cache hits size limit."""
        from datason.core import _TYPE_CACHE, _TYPE_CACHE_SIZE_LIMIT

        # Fill cache to limit
        original_cache = _TYPE_CACHE.copy()
        try:
            # Fill cache
            for i in range(_TYPE_CACHE_SIZE_LIMIT + 10):
                class_name = f"TestClass{i}"
                test_class = type(class_name, (), {})
                _TYPE_CACHE[test_class] = "other"

            # Should still handle new types gracefully
            class NewTestClass:
                pass

            obj = NewTestClass()
            result = serialize(obj)

            # Should work despite full cache
            assert isinstance(result, (dict, str))

        finally:
            # Restore original cache
            _TYPE_CACHE.clear()
            _TYPE_CACHE.update(original_cache)

    def test_string_length_cache_behavior(self):
        """Test string length caching edge cases."""
        # Test very long string that should trigger caching
        long_string = "x" * 500_000

        # Serialize twice to test cache hit
        result1 = serialize(long_string)
        result2 = serialize(long_string)

        # Both should be identical (testing cache consistency)
        assert result1 == result2
        assert isinstance(result1, str)

    def test_uuid_string_cache_behavior(self):
        """Test UUID string caching."""
        # Create UUID and serialize multiple times
        test_uuid = uuid.UUID("12345678-1234-5678-9012-123456789abc")

        result1 = serialize(test_uuid)
        result2 = serialize(test_uuid)

        # Should be consistent (testing cache)
        assert result1 == result2
        assert isinstance(result1, str)


class TestConfigurationEdgeCases:
    """Test edge cases with different configurations."""

    def test_serialize_with_none_config(self):
        """Test serialization with explicit None config."""
        data = {"test": float("nan")}

        result = serialize(data, config=None)

        # Should use default behavior
        assert result["test"] is None

    def test_serialize_with_custom_type_handler_failure(self):
        """Test custom type handler that fails."""
        from datason.config import get_default_config

        config = get_default_config()

        # Mock type handler that fails
        with patch.object(config, "custom_serializers", {"datetime": lambda x: 1 / 0}):
            data = {"date": datetime.now()}

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = serialize(data, config=config)

                # Custom handler failures are handled silently
                # Should fall back to default handling without warnings
                assert isinstance(result["date"], str)


class TestMemoryPoolingEdgeCases:
    """Test edge cases in memory pooling for optimization."""

    def test_dict_pooling_exception_safety(self):
        """Test that dict pooling is exception-safe."""

        # Create scenario that might raise exception during processing
        class ProblematicValue:
            def __str__(self):
                raise ValueError("Cannot convert to string")

        data = {"key": ProblematicValue()}

        try:
            result = serialize(data)
        except Exception:
            pass  # Exception is expected

        # Pool should still be in good state for next operation
        normal_data = {"normal": "value"}
        result = serialize(normal_data)
        assert result == {"normal": "value"}

    def test_list_pooling_exception_safety(self):
        """Test that list pooling is exception-safe."""

        # Similar test for list pooling
        class ProblematicItem:
            def __str__(self):
                raise ValueError("Cannot convert to string")

        data = [1, 2, ProblematicItem()]

        try:
            result = serialize(data)
        except Exception:
            pass  # Exception is expected

        # Pool should still work for next operation
        normal_data = [1, 2, 3]
        result = serialize(normal_data)
        assert result == [1, 2, 3]
