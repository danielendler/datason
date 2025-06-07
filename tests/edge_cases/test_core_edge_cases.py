"""
Core Module Edge Cases and Coverage Boosters

This file contains ALL tests specifically designed to cover edge cases and error paths
in the datason.core module. Consolidated from multiple scattered files to avoid duplication.

This includes:
- Import failure scenarios
- Object serialization edge cases
- Security and limits testing
- Circular reference handling
- Performance edge cases
- Error handling paths
"""

import os
import sys
import time
import unittest
import warnings
from io import BytesIO, StringIO
from unittest.mock import MagicMock, Mock, patch

from datason.core import (
    MAX_OBJECT_SIZE,
    MAX_SERIALIZATION_DEPTH,
    MAX_STRING_LENGTH,
    SecurityError,
    serialize,
)

# Environment detection for CI vs local testing
IS_CI = any(
    [
        os.getenv("CI") == "true",
        os.getenv("GITHUB_ACTIONS") == "true",
        os.getenv("TRAVIS") == "true",
        os.getenv("CIRCLECI") == "true",
        bool(os.getenv("JENKINS_URL")),
        os.getenv("GITLAB_CI") == "true",
    ]
)


class TestCoreImportFallbacks(unittest.TestCase):
    """Test import fallback paths in core module."""

    def test_ml_serializers_import_failure(self):
        """Test core module behavior when ml_serializers import fails."""
        # Test lines 14-15 in core.py - ML serializer import fallback
        original_modules = sys.modules.copy()

        # Remove ml_serializers from modules to force import error
        if "datason.ml_serializers" in sys.modules:
            del sys.modules["datason.ml_serializers"]

        # Temporarily patch __import__ to raise ImportError for ml_serializers
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "datason.ml_serializers":
                raise ImportError("ML serializers not available")
            return original_import(name, *args, **kwargs)

        try:
            __builtins__["__import__"] = mock_import

            # Force reimport of core module
            if "datason.core" in sys.modules:
                del sys.modules["datason.core"]

            # Import should succeed even without ml_serializers
            from datason.core import serialize as core_serialize

            # Test serialization still works
            result = core_serialize({"test": "value"})
            self.assertEqual(result, {"test": "value"})

        finally:
            # Restore original state
            __builtins__["__import__"] = original_import
            sys.modules.clear()
            sys.modules.update(original_modules)

    def test_ml_serializer_function_fallback(self):
        """Test fallback when detect_and_serialize_ml_object is not available."""

        # Test lines 19-20 in core.py - function fallback
        class CustomObject:
            def __init__(self):
                self.value = "test"

        obj = CustomObject()

        # Patch the function to be None (simulating import failure)
        with patch("datason.core.detect_and_serialize_ml_object", None):
            result = serialize(obj)

            # Should fall back to dict serialization
            self.assertEqual(result, {"value": "test"})


class TestObjectSerializationEdgeCases(unittest.TestCase):
    """Test edge cases in object serialization."""

    def test_object_dict_method_exception(self):
        """Test object with .dict() method that raises exception."""

        # Test line 93 in core.py - exception handling in dict() method
        class ObjectWithFailingDict:
            def dict(self):
                raise ValueError("dict() method failed")

            def __init__(self):
                self.fallback_value = "test"

        obj = ObjectWithFailingDict()
        result = serialize(obj)

        # Should fall back to __dict__ serialization
        self.assertEqual(result, {"fallback_value": "test"})

    def test_object_without_dict_attribute(self):
        """Test object without __dict__ attribute."""
        # Test line 100 in core.py - fallback when no __dict__
        obj = object()  # Basic object has no __dict__
        result = serialize(obj)

        # Should fall back to string representation
        self.assertIsInstance(result, str)
        self.assertIn("object", result)

    def test_object_vars_exception(self):
        """Test object where vars() raises exception."""

        # Test line 106 in core.py - vars() exception handling
        class BadVarsObject:
            def __getattribute__(self, name):
                if name == "__dict__":
                    raise AttributeError("No __dict__")
                return super().__getattribute__(name)

        obj = BadVarsObject()
        result = serialize(obj)

        # Should fall back to string representation
        self.assertIsInstance(result, str)

    def test_object_empty_dict(self):
        """Test object with empty __dict__."""

        class EmptyObject:
            pass

        obj = EmptyObject()
        result = serialize(obj)
        # With new type handler system, empty __dict__ returns empty dict
        self.assertEqual(result, {})

    def test_object_with_failing_dict_method(self):
        """Test object with .dict() method that raises exception."""

        class BadDictObject:
            def dict(self):
                raise ValueError("dict() method failed")

            def __init__(self):
                self.value = "test"

        obj = BadDictObject()
        result = serialize(obj)
        # Should fall back to __dict__ serialization
        self.assertIn("value", result)

    def test_object_with_failing_vars(self):
        """Test object with vars() that raises exception."""

        class BadVarsObject:
            def __init__(self):
                # Create an object that vars() might fail on
                self.value = "test"

            def __getattribute__(self, name):
                if name == "__dict__":
                    raise AttributeError("No __dict__ access")
                return super().__getattribute__(name)

        obj = BadVarsObject()
        result = serialize(obj)
        # Should fall back to string representation
        self.assertIsInstance(result, str)

    def test_unprintable_object(self):
        """Test object that fails to convert to string."""

        class UnprintableObject:
            def __str__(self):
                raise ValueError("Cannot convert to string")

            def __repr__(self):
                raise ValueError("Cannot convert to repr")

        obj = UnprintableObject()
        result = serialize(obj)
        # Should handle gracefully
        self.assertIsInstance(result, (str, dict))


class TestCircularReferenceEdgeCases(unittest.TestCase):
    """Test circular reference detection edge cases."""

    def test_circular_reference_in_object(self):
        """Test handling of circular references in custom objects."""

        class CircularObject:
            def __init__(self):
                self.value = "test"
                self.self_ref = self

        obj = CircularObject()
        result = serialize(obj)

        # Should handle circular reference gracefully
        self.assertIsInstance(result, (dict, str))
        if isinstance(result, dict):
            # Should not cause infinite recursion
            self.assertIn("value", result)

    def test_nested_circular_reference(self):
        """Test nested circular references."""

        class NodeA:
            def __init__(self):
                self.type = "A"

        class NodeB:
            def __init__(self):
                self.type = "B"

        a = NodeA()
        b = NodeB()
        a.ref = b
        b.ref = a

        # This should not cause infinite recursion
        result = serialize({"nodes": [a, b]})
        self.assertIsInstance(result, dict)

    def test_simple_circular_reference(self):
        """Test that simple circular references are handled safely."""
        a = {}
        b = {"a": a}
        a["b"] = b  # Create circular reference

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = serialize(a)

            # Should get a warning about circular reference
            assert len(w) >= 1
            assert "circular reference" in str(w[0].message).lower()

            # Result should be safe (no infinite recursion)
            assert isinstance(result, dict)

    def test_list_circular_reference(self):
        """Test circular references in lists."""
        a = []
        b = [a]
        a.append(b)  # Create circular reference

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = serialize(a)
            # Should handle gracefully without infinite recursion
            self.assertIsInstance(result, list)

    def test_dict_circular_reference(self):
        """Test circular references in dictionaries."""
        obj1 = {"name": "obj1"}
        obj2 = {"name": "obj2", "ref": obj1}
        obj1["ref"] = obj2  # Create circular reference

        result = serialize(obj1)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], "obj1")


class TestSecurityLimits(unittest.TestCase):
    """Test security limits and protections."""

    def test_excessive_depth_raises_error(self):
        """Test that excessive nesting depth raises SecurityError."""
        # Create deeply nested structure that exceeds limit
        nested_data = {}
        current = nested_data

        # Create nesting deeper than the limit
        for i in range(MAX_SERIALIZATION_DEPTH + 10):
            current["nested"] = {}
            current = current["nested"]
        current["value"] = "deep"

        # Should raise SecurityError
        with self.assertRaises(SecurityError):
            serialize(nested_data)

    def test_large_string_truncation(self):
        """Test that large strings are truncated."""
        large_string = "a" * (MAX_STRING_LENGTH + 100)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = serialize({"large_string": large_string})

            # Should be truncated
            self.assertLessEqual(len(result["large_string"]), MAX_STRING_LENGTH)
            # Should have warning
            self.assertTrue(any("truncat" in str(warning.message).lower() for warning in w))

    def test_security_constants_exist(self):
        """Test that security constants are properly defined."""
        self.assertIsInstance(MAX_SERIALIZATION_DEPTH, int)
        self.assertGreater(MAX_SERIALIZATION_DEPTH, 10)
        self.assertLess(MAX_SERIALIZATION_DEPTH, 10000)

        self.assertIsInstance(MAX_OBJECT_SIZE, int)
        self.assertGreater(MAX_OBJECT_SIZE, 1000)

        self.assertIsInstance(MAX_STRING_LENGTH, int)
        self.assertGreater(MAX_STRING_LENGTH, 100)

    def test_security_error_class(self):
        """Test that SecurityError class works properly."""
        with self.assertRaises(SecurityError):
            raise SecurityError("Test security error")

        # Test it's a proper exception subclass
        self.assertTrue(issubclass(SecurityError, Exception))


class TestProblematicObjects(unittest.TestCase):
    """Test handling of objects that could cause issues."""

    def test_mock_object_serialization(self):
        """Test that MagicMock objects don't cause hanging."""
        mock_obj = MagicMock()
        result = serialize(mock_obj)
        self.assertIsInstance(result, str)
        self.assertIn("MagicMock", result)

    def test_bytesio_object_serialization(self):
        """Test that BytesIO objects don't cause hanging."""
        bio = BytesIO(b"test data")
        result = serialize(bio)
        self.assertIsInstance(result, str)
        self.assertIn("BytesIO", result)

    def test_problematic_object_combination(self):
        """Test combination of problematic objects."""

        class ProblematicObject:
            def __init__(self):
                self.file_handle = BytesIO(b"test data")
                self.string_io = StringIO("test string")
                self.mock_connection = MagicMock()
                self.mock_object = Mock()

        obj = ProblematicObject()
        result = serialize(obj)
        self.assertIsInstance(result, dict)

        # Verify problematic objects are handled safely
        self.assertIn("file_handle", result)
        self.assertIn("mock_connection", result)


class TestMLSerializerIntegrationEdgeCases(unittest.TestCase):
    """Test ML serializer integration edge cases."""

    def test_ml_serializer_import_error_during_serialize(self):
        """Test ImportError during ML object serialization."""

        class CustomMLObject:
            def __init__(self):
                self.weights = [1, 2, 3]

        obj = CustomMLObject()

        # Temporarily patch to cause import error during serialization
        with patch(
            "datason.core.detect_and_serialize_ml_object",
            side_effect=ImportError("Module not found"),
        ):
            result = serialize(obj)

            # Should fall back to dict serialization
            self.assertEqual(result, {"weights": [1, 2, 3]})

    def test_ml_serializer_returns_none(self):
        """Test when ML serializer returns None."""

        class UnknownMLObject:
            def __init__(self):
                self.data = "unknown"

        obj = UnknownMLObject()

        # Mock ML serializer to return None (object not recognized)
        with patch("datason.core.detect_and_serialize_ml_object", return_value=None):
            result = serialize(obj)

            # Should fall back to dict serialization
            self.assertEqual(result, {"data": "unknown"})


class TestPerformanceEdgeCases(unittest.TestCase):
    """Test performance-related edge cases."""

    def test_serialization_speed_simple_objects(self):
        """Test that simple objects serialize very quickly."""
        data = {"simple": "data", "number": 42, "list": [1, 2, 3]}

        start_time = time.time()
        result = serialize(data)
        end_time = time.time()

        time_taken = end_time - start_time
        self.assertLess(time_taken, 1.0)  # Should be very fast
        self.assertEqual(result, data)

    def test_no_hanging_on_deep_nesting(self):
        """Test that deep nesting doesn't cause hanging."""
        # Create reasonably deep nesting (within limits)
        nested_data = {}
        current = nested_data

        depth = min(MAX_SERIALIZATION_DEPTH - 5, 50)  # Stay within limits
        for i in range(depth):
            current["level"] = i
            current["nested"] = {}
            current = current["nested"]
        current["end"] = True

        start_time = time.time()
        result = serialize(nested_data)
        end_time = time.time()

        # Should complete quickly
        self.assertLess(end_time - start_time, 5.0)
        self.assertIsInstance(result, dict)
