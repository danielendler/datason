"""
Core Module Edge Cases and Coverage Boosters

This file contains tests specifically designed to cover edge cases and error paths
in the datason.core module that are not covered by the main comprehensive tests.
These tests target specific lines for coverage improvement.
"""

import sys
import unittest
from unittest.mock import patch

from datason.core import serialize


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
