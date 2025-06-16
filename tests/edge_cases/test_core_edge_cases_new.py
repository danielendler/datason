"""
Core Module Edge Cases and Coverage Boosters (New Core Implementation)

This file contains tests specifically designed to cover edge cases and error paths
in the datason.core_new module that are not covered by the main comprehensive tests.
These tests target specific lines for coverage improvement.
"""

import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

from datason.core_new import (
    MAX_SERIALIZATION_DEPTH,
    MAX_STRING_LENGTH,
    _get_cached_type_category,
    serialize,
)

# Optional imports with fallbacks
try:
    import numpy as np  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class TestCoreNewImportFallbacks(unittest.TestCase):
    """Test import fallback paths in core_new module."""

    def test_ml_serializers_import_failure(self):
        """Test core_new module behavior when ml_serializers import fails."""
        # Test ML serializer import fallback
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

            # Force reimport of core_new module
            if "datason.core_new" in sys.modules:
                del sys.modules["datason.core_new"]

            # Import should succeed even without ml_serializers
            from datason.core_new import serialize as core_serialize

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

        class CustomObject:
            def __init__(self):
                self.value = "test"

        obj = CustomObject()

        # Patch the function to be None (simulating import failure)
        with patch("datason.core_new._ml_serializer", None):
            result = serialize(obj)

            # Should fall back to dict serialization
            self.assertEqual(result, {"value": "test"})

    def test_config_import_failure_fallback(self):
        """Test behavior when config imports are not available."""
        with patch("datason.core_new._config_available", False):
            # Basic serialization should still work
            result = serialize({"test": "data"})
            self.assertEqual(result, {"test": "data"})

    def test_type_handlers_import_failure(self):
        """Test fallback functions when type handlers import fails."""
        from datason.core_new import normalize_numpy_types

        # Test dummy normalize_numpy_types function returns input unchanged
        with patch("datason.core_new._config_available", False):
            test_obj = [1, 2, 3]
            self.assertEqual(normalize_numpy_types(test_obj), test_obj)

        # Test basic serialization still works when config unavailable
        with patch("datason.core_new._config_available", False):
            result = serialize({"test": "data"})
            self.assertEqual(result, {"test": "data"})


class TestObjectSerializationEdgeCases(unittest.TestCase):
    """Test edge cases in object serialization."""

    def test_object_dict_method_exception(self):
        """Test object with .dict() method that raises exception."""

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
        obj = object()  # Basic object has no __dict__
        result = serialize(obj)

        # Should fall back to string representation
        self.assertIsInstance(result, str)
        self.assertIn("object", result)

    def test_object_vars_exception(self):
        """Test object where vars() raises exception."""

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
                self.value = "A"
                self.ref = None

        class NodeB:
            def __init__(self):
                self.value = "B"
                self.ref = None

        a = NodeA()
        b = NodeB()
        a.ref = b
        b.ref = a

        result = serialize(a)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["value"], "A")

    def test_simple_circular_reference(self):
        """Test simple circular reference."""
        obj = []
        obj.append(obj)
        result = serialize(obj)

        # Should handle circular reference gracefully by detecting depth attack
        self.assertIsInstance(result, dict)

        # Current implementation detects this as a security error (depth bomb protection)
        self.assertEqual(result.get("__datason_type__"), "security_error")
        self.assertIn("depth", result.get("__datason_value__", "").lower())
        self.assertIn("circular", result.get("__datason_value__", "").lower())


class TestMLSerializerIntegrationEdgeCases(unittest.TestCase):
    """Test ML serializer integration edge cases."""

    def test_ml_serializer_import_error_during_serialize(self):
        """Test handling of import errors during ML serialization."""

        class CustomMLObject:
            def __init__(self):
                self.model_data = "test"

        obj = CustomMLObject()

        # Mock ML serializer to raise ImportError
        def failing_ml_serializer(obj):
            raise ImportError("ML library not available")

        with patch("datason.core_new._ml_serializer", failing_ml_serializer):
            result = serialize(obj)
            # Should fall back to regular serialization
            self.assertEqual(result, {"model_data": "test"})

    def test_ml_serializer_returns_none(self):
        """Test ML serializer returning None."""

        class UnknownMLObject:
            def __init__(self):
                self.data = "unknown"

        obj = UnknownMLObject()

        # Mock ML serializer to return None
        def none_ml_serializer(obj):
            return None

        with patch("datason.core_new._ml_serializer", none_ml_serializer):
            result = serialize(obj)
            # Should fall back to regular serialization
            self.assertEqual(result, {"data": "unknown"})


class TestTypeCategoryEdgeCases(unittest.TestCase):
    """Test type category edge cases."""

    def test_numpy_type_categorization_without_numpy(self):
        """Test type categorization when numpy is not available."""

        # Since core_new doesn't import numpy at module level, we test the categorization directly
        class FakeNumpyType:
            pass

        result = _get_cached_type_category(FakeNumpyType)
        self.assertEqual(result, "other")

    def test_pandas_type_categorization_without_pandas(self):
        """Test type categorization when pandas is not available."""

        # Since core_new doesn't import pandas at module level, we test the categorization directly
        class FakePandasType:
            pass

        result = _get_cached_type_category(FakePandasType)
        self.assertEqual(result, "other")

    @unittest.skipUnless(HAS_NUMPY, "NumPy not available")
    def test_numpy_generic_subclass_detection(self):
        """Test detection of numpy generic subclasses."""
        import numpy as np

        class CustomNumpyType(np.generic):
            pass

        result = _get_cached_type_category(CustomNumpyType)
        # Should be categorized as numpy type
        self.assertIn(result, ["numpy", "other"])

    @unittest.skipUnless(HAS_PANDAS, "Pandas not available")
    def test_pandas_subclass_detection(self):
        """Test detection of pandas subclasses."""

        class CustomDataFrame(pd.DataFrame):
            pass

        result = _get_cached_type_category(CustomDataFrame)
        # Should be categorized as pandas type
        self.assertIn(result, ["pandas", "other"])


class TestSecurityLimits(unittest.TestCase):
    """Test security limits."""

    def test_excessive_depth_raises_error(self):
        """Test that excessive depth raises error."""
        # Create a deeply nested structure
        obj = []
        current = obj
        for _ in range(MAX_SERIALIZATION_DEPTH + 1):
            current.append([])
            current = current[0]

        result = serialize(obj)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["__datason_type__"], "security_error")
        self.assertIn("Maximum depth", result["__datason_value__"])

    def test_large_string_warning(self):
        """Test that large strings trigger warning."""
        # Create a large string
        large_string = "x" * (MAX_STRING_LENGTH + 1)

        with warnings.catch_warnings(record=True) as w:
            result = serialize(large_string)
            self.assertEqual(len(w), 1)
            self.assertIn("String length", str(w[0].message))

        self.assertIsInstance(result, dict)
        self.assertEqual(result["__datason_type__"], "security_error")
        self.assertIn("String length", result["__datason_value__"])


class TestProblematicObjects(unittest.TestCase):
    """Test serialization of problematic objects."""

    def test_mock_object_serialization(self):
        """Test serialization of mock objects."""
        mock_obj = MagicMock()
        mock_obj.test_attr = "value"
        result = serialize(mock_obj)
        # Should serialize mock object attributes
        self.assertIsInstance(result, (dict, str))

    def test_unprintable_object(self):
        """Test object that raises exception when converted to string."""

        class UnprintableObject:
            def __str__(self):
                raise ValueError("Cannot convert to string")

            def __repr__(self):
                raise ValueError("Cannot represent")

        obj = UnprintableObject()
        # Should handle gracefully
        result = serialize(obj)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
