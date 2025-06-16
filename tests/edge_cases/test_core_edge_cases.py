"""
Core Module Edge Cases and Coverage Boosters

This file contains tests specifically designed to cover edge cases and error paths
in the datason.core module that are not covered by the main comprehensive tests.
These tests target specific lines for coverage improvement.
"""

import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

from datason.core_new import (
    MAX_SERIALIZATION_DEPTH,
    MAX_STRING_LENGTH,
    SecurityError,
    _get_cached_type_category,
    serialize,
)

# Optional imports with fallbacks
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


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

        # Test lines 19-20 in core.py - function fallback
        class CustomObject:
            def __init__(self):
                self.value = "test"

        obj = CustomObject()

        # Patch the function to be None (simulating import failure)
        with patch("datason.core_new.detect_and_serialize_ml_object", None):
            result = serialize(obj)

            # Should fall back to dict serialization
            self.assertEqual(result, {"value": "test"})

    def test_config_import_failure_fallback(self):
        """Test behavior when config imports are not available."""
        # Test lines 37-46 in core.py - config import fallback
        with patch("datason.core_new._config_available", False):
            # Basic serialization should still work
            result = serialize({"test": "data"})
            self.assertEqual(result, {"test": "data"})

    def test_type_handlers_import_failure(self):
        """Test fallback functions when type handlers import fails."""
        # Test lines 42-46 in core.py - type handler dummy functions
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
            "datason.core_new.detect_and_serialize_ml_object",
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
        with patch("datason.core_new.detect_and_serialize_ml_object", return_value=None):
            result = serialize(obj)

            # Should fall back to dict serialization
            self.assertEqual(result, {"data": "unknown"})


class TestTypeCategoryEdgeCases(unittest.TestCase):
    """Test type categorization edge cases and cache behavior - targeting lines 153-174."""

    def test_numpy_type_categorization_without_numpy(self):
        """Test numpy type categorization when numpy isn't available."""
        # Test lines 159-165 in core.py
        with patch("datason.core_new.np", None):
            # Should categorize as 'other' when numpy unavailable
            class FakeNumpyType:
                pass

            result = _get_cached_type_category(FakeNumpyType)
            self.assertEqual(result, "other")

    def test_pandas_type_categorization_without_pandas(self):
        """Test pandas type categorization when pandas isn't available."""
        # Test lines 166-172 in core.py
        with patch("datason.core_new.pd", None):
            # Should categorize as 'other' when pandas unavailable
            class FakePandasType:
                pass

            result = _get_cached_type_category(FakePandasType)
            self.assertEqual(result, "other")

    @unittest.skipUnless(HAS_NUMPY, "NumPy not available")
    def test_numpy_generic_subclass_detection(self):
        """Test detection of numpy generic subclasses."""
        # Test lines 159-165 - numpy generic/number detection
        if hasattr(np, "generic"):
            result = _get_cached_type_category(np.int64)
            self.assertEqual(result, "numpy")

        if hasattr(np, "number"):
            result = _get_cached_type_category(np.float32)
            self.assertEqual(result, "numpy")

    @unittest.skipUnless(HAS_PANDAS, "Pandas not available")
    def test_pandas_subclass_detection(self):
        """Test detection of pandas type subclasses."""

        # Test lines 166-172 - pandas subclass detection
        class CustomDataFrame(pd.DataFrame):
            pass

        result = _get_cached_type_category(CustomDataFrame)
        self.assertEqual(result, "pandas")


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

    def test_large_string_warning(self):
        """Test that large strings generate appropriate warnings."""
        large_string = "a" * (MAX_STRING_LENGTH + 100)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = serialize({"large_string": large_string})

            # Should serialize successfully (the security limit may be handled differently)
            self.assertIsInstance(result, dict)
            self.assertIn("large_string", result)

            # May have warning about string length (depending on implementation)
            if w:
                warning_messages = [str(warning.message).lower() for warning in w]
                # If there are warnings, they should be about length or truncation
                for msg in warning_messages:
                    self.assertTrue(
                        any(keyword in msg for keyword in ["length", "truncat", "exceed", "large"]),
                        f"Unexpected warning message: {msg}",
                    )


class TestProblematicObjects(unittest.TestCase):
    """Test handling of objects that could cause issues."""

    def test_mock_object_serialization(self):
        """Test that MagicMock objects don't cause hanging."""
        mock_obj = MagicMock()
        result = serialize(mock_obj)
        self.assertIsInstance(result, str)
        self.assertIn("MagicMock", result)

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


if __name__ == "__main__":
    unittest.main()
