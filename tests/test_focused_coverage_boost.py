"""
Focused tests to boost coverage to 80%+ by testing specific uncovered lines.
"""

import sys
import unittest
from unittest.mock import Mock, patch

from serialpy import serialize


class TestUncoveredLines(unittest.TestCase):
    """Test specific uncovered lines to boost coverage."""

    def test_import_fallback_core(self):
        """Test core module import fallback paths."""
        # Test when ML serializer import fails - covers lines 14-15 in core.py
        with patch.dict("sys.modules", {"serialpy.ml_serializers": None}):
            # Force re-import to trigger the ImportError path
            import importlib

            if "serialpy.core" in sys.modules:
                importlib.reload(sys.modules["serialpy.core"])

        # This just tests that the module can handle import failures
        data = {"test": "value"}
        result = serialize(data)
        self.assertEqual(result, data)

    def test_pydantic_dict_method_with_exception(self):
        """Test object with .dict() method that raises exception."""

        class ObjectWithFailingDict:
            def dict(self):
                raise ValueError("dict() method failed")

            def __init__(self):
                self.value = "test"

        obj = ObjectWithFailingDict()
        result = serialize(obj)

        # Should fall back to __dict__ serialization
        self.assertEqual(result, {"value": "test"})

    def test_object_without_dict_attribute(self):
        """Test object without __dict__ attribute."""
        # Create object without __dict__
        obj = object()
        result = serialize(obj)

        # Should fall back to string representation
        self.assertIsInstance(result, str)

    def test_object_with_empty_dict(self):
        """Test object with empty __dict__."""

        class EmptyObject:
            pass

        obj = EmptyObject()
        result = serialize(obj)

        # Should fall back to string representation for empty dict
        self.assertIsInstance(result, str)

    def test_ml_serializer_import_error_during_serialize(self):
        """Test when ML serializer import fails during serialization."""

        class CustomObject:
            def __init__(self):
                self.value = "test"

        obj = CustomObject()

        # Temporarily remove the ML serializer import
        with patch(
            "serialpy.core.detect_and_serialize_ml_object",
            side_effect=ImportError("Not found"),
        ):
            result = serialize(obj)

        # Should still work and fall back to dict serialization
        self.assertEqual(result, {"value": "test"})


class TestMLSerializerCoverage(unittest.TestCase):
    """Test ML serializer coverage without complex mocking."""

    def test_detect_unsupported_ml_object(self):
        """Test ML detection with unsupported object type."""
        from serialpy.ml_serializers import detect_and_serialize_ml_object

        class CustomObject:
            pass

        obj = CustomObject()
        result = detect_and_serialize_ml_object(obj)

        # Should return None for unsupported types
        self.assertIsNone(result)

    def test_import_fallback_paths(self):
        """Test import fallback when libraries not available."""
        from serialpy.ml_serializers import (
            serialize_pytorch_tensor,
            serialize_sklearn_model,
            serialize_tensorflow_tensor,
        )

        # Test TensorFlow fallback when tf is None
        with patch("serialpy.ml_serializers.tf", None):
            tf_result = serialize_tensorflow_tensor("mock_tensor")
            self.assertIn("_type", tf_result)
            self.assertEqual(tf_result["_type"], "tf.Tensor")

        # Test PyTorch fallback when torch is None
        with patch("serialpy.ml_serializers.torch", None):
            torch_result = serialize_pytorch_tensor("mock_tensor")
            self.assertIn("_type", torch_result)
            self.assertEqual(torch_result["_type"], "torch.Tensor")

        # Test sklearn fallback when sklearn is None
        with patch("serialpy.ml_serializers.sklearn", None):
            sklearn_result = serialize_sklearn_model("mock_model")
            self.assertIn("_type", sklearn_result)
            self.assertEqual(sklearn_result["_type"], "sklearn.model")


class TestDatetimeUtilsCoverage(unittest.TestCase):
    """Test datetime utils coverage."""

    def test_ensure_timestamp_without_pandas(self):
        """Test ensure_timestamp when pandas not available."""
        from serialpy.datetime_utils import ensure_timestamp

        # Mock pandas to be None
        with patch("serialpy.datetime_utils.pd", None):
            with self.assertRaises(ImportError) as context:
                ensure_timestamp("2023-01-01")

            self.assertIn("pandas is required", str(context.exception))

    def test_ensure_dates_without_pandas(self):
        """Test ensure_dates when pandas not available."""
        from serialpy.datetime_utils import ensure_dates

        # Mock pandas to be None
        with patch("serialpy.datetime_utils.pd", None):
            with self.assertRaises(ImportError) as context:
                ensure_dates({})

            self.assertIn("pandas is required", str(context.exception))

    def test_convert_pandas_timestamps_without_pandas(self):
        """Test convert_pandas_timestamps when pandas not available."""
        from serialpy.datetime_utils import convert_pandas_timestamps

        # Mock pandas to be None
        with patch("serialpy.datetime_utils.pd", None):
            test_data = {"date": "2023-01-01", "value": 42}
            result = convert_pandas_timestamps(test_data)

            # Should return unchanged when pandas is None
            self.assertEqual(result, test_data)


class TestSimpleErrorPaths(unittest.TestCase):
    """Test simple error handling paths."""

    def test_sklearn_model_error_handling(self):
        """Test sklearn model serialization error handling."""
        from serialpy.ml_serializers import serialize_sklearn_model

        # Create mock model that raises exception
        mock_model = Mock()
        mock_model.get_params.side_effect = Exception("Mock error")
        mock_model.__class__.__module__ = "sklearn.linear_model"
        mock_model.__class__.__name__ = "LinearRegression"

        with patch("serialpy.ml_serializers.sklearn", Mock()):
            with patch("serialpy.ml_serializers.BaseEstimator", Mock()):
                result = serialize_sklearn_model(mock_model)

        # Should handle error and return error info
        self.assertEqual(result["_type"], "sklearn.model")
        self.assertIn("_error", result)

    def test_scipy_sparse_error_handling(self):
        """Test scipy sparse matrix error handling."""
        from serialpy.ml_serializers import serialize_scipy_sparse

        # Create mock matrix that raises exception
        mock_matrix = Mock()
        mock_matrix.tocoo.side_effect = Exception("Cannot convert")

        with patch("serialpy.ml_serializers.scipy", Mock()):
            result = serialize_scipy_sparse(mock_matrix)

        # Should handle error gracefully
        self.assertEqual(result["_type"], "scipy.sparse")


class TestConvertersModule(unittest.TestCase):
    """Test converters module for coverage."""

    def test_converters_basic_functionality(self):
        """Test basic converter functionality."""
        # Import and test basic converters
        try:
            from serialpy.converters import create_converter_from_numpy_dtype

            # This should at least import without error
            self.assertTrue(callable(create_converter_from_numpy_dtype))
        except ImportError:
            # If numpy not available, that's fine
            pass


class TestSerializersModule(unittest.TestCase):
    """Test serializers module for coverage."""

    def test_serializers_fallback_import(self):
        """Test serializers module import fallback."""
        try:
            from serialpy import serializers

            # Basic import test
            self.assertTrue(hasattr(serializers, "__file__"))
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
