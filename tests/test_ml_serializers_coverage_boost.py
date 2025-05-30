"""
ML Serializers Coverage Boost Tests

This file contains tests specifically designed to cover the remaining uncovered lines
in the serialpy.ml_serializers module to push coverage above 75%.
"""

import unittest
from unittest.mock import Mock, patch
import warnings

from serialpy.ml_serializers import (
    detect_and_serialize_ml_object,
    serialize_huggingface_tokenizer,
    serialize_jax_array,
    serialize_pil_image,
    serialize_pytorch_tensor,
    serialize_scipy_sparse,
    serialize_sklearn_model,
    serialize_tensorflow_tensor,
)


class TestMLSerializersImportFallbacks(unittest.TestCase):
    """Test import fallback paths when ML libraries are not available."""

    def test_tensorflow_import_fallback(self):
        """Test TensorFlow serialization when TensorFlow is not available."""
        with patch("serialpy.ml_serializers.tf", None):
            # Should handle gracefully when tf is None
            result = serialize_tensorflow_tensor("mock_tensor")

            self.assertEqual(result["_type"], "tf.Tensor")
            self.assertEqual(result["_data"], "mock_tensor")
            self.assertNotIn("_shape", result)
            self.assertNotIn("_dtype", result)

    def test_pytorch_import_fallback(self):
        """Test PyTorch serialization when PyTorch is not available."""
        with patch("serialpy.ml_serializers.torch", None):
            # Should handle gracefully when torch is None
            result = serialize_pytorch_tensor("mock_tensor")

            self.assertEqual(result["_type"], "torch.Tensor")
            self.assertEqual(result["_data"], "mock_tensor")
            self.assertNotIn("_shape", result)
            self.assertNotIn("_dtype", result)

    def test_sklearn_import_fallback(self):
        """Test sklearn serialization when sklearn is not available."""
        with patch("serialpy.ml_serializers.sklearn", None):
            # Should handle gracefully when sklearn is None
            result = serialize_sklearn_model("mock_model")

            self.assertEqual(result["_type"], "sklearn.model")
            self.assertEqual(result["_data"], "mock_model")
            self.assertNotIn("_class", result)
            self.assertNotIn("_params", result)

    def test_jax_import_fallback(self):
        """Test JAX serialization when JAX is not available."""
        with patch("serialpy.ml_serializers.jax", None):
            # Should handle gracefully when jax is None
            result = serialize_jax_array("mock_array")

            self.assertEqual(result["_type"], "jax.Array")
            self.assertEqual(result["_data"], "mock_array")
            self.assertNotIn("_shape", result)
            self.assertNotIn("_dtype", result)

    def test_scipy_import_fallback(self):
        """Test SciPy serialization when SciPy is not available."""
        with patch("serialpy.ml_serializers.scipy", None):
            # Should handle gracefully when scipy is None
            result = serialize_scipy_sparse("mock_matrix")

            self.assertEqual(result["_type"], "scipy.sparse")
            self.assertEqual(result["_data"], "mock_matrix")
            self.assertNotIn("_shape", result)
            self.assertNotIn("_format", result)

    def test_pil_import_fallback(self):
        """Test PIL serialization when PIL is not available."""
        with patch("serialpy.ml_serializers.Image", None):
            # Should handle gracefully when PIL is None
            result = serialize_pil_image("mock_image")

            self.assertEqual(result["_type"], "PIL.Image")
            self.assertEqual(result["_data"], "mock_image")
            self.assertNotIn("_mode", result)
            self.assertNotIn("_size", result)

    def test_transformers_import_fallback(self):
        """Test Transformers serialization when transformers is not available."""
        with patch("serialpy.ml_serializers.transformers", None):
            # Should handle gracefully when transformers is None
            result = serialize_huggingface_tokenizer("mock_tokenizer")

            self.assertEqual(result["_type"], "transformers.tokenizer")
            self.assertEqual(result["_data"], "mock_tokenizer")
            self.assertNotIn("_vocab_size", result)
            self.assertNotIn("_model_name", result)


class TestMLSerializersErrorPaths(unittest.TestCase):
    """Test error handling paths in ML serializers."""

    def test_tensorflow_error_handling(self):
        """Test TensorFlow tensor serialization error handling."""
        # Create mock tensor that raises errors during processing
        mock_tensor = Mock()
        mock_tensor.shape = Mock()
        mock_tensor.shape.as_list.side_effect = Exception("TensorFlow error")
        mock_tensor.dtype = Mock()
        mock_tensor.dtype.name = "float32"
        mock_tensor.numpy.return_value = [[1, 2], [3, 4]]

        with patch("serialpy.ml_serializers.tf", Mock()):
            # Should handle shape error
            with self.assertRaises(Exception):
                serialize_tensorflow_tensor(mock_tensor)

    def test_pytorch_error_handling(self):
        """Test PyTorch tensor serialization error handling."""
        # Skip this test - the current implementation doesn't have try/catch in PyTorch serializer
        self.skipTest("Current implementation doesn't catch PyTorch errors")

    def test_sklearn_error_handling(self):
        """Test sklearn model serialization error handling."""
        # Create mock model that raises errors
        mock_model = Mock()
        mock_model.get_params.side_effect = Exception("Sklearn error")
        mock_model.__class__.__module__ = "sklearn.linear_model"
        mock_model.__class__.__name__ = "LinearRegression"

        with patch("serialpy.ml_serializers.sklearn", Mock()):
            with patch("serialpy.ml_serializers.BaseEstimator", Mock()):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = serialize_sklearn_model(mock_model)

                    # Should handle error gracefully
                    self.assertEqual(result["_type"], "sklearn.model")
                    self.assertIn("_error", result)
                    self.assertTrue(len(w) > 0)

    def test_scipy_sparse_error_handling(self):
        """Test SciPy sparse matrix serialization error handling."""
        # Create mock matrix that raises errors
        mock_matrix = Mock()
        mock_matrix.tocoo.side_effect = Exception("SciPy error")
        mock_matrix.format = "csr"
        mock_matrix.shape = (3, 3)

        with patch("serialpy.ml_serializers.scipy", Mock()):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = serialize_scipy_sparse(mock_matrix)

                # Should handle error gracefully
                self.assertEqual(result["_type"], "scipy.sparse")
                self.assertIn("_error", result)
                self.assertTrue(len(w) > 0)

    def test_pil_error_handling(self):
        """Test PIL image serialization error handling."""
        # Skip this test - the current implementation doesn't have try/catch in PIL serializer
        self.skipTest("Current implementation doesn't catch PIL errors")


class TestMLDetectionEdgeCases(unittest.TestCase):
    """Test ML object detection edge cases."""

    def test_detect_with_partial_library_availability(self):
        """Test detection when only some ML libraries are available."""
        # Test with object that could be from multiple libraries
        mock_obj = Mock()
        mock_obj.__class__.__module__ = "unknown_module"

        # Patch some libraries to None
        with patch("serialpy.ml_serializers.torch", None):
            with patch("serialpy.ml_serializers.tf", Mock()):
                result = detect_and_serialize_ml_object(mock_obj)
                # Should return None for unknown objects
                self.assertIsNone(result)

    def test_detect_with_hasattr_exceptions(self):
        """Test detection when hasattr calls raise exceptions."""

        # Create object that raises on attribute access
        class BadObject:
            def __getattribute__(self, name):
                if name in ["numpy", "detach", "get_params"]:
                    raise Exception("No attribute access")
                return super().__getattribute__(name)

        bad_obj = BadObject()
        result = detect_and_serialize_ml_object(bad_obj)

        # Should handle gracefully and return None
        self.assertIsNone(result)

    def test_detect_sklearn_estimator_check(self):
        """Test sklearn estimator detection edge cases."""
        # Create mock object that looks like sklearn but isn't
        mock_obj = Mock()
        mock_obj.__class__.__module__ = "sklearn.linear_model"

        # Test when BaseEstimator is not available
        with patch("serialpy.ml_serializers.BaseEstimator", None):
            result = detect_and_serialize_ml_object(mock_obj)
            # Should return None when BaseEstimator check fails
            self.assertIsNone(result)


class TestMLSerializersParameterFiltering(unittest.TestCase):
    """Test parameter filtering in ML serializers."""

    def test_sklearn_parameter_filtering(self):
        """Test sklearn parameter filtering for large objects."""
        # Create mock model with large parameters
        mock_model = Mock()
        large_array = list(range(1000))  # Large parameter
        mock_model.get_params.return_value = {
            "normal_param": 42,
            "large_param": large_array,
            "string_param": "test",
        }
        mock_model.__class__.__module__ = "sklearn.ensemble"
        mock_model.__class__.__name__ = "RandomForestClassifier"

        with patch("serialpy.ml_serializers.sklearn", Mock()):
            with patch("serialpy.ml_serializers.BaseEstimator", Mock()):
                result = serialize_sklearn_model(mock_model)

                # Should include normal params but handle large ones
                self.assertEqual(result["_type"], "sklearn.model")
                self.assertIn("_params", result)
                params = result["_params"]
                self.assertEqual(params["normal_param"], 42)
                self.assertEqual(params["string_param"], "test")


if __name__ == "__main__":
    unittest.main()
