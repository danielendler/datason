"""
Tests for ML/AI serializers in datason.

This module tests the ML serialization functionality with actual ML libraries
when available, and fallback behavior when they're not.
"""

import warnings
from unittest.mock import Mock, patch

import pytest

# Optional dependency imports
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from datason.ml_serializers import (
    detect_and_serialize_ml_object,
    get_ml_library_info,
    serialize_huggingface_tokenizer,
    serialize_jax_array,
    serialize_pil_image,
    serialize_pytorch_tensor,
    serialize_scipy_sparse,
    serialize_sklearn_model,
)

# Check which libraries are available
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import jax  # noqa: F401 - Used in patch testing on line 374
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import scipy.sparse

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class TestMLLibraryAvailability:
    """Test library availability detection."""

    def test_get_ml_library_info(self) -> None:
        """Test getting ML library availability information."""
        info = get_ml_library_info()

        assert isinstance(info, dict)
        expected_keys = {
            "torch",
            "tensorflow",
            "jax",
            "sklearn",
            "scipy",
            "PIL",
            "transformers",
        }
        assert set(info.keys()) == expected_keys

        # All values should be boolean
        for _key, value in info.items():
            assert isinstance(value, bool)

    def test_library_availability_matches_actual(self) -> None:
        """Test that library info matches actual availability."""
        info = get_ml_library_info()

        assert info["torch"] == HAS_TORCH
        assert info["sklearn"] == HAS_SKLEARN
        assert info["jax"] == HAS_JAX
        assert info["scipy"] == HAS_SCIPY
        assert info["PIL"] == HAS_PIL
        assert info["transformers"] == HAS_TRANSFORMERS


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestPyTorchSerialization:
    """Test PyTorch tensor serialization with actual PyTorch."""

    def test_serialize_simple_tensor(self) -> None:
        """Test serialization of a simple PyTorch tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = serialize_pytorch_tensor(tensor)

        assert result["__datason_type__"] == "torch.Tensor"
        assert result["__datason_value__"]["shape"] == [3]
        assert result["__datason_value__"]["dtype"] == "torch.float32"
        assert result["__datason_value__"]["data"] == [1.0, 2.0, 3.0]
        assert result["__datason_value__"]["device"] == "cpu"
        assert result["__datason_value__"]["requires_grad"] is False

    def test_serialize_multidimensional_tensor(self) -> None:
        """Test serialization of multidimensional tensor."""
        tensor = torch.randn(2, 3, 4)
        result = serialize_pytorch_tensor(tensor)

        assert result["__datason_type__"] == "torch.Tensor"
        assert result["__datason_value__"]["shape"] == [2, 3, 4]
        assert len(result["__datason_value__"]["data"]) == 2
        assert len(result["__datason_value__"]["data"][0]) == 3
        assert len(result["__datason_value__"]["data"][0][0]) == 4

    def test_serialize_tensor_with_gradients(self) -> None:
        """Test serialization of tensor with gradient tracking."""
        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = serialize_pytorch_tensor(tensor)

        assert result["__datason_value__"]["requires_grad"] is True

    def test_detect_pytorch_tensor(self) -> None:
        """Test automatic detection of PyTorch tensors."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = detect_and_serialize_ml_object(tensor)

        assert result is not None
        assert result["__datason_type__"] == "torch.Tensor"


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
class TestSklearnSerialization:
    """Test scikit-learn model serialization with actual sklearn."""

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
    def test_serialize_random_forest(self) -> None:
        """Test serialization of RandomForestClassifier."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Fit on simple data
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        model.fit(X, y)

        result = serialize_sklearn_model(model)

        assert result["__datason_type__"] == "sklearn.model"
        assert "RandomForestClassifier" in result["__datason_value__"]["class"]
        assert result["__datason_value__"]["fitted"] is True
        assert result["__datason_value__"]["params"]["n_estimators"] == 10
        assert result["__datason_value__"]["params"]["random_state"] == 42

    def test_serialize_unfitted_model(self) -> None:
        """Test serialization of unfitted model."""
        model = LogisticRegression(random_state=42)
        result = serialize_sklearn_model(model)

        assert result["__datason_type__"] == "sklearn.model"
        assert "LogisticRegression" in result["__datason_value__"]["class"]
        assert result["__datason_value__"]["fitted"] is False
        assert result["__datason_value__"]["params"]["random_state"] == 42

    def test_detect_sklearn_model(self) -> None:
        """Test automatic detection of sklearn models."""
        model = RandomForestClassifier(n_estimators=5)
        result = detect_and_serialize_ml_object(model)

        assert result is not None
        assert result["__datason_type__"] == "sklearn.model"


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestJAXSerialization:
    """Test JAX array serialization with actual JAX."""

    def test_serialize_jax_array(self) -> None:
        """Test serialization of JAX array."""
        array = jnp.array([1.0, 2.0, 3.0])
        result = serialize_jax_array(array)

        assert result["__datason_type__"] == "jax.Array"
        assert result["__datason_value__"]["shape"] == [3]
        assert "float32" in result["__datason_value__"]["dtype"]
        assert result["__datason_value__"]["data"] == [1.0, 2.0, 3.0]

    def test_serialize_jax_multidimensional(self) -> None:
        """Test serialization of multidimensional JAX array."""
        array = jnp.ones((2, 3))
        result = serialize_jax_array(array)

        assert result["__datason_value__"]["shape"] == [2, 3]
        assert result["__datason_value__"]["data"] == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

    def test_detect_jax_array(self) -> None:
        """Test automatic detection of JAX arrays."""
        array = jnp.array([1, 2, 3])
        result = detect_and_serialize_ml_object(array)

        assert result is not None
        assert result["__datason_type__"] == "jax.Array"


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
class TestScipySerialization:
    """Test scipy sparse matrix serialization with actual scipy."""

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
    def test_serialize_csr_matrix(self) -> None:
        """Test serialization of CSR sparse matrix."""
        data = np.array([1, 2, 3])
        indices = np.array([0, 1, 2])
        indptr = np.array([0, 1, 2, 3])
        matrix = scipy.sparse.csr_matrix((data, indices, indptr), shape=(3, 3))

        result = serialize_scipy_sparse(matrix)

        assert result["__datason_type__"] == "scipy.sparse"
        assert result["__datason_value__"]["format"] == "csr_matrix"
        assert result["__datason_value__"]["shape"] == [3, 3]
        assert result["__datason_value__"]["nnz"] == 3
        assert len(result["__datason_value__"]["data"]) == 3

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
    def test_serialize_coo_matrix(self) -> None:
        """Test serialization of COO sparse matrix."""
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2])
        data = np.array([1.0, 2.0, 3.0])
        matrix = scipy.sparse.coo_matrix((data, (row, col)), shape=(3, 3))

        result = serialize_scipy_sparse(matrix)

        assert result["__datason_type__"] == "scipy.sparse"
        assert result["__datason_value__"]["format"] == "coo_matrix"
        assert result["__datason_value__"]["shape"] == [3, 3]

    def test_detect_scipy_sparse(self) -> None:
        """Test automatic detection of scipy sparse matrices."""
        matrix = scipy.sparse.eye(3)
        result = detect_and_serialize_ml_object(matrix)

        assert result is not None
        assert result["__datason_type__"] == "scipy.sparse"


@pytest.mark.skipif(not HAS_PIL, reason="PIL not available")
class TestPILSerialization:
    """Test PIL Image serialization with actual PIL."""

    def test_serialize_rgb_image(self):
        """Test serialization of RGB PIL images."""
        pytest.importorskip("PIL")
        from PIL import Image

        # Create a simple RGB image
        img = Image.new("RGB", (64, 64), color="red")

        result = serialize_pil_image(img)

        assert result["__datason_type__"] == "PIL.Image"
        assert result["__datason_value__"]["mode"] == "RGB"
        # Size is returned as tuple, not list
        assert result["__datason_value__"]["size"] == (64, 64)
        assert isinstance(result["__datason_value__"]["data"], str)  # Base64 encoded

    def test_serialize_grayscale_image(self):
        """Test serialization of grayscale PIL images."""
        pytest.importorskip("PIL")
        from PIL import Image

        # Create a simple grayscale image
        img = Image.new("L", (32, 32), color=128)

        result = serialize_pil_image(img)

        assert result["__datason_type__"] == "PIL.Image"
        assert result["__datason_value__"]["mode"] == "L"
        # Size is returned as tuple, not list
        assert result["__datason_value__"]["size"] == (32, 32)
        assert isinstance(result["__datason_value__"]["data"], str)  # Base64 encoded

    def test_detect_pil_image(self) -> None:
        """Test automatic detection of PIL images."""
        image = Image.new("RGB", (10, 10))
        result = detect_and_serialize_ml_object(image)

        assert result is not None
        assert result["__datason_type__"] == "PIL.Image"


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not available")
class TestTransformersSerialization:
    """Test HuggingFace transformers serialization."""

    def test_serialize_tokenizer_metadata(self) -> None:
        """Test serialization of tokenizer metadata."""
        # Use a small, fast tokenizer for testing
        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", trust_remote_code=True)
            result = serialize_huggingface_tokenizer(tokenizer)

            assert result["__datason_type__"] == "transformers.tokenizer"
            assert "tokenizer" in result["__datason_value__"]["class"].lower()
            assert result["__datason_value__"]["vocab_size"] > 0
            assert result["__datason_value__"]["model_max_length"] is not None
        except Exception:
            # If download fails, test with mock
            pytest.skip("Could not download tokenizer for testing")

    def test_detect_transformers_tokenizer(self) -> None:
        """Test automatic detection of transformers tokenizers."""
        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", trust_remote_code=True)
            result = detect_and_serialize_ml_object(tokenizer)

            assert result is not None
            assert result["__datason_type__"] == "transformers.tokenizer"
        except Exception:
            pytest.skip("Could not download tokenizer for testing")


class TestMLSerializersFallbacks:
    """Test ML serializers fallback behavior when libraries aren't available."""

    def test_pytorch_tensor_fallback(self) -> None:
        """Test PyTorch tensor serialization fallback when torch not available."""
        mock_tensor = Mock()
        mock_tensor.__str__ = Mock(return_value="MockTensor([1, 2, 3])")

        with patch("datason.ml_serializers.torch", None):
            result = serialize_pytorch_tensor(mock_tensor)

        assert result == {"__datason_type__": "torch.Tensor", "__datason_value__": "MockTensor([1, 2, 3])"}

    def test_sklearn_model_fallback(self) -> None:
        """Test sklearn model serialization fallback when sklearn not available."""
        mock_model = Mock()
        mock_model.__str__ = Mock(return_value="MockSKLearnModel")

        with patch("datason.ml_serializers.sklearn", None), patch("datason.ml_serializers.BaseEstimator", None):
            result = serialize_sklearn_model(mock_model)

        assert result == {"__datason_type__": "sklearn.model", "__datason_value__": "MockSKLearnModel"}

    def test_jax_array_fallback(self) -> None:
        """Test JAX array serialization fallback when jax not available."""
        mock_array = Mock()
        mock_array.__str__ = Mock(return_value="MockJaxArray")

        with patch("datason.ml_serializers.jax", None):
            result = serialize_jax_array(mock_array)

        assert result == {"__datason_type__": "jax.Array", "__datason_value__": "MockJaxArray"}

    def test_scipy_sparse_fallback(self) -> None:
        """Test scipy sparse matrix serialization fallback when scipy not available."""
        mock_matrix = Mock()
        mock_matrix.__str__ = Mock(return_value="MockSparseMatrix")

        with patch("datason.ml_serializers.scipy", None):
            result = serialize_scipy_sparse(mock_matrix)

        assert result == {"__datason_type__": "scipy.sparse", "__datason_value__": "MockSparseMatrix"}

    def test_pil_image_fallback(self) -> None:
        """Test PIL Image serialization fallback when PIL not available."""
        mock_image = Mock()
        mock_image.__str__ = Mock(return_value="MockPILImage")

        with patch("datason.ml_serializers.Image", None):
            result = serialize_pil_image(mock_image)

        assert result == {"__datason_type__": "PIL.Image", "__datason_value__": "MockPILImage"}

    def test_transformers_tokenizer_fallback(self) -> None:
        """Test HuggingFace tokenizer serialization fallback when transformers not available."""
        mock_tokenizer = Mock()
        mock_tokenizer.__str__ = Mock(return_value="MockTokenizer")

        with patch("datason.ml_serializers.transformers", None):
            result = serialize_huggingface_tokenizer(mock_tokenizer)

        assert result == {"__datason_type__": "transformers.tokenizer", "__datason_value__": "MockTokenizer"}


class TestDetectAndSerializeMLObject:
    """Test the automatic ML object detection function."""

    def test_detect_returns_none_for_regular_objects(self) -> None:
        """Test that regular Python objects return None."""
        regular_objects = ["string", 123, [1, 2, 3], {"key": "value"}, None, Mock()]

        for obj in regular_objects:
            result = detect_and_serialize_ml_object(obj)
            assert result is None

    def test_detect_with_all_libraries_unavailable(self) -> None:
        """Test detection when all ML libraries are patched to None."""
        mock_obj = Mock()

        with patch.multiple(
            "datason.ml_serializers",
            torch=None,
            tf=None,
            jax=None,
            sklearn=None,
            BaseEstimator=None,
            scipy=None,
            Image=None,
            transformers=None,
        ):
            result = detect_and_serialize_ml_object(mock_obj)

        assert result is None


class TestMLSerializersErrorHandling:
    """Test error handling in ML serializers."""

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_sklearn_model_serialization_error_handling(self) -> None:
        """Test sklearn model serialization error handling."""
        # Create a model that will cause issues
        model = RandomForestClassifier()

        # Patch get_params to raise an exception
        with patch.object(model, "get_params", side_effect=Exception("Mock error")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = serialize_sklearn_model(model)

        assert result["__datason_type__"] == "sklearn.model"
        assert "error" in result["__datason_value__"]
        assert len(w) == 1
        assert "Could not serialize sklearn model" in str(w[0].message)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_scipy_sparse_serialization_error_handling(self) -> None:
        """Test scipy sparse matrix serialization error handling."""
        matrix = scipy.sparse.eye(3)

        # Patch tocoo to raise an exception
        with patch.object(matrix, "tocoo", side_effect=Exception("Mock sparse error")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = serialize_scipy_sparse(matrix)

        assert result["__datason_type__"] == "scipy.sparse"
        assert "error" in result["__datason_value__"]
        assert len(w) == 1
        assert "Could not serialize scipy sparse matrix" in str(w[0].message)

    @pytest.mark.skipif(not HAS_PIL, reason="PIL not available")
    def test_pil_image_serialization_error_handling(self) -> None:
        """Test PIL Image serialization error handling."""
        image = Image.new("RGB", (10, 10))

        # Patch save to raise an exception
        with patch.object(image, "save", side_effect=Exception("Mock image error")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = serialize_pil_image(image)

        assert result["__datason_type__"] == "PIL.Image"
        assert "error" in result["__datason_value__"]
        assert len(w) == 1
        assert "Could not serialize PIL Image" in str(w[0].message)
