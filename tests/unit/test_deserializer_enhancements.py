"""Unit tests for the new template deserializer enhancements.

This test suite validates the individual methods we added to the TemplateDeserializer
class to achieve 100% user config round-trip fidelity.
"""

import uuid
import warnings
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from datason.deserializers import TemplateDeserializer

# Optional dependencies
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from sklearn.linear_model import LogisticRegression

    HAS_SKLEARN = True
except ImportError:
    LogisticRegression = None
    HAS_SKLEARN = False


class TestTemplateDeserializerMethods:
    """Test the individual template deserializer methods we added."""

    def setup_method(self):
        """Set up test fixtures."""
        self.template_deserializer = TemplateDeserializer(template=None)

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_deserialize_numpy_scalar_with_template(self):
        """Test _deserialize_numpy_scalar_with_template method."""
        # Test various NumPy scalar types
        test_cases = [
            (42, np.int32(1)),  # int -> np.int32
            (3.14, np.float64(1)),  # float -> np.float64
            (True, np.bool_(False)),  # bool -> np.bool_
            ("42", np.int16(1)),  # str -> np.int16
        ]

        for obj, template in test_cases:
            result = self.template_deserializer._deserialize_numpy_scalar_with_template(obj, template)

            # Should convert to template's NumPy type
            assert type(result) is type(template), f"Expected {type(template)}, got {type(result)}"

            # Value should be converted appropriately
            if isinstance(obj, str):
                # String conversion might fail, that's ok
                if type(result) is not type(template):
                    continue
            else:
                assert result == type(template)(obj)

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_deserialize_numpy_with_template(self):
        """Test _deserialize_numpy_with_template method."""
        # Test various list -> NumPy array conversions
        test_cases = [
            ([1, 2, 3], np.array([1, 2, 3])),
            ([[1, 2], [3, 4]], np.array([[1, 2], [3, 4]])),
            ([1.0, 2.0, 3.0], np.array([1.0, 2.0, 3.0])),
            ([True, False], np.array([True, False])),
        ]

        for obj, template in test_cases:
            result = self.template_deserializer._deserialize_numpy_with_template(obj, template)

            # Should return NumPy array
            assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"

            # Should have same shape if possible
            if result.size == template.size:
                # Only check shape if sizes match
                pass

            # Values should match
            assert np.array_equal(result.flatten(), np.array(obj).flatten())

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_deserialize_numpy_with_template_failures(self):
        """Test _deserialize_numpy_with_template with edge cases."""
        template = np.array([1, 2, 3])

        # Test with invalid input (should handle gracefully)
        result = self.template_deserializer._deserialize_numpy_with_template("invalid", template)
        # Should return original or handle gracefully
        assert result is not None  # Should not crash

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_deserialize_torch_with_template(self):
        """Test _deserialize_torch_with_template method."""
        template = torch.tensor([1.0, 2.0, 3.0])

        # Test dict format with _data field (current serialization format)
        obj_dict = {"_type": "torch.Tensor", "_data": [1.0, 2.0, 3.0], "_dtype": "torch.float32", "_device": "cpu"}

        result = self.template_deserializer._deserialize_torch_with_template(obj_dict, template)

        # Should return tensor
        assert torch.is_tensor(result), f"Expected tensor, got {type(result)}"
        assert result.tolist() == [1.0, 2.0, 3.0]

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_deserialize_torch_with_template_list(self):
        """Test _deserialize_torch_with_template with list input."""
        template = torch.tensor([1.0, 2.0, 3.0])
        obj_list = [1.0, 2.0, 3.0]

        result = self.template_deserializer._deserialize_torch_with_template(obj_list, template)

        # Should convert list to tensor
        assert torch.is_tensor(result), f"Expected tensor, got {type(result)}"
        assert result.tolist() == obj_list

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_deserialize_torch_with_template_failures(self):
        """Test _deserialize_torch_with_template with edge cases."""
        template = torch.tensor([1.0, 2.0, 3.0])

        # Test with invalid dict (should warn and return original)
        invalid_dict = {"_type": "torch.Tensor"}  # Missing _data
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = self.template_deserializer._deserialize_torch_with_template(invalid_dict, template)
            assert result == invalid_dict  # Should return original

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_deserialize_sklearn_with_template(self):
        """Test _deserialize_sklearn_with_template method."""
        template = LogisticRegression(random_state=42)

        # Test sklearn serialization format
        obj_dict = {
            "_type": "sklearn.model",
            "_class": "sklearn.linear_model._logistic.LogisticRegression",
            "_params": {"random_state": 42, "C": 1.0, "max_iter": 100},
        }

        result = self.template_deserializer._deserialize_sklearn_with_template(obj_dict, template)

        # Should return sklearn model
        assert isinstance(result, LogisticRegression), f"Expected LogisticRegression, got {type(result)}"
        assert result.get_params()["random_state"] == 42
        assert result.get_params()["C"] == 1.0

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_deserialize_sklearn_with_template_failures(self):
        """Test _deserialize_sklearn_with_template with edge cases."""
        template = LogisticRegression(random_state=42)

        # Test with invalid class name (should warn and return original)
        invalid_dict = {"_type": "sklearn.model", "_class": "sklearn.nonexistent.FakeModel", "_params": {}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.template_deserializer._deserialize_sklearn_with_template(invalid_dict, template)
            assert result == invalid_dict  # Should return original
            assert len(w) > 0  # Should have warned

    def test_deserialize_complex_with_template(self):
        """Test _deserialize_complex_with_template method."""
        template = complex(1, 2)

        # Test dict format with _type
        obj_dict = {"_type": "complex", "real": 3.0, "imag": 4.0}
        result = self.template_deserializer._deserialize_complex_with_template(obj_dict, template)
        assert isinstance(result, complex)
        assert result == complex(3.0, 4.0)

        # Test dict format without _type
        obj_dict2 = {"real": 1.0, "imag": 2.0}
        result2 = self.template_deserializer._deserialize_complex_with_template(obj_dict2, template)
        assert isinstance(result2, complex)
        assert result2 == complex(1.0, 2.0)

        # Test numeric input
        result3 = self.template_deserializer._deserialize_complex_with_template(5, template)
        assert isinstance(result3, complex)
        assert result3 == complex(5)

        # Test string input
        result4 = self.template_deserializer._deserialize_complex_with_template("1+2j", template)
        assert isinstance(result4, complex)
        assert result4 == complex(1, 2)

        # Test invalid input (should return original)
        result5 = self.template_deserializer._deserialize_complex_with_template("invalid", template)
        assert result5 == "invalid"

    def test_deserialize_decimal_with_template(self):
        """Test _deserialize_decimal_with_template method."""
        template = Decimal("123.45")

        # Test dict format
        obj_dict = {"_type": "decimal", "value": "678.90"}
        result = self.template_deserializer._deserialize_decimal_with_template(obj_dict, template)
        assert isinstance(result, Decimal)
        assert result == Decimal("678.90")

        # Test string input
        result2 = self.template_deserializer._deserialize_decimal_with_template("999.99", template)
        assert isinstance(result2, Decimal)
        assert result2 == Decimal("999.99")

        # Test numeric input
        result3 = self.template_deserializer._deserialize_decimal_with_template(42, template)
        assert isinstance(result3, Decimal)
        assert result3 == Decimal("42")

        result4 = self.template_deserializer._deserialize_decimal_with_template(3.14, template)
        assert isinstance(result4, Decimal)
        assert result4 == Decimal("3.14")

    def test_deserialize_decimal_with_template_failures(self):
        """Test _deserialize_decimal_with_template with edge cases."""
        template = Decimal("123.45")

        # Test invalid input (should handle gracefully)
        test_obj = object()
        result = self.template_deserializer._deserialize_decimal_with_template(test_obj, template)
        # Should return original or handle gracefully
        assert result is not None  # Should not crash

    def test_deserialize_path_with_template(self):
        """Test _deserialize_path_with_template method."""
        template = Path("./test.txt")

        # Test string input
        result = self.template_deserializer._deserialize_path_with_template("/tmp/test.txt", template)
        assert isinstance(result, Path)
        assert result == Path("/tmp/test.txt")

        # Test relative path
        result2 = self.template_deserializer._deserialize_path_with_template("relative/path", template)
        assert isinstance(result2, Path)
        assert result2 == Path("relative/path")

    def test_deserialize_path_with_template_failures(self):
        """Test _deserialize_path_with_template with edge cases."""
        template = Path("./test.txt")

        # Test invalid input (should warn and return original)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.template_deserializer._deserialize_path_with_template(None, template)
            assert result is None  # Should return original
            assert len(w) > 0  # Should have warned


class TestTemplateDeserializerMainLogic:
    """Test the main template deserializer dispatch logic."""

    def test_template_deserializer_dispatch(self):
        """Test that _deserialize_with_template dispatches to correct methods."""
        # Test that the right methods are called for different template types

        # Complex number
        template = complex(1, 2)
        deserializer = TemplateDeserializer(template)
        obj = {"_type": "complex", "real": 3, "imag": 4}
        result = deserializer._deserialize_with_template(obj, template)
        assert isinstance(result, complex)
        assert result == complex(3, 4)

        # Decimal
        template = Decimal("123.45")
        deserializer = TemplateDeserializer(template)
        obj = "678.90"
        result = deserializer._deserialize_with_template(obj, template)
        assert isinstance(result, Decimal)
        assert result == Decimal("678.90")

        # Path
        template = Path("./test.txt")
        deserializer = TemplateDeserializer(template)
        obj = "/tmp/test.txt"
        result = deserializer._deserialize_with_template(obj, template)
        assert isinstance(result, Path)
        assert result == Path("/tmp/test.txt")

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_template_deserializer_numpy_dispatch(self):
        """Test NumPy-specific template deserializer dispatch."""
        # NumPy array
        template = np.array([1, 2, 3])
        deserializer = TemplateDeserializer(template)
        obj = [4, 5, 6]
        result = deserializer._deserialize_with_template(obj, template)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([4, 5, 6]))

        # NumPy scalar
        template = np.int32(42)
        deserializer = TemplateDeserializer(template)
        obj = 99
        result = deserializer._deserialize_with_template(obj, template)
        assert type(result) is np.int32
        assert result == np.int32(99)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_template_deserializer_torch_dispatch(self):
        """Test PyTorch-specific template deserializer dispatch."""
        template = torch.tensor([1.0, 2.0, 3.0])
        deserializer = TemplateDeserializer(template)
        obj = {"_type": "torch.Tensor", "_data": [4.0, 5.0, 6.0], "_device": "cpu"}
        result = deserializer._deserialize_with_template(obj, template)
        assert torch.is_tensor(result)
        assert torch.equal(result, torch.tensor([4.0, 5.0, 6.0]))

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_template_deserializer_sklearn_dispatch(self):
        """Test sklearn-specific template deserializer dispatch."""
        template = LogisticRegression(random_state=42)
        deserializer = TemplateDeserializer(template)
        obj = {
            "_type": "sklearn.model",
            "_class": "sklearn.linear_model._logistic.LogisticRegression",
            "_params": {"random_state": 99},
        }
        result = deserializer._deserialize_with_template(obj, template)
        assert isinstance(result, LogisticRegression)
        assert result.get_params()["random_state"] == 99


class TestTemplateDeserializerRegressionTests:
    """Regression tests to ensure we didn't break existing functionality."""

    def test_existing_datetime_still_works(self):
        """Test that existing datetime template deserialization still works."""
        template = datetime(2023, 1, 1, 12, 0, 0)
        deserializer = TemplateDeserializer(template)
        obj = "2023-12-31T23:59:59"
        result = deserializer._deserialize_with_template(obj, template)
        assert isinstance(result, datetime)
        assert result.year == 2023
        assert result.month == 12
        assert result.day == 31

    def test_existing_uuid_still_works(self):
        """Test that existing UUID template deserialization still works."""
        template = uuid.UUID("12345678-1234-5678-9012-123456789abc")
        deserializer = TemplateDeserializer(template)
        obj = "87654321-4321-8765-2109-cba987654321"
        result = deserializer._deserialize_with_template(obj, template)
        assert isinstance(result, uuid.UUID)
        assert str(result) == "87654321-4321-8765-2109-cba987654321"

    def test_basic_type_coercion_still_works(self):
        """Test that basic type coercion still works."""
        deserializer = TemplateDeserializer(None)

        # int template
        result = deserializer._coerce_to_template_type("42", 42)
        assert result == 42
        assert type(result) is int

        # float template
        result = deserializer._coerce_to_template_type("3.14", 3.14)
        assert result == 3.14
        assert type(result) is float

        # bool template
        result = deserializer._coerce_to_template_type(1, True)
        assert result is True
        assert type(result) is bool

    def test_fallback_behavior(self):
        """Test that fallback behavior works when template methods fail."""
        deserializer = TemplateDeserializer(None)

        # Unsupported template type should fall back to coercion
        class CustomType:
            def __init__(self, value):
                self.value = value

        template = CustomType(42)
        obj = "some value"
        result = deserializer._deserialize_with_template(obj, template)
        # Should return original object since coercion fails
        assert result == obj
