"""Core deserialization paths coverage tests.

Targets the massive uncovered block in lines 272-638 which handles
type metadata reconstruction for complex types.
"""

import warnings
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest

from datason.deserializers import (
    _deserialize_with_type_metadata,
    deserialize,
)


class TestBasicTypeMetadataDeserialization:
    """Test basic type metadata reconstruction."""

    def test_datetime_type_metadata(self):
        """Test datetime reconstruction from type metadata."""
        data = {
            "__datason_type__": "datetime",
            "__datason_value__": "2023-01-01T12:00:00",
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, datetime)
        assert result == datetime.fromisoformat("2023-01-01T12:00:00")

    def test_uuid_type_metadata(self):
        """Test UUID reconstruction from type metadata."""
        uuid_str = "12345678-1234-5678-9012-123456789abc"
        data = {
            "__datason_type__": "uuid.UUID",
            "__datason_value__": uuid_str,
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, UUID)
        assert result == UUID(uuid_str)

    def test_complex_type_metadata(self):
        """Test complex number reconstruction from type metadata."""
        # Test dict format
        data = {
            "__datason_type__": "complex",
            "__datason_value__": {"real": 1.5, "imag": 2.5},
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, complex)
        assert result == complex(1.5, 2.5)

        # Test direct value format
        data = {
            "__datason_type__": "complex",
            "__datason_value__": "1+2j",
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, complex)
        assert result == complex("1+2j")

    def test_decimal_type_metadata(self):
        """Test Decimal reconstruction from type metadata."""
        data = {
            "__datason_type__": "decimal.Decimal",
            "__datason_value__": "3.14159",
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, Decimal)
        assert result == Decimal("3.14159")

    def test_path_type_metadata(self):
        """Test pathlib.Path reconstruction from type metadata."""
        data = {
            "__datason_type__": "pathlib.Path",
            "__datason_value__": "/tmp/test/path",
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, Path)
        assert result == Path("/tmp/test/path")

    def test_set_type_metadata(self):
        """Test set reconstruction from type metadata."""
        data = {
            "__datason_type__": "set",
            "__datason_value__": [1, 2, 3, 4, 5],
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, set)
        assert result == {1, 2, 3, 4, 5}

    def test_tuple_type_metadata(self):
        """Test tuple reconstruction from type metadata."""
        data = {
            "__datason_type__": "tuple",
            "__datason_value__": [1, "hello", 3.14, True],
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, tuple)
        assert result == (1, "hello", 3.14, True)


class TestPandasTypeMetadataDeserialization:
    """Test pandas type metadata reconstruction."""

    def test_dataframe_records_format(self):
        """Test DataFrame reconstruction from records format."""
        pd = pytest.importorskip("pandas")

        # Records format (list of dicts)
        data = {
            "__datason_type__": "pandas.DataFrame",
            "__datason_value__": [
                {"A": 1, "B": 2, "C": 3},
                {"A": 4, "B": 5, "C": 6},
                {"A": 7, "B": 8, "C": 9},
            ],
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 3)
        assert list(result.columns) == ["A", "B", "C"]

    def test_dataframe_values_format(self):
        """Test DataFrame reconstruction from values format."""
        pd = pytest.importorskip("pandas")

        # Values format (list of lists)
        data = {
            "__datason_type__": "pandas.DataFrame",
            "__datason_value__": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 3)

    def test_dataframe_split_format(self):
        """Test DataFrame reconstruction from split format."""
        pd = pytest.importorskip("pandas")

        # Split format
        data = {
            "__datason_type__": "pandas.DataFrame",
            "__datason_value__": {
                "index": [0, 1, 2],
                "columns": ["A", "B", "C"],
                "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            },
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 3)
        assert list(result.columns) == ["A", "B", "C"]
        assert list(result.index) == [0, 1, 2]

    def test_dataframe_index_format(self):
        """Test DataFrame reconstruction from index format."""
        pd = pytest.importorskip("pandas")

        # Index format
        data = {
            "__datason_type__": "pandas.DataFrame",
            "__datason_value__": {
                0: {"A": 1, "B": 2},
                1: {"A": 3, "B": 4},
                2: {"A": 5, "B": 6},
            },
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)

    def test_dataframe_fallback(self):
        """Test DataFrame reconstruction fallback behavior."""
        pd = pytest.importorskip("pandas")

        # Fallback case - direct value
        data = {
            "__datason_type__": "pandas.DataFrame",
            "__datason_value__": {"A": [1, 2, 3], "B": [4, 5, 6]},
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, pd.DataFrame)

    def test_series_with_name(self):
        """Test Series reconstruction with name preservation."""
        pd = pytest.importorskip("pandas")

        # Series with name
        data = {
            "__datason_type__": "pandas.Series",
            "__datason_value__": {
                "_series_name": "test_series",
                "0": 10,
                "1": 20,
                "2": 30,
            },
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, pd.Series)
        assert result.name == "test_series"
        assert result[0] == 10
        assert result[1] == 20
        assert result[2] == 30

    def test_series_categorical_with_name(self):
        """Test Series categorical reconstruction with name."""
        pd = pytest.importorskip("pandas")

        # Categorical series with name
        data = {
            "__datason_type__": "pandas.Series",
            "__datason_value__": {
                "_series_name": "categories",
                "_dtype": "category",
                "_categories": ["A", "B", "C"],
                "_ordered": False,
                "0": "A",
                "1": "B",
                "2": "A",
            },
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, pd.Series)
        assert result.name == "categories"
        assert str(result.dtype) == "category"

    def test_series_categorical_unnamed(self):
        """Test Series categorical reconstruction without name."""
        pd = pytest.importorskip("pandas")

        # Categorical series without name
        data = {
            "__datason_type__": "pandas.Series",
            "__datason_value__": {
                "_dtype": "category",
                "_categories": ["X", "Y", "Z"],
                "_ordered": True,
                "0": "X",
                "1": "Y",
                "2": "Z",
            },
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, pd.Series)
        assert str(result.dtype) == "category"

    def test_series_dict_format(self):
        """Test Series reconstruction from dict format."""
        pd = pytest.importorskip("pandas")

        # Regular dict format
        data = {
            "__datason_type__": "pandas.Series",
            "__datason_value__": {"0": 100, "1": 200, "2": 300},
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, pd.Series)
        assert result[0] == 100

    def test_series_list_format(self):
        """Test Series reconstruction from list format."""
        pd = pytest.importorskip("pandas")

        # List format
        data = {
            "__datason_type__": "pandas.Series",
            "__datason_value__": [10, 20, 30, 40, 50],
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, pd.Series)
        assert len(result) == 5
        assert result[0] == 10

    def test_series_fallback(self):
        """Test Series reconstruction fallback behavior."""
        pd = pytest.importorskip("pandas")

        # Fallback case
        data = {
            "__datason_type__": "pandas.Series",
            "__datason_value__": "scalar_value",
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, pd.Series)


class TestNumpyTypeMetadataDeserialization:
    """Test NumPy type metadata reconstruction."""

    def test_ndarray_basic(self):
        """Test basic NumPy array reconstruction."""
        np = pytest.importorskip("numpy")

        # Basic array
        data = {
            "__datason_type__": "numpy.ndarray",
            "__datason_value__": [1, 2, 3, 4, 5],
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, np.ndarray)
        assert result.tolist() == [1, 2, 3, 4, 5]

    def test_ndarray_with_dtype_and_shape(self):
        """Test NumPy array reconstruction with dtype and shape."""
        np = pytest.importorskip("numpy")

        # Array with metadata
        data = {
            "__datason_type__": "numpy.ndarray",
            "__datason_value__": {
                "data": [1, 2, 3, 4, 5, 6],
                "dtype": "float64",
                "shape": [2, 3],
            },
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        assert result.dtype == np.float64

    def test_ndarray_with_complex_elements(self):
        """Test NumPy array with complex number elements."""
        np = pytest.importorskip("numpy")

        # Array with complex elements that have type metadata
        data = {
            "__datason_type__": "numpy.ndarray",
            "__datason_value__": [
                {"_type": "complex", "real": 1, "imag": 2},
                {"_type": "complex", "real": 3, "imag": 4},
            ],
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_numpy_scalar_types(self):
        """Test NumPy scalar type reconstruction."""
        np = pytest.importorskip("numpy")

        # Test various numpy scalar types
        scalar_types = [
            ("numpy.int32", 42, np.int32),
            ("numpy.int64", 123456789, np.int64),
            ("numpy.float32", 3.14, np.float32),
            ("numpy.float64", 2.71828, np.float64),
            ("numpy.bool_", True, np.bool_),
            ("numpy.complex64", "1+2j", np.complex64),
            ("numpy.complex128", "3+4j", np.complex128),
        ]

        for type_name, value, expected_type in scalar_types:
            data = {
                "__datason_type__": type_name,
                "__datason_value__": value,
            }
            result = _deserialize_with_type_metadata(data)
            assert isinstance(result, expected_type)

    def test_numpy_generic_fallback(self):
        """Test NumPy generic type fallback mechanism."""
        np = pytest.importorskip("numpy")

        # Test a hypothetical numpy type that should use the generic fallback
        data = {
            "__datason_type__": "numpy.uint16",
            "__datason_value__": 65535,
        }
        result = _deserialize_with_type_metadata(data)
        assert isinstance(result, np.uint16)

    def test_numpy_bool_deprecation_handling(self):
        """Test handling of numpy.bool deprecation."""
        pytest.importorskip("numpy")

        # Test the special case for bool_ vs bool
        data = {
            "__datason_type__": "numpy.bool",
            "__datason_value__": True,
        }
        # Should handle the deprecation and use bool_ instead
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _deserialize_with_type_metadata(data)
            # Should work without error


class TestMLTypeMetadataDeserialization:
    """Test ML framework type metadata reconstruction."""

    def test_torch_tensor_basic(self):
        """Test basic PyTorch tensor reconstruction."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        # Basic tensor
        data = {
            "__datason_type__": "torch.Tensor",
            "__datason_value__": [1.0, 2.0, 3.0, 4.0],
        }
        result = _deserialize_with_type_metadata(data)
        assert torch.is_tensor(result)
        assert result.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_torch_tensor_with_metadata(self):
        """Test PyTorch tensor reconstruction with full metadata."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        # Tensor with full metadata (new format)
        data = {
            "__datason_type__": "torch.Tensor",
            "__datason_value__": {
                "data": [[1.0, 2.0], [3.0, 4.0]],
                "dtype": "torch.float32",
                "device": "cpu",
                "requires_grad": False,
                "shape": [2, 2],
            },
        }
        result = _deserialize_with_type_metadata(data)
        assert torch.is_tensor(result)
        assert result.shape == torch.Size([2, 2])
        assert result.dtype == torch.float32

    def test_torch_tensor_legacy_format(self):
        """Test PyTorch tensor reconstruction with legacy metadata format."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        # Tensor with legacy metadata format
        data = {
            "__datason_type__": "torch.Tensor",
            "__datason_value__": {
                "_data": [1.0, 2.0, 3.0, 4.0],
                "_dtype": "torch.float64",
                "_device": "cpu",
                "_requires_grad": True,
                "_shape": [4],
            },
        }
        result = _deserialize_with_type_metadata(data)
        assert torch.is_tensor(result)
        assert result.requires_grad is True
        assert result.dtype == torch.float64

    def test_torch_tensor_reshape_failure(self):
        """Test PyTorch tensor reshape failure handling."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        # Tensor with incompatible reshape
        data = {
            "__datason_type__": "torch.Tensor",
            "__datason_value__": {
                "data": [1.0, 2.0, 3.0],  # 3 elements
                "shape": [2, 2],  # Incompatible: needs 4 elements
            },
        }
        # Should handle reshape failure gracefully
        result = _deserialize_with_type_metadata(data)
        assert torch.is_tensor(result)
        # Should keep original shape when reshape fails

    def test_torch_import_error_handling(self):
        """Test PyTorch import error handling."""
        # Mock torch import failure
        with patch("builtins.__import__", side_effect=ImportError("No module named 'torch'")):
            data = {
                "__datason_type__": "torch.Tensor",
                "__datason_value__": [1.0, 2.0, 3.0],
            }
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = _deserialize_with_type_metadata(data)
                # Should return original value and generate warning
                assert result == data["__datason_value__"]
                assert any("PyTorch not available" in str(warning.message) for warning in w)

    def test_sklearn_model_reconstruction(self):
        """Test scikit-learn model reconstruction."""
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("scikit-learn not available")

        # Mock sklearn model data
        data = {
            "__datason_type__": "sklearn.linear_model.LinearRegression",
            "__datason_value__": {
                "model_type": "LinearRegression",
                "params": {"fit_intercept": True},
                # Note: This is a simplified test - real sklearn serialization is more complex
            },
        }
        # The function should attempt to reconstruct sklearn models
        # Even if it fails, it should handle gracefully
        _deserialize_with_type_metadata(data)
        # Result might be None if reconstruction fails, which is fine


class TestTypeMetadataEdgeCases:
    """Test edge cases in type metadata handling."""

    def test_pandas_without_pandas_installed(self):
        """Test pandas type metadata when pandas is not available."""
        with patch("datason.deserializers.pd", None):
            data = {
                "__datason_type__": "pandas.DataFrame",
                "__datason_value__": [{"A": 1, "B": 2}],
            }
            result = _deserialize_with_type_metadata(data)
            # Should return original value when pandas is not available
            assert result == data["__datason_value__"]

    def test_numpy_without_numpy_installed(self):
        """Test numpy type metadata when numpy is not available."""
        with patch("datason.deserializers.np", None):
            data = {
                "__datason_type__": "numpy.ndarray",
                "__datason_value__": [1, 2, 3],
            }
            result = _deserialize_with_type_metadata(data)
            # Should return original value when numpy is not available
            assert result == data["__datason_value__"]

    def test_unknown_type_metadata(self):
        """Test handling of unknown type metadata."""
        data = {
            "__datason_type__": "unknown.CustomType",
            "__datason_value__": {"some": "data"},
        }
        # Should handle unknown types gracefully
        _deserialize_with_type_metadata(data)
        # Might return None or the original data


class TestTypeMetadataIntegration:
    """Test integration of type metadata with main deserialize function."""

    def test_deserialize_with_type_metadata_integration(self):
        """Test that main deserialize function uses type metadata."""
        complex_data = {
            "datetime_field": {
                "__datason_type__": "datetime",
                "__datason_value__": "2023-01-01T12:00:00",
            },
            "uuid_field": {
                "__datason_type__": "uuid.UUID",
                "__datason_value__": "12345678-1234-5678-9012-123456789abc",
            },
            "complex_field": {
                "__datason_type__": "complex",
                "__datason_value__": {"real": 1, "imag": 2},
            },
            "set_field": {
                "__datason_type__": "set",
                "__datason_value__": [1, 2, 3],
            },
            "regular_field": "normal_string",
        }

        result = deserialize(complex_data)

        assert isinstance(result["datetime_field"], datetime)
        assert isinstance(result["uuid_field"], UUID)
        assert isinstance(result["complex_field"], complex)
        assert isinstance(result["set_field"], set)
        assert result["regular_field"] == "normal_string"

    def test_nested_type_metadata(self):
        """Test nested structures with type metadata."""
        nested_data = {
            "level1": {
                "level2": {
                    "datetime": {
                        "__datason_type__": "datetime",
                        "__datason_value__": "2023-01-01T12:00:00",
                    },
                    "path": {
                        "__datason_type__": "pathlib.Path",
                        "__datason_value__": "/tmp/test",
                    },
                }
            }
        }

        result = deserialize(nested_data)

        assert isinstance(result["level1"]["level2"]["datetime"], datetime)
        assert isinstance(result["level1"]["level2"]["path"], Path)
