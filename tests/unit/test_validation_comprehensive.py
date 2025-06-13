"""Comprehensive tests for datason.validation module.

This module tests all validation functionality including Pydantic and Marshmallow
integration, lazy imports, error handling, and edge cases.
"""

from unittest.mock import Mock, patch

import pytest

import datason.validation as validation


class TestLazyImports:
    """Test lazy import functionality for optional dependencies."""

    def test_lazy_import_pydantic_base_model_success(self):
        """Test successful Pydantic BaseModel import."""
        # Reset the lazy import cache
        validation._LAZY_IMPORTS["BaseModel"] = None

        # Mock the import at the module level
        with patch("builtins.__import__") as mock_import:
            mock_base_model = Mock()
            mock_pydantic_module = Mock()
            mock_pydantic_module.BaseModel = mock_base_model
            mock_import.return_value = mock_pydantic_module

            result = validation._lazy_import_pydantic_base_model()
            assert result == mock_base_model
            # Should cache the result
            assert validation._LAZY_IMPORTS["BaseModel"] == mock_base_model

    def test_lazy_import_pydantic_base_model_failure(self):
        """Test Pydantic import failure handling."""
        # Reset the lazy import cache
        validation._LAZY_IMPORTS["BaseModel"] = None

        with patch("builtins.__import__", side_effect=ImportError("No module named 'pydantic'")):
            result = validation._lazy_import_pydantic_base_model()
            assert result is None
            # Should cache the failure
            assert validation._LAZY_IMPORTS["BaseModel"] is False

    def test_lazy_import_pydantic_cached_success(self):
        """Test that successful import is cached."""
        mock_base_model = Mock()
        validation._LAZY_IMPORTS["BaseModel"] = mock_base_model

        result = validation._lazy_import_pydantic_base_model()
        assert result == mock_base_model

    def test_lazy_import_pydantic_cached_failure(self):
        """Test that failed import is cached."""
        validation._LAZY_IMPORTS["BaseModel"] = False

        result = validation._lazy_import_pydantic_base_model()
        assert result is None

    def test_lazy_import_marshmallow_schema_success(self):
        """Test successful Marshmallow Schema import."""
        # Reset the lazy import cache
        validation._LAZY_IMPORTS["Schema"] = None

        # Mock the import at the module level
        with patch("builtins.__import__") as mock_import:
            mock_schema = Mock()
            mock_marshmallow_module = Mock()
            mock_marshmallow_module.Schema = mock_schema
            mock_import.return_value = mock_marshmallow_module

            result = validation._lazy_import_marshmallow_schema()
            assert result == mock_schema
            # Should cache the result
            assert validation._LAZY_IMPORTS["Schema"] == mock_schema

    def test_lazy_import_marshmallow_schema_failure(self):
        """Test Marshmallow import failure handling."""
        # Reset the lazy import cache
        validation._LAZY_IMPORTS["Schema"] = None

        with patch("builtins.__import__", side_effect=ImportError("No module named 'marshmallow'")):
            result = validation._lazy_import_marshmallow_schema()
            assert result is None
            # Should cache the failure
            assert validation._LAZY_IMPORTS["Schema"] is False

    def test_lazy_import_marshmallow_cached_success(self):
        """Test that successful Marshmallow import is cached."""
        mock_schema = Mock()
        validation._LAZY_IMPORTS["Schema"] = mock_schema

        result = validation._lazy_import_marshmallow_schema()
        assert result == mock_schema

    def test_lazy_import_marshmallow_cached_failure(self):
        """Test that failed Marshmallow import is cached."""
        validation._LAZY_IMPORTS["Schema"] = False

        result = validation._lazy_import_marshmallow_schema()
        assert result is None


class TestPydanticSerialization:
    """Test Pydantic model serialization functionality."""

    def test_serialize_pydantic_no_pydantic_available(self):
        """Test error when Pydantic is not available."""
        with patch.object(validation, "_lazy_import_pydantic_base_model", return_value=None):
            with pytest.raises(ImportError, match="Pydantic is required for serialize_pydantic"):
                validation.serialize_pydantic({"test": "data"})

    def test_serialize_pydantic_v2_model_dump(self):
        """Test Pydantic v2 model serialization using model_dump()."""
        mock_base_model = Mock()
        mock_model = Mock()
        mock_model.model_dump.return_value = {"field": "value"}

        with patch.object(validation, "_lazy_import_pydantic_base_model", return_value=mock_base_model):
            with patch("datason.validation.serialize") as mock_serialize:
                validation.serialize_pydantic(mock_model)
                mock_model.model_dump.assert_called_once()
                mock_serialize.assert_called_once_with({"field": "value"})

    def test_serialize_pydantic_v1_dict_fallback(self):
        """Test Pydantic v1 model serialization using dict() fallback."""
        mock_base_model = Mock()
        mock_model = Mock()
        # Simulate v2 method not available, but v1 method available
        mock_model.model_dump.side_effect = AttributeError("model_dump not available")
        mock_model.dict.return_value = {"field": "value"}

        with patch.object(validation, "_lazy_import_pydantic_base_model", return_value=mock_base_model):
            with patch("datason.validation.serialize") as mock_serialize:
                validation.serialize_pydantic(mock_model)
                mock_model.model_dump.assert_called_once()
                mock_model.dict.assert_called_once()
                mock_serialize.assert_called_once_with({"field": "value"})

    def test_serialize_pydantic_dict_fallback(self):
        """Test fallback to __dict__ when both model_dump and dict fail."""

        class BrokenPydanticModel:
            def __init__(self):
                self.__dict__ = {"field": "value"}

            def model_dump(self):
                raise AttributeError("model_dump not available")

            def dict(self):
                raise Exception("dict method failed")

        mock_base_model = Mock()
        broken_model = BrokenPydanticModel()

        with patch.object(validation, "_lazy_import_pydantic_base_model", return_value=mock_base_model):
            with patch("datason.validation.serialize") as mock_serialize:
                validation.serialize_pydantic(broken_model)
                mock_serialize.assert_called_once_with({"field": "value"})

    def test_serialize_pydantic_non_pydantic_object(self):
        """Test serialization of non-Pydantic objects."""
        mock_base_model = Mock()
        regular_obj = {"not": "pydantic"}

        with patch.object(validation, "_lazy_import_pydantic_base_model", return_value=mock_base_model):
            with patch("datason.validation.serialize") as mock_serialize:
                validation.serialize_pydantic(regular_obj)
                mock_serialize.assert_called_once_with(regular_obj)


class TestMarshmallowSerialization:
    """Test Marshmallow schema serialization functionality."""

    def test_serialize_marshmallow_no_marshmallow_available(self):
        """Test error when Marshmallow is not available."""
        with patch.object(validation, "_lazy_import_marshmallow_schema", return_value=None):
            with pytest.raises(ImportError, match="Marshmallow is required for serialize_marshmallow"):
                validation.serialize_marshmallow({"test": "data"})

    def test_serialize_marshmallow_schema_object(self):
        """Test Marshmallow schema object serialization."""
        mock_schema_class = Mock()
        mock_field1 = Mock()
        mock_field1.__class__.__name__ = "StringField"
        mock_field2 = Mock()
        mock_field2.__class__.__name__ = "IntegerField"

        mock_schema = Mock()
        mock_schema.fields = {"name": mock_field1, "age": mock_field2}

        expected_data = {
            "__datason_type__": "marshmallow.schema",
            "__datason_value__": {"fields": {"name": "StringField", "age": "IntegerField"}},
        }

        with patch.object(validation, "_lazy_import_marshmallow_schema", return_value=mock_schema_class):
            with patch("datason.validation.serialize") as mock_serialize:
                validation.serialize_marshmallow(mock_schema)
                mock_serialize.assert_called_once_with(expected_data)

    def test_serialize_marshmallow_schema_dict_fallback(self):
        """Test fallback to __dict__ when schema methods fail."""

        class BrokenMarshmallowSchema:
            def __init__(self):
                self.__dict__ = {"fields": {"name": "StringField", "age": "IntegerField"}}

            def dump(self):
                raise Exception("dump method failed")

        class FailingFieldsDict:
            def __init__(self):
                self._fields = {"name": "StringField", "age": "IntegerField"}

            def items(self):
                raise Exception("items method failed")

        mock_schema_class = Mock()
        broken_schema = BrokenMarshmallowSchema()

        with patch.object(validation, "_lazy_import_marshmallow_schema", return_value=mock_schema_class):
            with patch("datason.validation.serialize") as mock_serialize:
                validation.serialize_marshmallow(broken_schema)
                mock_serialize.assert_called_once_with(
                    {
                        "__datason_type__": "marshmallow.schema",
                        "__datason_value__": {"fields": {"name": "StringField", "age": "IntegerField"}},
                    }
                )


class TestAttributeAccess:
    """Test the __getattr__ functionality."""

    def test_getattr_base_model(self):
        """Test accessing BaseModel attribute."""
        mock_base_model = Mock()
        with patch.object(validation, "_lazy_import_pydantic_base_model", return_value=mock_base_model):
            result = validation.__getattr__("BaseModel")
            assert result == mock_base_model

    def test_getattr_schema(self):
        """Test accessing Schema attribute."""
        mock_schema = Mock()
        with patch.object(validation, "_lazy_import_marshmallow_schema", return_value=mock_schema):
            result = validation.__getattr__("Schema")
            assert result == mock_schema

    def test_getattr_invalid_attribute(self):
        """Test accessing invalid attribute raises AttributeError."""
        with pytest.raises(AttributeError, match="module 'datason.validation' has no attribute 'invalid_attr'"):
            validation.__getattr__("invalid_attr")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_lazy_import_exception_handling(self):
        """Test that any exception during import is handled gracefully."""
        validation._LAZY_IMPORTS["BaseModel"] = None

        with patch("builtins.__import__", side_effect=RuntimeError("Unexpected error")):
            result = validation._lazy_import_pydantic_base_model()
            assert result is None
            assert validation._LAZY_IMPORTS["BaseModel"] is False

    def test_serialize_pydantic_with_none_input(self):
        """Test Pydantic serialization with None input."""
        mock_base_model = Mock()

        with patch.object(validation, "_lazy_import_pydantic_base_model", return_value=mock_base_model):
            with patch("datason.validation.serialize") as mock_serialize:
                validation.serialize_pydantic(None)
                mock_serialize.assert_called_once_with(None)

    def test_serialize_marshmallow_with_none_input(self):
        """Test Marshmallow serialization with None input."""
        mock_schema_class = Mock()

        with patch.object(validation, "_lazy_import_marshmallow_schema", return_value=mock_schema_class):
            with patch("datason.validation.serialize") as mock_serialize:
                validation.serialize_marshmallow(None)
                mock_serialize.assert_called_once_with(None)


class TestImportCacheBehavior:
    """Test the import cache behavior under various conditions."""

    def setUp(self):
        """Reset import cache before each test."""
        validation._LAZY_IMPORTS["BaseModel"] = None
        validation._LAZY_IMPORTS["Schema"] = None

    def test_cache_persistence_across_calls(self):
        """Test that cache persists across multiple calls."""
        mock_base_model = Mock()
        validation._LAZY_IMPORTS["BaseModel"] = mock_base_model

        # Multiple calls should return the same cached object
        result1 = validation._lazy_import_pydantic_base_model()
        result2 = validation._lazy_import_pydantic_base_model()

        assert result1 == result2 == mock_base_model

    def test_independent_cache_entries(self):
        """Test that BaseModel and Schema caches are independent."""
        mock_base_model = Mock()

        validation._LAZY_IMPORTS["BaseModel"] = mock_base_model
        validation._LAZY_IMPORTS["Schema"] = False  # Different state

        assert validation._lazy_import_pydantic_base_model() == mock_base_model
        assert validation._lazy_import_marshmallow_schema() is None
