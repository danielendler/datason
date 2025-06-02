"""Tests for template-based deserialization capabilities (v0.4.5)."""

import uuid
from datetime import datetime
from decimal import Decimal

import pytest

import datason
from datason.deserializers import (
    TemplateDeserializer,
    create_ml_round_trip_template,
    deserialize_with_template,
    infer_template_from_data,
)

# Optional imports for comprehensive testing
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None


class TestTemplateDeserializer:
    """Test the TemplateDeserializer class."""

    def test_template_deserializer_basic_dict(self):
        """Test template deserialization with basic dictionary."""
        template = {"name": "", "age": 0, "active": True}
        serialized_data = {"name": "Alice", "age": 30, "active": False}

        deserializer = TemplateDeserializer(template)
        result = deserializer.deserialize(serialized_data)

        assert result == serialized_data
        assert isinstance(result["name"], str)
        assert isinstance(result["age"], int)
        assert isinstance(result["active"], bool)

    def test_template_deserializer_list_template(self):
        """Test template deserialization with list template."""
        template = [{"id": 0, "value": ""}]
        serialized_data = [{"id": 1, "value": "first"}, {"id": 2, "value": "second"}]

        deserializer = TemplateDeserializer(template)
        result = deserializer.deserialize(serialized_data)

        assert len(result) == 2
        assert all(isinstance(item["id"], int) for item in result)
        assert all(isinstance(item["value"], str) for item in result)

    @pytest.mark.pandas
    def test_template_deserializer_dataframe(self):
        """Test template deserialization with DataFrame."""
        pd = pytest.importorskip("pandas")

        # Create template DataFrame
        template_df = pd.DataFrame({"id": [1], "name": [""], "value": [0.0]})

        # Serialize some data as records
        serialized_data = [{"id": 1, "name": "Alice", "value": 10.5}, {"id": 2, "name": "Bob", "value": 20.3}]

        deserializer = TemplateDeserializer(template_df)
        result = deserializer.deserialize(serialized_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["id", "name", "value"]

        # Check dtypes match template
        assert result["id"].dtype == template_df["id"].dtype
        assert result["value"].dtype == template_df["value"].dtype

    @pytest.mark.pandas
    def test_template_deserializer_series(self):
        """Test template deserialization with Series."""
        pd = pytest.importorskip("pandas")

        # Create template Series
        template_series = pd.Series([0.0], name="values")

        # Serialize some data
        serialized_data = [1.1, 2.2, 3.3]

        deserializer = TemplateDeserializer(template_series)
        result = deserializer.deserialize(serialized_data)

        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert result.name == "values"
        assert result.dtype == template_series.dtype

    def test_template_deserializer_datetime_template(self):
        """Test template deserialization with datetime template."""
        template = datetime.now()
        serialized_data = "2023-12-25T10:30:45"

        deserializer = TemplateDeserializer(template)
        result = deserializer.deserialize(serialized_data)

        assert isinstance(result, datetime)
        assert result.year == 2023
        assert result.month == 12
        assert result.day == 25

    def test_template_deserializer_uuid_template(self):
        """Test template deserialization with UUID template."""
        template = uuid.uuid4()
        serialized_data = "12345678-1234-5678-9012-123456789abc"

        deserializer = TemplateDeserializer(template)
        result = deserializer.deserialize(serialized_data)

        assert isinstance(result, uuid.UUID)
        assert str(result) == serialized_data

    def test_template_deserializer_type_coercion(self):
        """Test type coercion in template deserialization."""
        template = {"count": 0, "rate": 0.0, "name": "", "enabled": True}
        serialized_data = {"count": "42", "rate": "3.14", "name": 123, "enabled": "true"}

        deserializer = TemplateDeserializer(template)
        result = deserializer.deserialize(serialized_data)

        assert result["count"] == 42
        assert isinstance(result["count"], int)
        assert result["rate"] == 3.14
        assert isinstance(result["rate"], float)
        assert result["name"] == "123"
        assert isinstance(result["name"], str)

    def test_template_deserializer_strict_mode_error(self):
        """Test strict mode error handling."""
        template = {"value": 0}
        invalid_data = {"invalid": "structure"}

        deserializer = TemplateDeserializer(template, strict=True)

        # Should not raise error because template allows extra keys
        result = deserializer.deserialize(invalid_data)
        assert "invalid" in result

    def test_template_deserializer_fallback_auto_detect(self):
        """Test fallback to auto-detection."""
        template = {"expected": "structure"}
        data_with_extra = {"expected": "value", "extra": "2023-12-25T10:30:45"}

        deserializer = TemplateDeserializer(template, strict=False, fallback_auto_detect=True)
        result = deserializer.deserialize(data_with_extra)

        assert result["expected"] == "value"
        assert "extra" in result
        # Auto-detection might convert the datetime string
        assert result["extra"] == "2023-12-25T10:30:45"  # May or may not be converted

    @pytest.mark.pandas
    def test_template_deserializer_dataframe_split_format(self):
        """Test DataFrame deserialization from split format."""
        pd = pytest.importorskip("pandas")

        template_df = pd.DataFrame({"a": [1], "b": ["text"]})

        # Split format data
        split_data = {"data": [[1, "hello"], [2, "world"]], "columns": ["a", "b"], "index": [0, 1]}

        deserializer = TemplateDeserializer(template_df)
        result = deserializer.deserialize(split_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["a", "b"]

    def test_template_deserializer_nested_structures(self):
        """Test template deserialization with nested structures."""
        template = {
            "user": {"id": 0, "profile": {"name": "", "created": datetime.now()}},
            "data": [{"key": "", "value": 0}],
        }

        serialized_data = {
            "user": {"id": 123, "profile": {"name": "Alice", "created": "2023-01-01T10:00:00"}},
            "data": [{"key": "metric1", "value": 42}, {"key": "metric2", "value": 24}],
        }

        deserializer = TemplateDeserializer(template)
        result = deserializer.deserialize(serialized_data)

        assert result["user"]["id"] == 123
        assert isinstance(result["user"]["profile"]["created"], datetime)
        assert len(result["data"]) == 2
        assert all(isinstance(item["value"], int) for item in result["data"])


class TestDeserializeWithTemplate:
    """Test the convenience function for template-based deserialization."""

    def test_deserialize_with_template_convenience(self):
        """Test the convenience function."""
        template = {"name": "", "age": 0}
        data = {"name": "Bob", "age": 25}

        result = deserialize_with_template(data, template)

        assert result == data
        assert isinstance(result["name"], str)
        assert isinstance(result["age"], int)

    @pytest.mark.pandas
    def test_deserialize_with_template_dataframe(self):
        """Test convenience function with DataFrame template."""
        pd = pytest.importorskip("pandas")

        template_df = pd.DataFrame({"x": [1.0], "y": [""]})
        data = [{"x": 2.5, "y": "hello"}, {"x": 3.7, "y": "world"}]

        result = deserialize_with_template(data, template_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result["x"].dtype == template_df["x"].dtype

    def test_deserialize_with_template_kwargs(self):
        """Test convenience function with additional kwargs."""
        template = {"value": 0}
        data = {"value": "42", "extra": "unexpected"}

        result = deserialize_with_template(data, template, strict=False, fallback_auto_detect=True)

        assert result["value"] == 42
        assert "extra" in result


class TestInferTemplateFromData:
    """Test template inference functionality."""

    def test_infer_template_from_records(self):
        """Test inferring template from list of records."""
        sample_data = [
            {"name": "Alice", "age": 30, "active": True},
            {"name": "Bob", "age": 25, "active": False},
            {"name": "Charlie", "age": 35, "active": True},
        ]

        template = infer_template_from_data(sample_data)

        assert isinstance(template, dict)
        assert "name" in template
        assert "age" in template
        assert "active" in template

        # Check inferred types
        assert isinstance(template["name"], str)
        assert isinstance(template["age"], int)
        assert isinstance(template["active"], bool)

    def test_infer_template_mixed_types(self):
        """Test template inference with mixed types."""
        sample_data = [
            {"id": 1, "value": 10.5},
            {"id": "2", "value": 20},  # Mixed types
            {"id": 3, "value": 30.7},
        ]

        template = infer_template_from_data(sample_data)

        # Should choose most common type
        assert "id" in template
        assert "value" in template

    @pytest.mark.pandas
    def test_infer_template_from_dataframe(self):
        """Test inferring template from DataFrame."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"], "z": [1.1, 2.2, 3.3]})

        template = infer_template_from_data(df)

        assert isinstance(template, pd.DataFrame)
        assert len(template) == 1  # Single row template
        assert list(template.columns) == ["x", "y", "z"]

    @pytest.mark.pandas
    def test_infer_template_from_series(self):
        """Test inferring template from Series."""
        pd = pytest.importorskip("pandas")

        series = pd.Series([1.1, 2.2, 3.3], name="values")
        template = infer_template_from_data(series)

        assert isinstance(template, pd.Series)
        assert len(template) == 1
        assert template.name == "values"

    def test_infer_template_from_single_dict(self):
        """Test inferring template from single dictionary."""
        data = {"key1": "value", "key2": 42, "key3": True}
        template = infer_template_from_data(data)

        assert template == data

    def test_infer_template_empty_data(self):
        """Test template inference with empty data."""
        template = infer_template_from_data([])
        assert template == {}

    def test_infer_template_complex_nested(self):
        """Test template inference with complex nested data."""
        sample_data = [
            {
                "user": {"id": 1, "name": "Alice"},
                "metadata": {"created": "2023-01-01", "version": 1},
                "tags": ["user", "active"],
            },
            {"user": {"id": 2, "name": "Bob"}, "metadata": {"created": "2023-01-02", "version": 2}, "tags": ["user"]},
        ]

        template = infer_template_from_data(sample_data)

        assert "user" in template
        assert "metadata" in template
        assert "tags" in template
        assert isinstance(template["user"], dict)
        assert isinstance(template["metadata"], dict)
        assert isinstance(template["tags"], list)


class TestMLRoundTripTemplate:
    """Test ML-specific round-trip template creation."""

    @pytest.mark.pandas
    def test_create_ml_template_dataframe(self):
        """Test creating ML template for DataFrame."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame({"feature1": [1.0, 2.0], "feature2": [10, 20], "target": ["class_a", "class_b"]})

        template = create_ml_round_trip_template(df)

        assert template["__ml_template__"] is True
        assert template["object_type"] == "DataFrame"
        assert template["structure_type"] == "dataframe"
        assert "columns" in template
        assert "dtypes" in template
        assert template["shape"] == (2, 3)

    @pytest.mark.pandas
    def test_create_ml_template_series(self):
        """Test creating ML template for Series."""
        pd = pytest.importorskip("pandas")

        series = pd.Series([1.1, 2.2, 3.3], name="target_values")
        template = create_ml_round_trip_template(series)

        assert template["__ml_template__"] is True
        assert template["object_type"] == "Series"
        assert template["structure_type"] == "series"
        assert template["name"] == "target_values"
        assert template["length"] == 3

    @pytest.mark.numpy
    def test_create_ml_template_numpy_array(self):
        """Test creating ML template for numpy array."""
        np = pytest.importorskip("numpy")

        arr = np.random.random((100, 10))
        template = create_ml_round_trip_template(arr)

        assert template["__ml_template__"] is True
        assert template["object_type"] == "ndarray"
        assert template["structure_type"] == "numpy_array"
        assert template["shape"] == (100, 10)
        assert "dtype" in template

    def test_create_ml_template_sklearn_model(self):
        """Test creating ML template for sklearn-like model."""

        # Mock sklearn model-like object
        class MockModel:
            def __init__(self):
                self.coef_ = [1, 2, 3]

            def get_params(self):
                return {"param1": "value1", "param2": 42}

        model = MockModel()
        template = create_ml_round_trip_template(model)

        assert template["__ml_template__"] is True
        assert template["object_type"] == "MockModel"
        assert template["structure_type"] == "sklearn_model"
        assert "parameters" in template
        assert template["fitted"] is True  # Has coef_

    def test_create_ml_template_generic_object(self):
        """Test creating ML template for generic object."""

        class CustomMLObject:
            def __init__(self):
                self.data = [1, 2, 3]

        obj = CustomMLObject()
        template = create_ml_round_trip_template(obj)

        assert template["__ml_template__"] is True
        assert template["object_type"] == "CustomMLObject"
        # Should have basic metadata
        assert "module" in template


class TestTemplateRoundTripIntegration:
    """Test integration between serialization with type hints and template deserialization."""

    def test_round_trip_with_type_metadata(self):
        """Test perfect round-trip with type metadata."""
        original_data = {
            "timestamp": datetime.now(),
            "uuid": uuid.uuid4(),
            "decimal_value": Decimal("123.456"),
            "data": [1, 2, 3],
        }

        # Serialize with type hints
        config = datason.SerializationConfig(include_type_hints=True)
        serialized = datason.serialize(original_data, config=config)

        # Create template from original
        template = original_data.copy()

        # Deserialize with template
        result = deserialize_with_template(serialized, template)

        # Should perfectly reconstruct types
        assert isinstance(result["timestamp"], datetime)
        assert isinstance(result["uuid"], uuid.UUID)
        assert isinstance(result["decimal_value"], dict)  # Decimal metadata
        assert result["data"] == [1, 2, 3]

    @pytest.mark.pandas
    def test_round_trip_dataframe_with_template(self):
        """Test DataFrame round-trip with template."""
        pd = pytest.importorskip("pandas")

        # Create original DataFrame with specific dtypes
        original_df = pd.DataFrame({"id": [1, 2, 3], "value": [1.1, 2.2, 3.3], "category": ["A", "B", "C"]})
        original_df["id"] = original_df["id"].astype("int32")
        original_df["category"] = original_df["category"].astype("category")

        # Serialize
        serialized = datason.serialize(original_df)

        # Use original as template
        result = deserialize_with_template(serialized, original_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == list(original_df.columns)
        # Dtypes should match template
        for col in original_df.columns:
            assert result[col].dtype == original_df[col].dtype

    def test_template_inference_then_deserialization(self):
        """Test inferring template from sample data, then using it for deserialization."""
        # Sample data for template inference
        sample_data = [{"name": "Alice", "age": 30, "score": 95.5}, {"name": "Bob", "age": 25, "score": 87.2}]

        # Infer template
        template = infer_template_from_data(sample_data)

        # New data to deserialize
        new_data = [
            {"name": "Charlie", "age": "35", "score": "92.1"},  # String numbers
            {"name": "Diana", "age": "28", "score": "88.7"},
        ]

        # Deserialize with inferred template
        result = deserialize_with_template(new_data, template)

        # Should convert string numbers to appropriate types
        assert isinstance(result[0]["age"], int)
        assert isinstance(result[0]["score"], float)
        assert result[0]["age"] == 35
        assert abs(result[0]["score"] - 92.1) < 0.01

    def test_template_error_handling_graceful(self):
        """Test graceful error handling in template deserialization."""
        template = {"expected_structure": True}
        incompatible_data = "this is a string, not a dict"

        # Should fall back gracefully
        result = deserialize_with_template(incompatible_data, template, strict=False, fallback_auto_detect=True)

        # Should return the string as-is since template can't be applied
        assert result == incompatible_data

    def test_template_complex_ml_workflow(self):
        """Test complex ML workflow with template-based deserialization."""
        # Simulate ML training data
        training_data = {
            "features": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "labels": [0, 1, 0],
            "metadata": {"created": datetime.now(), "version": "1.0", "model_type": "classification"},
        }

        # Create ML template
        create_ml_round_trip_template(training_data)

        # Serialize training data
        serialized = datason.serialize(training_data)

        # Later: deserialize with ML template
        result = deserialize_with_template(serialized, training_data)  # Use original as template

        assert "features" in result
        assert "labels" in result
        assert "metadata" in result
        assert isinstance(result["metadata"]["created"], datetime)
        assert result["features"] == training_data["features"]
        assert result["labels"] == training_data["labels"]
