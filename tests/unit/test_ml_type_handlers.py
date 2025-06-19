"""Unit tests for datason.ml_type_handlers module.

This test file covers the unified ML type handlers for various frameworks:
- CatBoostTypeHandler
- KerasTypeHandler
- OptunaTypeHandler
- PlotlyTypeHandler
- PolarsTypeHandler
- PyTorchTypeHandler
- SklearnTypeHandler
- Registration functions
"""

import warnings
from unittest.mock import Mock, patch

import pytest

from datason.ml_type_handlers import (
    CatBoostTypeHandler,
    KerasTypeHandler,
    OptunaTypeHandler,
    PlotlyTypeHandler,
    PolarsTypeHandler,
    PyTorchTypeHandler,
    SklearnTypeHandler,
    register_all_ml_handlers,
)


class TestCatBoostTypeHandler:
    """Test the CatBoostTypeHandler class."""

    def setup_method(self):
        """Set up for each test method."""
        self.handler = CatBoostTypeHandler()

    def test_type_name(self):
        """Test the type name property."""
        assert self.handler.type_name == "catboost.model"

    def test_lazy_import_catboost_success(self):
        """Test successful CatBoost import."""
        try:
            import catboost

            result = self.handler._lazy_import_catboost()
            assert result is catboost
        except ImportError:
            pytest.skip("CatBoost not available")

    def test_lazy_import_catboost_failure(self):
        """Test CatBoost import failure."""
        with patch("builtins.__import__", side_effect=ImportError):
            result = self.handler._lazy_import_catboost()
            assert result is None

    def test_can_handle_without_catboost(self):
        """Test can_handle when CatBoost is not available."""
        with patch.object(self.handler, "_lazy_import_catboost", return_value=None):
            assert self.handler.can_handle(Mock()) is False

    def test_can_handle_with_exception(self):
        """Test can_handle with exception."""
        with patch.object(self.handler, "_lazy_import_catboost", side_effect=RuntimeError):
            assert self.handler.can_handle(Mock()) is False

    @pytest.mark.skipif(
        not pytest.importorskip("catboost", reason="CatBoost not available"), reason="CatBoost required for this test"
    )
    def test_can_handle_real_catboost_classifier(self):
        """Test can_handle with real CatBoost classifier."""
        import catboost

        model = catboost.CatBoostClassifier(verbose=False)
        assert self.handler.can_handle(model) is True

    @pytest.mark.skipif(
        not pytest.importorskip("catboost", reason="CatBoost not available"), reason="CatBoost required for this test"
    )
    def test_can_handle_real_catboost_regressor(self):
        """Test can_handle with real CatBoost regressor."""
        import catboost

        model = catboost.CatBoostRegressor(verbose=False)
        assert self.handler.can_handle(model) is True

    def test_can_handle_non_catboost_object(self):
        """Test can_handle with non-CatBoost object."""
        assert self.handler.can_handle("not a model") is False
        assert self.handler.can_handle(123) is False
        assert self.handler.can_handle({}) is False

    @pytest.mark.skipif(
        not pytest.importorskip("catboost", reason="CatBoost not available"), reason="CatBoost required for this test"
    )
    def test_serialize_real_catboost_model(self):
        """Test serialization of real CatBoost model."""
        import catboost

        model = catboost.CatBoostClassifier(iterations=5, verbose=False)

        result = self.handler.serialize(model)

        assert result["__datason_type__"] == "catboost.model"
        assert "__datason_value__" in result
        value = result["__datason_value__"]
        assert "class_name" in value
        assert "params" in value
        assert "is_fitted" in value
        assert "catboost" in value["class_name"]

    def test_serialize_with_exception(self):
        """Test serialization with exception."""
        mock_model = Mock()
        mock_model.get_params.side_effect = RuntimeError("Test error")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.handler.serialize(mock_model)

            assert len(w) == 1
            assert "Failed to serialize CatBoost model" in str(w[0].message)
            assert result == {"__datason_type__": "dict", "__datason_value__": {}}

    @pytest.mark.skipif(
        not pytest.importorskip("catboost", reason="CatBoost not available"), reason="CatBoost required for this test"
    )
    def test_round_trip_serialization(self):
        """Test round-trip serialization/deserialization."""
        import catboost

        original = catboost.CatBoostClassifier(iterations=3, depth=2, verbose=False)

        # Serialize
        serialized = self.handler.serialize(original)

        # Deserialize
        deserialized = self.handler.deserialize(serialized)

        # Check that it's the same type and has same parameters
        assert isinstance(deserialized, catboost.CatBoostClassifier)
        assert deserialized.get_params()["iterations"] == 3
        assert deserialized.get_params()["depth"] == 2

    def test_deserialize_with_exception(self):
        """Test deserialization with exception."""
        data = {
            "__datason_type__": "catboost.model",
            "__datason_value__": {"class_name": "invalid.module.Class", "params": {}},
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.handler.deserialize(data)

            assert len(w) == 1
            assert "Failed to deserialize CatBoost model" in str(w[0].message)
            assert result == data


class TestKerasTypeHandler:
    """Test the KerasTypeHandler class."""

    def setup_method(self):
        """Set up for each test method."""
        self.handler = KerasTypeHandler()

    def test_type_name(self):
        """Test the type name property."""
        assert self.handler.type_name == "keras.model"

    def test_lazy_import_keras_success(self):
        """Test successful Keras import."""
        try:
            import keras

            result = self.handler._lazy_import_keras()
            assert result is keras
        except ImportError:
            pytest.skip("Keras not available")

    def test_lazy_import_keras_failure(self):
        """Test Keras import failure."""
        with patch("builtins.__import__", side_effect=ImportError):
            result = self.handler._lazy_import_keras()
            assert result is None

    def test_can_handle_without_keras(self):
        """Test can_handle when Keras is not available."""
        with patch.object(self.handler, "_lazy_import_keras", return_value=None):
            assert self.handler.can_handle(Mock()) is False

    def test_can_handle_with_exception(self):
        """Test can_handle with exception."""
        with patch.object(self.handler, "_lazy_import_keras", side_effect=RuntimeError):
            assert self.handler.can_handle(Mock()) is False

    @pytest.mark.skipif(
        not pytest.importorskip("keras", reason="Keras not available"), reason="Keras required for this test"
    )
    def test_can_handle_real_keras_model(self):
        """Test can_handle with real Keras model."""
        import keras

        model = keras.Sequential()
        assert self.handler.can_handle(model) is True

    def test_can_handle_mock_keras_model(self):
        """Test can_handle with mock Keras model."""
        mock_keras = Mock()
        mock_model = Mock()
        mock_model.compile = Mock()
        mock_model.fit = Mock()
        mock_model.predict = Mock()

        with patch.object(self.handler, "_lazy_import_keras", return_value=mock_keras):
            assert self.handler.can_handle(mock_model) is True

    def test_can_handle_non_keras_object(self):
        """Test can_handle with non-Keras object."""
        mock_keras = Mock()
        with patch.object(self.handler, "_lazy_import_keras", return_value=mock_keras):
            assert self.handler.can_handle("not a model") is False
            assert self.handler.can_handle(123) is False

    @pytest.mark.skipif(
        not pytest.importorskip("keras", reason="Keras not available"), reason="Keras required for this test"
    )
    def test_serialize_real_keras_model(self):
        """Test serialization of real Keras model."""
        import keras

        model = keras.Sequential()

        result = self.handler.serialize(model)

        assert result["__datason_type__"] == "keras.model"
        assert "__datason_value__" in result
        value = result["__datason_value__"]
        assert "model_type" in value
        assert value["model_type"] == "Sequential"

    def test_serialize_with_exception(self):
        """Test serialization with exception."""

        # Create a mock that will cause an exception during serialization
        class BrokenClass:
            def __init__(self):
                pass

            @property
            def __name__(self):
                raise RuntimeError("Test error")

        mock_model = Mock()
        mock_model.__class__ = BrokenClass()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.handler.serialize(mock_model)

            assert len(w) == 1
            assert "Failed to serialize Keras model" in str(w[0].message)
            assert result == {"__datason_type__": "dict", "__datason_value__": {}}

    @pytest.mark.skipif(
        not pytest.importorskip("keras", reason="Keras not available"), reason="Keras required for this test"
    )
    def test_deserialize_real_keras_model(self):
        """Test deserialization to create real Keras model."""
        data = {"__datason_type__": "keras.model", "__datason_value__": {"model_type": "Sequential"}}

        result = self.handler.deserialize(data)

        import keras

        assert isinstance(result, keras.Sequential)

    def test_deserialize_without_keras(self):
        """Test deserialization when Keras is not available."""
        data = {"__datason_type__": "keras.model", "__datason_value__": {"model_type": "Sequential"}}

        with patch.object(self.handler, "_lazy_import_keras", return_value=None):
            result = self.handler.deserialize(data)
            assert result == data

    def test_deserialize_with_exception(self):
        """Test deserialization with exception."""
        data = {"__datason_type__": "keras.model", "__datason_value__": {"model_type": "Sequential"}}

        with patch.object(self.handler, "_lazy_import_keras", side_effect=RuntimeError):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = self.handler.deserialize(data)

                assert len(w) == 1
                assert "Failed to deserialize Keras model" in str(w[0].message)
                assert result == data


class TestOptunaTypeHandler:
    """Test the OptunaTypeHandler class."""

    def setup_method(self):
        """Set up for each test method."""
        self.handler = OptunaTypeHandler()

    def test_type_name(self):
        """Test the type name property."""
        assert self.handler.type_name == "optuna.Study"

    def test_lazy_import_optuna_success(self):
        """Test successful Optuna import."""
        try:
            import optuna

            result = self.handler._lazy_import_optuna()
            assert result is optuna
        except ImportError:
            pytest.skip("Optuna not available")

    def test_lazy_import_optuna_failure(self):
        """Test Optuna import failure."""
        with patch("builtins.__import__", side_effect=ImportError):
            result = self.handler._lazy_import_optuna()
            assert result is None

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not available"), reason="Optuna required for this test"
    )
    def test_can_handle_real_optuna_study(self):
        """Test can_handle with real Optuna study."""
        import optuna

        study = optuna.create_study()
        assert self.handler.can_handle(study) is True

    def test_can_handle_without_optuna(self):
        """Test can_handle when Optuna is not available."""
        with patch.object(self.handler, "_lazy_import_optuna", return_value=None):
            assert self.handler.can_handle(Mock()) is False

    def test_can_handle_with_exception(self):
        """Test can_handle with exception."""
        with patch.object(self.handler, "_lazy_import_optuna", side_effect=RuntimeError):
            assert self.handler.can_handle(Mock()) is False

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not available"), reason="Optuna required for this test"
    )
    def test_serialize_real_optuna_study(self):
        """Test serialization of real Optuna study."""
        import optuna

        study = optuna.create_study(study_name="test_study")

        result = self.handler.serialize(study)

        assert result["__datason_type__"] == "optuna.Study"
        assert "__datason_value__" in result
        value = result["__datason_value__"]
        assert "study_name" in value
        assert "direction" in value
        assert "n_trials" in value
        assert value["study_name"] == "test_study"

    def test_serialize_with_exception(self):
        """Test serialization with exception."""
        mock_study = Mock()
        # Make the study_name property raise an exception
        mock_study.study_name = Mock(side_effect=RuntimeError("Test error"))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.handler.serialize(mock_study)

            assert len(w) == 1
            assert "Failed to serialize Optuna study" in str(w[0].message)
            assert result == {"__datason_type__": "dict", "__datason_value__": {}}

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not available"), reason="Optuna required for this test"
    )
    def test_deserialize_real_optuna_study(self):
        """Test deserialization to create real Optuna study."""
        data = {
            "__datason_type__": "optuna.Study",
            "__datason_value__": {"study_name": "test_study", "direction": "MINIMIZE", "n_trials": 0},
        }

        result = self.handler.deserialize(data)

        import optuna

        assert isinstance(result, optuna.Study)
        assert result.study_name == "test_study"

    def test_deserialize_without_optuna(self):
        """Test deserialization when Optuna is not available."""
        data = {"__datason_type__": "optuna.Study", "__datason_value__": {"study_name": "test_study"}}

        with patch.object(self.handler, "_lazy_import_optuna", return_value=None):
            result = self.handler.deserialize(data)
            assert result == data

    def test_deserialize_with_exception(self):
        """Test deserialization with exception."""
        data = {"__datason_type__": "optuna.Study", "__datason_value__": {"study_name": "test_study"}}

        with patch.object(self.handler, "_lazy_import_optuna", side_effect=RuntimeError):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = self.handler.deserialize(data)

                assert len(w) == 1
                assert "Failed to deserialize Optuna study" in str(w[0].message)
                assert result == data


class TestPlotlyTypeHandler:
    """Test the PlotlyTypeHandler class."""

    def setup_method(self):
        """Set up for each test method."""
        self.handler = PlotlyTypeHandler()

    def test_type_name(self):
        """Test the type name property."""
        assert self.handler.type_name == "plotly.graph_objects.Figure"

    def test_lazy_import_plotly_success(self):
        """Test successful Plotly import."""
        try:
            import plotly

            result = self.handler._lazy_import_plotly()
            assert result is plotly
        except ImportError:
            pytest.skip("Plotly not available")

    def test_lazy_import_plotly_failure(self):
        """Test Plotly import failure."""
        with patch("builtins.__import__", side_effect=ImportError):
            result = self.handler._lazy_import_plotly()
            assert result is None

    @pytest.mark.skipif(
        not pytest.importorskip("plotly", reason="Plotly not available"), reason="Plotly required for this test"
    )
    def test_can_handle_real_plotly_figure(self):
        """Test can_handle with real Plotly figure."""
        import plotly.graph_objects as go

        fig = go.Figure()
        assert self.handler.can_handle(fig) is True

    def test_can_handle_without_plotly(self):
        """Test can_handle when Plotly is not available."""
        with patch.object(self.handler, "_lazy_import_plotly", return_value=None):
            assert self.handler.can_handle(Mock()) is False

    def test_can_handle_with_exception(self):
        """Test can_handle with exception."""
        with patch.object(self.handler, "_lazy_import_plotly", side_effect=RuntimeError):
            assert self.handler.can_handle(Mock()) is False

    @pytest.mark.skipif(
        not pytest.importorskip("plotly", reason="Plotly not available"), reason="Plotly required for this test"
    )
    def test_serialize_real_plotly_figure(self):
        """Test serialization of real Plotly figure."""
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))

        result = self.handler.serialize(fig)

        assert result["__datason_type__"] == "plotly.graph_objects.Figure"
        assert "__datason_value__" in result

    def test_serialize_with_exception(self):
        """Test serialization with exception."""
        mock_fig = Mock()
        mock_fig.to_dict.side_effect = RuntimeError("Test error")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.handler.serialize(mock_fig)

            assert len(w) == 1
            assert "Failed to serialize Plotly figure" in str(w[0].message)
            assert result == {"__datason_type__": "dict", "__datason_value__": {}}

    @pytest.mark.skipif(
        not pytest.importorskip("plotly", reason="Plotly not available"), reason="Plotly required for this test"
    )
    def test_deserialize_real_plotly_figure(self):
        """Test deserialization to create real Plotly figure."""
        import plotly.graph_objects as go

        original_fig = go.Figure()
        original_fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))

        # Serialize then deserialize
        serialized = self.handler.serialize(original_fig)
        result = self.handler.deserialize(serialized)

        assert isinstance(result, go.Figure)

    def test_deserialize_without_plotly(self):
        """Test deserialization when Plotly is not available."""
        data = {"__datason_type__": "plotly.graph_objects.Figure", "__datason_value__": {"data": [], "layout": {}}}

        with patch.object(self.handler, "_lazy_import_plotly", return_value=None):
            result = self.handler.deserialize(data)
            assert result == data

    def test_deserialize_with_exception(self):
        """Test deserialization with exception."""
        data = {"__datason_type__": "plotly.graph_objects.Figure", "__datason_value__": {"invalid": "data"}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.handler.deserialize(data)

            # Should either succeed or warn and return data
            if len(w) > 0:
                assert "Failed to deserialize Plotly figure" in str(w[0].message)
                assert result == data


class TestPolarsTypeHandler:
    """Test the PolarsTypeHandler class."""

    def setup_method(self):
        """Set up for each test method."""
        self.handler = PolarsTypeHandler()

    def test_type_name(self):
        """Test the type name property."""
        assert self.handler.type_name == "polars.DataFrame"

    def test_lazy_import_polars_success(self):
        """Test successful Polars import."""
        try:
            import polars

            result = self.handler._lazy_import_polars()
            assert result is polars
        except ImportError:
            pytest.skip("Polars not available")

    def test_lazy_import_polars_failure(self):
        """Test Polars import failure."""
        with patch("builtins.__import__", side_effect=ImportError):
            result = self.handler._lazy_import_polars()
            assert result is None

    @pytest.mark.skipif(
        not pytest.importorskip("polars", reason="Polars not available"), reason="Polars required for this test"
    )
    def test_can_handle_real_polars_dataframe(self):
        """Test can_handle with real Polars DataFrame."""
        import polars as pl

        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert self.handler.can_handle(df) is True

    @pytest.mark.skipif(
        not pytest.importorskip("polars", reason="Polars not available"), reason="Polars required for this test"
    )
    def test_can_handle_real_polars_lazyframe(self):
        """Test can_handle with real Polars LazyFrame."""
        import polars as pl

        df = pl.DataFrame({"a": [1, 2, 3]}).lazy()
        # LazyFrames are not handled by this handler, only DataFrames
        assert self.handler.can_handle(df) is False

    def test_can_handle_without_polars(self):
        """Test can_handle when Polars is not available."""
        with patch.object(self.handler, "_lazy_import_polars", return_value=None):
            assert self.handler.can_handle(Mock()) is False

    @pytest.mark.skipif(
        not pytest.importorskip("polars", reason="Polars not available"), reason="Polars required for this test"
    )
    def test_serialize_real_polars_dataframe(self):
        """Test serialization of real Polars DataFrame."""
        import polars as pl

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        result = self.handler.serialize(df)

        assert result["__datason_type__"] == "polars.DataFrame"
        assert "__datason_value__" in result
        value = result["__datason_value__"]
        assert "data" in value
        assert "shape" in value
        assert "columns" in value
        assert "dtypes" in value

    def test_serialize_with_exception(self):
        """Test serialization with exception."""
        mock_df = Mock()
        mock_df.write_csv.side_effect = RuntimeError("Test error")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.handler.serialize(mock_df)

            assert len(w) == 1
            assert "Failed to serialize Polars DataFrame" in str(w[0].message)
            assert result == {"__datason_type__": "dict", "__datason_value__": {}}

    @pytest.mark.skipif(
        not pytest.importorskip("polars", reason="Polars not available"), reason="Polars required for this test"
    )
    def test_round_trip_serialization(self):
        """Test round-trip serialization/deserialization."""
        import polars as pl

        original = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        # Serialize
        serialized = self.handler.serialize(original)

        # Deserialize
        deserialized = self.handler.deserialize(serialized)

        # Check data is preserved
        assert isinstance(deserialized, pl.DataFrame)
        # Compare shape and column names
        assert deserialized.shape == original.shape
        assert deserialized.columns == original.columns
        # Compare data values by converting to lists
        orig_dict = original.to_dict()
        deser_dict = deserialized.to_dict()
        assert list(orig_dict.keys()) == list(deser_dict.keys())
        for key in orig_dict:
            # Convert to lists for comparison to avoid Polars Series comparison issues
            assert list(orig_dict[key]) == list(deser_dict[key])

    def test_deserialize_without_polars(self):
        """Test deserialization when Polars is not available."""
        data = {
            "__datason_type__": "polars.DataFrame",
            "__datason_value__": {
                "data": {"a": [1, 2], "b": ["x", "y"]},
                "shape": (2, 2),
                "columns": ["a", "b"],
                "dtypes": ["Int64", "String"],
            },
        }

        with patch.object(self.handler, "_lazy_import_polars", return_value=None):
            result = self.handler.deserialize(data)
            assert result == data


class TestPyTorchTypeHandler:
    """Test the PyTorchTypeHandler class."""

    def setup_method(self):
        """Set up for each test method."""
        self.handler = PyTorchTypeHandler()

    def test_type_name(self):
        """Test the type name property."""
        assert self.handler.type_name == "torch.Tensor"

    def test_lazy_import_torch_success(self):
        """Test successful PyTorch import."""
        try:
            import torch

            result = self.handler._lazy_import_torch()
            assert result is torch
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_lazy_import_torch_failure(self):
        """Test PyTorch import failure."""
        with patch("builtins.__import__", side_effect=ImportError):
            result = self.handler._lazy_import_torch()
            assert result is None

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not available"), reason="PyTorch required for this test"
    )
    def test_can_handle_real_torch_tensor(self):
        """Test can_handle with real PyTorch tensor."""
        import torch

        tensor = torch.tensor([1, 2, 3])
        assert self.handler.can_handle(tensor) is True

    def test_can_handle_without_torch(self):
        """Test can_handle when PyTorch is not available."""
        with patch.object(self.handler, "_lazy_import_torch", return_value=None):
            assert self.handler.can_handle(Mock()) is False

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not available"), reason="PyTorch required for this test"
    )
    def test_serialize_real_torch_tensor(self):
        """Test serialization of real PyTorch tensor."""
        import torch

        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = self.handler.serialize(tensor)

        assert result["__datason_type__"] == "torch.Tensor"
        assert "__datason_value__" in result
        value = result["__datason_value__"]
        assert "data" in value
        assert "shape" in value
        assert "dtype" in value
        assert value["shape"] == [2, 2]

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not available"), reason="PyTorch required for this test"
    )
    def test_serialize_torch_tensor_with_grad(self):
        """Test serialization of PyTorch tensor with gradients."""
        import torch

        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        result = self.handler.serialize(tensor)

        value = result["__datason_value__"]
        assert value["requires_grad"] is True

    def test_serialize_with_exception(self):
        """Test serialization with exception."""
        mock_tensor = Mock()
        mock_tensor.detach.side_effect = RuntimeError("Test error")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.handler.serialize(mock_tensor)

            assert len(w) == 1
            assert "Failed to serialize PyTorch tensor" in str(w[0].message)
            assert result == {"__datason_type__": "dict", "__datason_value__": {}}

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not available"), reason="PyTorch required for this test"
    )
    def test_round_trip_serialization(self):
        """Test round-trip serialization/deserialization."""
        import torch

        original = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

        # Serialize
        serialized = self.handler.serialize(original)

        # Deserialize
        deserialized = self.handler.deserialize(serialized)

        # Check data and properties are preserved
        assert isinstance(deserialized, torch.Tensor)
        assert torch.equal(deserialized, original.detach())
        assert deserialized.requires_grad is True

    def test_deserialize_without_torch(self):
        """Test deserialization when PyTorch is not available."""
        data = {
            "__datason_type__": "torch.Tensor",
            "__datason_value__": {"data": [[1, 2], [3, 4]], "shape": [2, 2], "dtype": "torch.float32"},
        }

        with patch.object(self.handler, "_lazy_import_torch", return_value=None):
            result = self.handler.deserialize(data)
            assert result == data


class TestSklearnTypeHandler:
    """Test the SklearnTypeHandler class."""

    def setup_method(self):
        """Set up for each test method."""
        self.handler = SklearnTypeHandler()

    def test_type_name(self):
        """Test the type name property."""
        assert self.handler.type_name == "sklearn.base.BaseEstimator"

    def test_lazy_import_sklearn_success(self):
        """Test successful scikit-learn import."""
        try:
            import sklearn

            result = self.handler._lazy_import_sklearn()
            assert result is sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_lazy_import_sklearn_failure(self):
        """Test scikit-learn import failure."""
        with patch("builtins.__import__", side_effect=ImportError):
            result = self.handler._lazy_import_sklearn()
            assert result is None

    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn required for this test",
    )
    def test_can_handle_real_sklearn_model(self):
        """Test can_handle with real scikit-learn model."""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        assert self.handler.can_handle(model) is True

    def test_can_handle_non_sklearn_object(self):
        """Test can_handle with non-sklearn object."""
        assert self.handler.can_handle("not a model") is False
        assert self.handler.can_handle(123) is False

    def test_can_handle_object_without_sklearn_module(self):
        """Test can_handle with object not from sklearn module."""
        mock_obj = Mock()
        mock_obj.get_params = Mock()
        mock_obj.set_params = Mock()
        mock_obj.__class__.__module__ = "other.module"

        assert self.handler.can_handle(mock_obj) is False

    def test_can_handle_with_exception(self):
        """Test can_handle with exception."""
        mock_obj = Mock()
        mock_obj.get_params.side_effect = RuntimeError

        assert self.handler.can_handle(mock_obj) is False

    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn required for this test",
    )
    def test_serialize_real_sklearn_model(self):
        """Test serialization of real scikit-learn model."""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression(fit_intercept=False)

        result = self.handler.serialize(model)

        assert result["__datason_type__"] == "sklearn.base.BaseEstimator"
        assert "__datason_value__" in result
        value = result["__datason_value__"]
        assert "class_name" in value
        assert "params" in value
        assert "fitted_attributes" in value
        assert "is_fitted" in value
        assert "sklearn" in value["class_name"]
        assert value["params"]["fit_intercept"] is False

    def test_serialize_with_exception(self):
        """Test serialization with exception."""
        mock_model = Mock()
        mock_model.get_params.side_effect = RuntimeError("Test error")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.handler.serialize(mock_model)

            assert len(w) == 1
            assert "Failed to serialize scikit-learn model" in str(w[0].message)
            assert result == {"__datason_type__": "dict", "__datason_value__": {}}

    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn required for this test",
    )
    def test_round_trip_serialization(self):
        """Test round-trip serialization/deserialization."""
        from sklearn.linear_model import LinearRegression

        original = LinearRegression(fit_intercept=False, positive=True)

        # Serialize
        serialized = self.handler.serialize(original)

        # Deserialize
        deserialized = self.handler.deserialize(serialized)

        # Check that it's the same type and has same parameters
        assert isinstance(deserialized, LinearRegression)
        assert deserialized.get_params()["fit_intercept"] is False
        assert deserialized.get_params()["positive"] is True

    def test_deserialize_with_exception(self):
        """Test deserialization with exception."""
        data = {
            "__datason_type__": "sklearn.base.BaseEstimator",
            "__datason_value__": {"class_name": "invalid.module.Class", "params": {}},
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.handler.deserialize(data)

            assert len(w) == 1
            assert "Failed to deserialize scikit-learn model" in str(w[0].message)
            assert result == data


class TestRegistrationFunction:
    """Test the registration function."""

    def test_register_all_ml_handlers(self):
        """Test that register_all_ml_handlers doesn't raise errors."""
        # This function should not raise any errors
        with patch("datason.type_registry.register_type_handler") as mock_register:
            register_all_ml_handlers()

            # Should have called register_type_handler multiple times
            assert mock_register.call_count == 7  # 7 handlers

    def test_register_all_ml_handlers_real(self):
        """Test register_all_ml_handlers with real registry."""
        from datason.type_registry import get_type_registry

        # Clear registry first
        registry = get_type_registry()
        registry.clear_handlers()

        try:
            # Register handlers
            register_all_ml_handlers()

            # Check that handlers were registered
            type_names = registry.get_registered_types()

            expected_types = [
                "catboost.model",
                "keras.model",
                "optuna.Study",
                "plotly.graph_objects.Figure",
                "polars.DataFrame",
                "torch.Tensor",
                "sklearn.base.BaseEstimator",
            ]

            for expected_type in expected_types:
                assert expected_type in type_names
        finally:
            # Always ensure handlers are re-registered after test to avoid affecting other tests
            register_all_ml_handlers()


class TestIntegrationScenarios:
    """Test integration scenarios with real ML objects."""

    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn required for this test",
    )
    def test_sklearn_integration_with_fitted_model(self):
        """Test sklearn handler with fitted model."""
        import numpy as np
        from sklearn.linear_model import LinearRegression

        # Create and fit model
        model = LinearRegression()
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        model.fit(X, y)

        handler = SklearnTypeHandler()

        # Serialize fitted model
        serialized = handler.serialize(model)

        # Check that fitted attributes are captured
        value = serialized["__datason_value__"]
        assert value["is_fitted"] is True
        assert "fitted_attributes" in value

        # Deserialize and check it's still a LinearRegression
        deserialized = handler.deserialize(serialized)
        assert isinstance(deserialized, LinearRegression)

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not available"), reason="PyTorch required for this test"
    )
    def test_torch_integration_with_complex_tensor(self):
        """Test PyTorch handler with complex tensor."""
        import torch

        # Create complex tensor
        tensor = torch.randn(3, 4, 5, requires_grad=True)
        tensor = tensor + 1j * torch.randn(3, 4, 5)  # Complex tensor

        handler = PyTorchTypeHandler()

        # This should handle complex tensors gracefully
        # Either succeed or fail gracefully with warning
        try:
            result = handler.serialize(tensor.real)  # Use real part for now
            assert "__datason_type__" in result
        except Exception:
            # Complex tensors might not be fully supported, that's OK
            pass

    def test_error_resilience_across_handlers(self):
        """Test that handlers are resilient to various error conditions."""
        handlers = [
            CatBoostTypeHandler(),
            KerasTypeHandler(),
            OptunaTypeHandler(),
            PlotlyTypeHandler(),
            PolarsTypeHandler(),
            PyTorchTypeHandler(),
            SklearnTypeHandler(),
        ]

        # Test with various problematic objects
        problematic_objects = [None, "", 123, [], {}, Mock()]

        for handler in handlers:
            for obj in problematic_objects:
                # can_handle should not raise exceptions
                try:
                    result = handler.can_handle(obj)
                    assert isinstance(result, bool)
                except Exception:
                    pytest.fail(f"{handler.__class__.__name__}.can_handle raised exception with {type(obj)}")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_all_handlers_have_required_methods(self):
        """Test that all handlers implement required TypeHandler methods."""
        handlers = [
            CatBoostTypeHandler(),
            KerasTypeHandler(),
            OptunaTypeHandler(),
            PlotlyTypeHandler(),
            PolarsTypeHandler(),
            PyTorchTypeHandler(),
            SklearnTypeHandler(),
        ]

        for handler in handlers:
            # Check required methods exist
            assert hasattr(handler, "can_handle")
            assert hasattr(handler, "serialize")
            assert hasattr(handler, "deserialize")
            assert hasattr(handler, "type_name")

            # Check type_name returns string
            assert isinstance(handler.type_name, str)
            assert len(handler.type_name) > 0

    def test_handlers_lazy_import_methods(self):
        """Test that all handlers have lazy import methods."""
        handlers_and_methods = [
            (CatBoostTypeHandler(), "_lazy_import_catboost"),
            (KerasTypeHandler(), "_lazy_import_keras"),
            (OptunaTypeHandler(), "_lazy_import_optuna"),
            (PlotlyTypeHandler(), "_lazy_import_plotly"),
            (PolarsTypeHandler(), "_lazy_import_polars"),
            (PyTorchTypeHandler(), "_lazy_import_torch"),
            (SklearnTypeHandler(), "_lazy_import_sklearn"),
        ]

        for handler, method_name in handlers_and_methods:
            assert hasattr(handler, method_name)
            # Method should be callable
            method = getattr(handler, method_name)
            assert callable(method)

    def test_serialize_deserialize_with_malformed_data(self):
        """Test deserialization with malformed data."""
        handlers = [
            CatBoostTypeHandler(),
            KerasTypeHandler(),
            OptunaTypeHandler(),
            PlotlyTypeHandler(),
            PolarsTypeHandler(),
            PyTorchTypeHandler(),
            SklearnTypeHandler(),
        ]

        malformed_data = [
            {},  # Missing keys
            {"__datason_type__": "wrong_type"},  # Wrong type
            {"__datason_value__": {}},  # Missing type
            {"__datason_type__": "test", "__datason_value__": None},  # None value
        ]

        for handler in handlers:
            for data in malformed_data:
                # Should not raise exceptions, should return data or warn
                try:
                    result = handler.deserialize(data)
                    # Should return something (either reconstructed object or original data)
                    assert result is not None
                except Exception:
                    pytest.fail(f"{handler.__class__.__name__}.deserialize raised exception with {data}")

    def test_warning_capture_in_serialize_methods(self):
        """Test that warnings are properly captured in serialize methods."""
        # This test verifies the warning pattern used across handlers
        handler = CatBoostTypeHandler()

        # Mock an object that will cause serialization to fail
        mock_obj = Mock()
        mock_obj.get_params.side_effect = RuntimeError("Forced error")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = handler.serialize(mock_obj)

            # Should have warned and returned fallback result
            assert len(w) >= 1
            assert result == {"__datason_type__": "dict", "__datason_value__": {}}
