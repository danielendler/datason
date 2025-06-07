"""
Comprehensive tests for unified ML type handlers.

This module tests the new unified architecture where both serialization
and deserialization logic are co-located in the same handler classes,
preventing the split-brain architecture problem.
"""

import pytest

# Optional dependency imports with availability checks
try:
    import catboost  # noqa: F401

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import keras

    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from datason.ml_type_handlers import (
    CatBoostTypeHandler,
    KerasTypeHandler,
    OptunaTypeHandler,
    PlotlyTypeHandler,
    PolarsTypeHandler,
)
from datason.type_registry import TypeHandler, get_type_registry


class TestCatBoostTypeHandler:
    """Test CatBoost unified type handler."""

    @pytest.mark.skipif(not HAS_CATBOOST, reason="CatBoost not available")
    def test_catboost_classifier_round_trip(self):
        """Test CatBoost classifier serialization and deserialization."""
        import catboost

        # Create a CatBoost classifier
        model = catboost.CatBoostClassifier(iterations=2, verbose=False, random_seed=42)

        # Test unified handler
        handler = CatBoostTypeHandler()

        # Test detection
        assert handler.can_handle(model)
        assert handler.type_name == "catboost.model"

        # Test serialization
        serialized = handler.serialize(model)
        assert serialized["__datason_type__"] == "catboost.model"
        assert "class_name" in serialized["__datason_value__"]
        assert "params" in serialized["__datason_value__"]
        assert "CatBoostClassifier" in serialized["__datason_value__"]["class_name"]

        # Test deserialization
        deserialized = handler.deserialize(serialized)
        assert isinstance(deserialized, catboost.CatBoostClassifier)
        assert type(deserialized) is type(model)

        # Verify parameters are preserved
        original_params = model.get_params()
        deserialized_params = deserialized.get_params()
        assert original_params["iterations"] == deserialized_params["iterations"]
        assert original_params["random_seed"] == deserialized_params["random_seed"]

    @pytest.mark.skipif(not HAS_CATBOOST, reason="CatBoost not available")
    def test_catboost_regressor_round_trip(self):
        """Test CatBoost regressor serialization and deserialization."""
        import catboost

        # Create a CatBoost regressor
        model = catboost.CatBoostRegressor(iterations=3, verbose=False, random_seed=123)

        handler = CatBoostTypeHandler()

        # Full round-trip test
        assert handler.can_handle(model)
        serialized = handler.serialize(model)
        deserialized = handler.deserialize(serialized)

        assert isinstance(deserialized, catboost.CatBoostRegressor)
        assert deserialized.get_params()["iterations"] == 3
        assert deserialized.get_params()["random_seed"] == 123


class TestKerasTypeHandler:
    """Test Keras unified type handler."""

    @pytest.mark.skipif(not HAS_KERAS, reason="Keras not available")
    def test_keras_sequential_round_trip(self):
        """Test Keras Sequential model serialization and deserialization."""

        # Create a Keras Sequential model
        model = keras.Sequential(
            [keras.layers.Dense(10, input_shape=(5,), activation="relu"), keras.layers.Dense(1, activation="sigmoid")]
        )

        handler = KerasTypeHandler()

        # Test detection
        assert handler.can_handle(model)
        assert handler.type_name == "keras.model"

        # Test serialization
        serialized = handler.serialize(model)
        assert serialized["__datason_type__"] == "keras.model"
        assert serialized["__datason_value__"]["model_type"] == "Sequential"

        # Test deserialization
        deserialized = handler.deserialize(serialized)
        assert isinstance(deserialized, keras.Sequential)

    @pytest.mark.skipif(not HAS_KERAS, reason="Keras not available")
    def test_keras_functional_model_round_trip(self):
        """Test Keras functional model serialization and deserialization."""

        # Create a functional model
        inputs = keras.Input(shape=(10,))
        outputs = keras.layers.Dense(1)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        handler = KerasTypeHandler()

        # Test that handler can detect functional models
        assert handler.can_handle(model)

        # Full round-trip test
        serialized = handler.serialize(model)
        deserialized = handler.deserialize(serialized)

        # Note: Functional models are reconstructed as Sequential for simplicity
        assert hasattr(deserialized, "compile")
        assert hasattr(deserialized, "fit")
        assert hasattr(deserialized, "predict")


class TestOptunaTypeHandler:
    """Test Optuna unified type handler."""

    @pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not available")
    def test_optuna_study_round_trip(self):
        """Test Optuna study serialization and deserialization."""

        # Create an Optuna study
        study = optuna.create_study(
            study_name="test_unified_study", direction="minimize", sampler=optuna.samplers.RandomSampler(seed=42)
        )

        handler = OptunaTypeHandler()

        # Test detection
        assert handler.can_handle(study)
        assert handler.type_name == "optuna.Study"

        # Test serialization
        serialized = handler.serialize(study)
        assert serialized["__datason_type__"] == "optuna.Study"
        assert serialized["__datason_value__"]["study_name"] == "test_unified_study"
        assert serialized["__datason_value__"]["direction"] == "MINIMIZE"

        # Test deserialization
        deserialized = handler.deserialize(serialized)
        assert isinstance(deserialized, optuna.Study)
        assert deserialized.study_name == "test_unified_study"
        assert deserialized.direction == optuna.study.StudyDirection.MINIMIZE

    @pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not available")
    def test_optuna_maximize_study_round_trip(self):
        """Test Optuna maximize study serialization and deserialization."""

        # Create a maximization study
        study = optuna.create_study(study_name="maximize_test", direction="maximize")

        handler = OptunaTypeHandler()

        # Full round-trip test
        serialized = handler.serialize(study)
        assert serialized["__datason_value__"]["direction"] == "MAXIMIZE"

        deserialized = handler.deserialize(serialized)
        assert deserialized.direction == optuna.study.StudyDirection.MAXIMIZE


class TestPlotlyTypeHandler:
    """Test Plotly unified type handler."""

    @pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available")
    def test_plotly_figure_round_trip(self):
        """Test Plotly figure serialization and deserialization."""

        # Create a Plotly figure
        fig = go.Figure(data=[go.Bar(x=[1, 2, 3], y=[4, 5, 6], name="test_bar")], layout=go.Layout(title="Test Figure"))

        handler = PlotlyTypeHandler()

        # Test detection
        assert handler.can_handle(fig)
        assert handler.type_name == "plotly.graph_objects.Figure"

        # Test serialization
        serialized = handler.serialize(fig)
        assert serialized["__datason_type__"] == "plotly.graph_objects.Figure"
        assert "data" in serialized["__datason_value__"]
        assert "layout" in serialized["__datason_value__"]

        # Test deserialization
        deserialized = handler.deserialize(serialized)
        assert isinstance(deserialized, go.Figure)

        # Verify data is preserved
        original_data = fig.to_dict()["data"]
        deserialized_data = deserialized.to_dict()["data"]
        assert len(original_data) == len(deserialized_data)
        assert original_data[0]["x"] == deserialized_data[0]["x"]
        assert original_data[0]["y"] == deserialized_data[0]["y"]

    @pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available")
    def test_plotly_scatter_plot_round_trip(self):
        """Test Plotly scatter plot serialization and deserialization."""

        # Create a scatter plot
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], mode="markers"))

        handler = PlotlyTypeHandler()

        # Full round-trip test
        serialized = handler.serialize(fig)
        deserialized = handler.deserialize(serialized)

        assert isinstance(deserialized, go.Figure)
        # Verify scatter data
        original_trace = fig.data[0]
        deserialized_trace = deserialized.data[0]
        assert list(original_trace.x) == list(deserialized_trace.x)
        assert list(original_trace.y) == list(deserialized_trace.y)


class TestPolarsTypeHandler:
    """Test Polars unified type handler."""

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_polars_dataframe_round_trip(self):
        """Test Polars DataFrame serialization and deserialization."""

        # Create a Polars DataFrame
        df = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "w"], "c": [1.1, 2.2, 3.3, 4.4]})

        handler = PolarsTypeHandler()

        # Test detection
        assert handler.can_handle(df)
        assert handler.type_name == "polars.DataFrame"

        # Test serialization
        serialized = handler.serialize(df)
        assert serialized["__datason_type__"] == "polars.DataFrame"
        assert "data" in serialized["__datason_value__"]
        assert "shape" in serialized["__datason_value__"]
        assert "columns" in serialized["__datason_value__"]

        # Test deserialization
        deserialized = handler.deserialize(serialized)
        assert isinstance(deserialized, pl.DataFrame)

        # Verify data integrity
        assert deserialized.shape == df.shape
        assert deserialized.columns == df.columns
        # Use equals for DataFrame comparison to avoid Series ambiguity
        assert deserialized.equals(df)

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_polars_complex_dataframe_round_trip(self):
        """Test Polars DataFrame with complex data types."""
        from datetime import date

        # Create a more complex DataFrame
        df = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "date_col": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            }
        )

        handler = PolarsTypeHandler()

        # Full round-trip test
        serialized = handler.serialize(df)
        deserialized = handler.deserialize(serialized)

        assert isinstance(deserialized, pl.DataFrame)
        assert deserialized.shape == df.shape
        # Note: Some dtypes might be different after round-trip, but data should be preserved
        assert len(deserialized.columns) == len(df.columns)


class TestUnifiedArchitectureIntegration:
    """Test the overall unified architecture integration."""

    def test_registry_has_all_handlers(self):
        """Test that all ML handlers are registered in the global registry."""
        registry = get_type_registry()
        registered_types = registry.get_registered_types()

        # Check that all our new unified handlers are registered
        expected_types = [
            "catboost.model",
            "keras.model",
            "optuna.Study",
            "plotly.graph_objects.Figure",
            "polars.DataFrame",
        ]

        for expected_type in expected_types:
            assert expected_type in registered_types, f"Type {expected_type} not registered"

    def test_registry_find_handler_functionality(self):
        """Test that registry can find appropriate handlers."""
        registry = get_type_registry()

        # Test each handler type if available
        if HAS_CATBOOST:
            import catboost

            model = catboost.CatBoostClassifier(iterations=1, verbose=False)
            handler = registry.find_handler(model)
            assert handler is not None
            assert handler.type_name == "catboost.model"

        if HAS_KERAS:
            import keras

            model = keras.Sequential([keras.layers.Dense(1, input_shape=(5,))])
            handler = registry.find_handler(model)
            assert handler is not None
            assert handler.type_name == "keras.model"

        if HAS_OPTUNA:
            import optuna

            study = optuna.create_study()
            handler = registry.find_handler(study)
            assert handler is not None
            assert handler.type_name == "optuna.Study"

        if HAS_PLOTLY:
            import plotly.graph_objects as go

            fig = go.Figure()
            handler = registry.find_handler(fig)
            assert handler is not None
            assert handler.type_name == "plotly.graph_objects.Figure"

        if HAS_POLARS:
            import polars as pl

            df = pl.DataFrame({"a": [1, 2, 3]})
            handler = registry.find_handler(df)
            assert handler is not None
            assert handler.type_name == "polars.DataFrame"

    def test_registry_get_handler_by_type(self):
        """Test getting handlers by type name."""
        registry = get_type_registry()

        # Test getting handlers by type name
        type_names = [
            "catboost.model",
            "keras.model",
            "optuna.Study",
            "plotly.graph_objects.Figure",
            "polars.DataFrame",
        ]

        for type_name in type_names:
            handler = registry.find_handler_by_type_name(type_name)
            assert handler is not None
            assert handler.type_name == type_name

    def test_registry_no_handler_found(self):
        """Test registry behavior when no handler is found."""
        registry = get_type_registry()

        # Test with object that no handler can handle
        class UnknownObject:
            pass

        handler = registry.find_handler(UnknownObject())
        assert handler is None

        # Test getting non-existent handler type
        handler = registry.find_handler_by_type_name("non.existent.Type")
        assert handler is None

    def test_no_split_brain_problem(self):
        """Test that we can't have serialization without deserialization."""

        # This test ensures the architecture prevents split-brain problems
        class IncompleteHandler(TypeHandler):
            def can_handle(self, obj):
                return False

            def serialize(self, obj):
                return {"__datason_type__": "test", "__datason_value__": {}}

            # Missing deserialize method will cause TypeError when instantiated
            # This is enforced by the ABC (Abstract Base Class)

            @property
            def type_name(self):
                return "test.incomplete"

        # This should fail because deserialize is not implemented
        with pytest.raises(TypeError):
            IncompleteHandler()

    def test_unified_handlers_vs_old_split_brain(self):
        """Demonstrate the benefits of unified handlers vs split-brain architecture."""

        # OLD WAY (split-brain): Serialization in one file, deserialization in another
        # ❌ Easy to forget deserialization when adding new type
        # ❌ Logic scattered across files
        # ❌ Hard to maintain

        # NEW WAY (unified): Both in same handler class
        # ✅ Impossible to forget deserialization
        # ✅ Logic co-located
        # ✅ Easy to maintain

        handler = CatBoostTypeHandler()

        # Both methods must exist and be implemented
        assert hasattr(handler, "serialize")
        assert hasattr(handler, "deserialize")
        assert hasattr(handler, "can_handle")
        assert hasattr(handler, "type_name")

        # Methods are callable
        assert callable(handler.serialize)
        assert callable(handler.deserialize)
        assert callable(handler.can_handle)

        print("✅ Unified architecture prevents split-brain problems!")


class TestErrorHandling:
    """Test error handling in unified type handlers."""

    def test_serialization_failure_handling(self):
        """Test handling of serialization failures."""
        handler = CatBoostTypeHandler()

        # Create a mock object that will cause serialization to fail
        class MockFailingObject:
            def get_params(self):
                raise RuntimeError("Intentional failure")

        # Should return a safe fallback
        result = handler.serialize(MockFailingObject())
        assert result["__datason_type__"] == "dict"
        assert result["__datason_value__"] == {}

    def test_deserialization_failure_handling(self):
        """Test handling of deserialization failures."""
        handler = CatBoostTypeHandler()

        # Create malformed data that will cause deserialization to fail
        bad_data = {
            "__datason_type__": "catboost.model",
            "__datason_value__": {"class_name": "non.existent.Class", "params": {}},
        }

        # Should return the original data as fallback
        result = handler.deserialize(bad_data)
        assert result == bad_data

    def test_import_failure_handling(self):
        """Test handling when ML libraries are not available."""

        # Create a handler that simulates missing library
        class TestHandler(CatBoostTypeHandler):
            def _lazy_import_catboost(self):
                return None  # Simulate missing library

        handler = TestHandler()

        # Should return False when library is missing
        assert not handler.can_handle("any_object")

    def test_keras_serialization_with_config(self):
        """Test Keras serialization with model config."""
        if not HAS_KERAS:
            pytest.skip("Keras not available")

        import keras

        model = keras.Sequential([keras.layers.Dense(1, input_shape=(5,))])
        handler = KerasTypeHandler()

        # Test serialization includes config
        serialized = handler.serialize(model)
        assert "config" in serialized["__datason_value__"]

    def test_keras_missing_library_handling(self):
        """Test Keras handler when library is missing."""

        class TestKerasHandler(KerasTypeHandler):
            def _lazy_import_keras(self):
                return None

        handler = TestKerasHandler()
        assert not handler.can_handle("any_object")

        # Test deserialize when library is missing
        data = {"__datason_type__": "keras.model", "__datason_value__": {}}
        result = handler.deserialize(data)
        assert result == data

    def test_plotly_missing_library_handling(self):
        """Test Plotly handler when library is missing."""

        class TestPlotlyHandler(PlotlyTypeHandler):
            def _lazy_import_plotly(self):
                return None

        handler = TestPlotlyHandler()
        assert not handler.can_handle("any_object")

        # Test deserialize when library is missing
        data = {"__datason_type__": "plotly.graph_objects.Figure", "__datason_value__": {}}
        result = handler.deserialize(data)
        assert result == data

    def test_polars_missing_library_handling(self):
        """Test Polars handler when library is missing."""

        class TestPolarsHandler(PolarsTypeHandler):
            def _lazy_import_polars(self):
                return None

        handler = TestPolarsHandler()
        assert not handler.can_handle("any_object")

        # Test deserialize when library is missing
        data = {"__datason_type__": "polars.DataFrame", "__datason_value__": {}}
        result = handler.deserialize(data)
        assert result == data

    def test_optuna_missing_library_handling(self):
        """Test Optuna handler when library is missing."""

        class TestOptunaHandler(OptunaTypeHandler):
            def _lazy_import_optuna(self):
                return None

        handler = TestOptunaHandler()
        assert not handler.can_handle("any_object")

        # Test deserialize when library is missing
        data = {"__datason_type__": "optuna.Study", "__datason_value__": {}}
        result = handler.deserialize(data)
        assert result == data

    def test_catboost_with_fitted_model(self):
        """Test CatBoost serialization of fitted model."""
        if not HAS_CATBOOST:
            pytest.skip("CatBoost not available")

        import catboost
        import numpy as np

        # Create and fit a model
        model = catboost.CatBoostClassifier(iterations=2, verbose=False)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        model.fit(X, y)

        handler = CatBoostTypeHandler()
        serialized = handler.serialize(model)

        # Should include fitted status if available
        assert "is_fitted" in serialized["__datason_value__"]

    def test_optuna_direction_enum_handling(self):
        """Test Optuna direction enum conversion."""
        if not HAS_OPTUNA:
            pytest.skip("Optuna not available")

        import optuna

        # Test both minimize and maximize directions
        for direction in ["minimize", "maximize"]:
            study = optuna.create_study(direction=direction)
            handler = OptunaTypeHandler()

            serialized = handler.serialize(study)
            deserialized = handler.deserialize(serialized)

            expected_direction = (
                optuna.study.StudyDirection.MINIMIZE
                if direction == "minimize"
                else optuna.study.StudyDirection.MAXIMIZE
            )
            assert deserialized.direction == expected_direction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
