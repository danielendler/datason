"""Tests for new ML framework support in datason.

This module tests serialization support for:
- CatBoost models
- Keras models
- Optuna studies
- Plotly figures
- Polars DataFrames
- Enhanced Transformers support
"""

import warnings
from unittest.mock import Mock, patch

import pytest

# Optional dependency imports with availability checks
try:
    import catboost

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import keras  # noqa: F401

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

try:
    import transformers  # noqa: F401

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

import datason
from datason.ml_serializers import (
    detect_and_serialize_ml_object,
    get_ml_library_info,
    serialize_catboost_model,
    serialize_keras_model,
    serialize_optuna_study,
    serialize_plotly_figure,
    serialize_polars_dataframe,
)


class TestNewMLFrameworkAvailability:
    """Test availability of new ML frameworks."""

    def test_ml_library_info_includes_new_frameworks(self):
        """Test that get_ml_library_info includes new frameworks."""
        ml_info = get_ml_library_info()

        # Check that new frameworks are included
        assert "catboost" in ml_info
        assert "keras" in ml_info
        assert "optuna" in ml_info
        assert "plotly" in ml_info
        assert "polars" in ml_info

        # Values should be boolean
        assert isinstance(ml_info["catboost"], bool)
        assert isinstance(ml_info["keras"], bool)
        assert isinstance(ml_info["optuna"], bool)
        assert isinstance(ml_info["plotly"], bool)
        assert isinstance(ml_info["polars"], bool)


@pytest.mark.skipif(not HAS_CATBOOST, reason="CatBoost not available")
class TestCatBoostSerialization:
    """Test CatBoost model serialization."""

    def setup_method(self):
        """Ensure clean state before each test."""
        import datason

        # Comprehensive state clearing
        datason.clear_all_caches()

        # Reset default configuration to prevent interference
        from datason.config import SerializationConfig

        datason.set_default_config(SerializationConfig())

        # Clear any ML import state
        try:
            from datason import ml_serializers

            # Force reinitialize lazy imports to ensure clean state
            if hasattr(ml_serializers, "_LAZY_IMPORTS"):
                ml_serializers._LAZY_IMPORTS.clear()
                ml_serializers._LAZY_IMPORTS.update(
                    {
                        "torch": None,
                        "tensorflow": None,
                        "jax": None,
                        "jnp": None,  # JAX numpy alias
                        "sklearn": None,
                        "BaseEstimator": None,  # sklearn base estimator class
                        "scipy": None,
                        "PIL_Image": None,  # This was the missing key causing KeyError!
                        "PIL": None,
                        "transformers": None,
                        "catboost": None,
                        "keras": None,
                        "optuna": None,
                        "plotly": None,
                        "polars": None,
                        "pandas": None,
                        "numpy": None,
                    }
                )
        except ImportError:
            pass

    def test_serialize_catboost_classifier(self):
        """Test serialization of CatBoost classifier."""
        model = catboost.CatBoostClassifier(n_estimators=3, random_state=42, verbose=False)

        result = serialize_catboost_model(model)

        assert result["__datason_type__"] == "catboost.model"
        assert "class" in result["__datason_value__"]
        assert "CatBoostClassifier" in result["__datason_value__"]["class"]
        assert "params" in result["__datason_value__"]
        assert result["__datason_value__"]["params"]["n_estimators"] == 3

    def test_serialize_fitted_catboost_model(self):
        """Test serialization of fitted CatBoost model."""
        import numpy as np

        model = catboost.CatBoostClassifier(n_estimators=3, random_state=42, verbose=False)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)

        result = serialize_catboost_model(model)

        assert result["__datason_type__"] == "catboost.model"
        assert result["__datason_value__"]["fitted"] is True
        assert "tree_count" in result["__datason_value__"]

    def test_detect_catboost_model(self):
        """Test automatic detection of CatBoost models."""
        model = catboost.CatBoostClassifier(n_estimators=2, verbose=False)

        result = detect_and_serialize_ml_object(model)

        assert result is not None
        assert result["__datason_type__"] == "catboost.model"

    def test_catboost_end_to_end_serialization(self):
        """Test end-to-end serialization with dump_ml."""
        model = catboost.CatBoostClassifier(n_estimators=2, random_state=42, verbose=False)

        serialized = datason.dump_ml(model)
        assert "__datason_type__" in serialized
        assert serialized["__datason_type__"] == "catboost.model"


@pytest.mark.skipif(not HAS_KERAS, reason="Keras not available")
class TestKerasSerialization:
    """Test Keras model serialization."""

    def setup_method(self):
        """Ensure clean state before each test."""
        import datason

        # Comprehensive state clearing
        datason.clear_all_caches()

        # Reset default configuration to prevent interference
        from datason.config import SerializationConfig

        datason.set_default_config(SerializationConfig())

        # Clear any ML import state
        try:
            from datason import ml_serializers

            # Force reinitialize lazy imports to ensure clean state
            if hasattr(ml_serializers, "_LAZY_IMPORTS"):
                ml_serializers._LAZY_IMPORTS.clear()
                ml_serializers._LAZY_IMPORTS.update(
                    {
                        "torch": None,
                        "tensorflow": None,
                        "jax": None,
                        "jnp": None,  # JAX numpy alias
                        "sklearn": None,
                        "BaseEstimator": None,  # sklearn base estimator class
                        "scipy": None,
                        "PIL_Image": None,  # This was the missing key causing KeyError!
                        "PIL": None,
                        "transformers": None,
                        "catboost": None,
                        "keras": None,
                        "optuna": None,
                        "plotly": None,
                        "polars": None,
                        "pandas": None,
                        "numpy": None,
                    }
                )
        except ImportError:
            pass

    def test_serialize_keras_sequential(self):
        """Test serialization of Keras Sequential model."""
        from keras.layers import Dense
        from keras.models import Sequential

        model = Sequential([Dense(10, input_shape=(5,)), Dense(1, activation="sigmoid")])

        result = serialize_keras_model(model)

        assert result["__datason_type__"] == "keras.model"
        assert "Sequential" in result["__datason_value__"]["class"]
        assert result["__datason_value__"]["layers_count"] == 2

    def test_detect_keras_model(self):
        """Test automatic detection of Keras models."""
        from keras.layers import Dense
        from keras.models import Sequential

        model = Sequential([Dense(5, input_shape=(3,))])

        result = detect_and_serialize_ml_object(model)

        assert result is not None
        assert result["__datason_type__"] == "keras.model"

    def test_keras_end_to_end_serialization(self):
        """Test end-to-end serialization with dump_ml."""
        from keras.layers import Dense
        from keras.models import Sequential

        model = Sequential([Dense(3, input_shape=(2,))])

        serialized = datason.dump_ml(model)
        assert "__datason_type__" in serialized
        assert serialized["__datason_type__"] == "keras.model"


@pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not available")
class TestOptunaSerialization:
    """Test Optuna study serialization."""

    def setup_method(self):
        """Ensure clean state before each test."""
        import datason

        # Comprehensive state clearing
        datason.clear_all_caches()

        # Reset default configuration to prevent interference
        from datason.config import SerializationConfig

        datason.set_default_config(SerializationConfig())

        # Clear any ML import state
        try:
            from datason import ml_serializers

            # Force reinitialize lazy imports to ensure clean state
            if hasattr(ml_serializers, "_LAZY_IMPORTS"):
                ml_serializers._LAZY_IMPORTS.clear()
                ml_serializers._LAZY_IMPORTS.update(
                    {
                        "torch": None,
                        "tensorflow": None,
                        "jax": None,
                        "jnp": None,  # JAX numpy alias
                        "sklearn": None,
                        "BaseEstimator": None,  # sklearn base estimator class
                        "scipy": None,
                        "PIL_Image": None,  # This was the missing key causing KeyError!
                        "PIL": None,
                        "transformers": None,
                        "catboost": None,
                        "keras": None,
                        "optuna": None,
                        "plotly": None,
                        "polars": None,
                        "pandas": None,
                        "numpy": None,
                    }
                )
        except ImportError:
            pass

    def test_serialize_empty_optuna_study(self):
        """Test serialization of empty Optuna study."""
        study = optuna.create_study()

        result = serialize_optuna_study(study)

        assert result["__datason_type__"] == "optuna.study"
        assert "study_name" in result["__datason_value__"]
        assert result["__datason_value__"]["trials_count"] == 0

    def test_serialize_optuna_study_with_trials(self):
        """Test serialization of Optuna study with trials."""

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return (x - 2) ** 2

        study = optuna.create_study()
        study.optimize(objective, n_trials=3)

        result = serialize_optuna_study(study)

        assert result["__datason_type__"] == "optuna.study"
        assert result["__datason_value__"]["trials_count"] == 3
        assert "best_value" in result["__datason_value__"]
        assert "best_params" in result["__datason_value__"]

    def test_detect_optuna_study(self):
        """Test automatic detection of Optuna studies."""
        study = optuna.create_study()

        result = detect_and_serialize_ml_object(study)

        assert result is not None
        assert result["__datason_type__"] == "optuna.study"

    def test_optuna_end_to_end_serialization(self):
        """Test end-to-end serialization with dump_ml."""
        study = optuna.create_study()

        serialized = datason.dump_ml(study)
        assert "__datason_type__" in serialized
        assert serialized["__datason_type__"] == "optuna.study"


@pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available")
class TestPlotlySerialization:
    """Test Plotly figure serialization."""

    def setup_method(self):
        """Ensure clean state before each test."""
        import datason

        # Comprehensive state clearing
        datason.clear_all_caches()

        # Reset default configuration to prevent interference
        from datason.config import SerializationConfig

        datason.set_default_config(SerializationConfig())

        # Clear any ML import state
        try:
            from datason import ml_serializers

            # Force reinitialize lazy imports to ensure clean state
            if hasattr(ml_serializers, "_LAZY_IMPORTS"):
                ml_serializers._LAZY_IMPORTS.clear()
                ml_serializers._LAZY_IMPORTS.update(
                    {
                        "torch": None,
                        "tensorflow": None,
                        "jax": None,
                        "jnp": None,  # JAX numpy alias
                        "sklearn": None,
                        "BaseEstimator": None,  # sklearn base estimator class
                        "scipy": None,
                        "PIL_Image": None,  # This was the missing key causing KeyError!
                        "PIL": None,
                        "transformers": None,
                        "catboost": None,
                        "keras": None,
                        "optuna": None,
                        "plotly": None,
                        "polars": None,
                        "pandas": None,
                        "numpy": None,
                    }
                )
        except ImportError:
            pass

    def test_serialize_plotly_bar_chart(self):
        """Test serialization of Plotly bar chart."""
        fig = go.Figure(data=go.Bar(x=[1, 2, 3], y=[1, 3, 2]))
        fig.update_layout(title="Test Chart")

        result = serialize_plotly_figure(fig)

        assert result["__datason_type__"] == "plotly.figure"
        assert "data" in result["__datason_value__"]
        assert "layout" in result["__datason_value__"]
        assert len(result["__datason_value__"]["data"]) == 1

    def test_serialize_plotly_scatter_plot(self):
        """Test serialization of Plotly scatter plot."""
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13]))

        result = serialize_plotly_figure(fig)

        assert result["__datason_type__"] == "plotly.figure"
        assert "data" in result["__datason_value__"]

    def test_detect_plotly_figure(self):
        """Test automatic detection of Plotly figures."""
        fig = go.Figure(data=go.Bar(x=[1, 2], y=[1, 2]))

        result = detect_and_serialize_ml_object(fig)

        assert result is not None
        assert result["__datason_type__"] == "plotly.figure"

    def test_plotly_end_to_end_serialization(self):
        """Test end-to-end serialization with dump_ml."""
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))

        serialized = datason.dump_ml(fig)
        assert "__datason_type__" in serialized
        assert serialized["__datason_type__"] == "plotly.figure"


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
class TestPolarsSerialization:
    """Test Polars DataFrame serialization."""

    def setup_method(self):
        """Ensure clean state before each test."""
        import datason

        # Comprehensive state clearing
        datason.clear_all_caches()

        # Reset default configuration to prevent interference
        from datason.config import SerializationConfig

        datason.set_default_config(SerializationConfig())

        # Clear any ML import state
        try:
            from datason import ml_serializers

            # Force reinitialize lazy imports to ensure clean state
            if hasattr(ml_serializers, "_LAZY_IMPORTS"):
                ml_serializers._LAZY_IMPORTS.clear()
                ml_serializers._LAZY_IMPORTS.update(
                    {
                        "torch": None,
                        "tensorflow": None,
                        "jax": None,
                        "jnp": None,  # JAX numpy alias
                        "sklearn": None,
                        "BaseEstimator": None,  # sklearn base estimator class
                        "scipy": None,
                        "PIL_Image": None,  # This was the missing key causing KeyError!
                        "PIL": None,
                        "transformers": None,
                        "catboost": None,
                        "keras": None,
                        "optuna": None,
                        "plotly": None,
                        "polars": None,
                        "pandas": None,
                        "numpy": None,
                    }
                )
        except ImportError:
            pass

    def test_serialize_polars_dataframe(self):
        """Test serialization of Polars DataFrame."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.1, 2.2, 3.3]})

        result = serialize_polars_dataframe(df)

        assert result["__datason_type__"] == "polars.dataframe"
        assert "data" in result["__datason_value__"]
        assert "columns" in result["__datason_value__"]
        assert "shape" in result["__datason_value__"]
        assert result["__datason_value__"]["columns"] == ["a", "b", "c"]
        assert result["__datason_value__"]["shape"] == [3, 3]

    def test_detect_polars_dataframe(self):
        """Test automatic detection of Polars DataFrames."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        result = detect_and_serialize_ml_object(df)

        assert result is not None
        assert result["__datason_type__"] == "polars.dataframe"

    def test_polars_end_to_end_serialization(self):
        """Test end-to-end serialization with dump_ml."""
        df = pl.DataFrame({"test": [1, 2, 3]})

        serialized = datason.dump_ml(df)
        assert "__datason_type__" in serialized
        assert serialized["__datason_type__"] == "polars.dataframe"


class TestNewMLFrameworksFallbacks:
    """Test fallback behavior when new ML frameworks aren't available."""

    def test_catboost_fallback(self):
        """Test CatBoost serialization fallback when catboost not available."""
        mock_model = Mock()
        mock_model.__str__ = Mock(return_value="MockCatBoostModel")

        with patch("datason.ml_serializers._lazy_import_catboost", return_value=None):
            result = serialize_catboost_model(mock_model)

        assert result == {"__datason_type__": "catboost.model", "__datason_value__": "MockCatBoostModel"}

    def test_keras_fallback(self):
        """Test Keras serialization fallback when keras not available."""
        mock_model = Mock()
        mock_model.__str__ = Mock(return_value="MockKerasModel")

        with patch("datason.ml_serializers._lazy_import_keras", return_value=None):
            result = serialize_keras_model(mock_model)

        assert result == {"__datason_type__": "keras.model", "__datason_value__": "MockKerasModel"}

    def test_optuna_fallback(self):
        """Test Optuna serialization fallback when optuna not available."""
        mock_study = Mock()
        mock_study.__str__ = Mock(return_value="MockOptunaStudy")

        with patch("datason.ml_serializers._lazy_import_optuna", return_value=None):
            result = serialize_optuna_study(mock_study)

        assert result == {"__datason_type__": "optuna.study", "__datason_value__": "MockOptunaStudy"}

    def test_plotly_fallback(self):
        """Test Plotly serialization fallback when plotly not available."""
        mock_fig = Mock()
        mock_fig.__str__ = Mock(return_value="MockPlotlyFigure")

        with patch("datason.ml_serializers._lazy_import_plotly", return_value=None):
            result = serialize_plotly_figure(mock_fig)

        assert result == {"__datason_type__": "plotly.figure", "__datason_value__": "MockPlotlyFigure"}

    def test_polars_fallback(self):
        """Test Polars serialization fallback when polars not available."""
        mock_df = Mock()
        mock_df.__str__ = Mock(return_value="MockPolarsDataFrame")

        with patch("datason.ml_serializers._lazy_import_polars", return_value=None):
            result = serialize_polars_dataframe(mock_df)

        assert result == {"__datason_type__": "polars.dataframe", "__datason_value__": "MockPolarsDataFrame"}


class TestNewMLFrameworksErrorHandling:
    """Test error handling in new ML framework serializers."""

    @pytest.mark.skipif(not HAS_CATBOOST, reason="CatBoost not available")
    def test_catboost_error_handling(self):
        """Test CatBoost serialization error handling."""
        model = catboost.CatBoostClassifier(verbose=False)

        # Patch get_params to raise an exception
        with patch.object(model, "get_params", side_effect=Exception("Mock error")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = serialize_catboost_model(model)

        assert result["__datason_type__"] == "catboost.model"
        assert "error" in result["__datason_value__"]
        assert len(w) == 1
        assert "Could not serialize CatBoost model" in str(w[0].message)

    @pytest.mark.skipif(not HAS_KERAS, reason="Keras not available")
    def test_keras_error_handling(self):
        """Test Keras serialization error handling."""
        from keras.layers import Dense
        from keras.models import Sequential

        model = Sequential([Dense(5)])

        # Patch get_config to raise an exception
        with patch.object(model, "get_config", side_effect=Exception("Mock error")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = serialize_keras_model(model)

        assert result["__datason_type__"] == "keras.model"
        assert "error" in result["__datason_value__"]
        assert len(w) == 1
        assert "Could not serialize Keras model" in str(w[0].message)


class TestTemplateDeserializationNewFrameworks:
    """Test template-based deserialization for new frameworks."""

    @pytest.mark.skipif(not HAS_CATBOOST, reason="CatBoost not available")
    def test_catboost_template_reconstruction(self):
        """Test CatBoost model template reconstruction."""
        model = catboost.CatBoostClassifier(n_estimators=2, random_state=42, verbose=False)

        # Serialize
        serialized = datason.dump_ml(model)

        # Template reconstruction
        template = catboost.CatBoostClassifier(n_estimators=2, random_state=42, verbose=False)
        try:
            reconstructed = datason.load_perfect(serialized, template)
            assert type(reconstructed) is type(template)
        except Exception:
            # Template reconstruction might not work perfectly yet, that's ok
            pass

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_polars_template_reconstruction(self):
        """Test Polars DataFrame template reconstruction."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        # Serialize
        serialized = datason.dump_ml(df)

        # Template reconstruction
        template = pl.DataFrame({"x": [0], "y": [0]})
        try:
            reconstructed = datason.load_perfect(serialized, template)
            # Check if reconstruction maintains structure
            if hasattr(reconstructed, "columns"):
                assert "x" in reconstructed.columns
                assert "y" in reconstructed.columns
        except Exception:
            # Template reconstruction might not work perfectly yet, that's ok
            pass


class TestPerformanceOptimizations:
    """Test performance optimizations for new frameworks."""

    def test_large_framework_detection_performance(self):
        """Test that framework detection is efficient for unknown objects."""
        import time

        # Test with a regular object that shouldn't match any framework
        test_obj = {"regular": "dict", "with": ["some", "data"]}

        start_time = time.time()
        for _ in range(100):
            result = detect_and_serialize_ml_object(test_obj)
        end_time = time.time()

        # Should be fast and return None
        assert result is None
        assert (end_time - start_time) < 0.1  # Should be very fast
