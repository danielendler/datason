"""Tests for the ml_misc plugin (Polars, JAX, CatBoost, Optuna, Plotly)."""

from __future__ import annotations

from typing import Any

import pytest

import datason

# =========================================================================
# Polars
# =========================================================================

polars = pytest.importorskip("polars")


class TestPolarsDataFrame:
    """Polars DataFrame serialization."""

    def test_basic_roundtrip(self) -> None:
        df = polars.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        s = datason.dumps(df)
        restored = datason.loads(s)
        assert isinstance(restored, polars.DataFrame)
        assert restored.shape == (3, 2)
        assert restored.columns == ["a", "b"]

    def test_string_columns(self) -> None:
        df = polars.DataFrame({"name": ["Alice", "Bob"], "score": [95, 87]})
        s = datason.dumps(df)
        restored = datason.loads(s)
        assert isinstance(restored, polars.DataFrame)
        assert restored["name"].to_list() == ["Alice", "Bob"]

    def test_without_type_hints(self) -> None:
        df = polars.DataFrame({"x": [1, 2]})
        s = datason.dumps(df, include_type_hints=False)
        restored = datason.loads(s)
        # Without hints, returns list of dicts
        assert isinstance(restored, list)

    def test_empty_dataframe(self) -> None:
        df = polars.DataFrame({"a": [], "b": []}).cast({"a": polars.Int64, "b": polars.Float64})
        s = datason.dumps(df)
        restored = datason.loads(s)
        assert isinstance(restored, polars.DataFrame)
        assert restored.shape[0] == 0


class TestPolarsSeries:
    """Polars Series serialization."""

    def test_basic_roundtrip(self) -> None:
        s = polars.Series("scores", [1.0, 2.0, 3.0])
        json_str = datason.dumps(s)
        restored = datason.loads(json_str)
        assert isinstance(restored, polars.Series)
        assert restored.name == "scores"
        assert restored.to_list() == [1.0, 2.0, 3.0]

    def test_integer_series(self) -> None:
        s = polars.Series("ids", [10, 20, 30])
        json_str = datason.dumps(s)
        restored = datason.loads(json_str)
        assert isinstance(restored, polars.Series)
        assert restored.to_list() == [10, 20, 30]


# =========================================================================
# JAX
# =========================================================================

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class TestJaxArray:
    """JAX array serialization."""

    def test_1d_roundtrip(self) -> None:
        arr = jnp.array([1.0, 2.0, 3.0])
        s = datason.dumps(arr)
        restored = datason.loads(s)
        assert isinstance(restored, jax.Array)
        assert list(restored) == [1.0, 2.0, 3.0]

    def test_2d_roundtrip(self) -> None:
        arr = jnp.array([[1, 2], [3, 4]])
        s = datason.dumps(arr)
        restored = datason.loads(s)
        assert isinstance(restored, jax.Array)
        assert restored.shape == (2, 2)

    def test_without_type_hints(self) -> None:
        arr = jnp.array([10, 20])
        s = datason.dumps(arr, include_type_hints=False)
        restored = datason.loads(s)
        assert isinstance(restored, list)

    def test_float32_dtype(self) -> None:
        arr = jnp.array([1.5, 2.5], dtype=jnp.float32)
        s = datason.dumps(arr)
        restored = datason.loads(s)
        assert isinstance(restored, jax.Array)


# =========================================================================
# CatBoost
# =========================================================================

catboost = pytest.importorskip("catboost")


class TestCatBoostModel:
    """CatBoost model serialization (metadata only)."""

    def test_unfitted_model(self) -> None:
        model = catboost.CatBoostClassifier(iterations=10, verbose=0)
        s = datason.dumps(model)
        restored = datason.loads(s)
        # CatBoost models serialize as metadata dicts (not reconstructable)
        assert isinstance(restored, dict)
        assert restored["class"] == "CatBoostClassifier"

    def test_fitted_model(self) -> None:
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        model = catboost.CatBoostClassifier(iterations=5, verbose=0)
        model.fit(X, y)
        s = datason.dumps(model)
        restored = datason.loads(s)
        assert isinstance(restored, dict)
        assert restored["tree_count"] == 5


# =========================================================================
# Optuna
# =========================================================================

optuna = pytest.importorskip("optuna")


class TestOptunaStudy:
    """Optuna study serialization (metadata only)."""

    def test_empty_study(self) -> None:
        study = optuna.create_study(direction="minimize")
        s = datason.dumps(study)
        restored = datason.loads(s)
        assert isinstance(restored, dict)
        assert restored["direction"] == "MINIMIZE"
        assert restored["n_trials"] == 0

    def test_study_with_trials(self) -> None:
        def objective(trial: Any) -> float:
            x = trial.suggest_float("x", -10, 10)
            return x**2

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5, show_progress_bar=False)
        s = datason.dumps(study)
        restored = datason.loads(s)
        assert isinstance(restored, dict)
        assert restored["n_trials"] == 5
        assert "best_value" in restored
        assert "best_params" in restored
        assert len(restored["trials"]) == 5


# =========================================================================
# Plotly
# =========================================================================

go = pytest.importorskip("plotly.graph_objects")


class TestPlotlyFigure:
    """Plotly figure serialization."""

    def test_scatter_roundtrip(self) -> None:
        fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])
        s = datason.dumps(fig)
        restored = datason.loads(s)
        assert isinstance(restored, go.Figure)

    def test_bar_chart(self) -> None:
        fig = go.Figure(data=[go.Bar(x=["a", "b"], y=[10, 20])])
        s = datason.dumps(fig)
        restored = datason.loads(s)
        assert isinstance(restored, go.Figure)

    def test_without_type_hints(self) -> None:
        fig = go.Figure(data=[go.Scatter(x=[1], y=[2])])
        s = datason.dumps(fig, include_type_hints=False)
        restored = datason.loads(s)
        # Without hints, returns raw dict
        assert isinstance(restored, dict)
