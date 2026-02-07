"""Plugin for miscellaneous ML framework types.

Handles Polars DataFrames/Series, JAX arrays, CatBoost models,
Optuna studies, and Plotly figures. Each library is imported lazily
and guarded by availability checks.

This module imports all five libraries directly — if any is not
installed, only those types are skipped (not the whole plugin).
"""

# pyright: reportOptionalMemberAccess=false
# pyright: reportConstantRedefinition=false
from __future__ import annotations

from typing import Any

from .._errors import PluginError
from .._protocols import DeserializeContext, SerializeContext
from .._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY

# Lazy import flags — pyright can't model try/except import patterns,
# so we suppress the relevant warnings at module level above.
_HAS_POLARS = False
_HAS_JAX = False
_HAS_CATBOOST = False
_HAS_OPTUNA = False
_HAS_PLOTLY = False

pl: Any = None
jax: Any = None
jnp: Any = None
catboost: Any = None
optuna: Any = None
go: Any = None

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:
    pass

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    pass

try:
    import catboost

    _HAS_CATBOOST = True
except ImportError:
    pass

try:
    import optuna

    _HAS_OPTUNA = True
except ImportError:
    pass

try:
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:
    pass


_TYPE_NAMES = frozenset(
    {
        "polars.DataFrame",
        "polars.Series",
        "jax.Array",
        "catboost.Model",
        "optuna.Study",
        "plotly.Figure",
    }
)


class MlMiscPlugin:
    """Handles Polars, JAX, CatBoost, Optuna, and Plotly types."""

    @property
    def name(self) -> str:
        return "ml_misc"

    @property
    def priority(self) -> int:
        return 350

    def can_handle(self, obj: Any) -> bool:
        if _HAS_POLARS and isinstance(obj, pl.DataFrame | pl.Series):
            return True
        if _HAS_JAX and isinstance(obj, jax.Array):
            return True
        if _HAS_CATBOOST and isinstance(obj, catboost.CatBoost):
            return True
        if _HAS_OPTUNA and isinstance(obj, optuna.study.Study):
            return True
        return bool(_HAS_PLOTLY and isinstance(obj, go.Figure))

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        return _serialize_ml_misc(obj, ctx)

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        return data.get(TYPE_METADATA_KEY, "") in _TYPE_NAMES

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        return _deserialize_ml_misc(data)


def _serialize_ml_misc(obj: Any, ctx: SerializeContext) -> Any:
    """Route to type-specific serializer."""
    if _HAS_POLARS and isinstance(obj, pl.DataFrame):
        return _serialize_polars_df(obj, ctx)
    if _HAS_POLARS and isinstance(obj, pl.Series):
        return _serialize_polars_series(obj, ctx)
    if _HAS_JAX and isinstance(obj, jax.Array):
        return _serialize_jax_array(obj, ctx)
    if _HAS_CATBOOST and isinstance(obj, catboost.CatBoost):
        return _serialize_catboost(obj, ctx)
    if _HAS_OPTUNA and isinstance(obj, optuna.study.Study):
        return _serialize_optuna_study(obj, ctx)
    if _HAS_PLOTLY and isinstance(obj, go.Figure):
        return _serialize_plotly_figure(obj, ctx)
    raise PluginError(f"Unsupported ml_misc type: {type(obj).__name__}")


# =========================================================================
# Polars
# =========================================================================


def _serialize_polars_df(df: Any, ctx: SerializeContext) -> Any:
    """Serialize a Polars DataFrame."""
    value = {
        "columns": df.columns,
        "data": {col: df[col].to_list() for col in df.columns},
        "schema": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes, strict=True)},
    }
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "polars.DataFrame", VALUE_METADATA_KEY: value}
    return df.to_dicts()


def _serialize_polars_series(series: Any, ctx: SerializeContext) -> Any:
    """Serialize a Polars Series."""
    value = {
        "name": series.name,
        "data": series.to_list(),
        "dtype": str(series.dtype),
    }
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "polars.Series", VALUE_METADATA_KEY: value}
    return series.to_list()


# =========================================================================
# JAX
# =========================================================================


def _serialize_jax_array(arr: Any, ctx: SerializeContext) -> Any:
    """Serialize a JAX array."""
    import numpy as np

    np_arr = np.asarray(arr)
    value = {
        "data": np_arr.tolist(),
        "dtype": str(np_arr.dtype),
        "shape": list(np_arr.shape),
    }
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "jax.Array", VALUE_METADATA_KEY: value}
    return np_arr.tolist()


# =========================================================================
# CatBoost
# =========================================================================


def _serialize_catboost(model: Any, ctx: SerializeContext) -> Any:
    """Serialize a CatBoost model via its JSON export."""
    value: dict[str, Any] = {"class": type(model).__name__}
    if model.is_fitted():
        value["params"] = model.get_all_params()
        value["tree_count"] = model.tree_count_
    else:
        value["params"] = model.get_params()
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "catboost.Model", VALUE_METADATA_KEY: value}
    return value


# =========================================================================
# Optuna
# =========================================================================


def _serialize_optuna_study(study: Any, ctx: SerializeContext) -> Any:
    """Serialize Optuna study metadata (not the full storage)."""
    trials_data = []
    for trial in study.trials:
        trials_data.append(
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            }
        )
    value: dict[str, Any] = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "trials": trials_data,
    }
    if study.trials:
        value["best_value"] = study.best_value
        value["best_params"] = study.best_params
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "optuna.Study", VALUE_METADATA_KEY: value}
    return value


# =========================================================================
# Plotly
# =========================================================================


def _serialize_plotly_figure(fig: Any, ctx: SerializeContext) -> Any:
    """Serialize a Plotly figure to its JSON dict representation."""
    fig_dict = fig.to_dict()
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "plotly.Figure", VALUE_METADATA_KEY: fig_dict}
    return fig_dict


# =========================================================================
# Deserialization
# =========================================================================


def _deserialize_ml_misc(data: dict[str, Any]) -> Any:
    """Route to type-specific deserializer."""
    type_name = data[TYPE_METADATA_KEY]
    value = data[VALUE_METADATA_KEY]

    match type_name:
        case "polars.DataFrame":
            return _reconstruct_polars_df(value)
        case "polars.Series":
            return _reconstruct_polars_series(value)
        case "jax.Array":
            return _reconstruct_jax_array(value)
        case "catboost.Model":
            return value  # Metadata only — cannot reconstruct fitted model
        case "optuna.Study":
            return value  # Metadata only — cannot reconstruct study storage
        case "plotly.Figure":
            return _reconstruct_plotly_figure(value)
        case _:
            raise PluginError(f"Unknown ml_misc type: {type_name}")


def _reconstruct_polars_df(value: Any) -> Any:
    """Reconstruct a Polars DataFrame."""
    if not _HAS_POLARS:
        raise PluginError("polars is not installed")
    return pl.DataFrame(value["data"])


def _reconstruct_polars_series(value: Any) -> Any:
    """Reconstruct a Polars Series."""
    if not _HAS_POLARS:
        raise PluginError("polars is not installed")
    return pl.Series(value["name"], value["data"])


def _reconstruct_jax_array(value: Any) -> Any:
    """Reconstruct a JAX array."""
    if not _HAS_JAX:
        raise PluginError("jax is not installed")
    import numpy as np

    np_arr = np.array(value["data"], dtype=value["dtype"])
    return jnp.array(np_arr)


def _reconstruct_plotly_figure(value: Any) -> Any:
    """Reconstruct a Plotly Figure from dict."""
    if not _HAS_PLOTLY:
        raise PluginError("plotly is not installed")
    return go.Figure(value)
