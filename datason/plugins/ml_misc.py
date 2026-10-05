"""Plugin for miscellaneous ML framework types.

Handles Polars DataFrames/Series, JAX arrays, CatBoost models,
Optuna studies, and Plotly figures. Libraries are imported per family on first
use. If a library is unavailable, only its types are skipped.
"""

# pyright: reportOptionalMemberAccess=false
# pyright: reportConstantRedefinition=false
from __future__ import annotations

# Imports are selected per family, including direct plugin use.
import importlib
import threading
from typing import Any, Literal

from .._errors import PluginError
from .._protocols import DeserializeContext, SerializeContext
from .._reconstruction import check_dense_allocation
from .._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY
from ._lazy import matches_family

_Framework = Literal["polars", "jax", "catboost", "optuna", "plotly"]
_FRAMEWORKS: tuple[_Framework, ...] = ("polars", "jax", "catboost", "optuna", "plotly")
_loaded: set[_Framework] = set()
_import_lock = threading.Lock()
_HAS_POLARS = _HAS_JAX = _HAS_CATBOOST = _HAS_OPTUNA = _HAS_PLOTLY = False
pl: Any = None
jax: Any = None
jnp: Any = None
catboost: Any = None
optuna: Any = None
go: Any = None


def _load_framework(name: _Framework) -> None:
    global pl, jax, jnp, catboost, optuna, go
    global _HAS_POLARS, _HAS_JAX, _HAS_CATBOOST, _HAS_OPTUNA, _HAS_PLOTLY
    if name in _loaded:
        return
    with _import_lock:
        if name in _loaded:
            return
        try:
            match name:
                case "polars":
                    pl = importlib.import_module("polars")
                    _HAS_POLARS = True
                case "jax":
                    jax = importlib.import_module("jax")
                    jnp = importlib.import_module("jax.numpy")
                    _HAS_JAX = True
                case "catboost":
                    catboost = importlib.import_module("catboost")
                    _HAS_CATBOOST = True
                case "optuna":
                    optuna = importlib.import_module("optuna")
                    _HAS_OPTUNA = True
                case "plotly":
                    go = importlib.import_module("plotly.graph_objects")
                    _HAS_PLOTLY = True
                case _:
                    raise ValueError(f"Unknown optional ML family: {name}")
        except ImportError:
            pass
        _loaded.add(name)


def _load_for_object(obj: Any) -> None:
    for name in _FRAMEWORKS:
        if name in _loaded:
            continue
        roots = ("jax", "jaxlib") if name == "jax" else (name,)
        if matches_family(obj, roots):
            _load_framework(name)


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
        _load_for_object(obj)
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
        return _deserialize_ml_misc(data, ctx)


def _serialize_ml_misc(obj: Any, ctx: SerializeContext) -> Any:
    """Route to type-specific serializer."""
    _load_for_object(obj)
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


def _deserialize_ml_misc(data: dict[str, Any], ctx: DeserializeContext) -> Any:
    """Route to type-specific deserializer."""
    type_name = data[TYPE_METADATA_KEY]
    value = data[VALUE_METADATA_KEY]

    match type_name:
        case "polars.DataFrame":
            return _reconstruct_polars_df(value)
        case "polars.Series":
            return _reconstruct_polars_series(value)
        case "jax.Array":
            return _reconstruct_jax_array(value, ctx)
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
    _load_framework("polars")
    if not _HAS_POLARS:
        raise PluginError("polars is not installed")
    return pl.DataFrame(value["data"])


def _reconstruct_polars_series(value: Any) -> Any:
    """Reconstruct a Polars Series."""
    _load_framework("polars")
    if not _HAS_POLARS:
        raise PluginError("polars is not installed")
    return pl.Series(value["name"], value["data"])


def _reconstruct_jax_array(value: Any, ctx: DeserializeContext) -> Any:
    """Reconstruct a JAX array."""
    _load_framework("jax")
    if not _HAS_JAX:
        raise PluginError("jax is not installed")
    import numpy as np

    from .._errors import DeserializationError

    dtype = np.dtype(value["dtype"])
    if dtype.fields is not None or not (jnp.issubdtype(dtype, jnp.number) or jnp.issubdtype(dtype, jnp.bool_)):
        raise DeserializationError("JAX reconstruction requires a numeric or boolean dtype")
    if not jax.config.x64_enabled and (
        (dtype.kind in "iuf" and dtype.itemsize > 4) or (dtype.kind == "c" and dtype.itemsize > 8)
    ):
        raise DeserializationError("JAX dtype requires enabling x64 in the application")
    shape = check_dense_allocation(value["data"], value.get("shape"), dtype.itemsize, ctx)
    np_arr = np.array(value["data"], dtype=dtype)
    if shape is not None:
        np_arr = np_arr.reshape(shape)
    return jnp.array(np_arr)


def _reconstruct_plotly_figure(value: Any) -> Any:
    """Reconstruct a Plotly Figure from dict."""
    _load_framework("plotly")
    if not _HAS_PLOTLY:
        raise PluginError("plotly is not installed")
    return go.Figure(value)
