"""Plugin for Pandas type serialization.

Handles DataFrame, Series, Timestamp, Timedelta, and Categorical.
This module imports pandas directly — if pandas is not installed,
the ImportError is caught by plugins/__init__.py and this plugin
is simply not registered.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._config import DataFrameOrient
from .._errors import PluginError
from .._protocols import DeserializeContext, SerializeContext
from .._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY


class PandasPlugin:
    """Handles serialization/deserialization of Pandas types."""

    @property
    def name(self) -> str:
        return "pandas"

    @property
    def priority(self) -> int:
        return 201

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, pd.DataFrame | pd.Series | pd.Timestamp | pd.Timedelta)

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        return _serialize_pandas(obj, ctx)

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        type_name = data.get(TYPE_METADATA_KEY, "")
        return isinstance(type_name, str) and type_name.startswith("pandas.")

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        return _deserialize_pandas(data)


def _serialize_pandas(obj: Any, ctx: SerializeContext) -> Any:
    """Serialize a Pandas object to JSON-safe representation."""
    if isinstance(obj, pd.DataFrame):
        return _serialize_dataframe(obj, ctx)
    if isinstance(obj, pd.Series):
        return _serialize_series(obj, ctx)
    if isinstance(obj, pd.Timestamp):
        return _serialize_timestamp(obj, ctx)
    if isinstance(obj, pd.Timedelta):
        return _serialize_timedelta(obj, ctx)
    raise PluginError(f"Unexpected Pandas type: {type(obj).__name__}")


def _serialize_dataframe(df: Any, ctx: SerializeContext) -> Any:
    """Serialize a DataFrame using the configured orientation."""
    orient = ctx.config.dataframe_orient
    value = _dataframe_to_dict(df, orient)
    if ctx.config.include_type_hints:
        return {
            TYPE_METADATA_KEY: "pandas.DataFrame",
            VALUE_METADATA_KEY: {"data": value, "orient": orient.value},
        }
    return value


def _dataframe_to_dict(df: Any, orient: DataFrameOrient) -> Any:
    """Convert DataFrame to dict using the specified orientation."""
    match orient:
        case DataFrameOrient.RECORDS:
            return df.to_dict(orient="records")
        case DataFrameOrient.SPLIT:
            return df.to_dict(orient="split")
        case DataFrameOrient.DICT:
            return df.to_dict(orient="dict")
        case DataFrameOrient.LIST:
            return df.to_dict(orient="list")
        case DataFrameOrient.VALUES:
            return df.values.tolist()


def _serialize_series(series: Any, ctx: SerializeContext) -> Any:
    """Serialize a Series with name and dtype metadata."""
    value = {
        "data": series.tolist(),
        "name": series.name,
        "dtype": str(series.dtype),
    }
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "pandas.Series", VALUE_METADATA_KEY: value}
    return series.tolist()


def _serialize_timestamp(ts: Any, ctx: SerializeContext) -> Any:
    """Serialize a Pandas Timestamp."""
    value = ts.isoformat()
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "pandas.Timestamp", VALUE_METADATA_KEY: value}
    return value


def _serialize_timedelta(td: Any, ctx: SerializeContext) -> Any:
    """Serialize a Pandas Timedelta as total seconds."""
    value = td.total_seconds()
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "pandas.Timedelta", VALUE_METADATA_KEY: value}
    return value


def _deserialize_pandas(data: dict[str, Any]) -> Any:
    """Reconstruct a Pandas object from serialized data."""
    type_name = data[TYPE_METADATA_KEY]
    value = data[VALUE_METADATA_KEY]

    match type_name:
        case "pandas.DataFrame":
            return _deserialize_dataframe(value)
        case "pandas.Series":
            return _deserialize_series(value)
        case "pandas.Timestamp":
            if not isinstance(value, str):
                raise PluginError(f"Expected string for Timestamp, got {type(value).__name__}")
            return pd.Timestamp(value)
        case "pandas.Timedelta":
            if not isinstance(value, int | float):
                raise PluginError(f"Expected number for Timedelta, got {type(value).__name__}")
            return pd.Timedelta(seconds=value)
        case _:
            raise PluginError(f"Unknown pandas type: {type_name}")


def _deserialize_dataframe(value: Any) -> Any:
    """Reconstruct a DataFrame from serialized data."""
    if not isinstance(value, dict):
        raise PluginError(f"Expected dict for DataFrame, got {type(value).__name__}")

    orient = value.get("orient", "records")
    raw = value.get("data", value)

    match orient:
        case "records":
            return pd.DataFrame.from_records(raw)
        case "split":
            return pd.DataFrame(**raw)
        case "dict" | "list":
            return pd.DataFrame.from_dict(raw, orient=orient)
        case "values":
            return pd.DataFrame(raw)
        case _:
            return pd.DataFrame(raw)


def _deserialize_series(value: Any) -> Any:
    """Reconstruct a Series from serialized data."""
    if not isinstance(value, dict):
        raise PluginError(f"Expected dict for Series, got {type(value).__name__}")
    return pd.Series(
        value["data"],
        name=value.get("name"),
        dtype=value.get("dtype"),
    )
