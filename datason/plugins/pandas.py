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
from .._errors import DeserializationError, PluginError
from .._protocols import DeserializeContext, SerializeContext
from .._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY
from ..security.redaction import should_redact_field
from ._pandas_metadata import describe_dtype, describe_index, restore_dtype, restore_index


class PandasPlugin:
    """Handles serialization/deserialization of Pandas types."""

    @property
    def name(self) -> str:
        return "pandas"

    @property
    def priority(self) -> int:
        return 201

    def can_handle(self, obj: Any) -> bool:
        return obj is pd.NA or obj is pd.NaT or isinstance(obj, pd.DataFrame | pd.Series | pd.Timestamp | pd.Timedelta)

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        return _serialize_pandas(obj, ctx)

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        type_name = data.get(TYPE_METADATA_KEY, "")
        return isinstance(type_name, str) and type_name.startswith("pandas.")

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        from .._deserialize import _deserialize_recursive

        try:
            value = _deserialize_recursive(data[VALUE_METADATA_KEY], ctx.child())
            return _deserialize_pandas({**data, VALUE_METADATA_KEY: value}, ctx)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise DeserializationError("Invalid Pandas payload") from exc


def _serialize_pandas(obj: Any, ctx: SerializeContext) -> Any:
    """Serialize a Pandas object to JSON-safe representation."""
    if obj is pd.NA or obj is pd.NaT:
        if ctx.config.include_type_hints:
            return {TYPE_METADATA_KEY: "pandas.NA" if obj is pd.NA else "pandas.NaT", VALUE_METADATA_KEY: None}
        return None
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
    positions = [i for i, label in enumerate(df.columns) if should_redact_field(str(label), ctx.config.redact_fields)]
    if positions:
        df = df.copy()
        for position in positions:
            df.isetitem(position, ["[REDACTED]"] * len(df))
    orient = ctx.config.dataframe_orient
    if ctx.config.include_type_hints and (
        df.empty
        or not df.columns.is_unique
        or not df.index.is_unique
        or any(not isinstance(c, str) for c in df.columns)
    ):
        orient = DataFrameOrient.SPLIT
    value = _dataframe_to_dict(df, orient)
    if ctx.config.include_type_hints:
        return {
            TYPE_METADATA_KEY: "pandas.DataFrame",
            VALUE_METADATA_KEY: {
                "data": value,
                "orient": orient.value,
                "index": describe_index(df.index),
                "columns": describe_index(df.columns),
                "dtypes": [describe_dtype(dtype) for dtype in df.dtypes],
            },
        }
    return value


def _dataframe_to_dict(df: Any, orient: DataFrameOrient) -> Any:
    """Convert DataFrame to dict using the specified orientation."""
    match orient:
        case DataFrameOrient.RECORDS:
            return df.to_dict(orient="records")
        case DataFrameOrient.SPLIT:
            return {"index": df.index.tolist(), "columns": df.columns.tolist(), "data": df.values.tolist()}
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
        "index": describe_index(series.index),
        "dtype_metadata": describe_dtype(series.dtype),
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
        return {TYPE_METADATA_KEY: "pandas.Timedelta", VALUE_METADATA_KEY: value, "nanoseconds": td.value}
    return value


def _deserialize_pandas(data: dict[str, Any], ctx: DeserializeContext) -> Any:
    """Reconstruct a Pandas object from serialized data."""
    type_name = data[TYPE_METADATA_KEY]
    value = data[VALUE_METADATA_KEY]

    match type_name:
        case "pandas.DataFrame":
            return _deserialize_dataframe(value, ctx)
        case "pandas.Series":
            return _deserialize_series(value, ctx)
        case "pandas.NA":
            return pd.NA
        case "pandas.NaT":
            return pd.NaT
        case "pandas.Timestamp":
            if not isinstance(value, str):
                raise PluginError(f"Expected string for Timestamp, got {type(value).__name__}")
            return pd.Timestamp(value)
        case "pandas.Timedelta":
            if "nanoseconds" in data:
                return pd.Timedelta(data["nanoseconds"], unit="ns")
            if not isinstance(value, int | float):
                raise PluginError(f"Expected number for Timedelta, got {type(value).__name__}")
            return pd.Timedelta(seconds=value)
        case _:
            raise PluginError(f"Unknown pandas type: {type_name}")


def _deserialize_dataframe(value: Any, ctx: DeserializeContext) -> Any:
    """Reconstruct a DataFrame from serialized data."""
    if not isinstance(value, dict):
        raise PluginError(f"Expected dict for DataFrame, got {type(value).__name__}")

    orient = value.get("orient", "records")
    raw = value.get("data", value)

    match orient:
        case "records":
            result = pd.DataFrame.from_records(raw)
        case "split":
            result = pd.DataFrame(**raw)
        case "dict" | "list":
            result = pd.DataFrame.from_dict(raw, orient="columns")
        case "values":
            result = pd.DataFrame(raw)
        case _:
            raise DeserializationError("Unknown DataFrame orientation")
    if "index" in value:
        result.index = restore_index(value["index"], ctx.config.max_size)
    if "columns" in value:
        result.columns = restore_index(value["columns"], ctx.config.max_size)
    if "dtypes" in value:
        if len(value["dtypes"]) != len(result.columns):
            raise DeserializationError("DataFrame dtype count does not match columns")
        for position, dtype in enumerate(value["dtypes"]):
            result.isetitem(position, result.iloc[:, position].astype(restore_dtype(dtype)))
    return result


def _deserialize_series(value: Any, ctx: DeserializeContext) -> Any:
    """Reconstruct a Series from serialized data."""
    if not isinstance(value, dict):
        raise PluginError(f"Expected dict for Series, got {type(value).__name__}")
    index = restore_index(value["index"], ctx.config.max_size) if "index" in value else None
    dtype = restore_dtype(value["dtype_metadata"]) if "dtype_metadata" in value else value.get("dtype")
    return pd.Series(
        value["data"],
        name=value.get("name"),
        dtype=dtype,
        index=index,
    )
