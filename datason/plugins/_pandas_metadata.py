"""Index and dtype metadata used by the optional Pandas plugin."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._errors import DeserializationError, SerializationError


def describe_dtype(dtype: Any) -> dict[str, Any]:
    """Keep categorical domains and nullable string storage explicit."""
    if isinstance(dtype, pd.CategoricalDtype):
        return {"dtype": "category", "categories": dtype.categories.tolist(), "ordered": dtype.ordered}
    if isinstance(dtype, pd.StringDtype):
        meta = {"dtype": "string", "storage": dtype.storage}
        if getattr(dtype, "na_value", pd.NA) is not pd.NA:
            meta["na_value"] = "nan"
        return meta
    return {"dtype": str(dtype)}


def restore_dtype(meta: dict[str, Any]) -> Any:
    if meta["dtype"] == "category":
        return pd.CategoricalDtype(categories=meta["categories"], ordered=meta["ordered"])
    if meta["dtype"] == "string":
        if meta.get("na_value") == "nan":
            return pd.StringDtype(storage=meta.get("storage", "python"), na_value=float("nan"))
        return pd.StringDtype(storage=meta.get("storage", "python"))
    return meta["dtype"]


def describe_index(index: Any) -> dict[str, Any]:
    """Describe common indexes without inferring their identity from values."""
    if isinstance(index, pd.RangeIndex):
        return {"kind": "range", "start": index.start, "stop": index.stop, "step": index.step, "name": index.name}
    if isinstance(index, pd.MultiIndex):
        return {
            "kind": "multi",
            "levels": [describe_index(level) for level in index.levels],
            "codes": [code.tolist() for code in index.codes],
            "names": list(index.names),
            "sortorder": getattr(index, "sortorder", None),
        }
    if isinstance(index, pd.PeriodIndex | pd.IntervalIndex):
        raise SerializationError("PeriodIndex and IntervalIndex require an explicit custom plugin")
    meta = {"kind": "index", "values": index.tolist(), "name": index.name, "dtype": describe_dtype(index.dtype)}
    if isinstance(index, pd.DatetimeIndex | pd.TimedeltaIndex):
        meta["kind"] = "datetime" if isinstance(index, pd.DatetimeIndex) else "timedelta"
        meta["freq"] = index.freqstr
    if isinstance(index, pd.CategoricalIndex):
        meta["kind"] = "category"
    return meta


def restore_index(meta: dict[str, Any], max_size: int) -> Any:
    """Validate range lengths before an index can drive frame allocation."""
    kind = meta["kind"]
    if kind == "range":
        parts = [meta[key] for key in ("start", "stop", "step")]
        if any(type(part) is not int for part in parts) or not parts[2]:
            raise DeserializationError("Invalid RangeIndex parameters")
        index = pd.RangeIndex(*parts, name=meta.get("name"))
        if len(index) > max_size:
            raise DeserializationError("RangeIndex exceeds container limit")
        return index
    if kind == "multi":
        return pd.MultiIndex(
            levels=[restore_index(level, max_size) for level in meta["levels"]],
            codes=meta["codes"],
            names=meta["names"],
            sortorder=meta.get("sortorder"),
        )
    constructors: dict[str, Any] = {
        "index": pd.Index,
        "datetime": pd.DatetimeIndex,
        "timedelta": pd.TimedeltaIndex,
        "category": pd.CategoricalIndex,
    }
    if kind not in constructors:
        raise DeserializationError("Unknown Pandas index kind")
    kwargs: dict[str, Any] = {"dtype": restore_dtype(meta["dtype"]), "name": meta.get("name")}
    if kind in ("datetime", "timedelta"):
        kwargs["freq"] = meta.get("freq")
    return constructors[kind](meta["values"], **kwargs)
