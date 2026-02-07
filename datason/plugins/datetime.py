"""Plugin for datetime, date, time, and timedelta serialization."""

from __future__ import annotations

import datetime as dt
from typing import Any

from .._config import DateFormat
from .._errors import PluginError
from .._protocols import DeserializeContext, SerializeContext
from .._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY

_HANDLED_TYPES = (dt.datetime, dt.date, dt.time, dt.timedelta)

_TYPE_NAMES = {
    dt.datetime: "datetime",
    dt.date: "date",
    dt.time: "time",
    dt.timedelta: "timedelta",
}


class DatetimePlugin:
    """Handles serialization/deserialization of datetime family types."""

    @property
    def name(self) -> str:
        return "datetime"

    @property
    def priority(self) -> int:
        return 100

    def can_handle(self, obj: Any) -> bool:
        # Use exact type lookup to avoid claiming subclasses like pd.Timestamp
        return type(obj) in _TYPE_NAMES

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        type_name = _TYPE_NAMES.get(type(obj))
        if type_name is None:
            raise PluginError(f"Unexpected type: {type(obj).__name__}")

        value = _serialize_value(obj, ctx.config.date_format)

        if ctx.config.include_type_hints:
            return {TYPE_METADATA_KEY: type_name, VALUE_METADATA_KEY: value}
        return value

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        return data.get(TYPE_METADATA_KEY) in _TYPE_NAMES.values()

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        type_name = data[TYPE_METADATA_KEY]
        value = data[VALUE_METADATA_KEY]
        return _deserialize_value(type_name, value)


def _serialize_value(obj: Any, fmt: DateFormat) -> str | float:
    """Serialize a datetime-family object according to the format config."""
    if isinstance(obj, dt.timedelta):
        return obj.total_seconds()

    if isinstance(obj, dt.time):
        return obj.isoformat()

    # datetime and date
    match fmt:
        case DateFormat.ISO:
            return obj.isoformat()
        case DateFormat.UNIX:
            if isinstance(obj, dt.datetime):
                return obj.timestamp()
            return obj.isoformat()
        case DateFormat.UNIX_MS:
            if isinstance(obj, dt.datetime):
                return obj.timestamp() * 1000
            return obj.isoformat()
        case DateFormat.STRING:
            return str(obj)
        case _:
            return obj.isoformat()


def _deserialize_value(type_name: str, value: Any) -> Any:
    """Reconstruct a datetime-family object from its serialized value."""
    match type_name:
        case "datetime":
            if isinstance(value, str):
                return dt.datetime.fromisoformat(value)
            if isinstance(value, int | float):
                # Detect millisecond timestamps (> year 2100 in seconds)
                ts = value / 1000 if abs(value) > 4_102_444_800 else value
                return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
            raise PluginError(f"Cannot deserialize datetime from {type(value).__name__}")
        case "date":
            if isinstance(value, str):
                return dt.date.fromisoformat(value)
            raise PluginError(f"Cannot deserialize date from {type(value).__name__}")
        case "time":
            if isinstance(value, str):
                return dt.time.fromisoformat(value)
            raise PluginError(f"Cannot deserialize time from {type(value).__name__}")
        case "timedelta":
            if isinstance(value, int | float):
                return dt.timedelta(seconds=value)
            raise PluginError(f"Cannot deserialize timedelta from {type(value).__name__}")
        case _:
            raise PluginError(f"Unknown datetime type: {type_name}")
