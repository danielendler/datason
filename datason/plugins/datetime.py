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
            meta: dict[str, Any] = {TYPE_METADATA_KEY: type_name, VALUE_METADATA_KEY: value}
            # Track naive/aware for numeric formats so round-trip is lossless
            if isinstance(obj, dt.datetime) and isinstance(value, int | float):
                meta["tz_aware"] = obj.tzinfo is not None
                meta["timestamp_unit"] = "milliseconds" if ctx.config.date_format == DateFormat.UNIX_MS else "seconds"
                meta["datetime_iso"] = obj.isoformat()
                meta["fold"] = obj.fold
            return meta
        return value

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        return data.get(TYPE_METADATA_KEY) in _TYPE_NAMES.values()

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        type_name = data[TYPE_METADATA_KEY]
        value = data[VALUE_METADATA_KEY]
        if type_name == "datetime" and "datetime_iso" in data:
            return dt.datetime.fromisoformat(data["datetime_iso"]).replace(fold=data.get("fold", 0))
        tz_aware = data.get("tz_aware")  # None = old format (assume aware, backward compat)
        return _deserialize_value(type_name, value, tz_aware=tz_aware, unit=data.get("timestamp_unit"))


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
                return _timestamp(obj)
            return obj.isoformat()
        case DateFormat.UNIX_MS:
            if isinstance(obj, dt.datetime):
                return _timestamp(obj) * 1000
            return obj.isoformat()
        case DateFormat.STRING:
            return str(obj)
        case _:
            return obj.isoformat()


def _timestamp(obj: dt.datetime) -> float:
    """Interpret naive timestamps as UTC rather than the machine's local zone."""
    return obj.replace(tzinfo=dt.timezone.utc).timestamp() if obj.tzinfo is None else obj.timestamp()


def _deserialize_value(type_name: str, value: Any, tz_aware: bool | None = None, unit: str | None = None) -> Any:
    """Reconstruct a datetime-family object from its serialized value."""
    match type_name:
        case "datetime":
            if isinstance(value, str):
                return dt.datetime.fromisoformat(value)
            if isinstance(value, int | float):
                # Detect millisecond timestamps (> year 2100 in seconds)
                if unit not in (None, "seconds", "milliseconds"):
                    raise PluginError("Unknown datetime timestamp unit")
                is_ms = unit == "milliseconds" if unit is not None else abs(value) > 4_102_444_800
                ts = value / 1000 if is_ms else value
                # tz_aware=False → naive (local time); None or True → UTC (backward compat)
                if tz_aware is False:
                    return dt.datetime.fromtimestamp(ts)
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
