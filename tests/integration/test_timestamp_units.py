"""Timestamp units must not depend on date magnitude or machine timezone."""

import datetime as dt
import json
import os
import time

import pytest

import datason
from datason._config import DateFormat


@pytest.mark.parametrize("fmt", [DateFormat.UNIX, DateFormat.UNIX_MS])
@pytest.mark.parametrize(
    "value",
    [
        dt.datetime(1970, 1, 1, 0, 0, 1, 123456),
        dt.datetime(1960, 1, 1, tzinfo=dt.timezone.utc),
        dt.datetime(2200, 1, 1, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))),
    ],
)
def test_numeric_dates_restore_unit_offset_and_precision(fmt, value):
    restored = datason.loads(datason.dumps(value, date_format=fmt))
    assert restored == value
    assert restored.tzinfo == value.tzinfo


def test_explicit_millisecond_unit_without_iso_metadata():
    payload = {"__datason_type__": "datetime", "__datason_value__": 1000, "timestamp_unit": "milliseconds"}
    assert datason.loads(json.dumps(payload)) == dt.datetime(1970, 1, 1, 0, 0, 1, tzinfo=dt.timezone.utc)


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="Platform has no tzset")
def test_naive_numeric_encoding_is_independent_of_local_timezone(monkeypatch):
    value = dt.datetime(1970, 1, 1, 0, 0, 1)
    old_tz = os.environ.get("TZ")
    try:
        monkeypatch.setenv("TZ", "UTC+8")
        time.tzset()
        encoded = datason.dumps(value, date_format=DateFormat.UNIX)
        assert json.loads(encoded)["__datason_value__"] == 1.0
        monkeypatch.setenv("TZ", "UTC-5")
        time.tzset()
        assert datason.loads(encoded) == value
    finally:
        if old_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", old_tz)
        time.tzset()
