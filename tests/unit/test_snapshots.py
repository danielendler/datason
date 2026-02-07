"""Snapshot tests for datason serialization output.

Uses syrupy to lock the serialization wire format. If the output
format changes, these tests will fail and require explicit snapshot
updates (--snapshot-update), making format changes deliberate.
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

import datason
from datason import DataFrameOrient, DateFormat, NanHandling


class TestPrimitiveSnapshots:
    """Lock JSON output for primitive types."""

    def test_simple_dict(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"name": "Alice", "age": 30}, sort_keys=True) == snapshot

    def test_nested_dict(self, snapshot: SnapshotAssertion) -> None:
        data = {"a": {"b": {"c": 1}}, "d": [1, 2, 3]}
        assert datason.dumps(data, sort_keys=True) == snapshot

    def test_empty_structures(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"dict": {}, "list": [], "null": None}, sort_keys=True) == snapshot

    def test_tuple_and_set(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"tuple": (1, 2, 3), "set": {3, 1, 2}}, sort_keys=True) == snapshot

    def test_unicode_strings(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"emoji": "hello", "cjk": "test"}, sort_keys=True) == snapshot


class TestDatetimeSnapshots:
    """Lock output format for datetime types across DateFormat modes."""

    @pytest.fixture()
    def dt_value(self) -> dt.datetime:
        return dt.datetime(2024, 6, 15, 10, 30, 0, tzinfo=dt.timezone.utc)

    def test_iso_format(self, snapshot: SnapshotAssertion, dt_value: dt.datetime) -> None:
        assert datason.dumps({"ts": dt_value}, date_format=DateFormat.ISO) == snapshot

    def test_unix_format(self, snapshot: SnapshotAssertion, dt_value: dt.datetime) -> None:
        assert datason.dumps({"ts": dt_value}, date_format=DateFormat.UNIX) == snapshot

    def test_unix_ms_format(self, snapshot: SnapshotAssertion, dt_value: dt.datetime) -> None:
        assert datason.dumps({"ts": dt_value}, date_format=DateFormat.UNIX_MS) == snapshot

    def test_string_format(self, snapshot: SnapshotAssertion, dt_value: dt.datetime) -> None:
        assert datason.dumps({"ts": dt_value}, date_format=DateFormat.STRING) == snapshot

    def test_date_only(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"d": dt.date(2024, 6, 15)}) == snapshot

    def test_time_only(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"t": dt.time(10, 30, 0)}) == snapshot

    def test_timedelta(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"td": dt.timedelta(hours=2, minutes=30)}) == snapshot


class TestStdlibTypeSnapshots:
    """Lock output for UUID, Decimal, complex, Path."""

    def test_uuid(self, snapshot: SnapshotAssertion) -> None:
        u = uuid.UUID("12345678-1234-5678-1234-567812345678")
        assert datason.dumps({"id": u}) == snapshot

    def test_decimal(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"price": Decimal("19.99")}) == snapshot

    def test_complex(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"z": complex(3.5, 2.1)}) == snapshot

    def test_path(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"p": PurePosixPath("/data/models/v2")}) == snapshot


class TestNumpySnapshots:
    """Lock output for NumPy types."""

    def test_1d_array(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps(np.array([1.0, 2.0, 3.0])) == snapshot

    def test_2d_array(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps(np.array([[1, 2], [3, 4]])) == snapshot

    def test_int_scalar(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"v": np.int64(42)}) == snapshot

    def test_float_scalar(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"v": np.float64(3.14)}) == snapshot


class TestPandasSnapshots:
    """Lock output for Pandas types."""

    def test_dataframe_records(self, snapshot: SnapshotAssertion) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        assert datason.dumps(df) == snapshot

    def test_dataframe_split(self, snapshot: SnapshotAssertion) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        assert datason.dumps(df, dataframe_orient=DataFrameOrient.SPLIT) == snapshot

    def test_series(self, snapshot: SnapshotAssertion) -> None:
        s = pd.Series([10, 20, 30], name="scores")
        assert datason.dumps(s) == snapshot


class TestNanHandlingSnapshots:
    """Lock NaN/Inf handling across modes."""

    def test_nan_null(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"v": float("nan")}, nan_handling=NanHandling.NULL) == snapshot

    def test_nan_string(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"v": float("nan")}, nan_handling=NanHandling.STRING) == snapshot

    def test_inf_null(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"v": float("inf")}, nan_handling=NanHandling.NULL) == snapshot

    def test_inf_string(self, snapshot: SnapshotAssertion) -> None:
        assert datason.dumps({"v": float("inf")}, nan_handling=NanHandling.STRING) == snapshot


class TestConfigPresetSnapshots:
    """Lock output differences between config presets."""

    @pytest.fixture()
    def data(self) -> dict[str, object]:
        return {"ts": dt.datetime(2024, 1, 15, tzinfo=dt.timezone.utc), "score": 95.5}

    def test_default_config(self, snapshot: SnapshotAssertion, data: dict[str, object]) -> None:
        assert datason.dumps(data, sort_keys=True) == snapshot

    def test_api_config(self, snapshot: SnapshotAssertion, data: dict[str, object]) -> None:
        from datason import api_config

        with datason.config(**api_config().__dict__):
            assert datason.dumps(data) == snapshot

    def test_ml_config(self, snapshot: SnapshotAssertion, data: dict[str, object]) -> None:
        from datason import ml_config

        with datason.config(**ml_config().__dict__):
            assert datason.dumps(data) == snapshot
