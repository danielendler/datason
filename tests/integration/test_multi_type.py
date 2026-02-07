"""Integration tests for multi-type serialization.

Tests that multiple plugin types can coexist in a single data
structure and roundtrip correctly through the full pipeline.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import uuid
from decimal import Decimal
from typing import Any

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
import datason


class TestMixedStdlibTypes:
    """Dicts containing multiple stdlib plugin types."""

    def test_datetime_uuid_decimal_path_in_dict(self) -> None:
        data = {
            "created": dt.datetime(2024, 1, 15, 10, 30),  # noqa: DTZ001
            "date": dt.date(2024, 1, 15),
            "id": uuid.UUID("12345678-1234-5678-1234-567812345678"),
            "price": Decimal("19.99"),
            "config": pathlib.PurePosixPath("/etc/config.yml"),
        }
        s = datason.dumps(data, include_type_hints=True)
        result = datason.loads(s)

        assert isinstance(result["created"], dt.datetime)
        assert isinstance(result["date"], dt.date)
        assert isinstance(result["id"], uuid.UUID)
        assert isinstance(result["price"], Decimal)
        assert isinstance(result["config"], pathlib.Path)

    def test_list_of_mixed_stdlib_objects(self) -> None:
        data = [
            dt.datetime(2024, 1, 1),  # noqa: DTZ001
            uuid.uuid4(),
            Decimal("3.14"),
            dt.timedelta(hours=2),
        ]
        s = datason.dumps(data, include_type_hints=True)
        result = datason.loads(s)

        assert isinstance(result[0], dt.datetime)
        assert isinstance(result[1], uuid.UUID)
        assert isinstance(result[2], Decimal)
        assert isinstance(result[3], dt.timedelta)

    def test_deeply_nested_mixed_types(self) -> None:
        data = {
            "level1": {
                "level2": {
                    "timestamp": dt.datetime(2024, 6, 1),  # noqa: DTZ001
                    "items": [
                        {"id": uuid.uuid4(), "value": Decimal("1.5")},
                        {"id": uuid.uuid4(), "value": Decimal("2.5")},
                    ],
                }
            }
        }
        s = datason.dumps(data, include_type_hints=True)
        result = datason.loads(s)

        assert isinstance(result["level1"]["level2"]["timestamp"], dt.datetime)
        assert isinstance(result["level1"]["level2"]["items"][0]["id"], uuid.UUID)
        assert isinstance(result["level1"]["level2"]["items"][1]["value"], Decimal)


class TestMixedDataScienceTypes:
    """Dicts containing numpy + pandas together."""

    def test_numpy_array_with_pandas_dataframe(self) -> None:
        data = {
            "weights": np.array([0.1, 0.9, 0.5]),
            "metrics": pd.DataFrame({"accuracy": [0.95, 0.96], "loss": [0.1, 0.08]}),
        }
        s = datason.dumps(data, include_type_hints=True)
        result = datason.loads(s)

        assert isinstance(result["weights"], np.ndarray)
        assert isinstance(result["metrics"], pd.DataFrame)
        np.testing.assert_array_almost_equal(result["weights"], [0.1, 0.9, 0.5])

    def test_numpy_scalar_with_pandas_timestamp(self) -> None:
        data = {
            "count": np.int64(42),
            "timestamp": pd.Timestamp("2024-06-15 12:00:00"),
        }
        s = datason.dumps(data, include_type_hints=True)
        result = datason.loads(s)

        # np.int64 may deserialize as int (subclass fast-path)
        assert result["count"] == 42  # noqa: PLR2004
        # pd.Timestamp deserializes as datetime
        assert isinstance(result["timestamp"], dt.datetime)


class TestKitchenSink:
    """The 'everything at once' tests."""

    @pytest.fixture()
    def kitchen_sink_data(self) -> dict[str, Any]:
        return {
            "id": uuid.UUID("12345678-1234-5678-1234-567812345678"),
            "created": dt.datetime(2024, 1, 15, 10, 30),  # noqa: DTZ001
            "date": dt.date(2024, 1, 15),
            "duration": dt.timedelta(hours=2),
            "time": dt.time(14, 30),
            "price": Decimal("19.99"),
            "config_path": pathlib.PurePosixPath("/etc/config.yml"),
            "weights": np.array([0.1, 0.9, 0.5]),
            "count": np.int64(42),
            "tags": ["production", "v2"],
            "nested": {"x": 1, "y": None},
            "active": True,
            "score": 3.14,
        }

    def test_kitchen_sink_roundtrip(self, kitchen_sink_data: dict[str, Any]) -> None:
        s = datason.dumps(kitchen_sink_data, include_type_hints=True)
        result = datason.loads(s)

        assert isinstance(result["id"], uuid.UUID)
        assert isinstance(result["created"], dt.datetime)
        assert isinstance(result["date"], dt.date)
        assert isinstance(result["duration"], dt.timedelta)
        assert isinstance(result["time"], dt.time)
        assert isinstance(result["price"], Decimal)
        assert isinstance(result["config_path"], pathlib.Path)
        assert isinstance(result["weights"], np.ndarray)
        assert result["tags"] == ["production", "v2"]
        assert result["active"] is True
        assert result["score"] == pytest.approx(3.14)

    def test_kitchen_sink_json_valid(self, kitchen_sink_data: dict[str, Any]) -> None:
        s = datason.dumps(kitchen_sink_data, include_type_hints=True)
        parsed = json.loads(s)
        assert isinstance(parsed, dict)
        # All keys present
        assert set(parsed.keys()) == set(kitchen_sink_data.keys())

    def test_nested_containers_with_mixed_types(self) -> None:
        data = [
            {"timestamp": dt.datetime(2024, 1, 1), "value": np.float64(3.14)},  # noqa: DTZ001
            {"timestamp": dt.datetime(2024, 1, 2), "value": np.float64(2.71)},  # noqa: DTZ001
        ]
        s = datason.dumps(data, include_type_hints=True)
        result = datason.loads(s)

        assert len(result) == 2  # noqa: PLR2004
        for item in result:
            assert isinstance(item["timestamp"], dt.datetime)
