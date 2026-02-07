"""Tests for the Pandas plugin."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
import datason
from datason._config import DataFrameOrient, SerializationConfig
from datason._errors import PluginError
from datason._protocols import DeserializeContext, SerializeContext
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY
from datason.plugins.pandas import PandasPlugin


@pytest.fixture()
def plugin() -> PandasPlugin:
    return PandasPlugin()


@pytest.fixture()
def ser_ctx() -> SerializeContext:
    return SerializeContext(config=SerializationConfig())


@pytest.fixture()
def deser_ctx() -> DeserializeContext:
    return DeserializeContext(config=SerializationConfig())


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})


class TestPandasPluginProperties:
    def test_name(self, plugin: PandasPlugin) -> None:
        assert plugin.name == "pandas"

    def test_priority(self, plugin: PandasPlugin) -> None:
        assert plugin.priority == 201


class TestCanHandle:
    def test_dataframe(self, plugin: PandasPlugin, sample_df: pd.DataFrame) -> None:
        assert plugin.can_handle(sample_df)

    def test_series(self, plugin: PandasPlugin) -> None:
        assert plugin.can_handle(pd.Series([1, 2, 3], name="vals"))

    def test_timestamp(self, plugin: PandasPlugin) -> None:
        assert plugin.can_handle(pd.Timestamp("2024-01-15"))

    def test_timedelta(self, plugin: PandasPlugin) -> None:
        assert plugin.can_handle(pd.Timedelta(hours=2))

    def test_rejects_dict(self, plugin: PandasPlugin) -> None:
        assert not plugin.can_handle({"a": [1, 2]})

    def test_rejects_list(self, plugin: PandasPlugin) -> None:
        assert not plugin.can_handle([1, 2, 3])

    def test_rejects_numpy_array(self, plugin: PandasPlugin) -> None:
        assert not plugin.can_handle(np.array([1, 2, 3]))


class TestSerializeDataFrame:
    def test_records_orient(self, plugin: PandasPlugin, sample_df: pd.DataFrame) -> None:
        ctx = SerializeContext(config=SerializationConfig(dataframe_orient=DataFrameOrient.RECORDS))
        result = plugin.serialize(sample_df, ctx)
        assert result[TYPE_METADATA_KEY] == "pandas.DataFrame"
        assert result[VALUE_METADATA_KEY]["orient"] == "records"
        data = result[VALUE_METADATA_KEY]["data"]
        assert len(data) == 3
        assert data[0] == {"a": 1, "b": 4.0, "c": "x"}

    def test_split_orient(self, plugin: PandasPlugin, sample_df: pd.DataFrame) -> None:
        ctx = SerializeContext(config=SerializationConfig(dataframe_orient=DataFrameOrient.SPLIT))
        result = plugin.serialize(sample_df, ctx)
        data = result[VALUE_METADATA_KEY]["data"]
        assert "columns" in data
        assert "data" in data

    def test_dict_orient(self, plugin: PandasPlugin, sample_df: pd.DataFrame) -> None:
        ctx = SerializeContext(config=SerializationConfig(dataframe_orient=DataFrameOrient.DICT))
        result = plugin.serialize(sample_df, ctx)
        data = result[VALUE_METADATA_KEY]["data"]
        assert "a" in data
        assert data["a"] == {0: 1, 1: 2, 2: 3}

    def test_list_orient(self, plugin: PandasPlugin, sample_df: pd.DataFrame) -> None:
        ctx = SerializeContext(config=SerializationConfig(dataframe_orient=DataFrameOrient.LIST))
        result = plugin.serialize(sample_df, ctx)
        data = result[VALUE_METADATA_KEY]["data"]
        assert data["a"] == [1, 2, 3]

    def test_values_orient(self, plugin: PandasPlugin, sample_df: pd.DataFrame) -> None:
        ctx = SerializeContext(config=SerializationConfig(dataframe_orient=DataFrameOrient.VALUES))
        result = plugin.serialize(sample_df, ctx)
        data = result[VALUE_METADATA_KEY]["data"]
        assert data == [[1, 4.0, "x"], [2, 5.0, "y"], [3, 6.0, "z"]]

    def test_without_hints(self, plugin: PandasPlugin, sample_df: pd.DataFrame) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        result = plugin.serialize(sample_df, ctx)
        assert isinstance(result, list)  # Default records orient returns list of dicts
        assert len(result) == 3


class TestSerializeSeries:
    def test_with_hints(self, plugin: PandasPlugin, ser_ctx: SerializeContext) -> None:
        series = pd.Series([10, 20, 30], name="scores")
        result = plugin.serialize(series, ser_ctx)
        assert result[TYPE_METADATA_KEY] == "pandas.Series"
        value = result[VALUE_METADATA_KEY]
        assert value["data"] == [10, 20, 30]
        assert value["name"] == "scores"
        assert "dtype" in value

    def test_without_hints(self, plugin: PandasPlugin) -> None:
        series = pd.Series([10, 20, 30])
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        result = plugin.serialize(series, ctx)
        assert result == [10, 20, 30]


class TestSerializeTimestamp:
    def test_with_hints(self, plugin: PandasPlugin, ser_ctx: SerializeContext) -> None:
        ts = pd.Timestamp("2024-01-15 10:30:00")
        result = plugin.serialize(ts, ser_ctx)
        assert result[TYPE_METADATA_KEY] == "pandas.Timestamp"
        assert isinstance(result[VALUE_METADATA_KEY], str)

    def test_without_hints(self, plugin: PandasPlugin) -> None:
        ts = pd.Timestamp("2024-01-15")
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        result = plugin.serialize(ts, ctx)
        assert isinstance(result, str)


class TestSerializeTimedelta:
    def test_with_hints(self, plugin: PandasPlugin, ser_ctx: SerializeContext) -> None:
        td = pd.Timedelta(hours=2, minutes=30)
        result = plugin.serialize(td, ser_ctx)
        assert result[TYPE_METADATA_KEY] == "pandas.Timedelta"
        assert result[VALUE_METADATA_KEY] == 9000.0

    def test_without_hints(self, plugin: PandasPlugin) -> None:
        td = pd.Timedelta(seconds=60)
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        result = plugin.serialize(td, ctx)
        assert result == 60.0


class TestCanDeserialize:
    def test_dataframe(self, plugin: PandasPlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "pandas.DataFrame"})

    def test_series(self, plugin: PandasPlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "pandas.Series"})

    def test_timestamp(self, plugin: PandasPlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "pandas.Timestamp"})

    def test_timedelta(self, plugin: PandasPlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "pandas.Timedelta"})

    def test_rejects_other(self, plugin: PandasPlugin) -> None:
        assert not plugin.can_deserialize({TYPE_METADATA_KEY: "datetime"})

    def test_rejects_missing_key(self, plugin: PandasPlugin) -> None:
        assert not plugin.can_deserialize({"data": [1, 2, 3]})


class TestDeserialize:
    def test_dataframe_records(self, plugin: PandasPlugin, deser_ctx: DeserializeContext) -> None:
        data = {
            TYPE_METADATA_KEY: "pandas.DataFrame",
            VALUE_METADATA_KEY: {
                "data": [{"a": 1, "b": 4}, {"a": 2, "b": 5}],
                "orient": "records",
            },
        }
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 2

    def test_dataframe_split(self, plugin: PandasPlugin, deser_ctx: DeserializeContext) -> None:
        data = {
            TYPE_METADATA_KEY: "pandas.DataFrame",
            VALUE_METADATA_KEY: {
                "data": {"columns": ["x", "y"], "data": [[1, 2], [3, 4]], "index": [0, 1]},
                "orient": "split",
            },
        }
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["x", "y"]

    def test_series(self, plugin: PandasPlugin, deser_ctx: DeserializeContext) -> None:
        data = {
            TYPE_METADATA_KEY: "pandas.Series",
            VALUE_METADATA_KEY: {"data": [10, 20, 30], "name": "scores", "dtype": "int64"},
        }
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, pd.Series)
        assert result.name == "scores"
        assert list(result) == [10, 20, 30]

    def test_timestamp(self, plugin: PandasPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "pandas.Timestamp", VALUE_METADATA_KEY: "2024-01-15T10:30:00"}
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, pd.Timestamp)
        assert result.year == 2024

    def test_timedelta(self, plugin: PandasPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "pandas.Timedelta", VALUE_METADATA_KEY: 9000.0}
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, pd.Timedelta)
        assert result.total_seconds() == 9000.0

    def test_timestamp_bad_type_raises(self, plugin: PandasPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "pandas.Timestamp", VALUE_METADATA_KEY: 12345}
        with pytest.raises(PluginError, match="Expected string for Timestamp"):
            plugin.deserialize(data, deser_ctx)

    def test_timedelta_bad_type_raises(self, plugin: PandasPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "pandas.Timedelta", VALUE_METADATA_KEY: "2 hours"}
        with pytest.raises(PluginError, match="Expected number for Timedelta"):
            plugin.deserialize(data, deser_ctx)

    def test_dataframe_bad_value_raises(self, plugin: PandasPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "pandas.DataFrame", VALUE_METADATA_KEY: "not a dict"}
        with pytest.raises(PluginError, match="Expected dict for DataFrame"):
            plugin.deserialize(data, deser_ctx)

    def test_series_bad_value_raises(self, plugin: PandasPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "pandas.Series", VALUE_METADATA_KEY: [1, 2, 3]}
        with pytest.raises(PluginError, match="Expected dict for Series"):
            plugin.deserialize(data, deser_ctx)

    def test_unknown_type_raises(self, plugin: PandasPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "pandas.Unknown", VALUE_METADATA_KEY: "value"}
        with pytest.raises(PluginError, match="Unknown pandas type"):
            plugin.deserialize(data, deser_ctx)


class TestRoundTrip:
    def test_dataframe_roundtrip(self, sample_df: pd.DataFrame) -> None:
        serialized = datason.dumps(sample_df)
        result = datason.loads(serialized)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_series_roundtrip(self) -> None:
        series = pd.Series([1.1, 2.2, 3.3], name="values")
        serialized = datason.dumps(series)
        result = datason.loads(serialized)
        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, series)

    def test_timestamp_roundtrip(self) -> None:
        ts = pd.Timestamp("2024-01-15 10:30:00")
        serialized = datason.dumps(ts)
        result = datason.loads(serialized)
        assert isinstance(result, pd.Timestamp)
        assert result == ts

    def test_timedelta_roundtrip(self) -> None:
        td = pd.Timedelta(hours=5, minutes=30)
        serialized = datason.dumps(td)
        result = datason.loads(serialized)
        assert isinstance(result, pd.Timedelta)
        assert result.total_seconds() == td.total_seconds()

    def test_dataframe_in_dict(self, sample_df: pd.DataFrame) -> None:
        data = {"table": sample_df, "name": "test_data"}
        serialized = datason.dumps(data)
        result = datason.loads(serialized)
        assert result["name"] == "test_data"
        pd.testing.assert_frame_equal(result["table"], sample_df)

    def test_json_valid(self, sample_df: pd.DataFrame) -> None:
        serialized = datason.dumps(sample_df)
        parsed = json.loads(serialized)
        assert isinstance(parsed, dict)

    def test_dataframe_with_split_orient(self) -> None:
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with datason.config(dataframe_orient=DataFrameOrient.SPLIT):
            serialized = datason.dumps(df)
        result = datason.loads(serialized)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)

    def test_mixed_pandas_types(self) -> None:
        data = {
            "timestamp": pd.Timestamp("2024-06-15"),
            "duration": pd.Timedelta(minutes=90),
            "series": pd.Series([1, 2, 3], name="vals"),
        }
        serialized = datason.dumps(data)
        result = datason.loads(serialized)
        assert isinstance(result["timestamp"], pd.Timestamp)
        assert isinstance(result["duration"], pd.Timedelta)
        assert isinstance(result["series"], pd.Series)
