"""Preserve Pandas labels and dtypes through the public serialization API."""

import json

import pytest

pd = pytest.importorskip("pandas")

import datason
from datason._config import DataFrameOrient


@pytest.mark.parametrize("orient", list(DataFrameOrient))
def test_frame_index_columns_and_dtypes(orient):
    frame = pd.DataFrame(
        {
            "small": pd.array([1, None], dtype="Int32"),
            "text": pd.array(["a", None], dtype="string"),
            "day": pd.to_datetime(["2026-10-03", None]),
            "category": pd.Categorical(["low", "high"], categories=["low", "medium", "high"], ordered=True),
        }
    )
    frame.index = pd.Index(["r1", "r2"], name="row")
    frame.columns.name = "measure"
    restored = datason.loads(datason.dumps(frame, dataframe_orient=orient))
    pd.testing.assert_frame_equal(restored, frame)


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(2, 8, 2, name="id"),
        pd.date_range("2026-10-03", periods=3, tz="UTC", name="day"),
        pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("c", 3)], names=["group", "id"]),
        pd.CategoricalIndex(["a", "b", "a"], categories=["a", "b", "c"], name="group"),
    ],
)
def test_series_indexes(index):
    series = pd.Series(pd.array([1, None, 3], dtype="Int16"), index=index, name="value")
    pd.testing.assert_series_equal(datason.loads(datason.dumps(series)), series)


@pytest.mark.parametrize("orient", list(DataFrameOrient))
def test_duplicate_columns_and_rows_use_lossless_split(orient):
    frame = pd.DataFrame([[1, 2], [3, 4]], columns=["x", "x"], index=["r", "r"])
    pd.testing.assert_frame_equal(datason.loads(datason.dumps(frame, dataframe_orient=orient)), frame)


def test_non_string_columns_do_not_collide():
    frame = pd.DataFrame([[1, 2]], columns=[1, "1"])
    pd.testing.assert_frame_equal(datason.loads(datason.dumps(frame)), frame)


def test_empty_frame_keeps_columns_and_dtypes():
    frame = pd.DataFrame({"a": pd.Series(dtype="Int32"), "b": pd.Series(dtype="float32")})
    pd.testing.assert_frame_equal(datason.loads(datason.dumps(frame)), frame)


def test_timedelta_nanosecond_precision():
    original = pd.Timedelta(123456789, unit="ns")
    assert datason.loads(datason.dumps(original)) == original


def test_missing_scalar_identity():
    assert datason.loads(datason.dumps(pd.NA)) is pd.NA
    assert datason.loads(datason.dumps(pd.NaT)) is pd.NaT


@pytest.mark.parametrize("orient", list(DataFrameOrient))
def test_field_redaction_is_independent_of_orientation(orient):
    original = pd.DataFrame({"password": ["sensitive-value"], "visible": [1]})
    encoded = datason.dumps(original, dataframe_orient=orient, redact_fields=("password",))
    assert "sensitive-value" not in encoded
    assert "[REDACTED]" in encoded
    assert original.iloc[0, 0] == "sensitive-value"
    json.loads(encoded)
