"""Hypothesis property-based tests for numpy and pandas types.

Generates arrays with varying shapes, dtypes, and values to
find edge cases in serialization/deserialization roundtrips.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
from hypothesis import given
from hypothesis import strategies as st

import datason
from datason._config import DataFrameOrient
from tests.conftest import st_dataframes, st_numpy_arrays, st_pandas_timestamps, st_series


class TestNumpyArrayRoundtrip:
    """Numpy arrays of various shapes and dtypes roundtrip."""

    @given(st_numpy_arrays())
    def test_ndarray_roundtrip(self, arr: np.ndarray) -> None:
        s = datason.dumps(arr, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, arr)

    @given(st_numpy_arrays(dtype=np.float32))
    def test_float32_roundtrip(self, arr: np.ndarray) -> None:
        s = datason.dumps(arr, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, arr, decimal=5)

    @given(st_numpy_arrays(dtype=np.int64))
    def test_int64_array_roundtrip(self, arr: np.ndarray) -> None:
        s = datason.dumps(arr, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    @given(st_numpy_arrays(ndim=1))
    def test_1d_roundtrip(self, arr: np.ndarray) -> None:
        s = datason.dumps(arr, include_type_hints=True)
        result = datason.loads(s)
        assert result.ndim == 1
        assert result.shape == arr.shape

    @given(st_numpy_arrays(ndim=2))
    def test_2d_roundtrip(self, arr: np.ndarray) -> None:
        s = datason.dumps(arr, include_type_hints=True)
        result = datason.loads(s)
        assert result.ndim == 2  # noqa: PLR2004
        assert result.shape == arr.shape

    @given(st_numpy_arrays(ndim=3))
    def test_3d_roundtrip(self, arr: np.ndarray) -> None:
        s = datason.dumps(arr, include_type_hints=True)
        result = datason.loads(s)
        assert result.ndim == 3  # noqa: PLR2004
        assert result.shape == arr.shape


class TestNumpyScalarRoundtrip:
    """Numpy scalar types roundtrip."""

    @given(st.integers(min_value=np.iinfo(np.int64).min, max_value=np.iinfo(np.int64).max))
    def test_int64_scalar(self, n: int) -> None:
        scalar = np.int64(n)
        s = datason.dumps(scalar, include_type_hints=True)
        result = datason.loads(s)
        assert result == n

    @given(st.floats(min_value=-1e15, max_value=1e15, allow_nan=False, allow_infinity=False))
    def test_float64_scalar(self, f: float) -> None:
        scalar = np.float64(f)
        s = datason.dumps(scalar, include_type_hints=True)
        result = datason.loads(s)
        assert abs(result - f) < 1e-10 or result == f


class TestNumpyEdgeCases:
    """Edge case arrays (empty, single element, various dtypes)."""

    @given(st.integers(min_value=0, max_value=5))
    def test_small_arrays(self, size: int) -> None:
        arr = np.arange(size, dtype=np.float64)
        s = datason.dumps(arr, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    @given(st.sampled_from([np.int32, np.int64, np.float32, np.float64]))
    def test_various_dtypes(self, dtype: type) -> None:
        arr = np.array([1, 2, 3], dtype=dtype)
        s = datason.dumps(arr, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, arr)


class TestNumpyJsonValidity:
    """Numpy serialization always produces valid JSON."""

    @given(st_numpy_arrays())
    def test_output_is_valid_json(self, arr: np.ndarray) -> None:
        s = datason.dumps(arr, include_type_hints=True)
        parsed = json.loads(s)
        assert isinstance(parsed, dict)


class TestPandasDataFrameRoundtrip:
    """DataFrames with random columns and rows roundtrip."""

    @given(st_dataframes())
    def test_dataframe_roundtrip(self, df: pd.DataFrame) -> None:
        s = datason.dumps(df, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(df.columns)
        assert len(result) == len(df)


class TestPandasSeriesRoundtrip:
    """Series with random values roundtrip."""

    @given(st_series())
    def test_series_roundtrip(self, series: pd.Series) -> None:
        s = datason.dumps(series, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)


class TestPandasTimestampRoundtrip:
    """Pandas Timestamps roundtrip."""

    @given(st_pandas_timestamps())
    def test_timestamp_roundtrip(self, ts: pd.Timestamp) -> None:
        s = datason.dumps(ts, include_type_hints=True)
        result = datason.loads(s)
        # Timestamp deserializes as datetime; compare values
        assert abs(pd.Timestamp(result) - ts) < pd.Timedelta(seconds=1)


class TestDataFrameOrientProperty:
    """All orient modes produce valid JSON."""

    @given(st_dataframes(), st.sampled_from(list(DataFrameOrient)))
    def test_all_orients_produce_valid_json(self, df: pd.DataFrame, orient: DataFrameOrient) -> None:
        s = datason.dumps(df, include_type_hints=True, dataframe_orient=orient)
        parsed = json.loads(s)
        assert isinstance(parsed, dict)
