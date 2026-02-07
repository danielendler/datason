"""Hypothesis property-based tests for JSON primitives and stdlib types.

Uses generative strategies to find edge cases in roundtrip serialization
that example-based tests may miss.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import uuid
from decimal import Decimal
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

import datason
from tests.conftest import (
    st_dates,
    st_datetimes,
    st_decimals_finite,
    st_mixed_serializable_data,
    st_paths,
    st_timedeltas,
    st_times,
)


class TestJsonPrimitiveRoundtrip:
    """Any valid JSON primitive roundtrips through dumps/loads."""

    @given(st.text())
    def test_string_roundtrip(self, s: str) -> None:
        result = datason.loads(datason.dumps(s))
        assert result == s

    @given(st.integers())
    def test_integer_roundtrip(self, n: int) -> None:
        result = datason.loads(datason.dumps(n))
        assert result == n

    @given(st.floats(allow_nan=False, allow_infinity=False))
    def test_float_roundtrip(self, f: float) -> None:
        result = datason.loads(datason.dumps(f))
        assert result == f

    @given(st.booleans())
    def test_boolean_roundtrip(self, b: bool) -> None:
        result = datason.loads(datason.dumps(b))
        assert result is b

    def test_none_roundtrip(self) -> None:
        result = datason.loads(datason.dumps(None))
        assert result is None


class TestJsonCompositeRoundtrip:
    """Composite JSON structures roundtrip."""

    @given(
        st.recursive(
            st.none()
            | st.booleans()
            | st.integers()
            | st.floats(allow_nan=False, allow_infinity=False)
            | st.text(max_size=50),
            lambda children: (
                st.lists(children, max_size=5) | st.dictionaries(st.text(max_size=20), children, max_size=5)
            ),
            max_leaves=30,
        )
    )
    def test_arbitrary_json_roundtrip(self, data: object) -> None:
        s = datason.dumps(data)
        # Output must be valid JSON
        json.loads(s)
        # Roundtrip must reconstruct the value
        result = datason.loads(s)
        assert result == data


class TestDatetimeRoundtrip:
    """All valid datetime objects roundtrip with type hints."""

    @given(st_datetimes())
    def test_datetime_roundtrip(self, dt_val: dt.datetime) -> None:
        s = datason.dumps(dt_val, include_type_hints=True)
        result = datason.loads(s)
        assert result == dt_val

    @given(st_dates())
    def test_date_roundtrip(self, d: dt.date) -> None:
        s = datason.dumps(d, include_type_hints=True)
        result = datason.loads(s)
        assert result == d

    @given(st_times())
    def test_time_roundtrip(self, t: dt.time) -> None:
        s = datason.dumps(t, include_type_hints=True)
        result = datason.loads(s)
        assert result == t

    @given(st_timedeltas())
    def test_timedelta_roundtrip(self, td: dt.timedelta) -> None:
        s = datason.dumps(td, include_type_hints=True)
        result = datason.loads(s)
        assert result == td


class TestUuidRoundtrip:
    """All UUIDs roundtrip with type hints."""

    @given(st.uuids())
    def test_uuid_roundtrip(self, u: uuid.UUID) -> None:
        s = datason.dumps(u, include_type_hints=True)
        result = datason.loads(s)
        assert result == u


class TestDecimalRoundtrip:
    """Finite decimals roundtrip with type hints."""

    @given(st_decimals_finite())
    def test_decimal_roundtrip(self, d: Decimal) -> None:
        s = datason.dumps(d, include_type_hints=True)
        result = datason.loads(s)
        assert result == d

    @given(st.complex_numbers(allow_nan=False, allow_infinity=False))
    def test_complex_roundtrip(self, c: complex) -> None:
        s = datason.dumps(c, include_type_hints=True)
        result = datason.loads(s)
        assert result.real == c.real
        assert result.imag == c.imag


class TestPathRoundtrip:
    """Path objects roundtrip with type hints."""

    @given(st_paths())
    def test_path_roundtrip(self, p: pathlib.PurePosixPath) -> None:
        path = pathlib.PurePosixPath(p)
        s = datason.dumps(path, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result, pathlib.Path)
        assert str(result) == str(path)


class TestJsonOutputValidity:
    """dumps() always produces valid JSON for any serializable data."""

    @given(st_mixed_serializable_data())
    def test_output_is_valid_json(self, data: dict[str, Any]) -> None:
        s = datason.dumps(data, include_type_hints=True)
        parsed = json.loads(s)
        assert isinstance(parsed, dict)

    @given(st.lists(st.integers(), max_size=50))
    def test_integer_list_valid_json(self, data: list[Any]) -> None:
        s = datason.dumps(data)
        parsed = json.loads(s)
        assert parsed == data

    @given(st.dictionaries(st.text(max_size=20), st.integers(), max_size=20))
    def test_string_int_dict_valid_json(self, data: dict[str, Any]) -> None:
        s = datason.dumps(data)
        parsed = json.loads(s)
        assert parsed == data
