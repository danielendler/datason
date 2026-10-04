"""Read persisted strings produced by the actual first-alpha release source."""

import datetime as dt
import json
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

import datason
from datason.security.integrity import verify_integrity

_FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures" / "v2.0.0a1.json").read_text(encoding="utf-8"))
_PAYLOADS = _FIXTURE["payloads"]


def test_published_alpha_stdlib_payload():
    actual = datason.loads(_PAYLOADS["stdlib"])
    expected = {
        "datetime": dt.datetime(2026, 2, 7, 12, 34, 56, 123456, tzinfo=dt.timezone.utc),
        "date": dt.date(2026, 2, 7),
        "time": dt.time(12, 34, 56, 123456),
        "timedelta": dt.timedelta(seconds=3, microseconds=5),
        "uuid": UUID("12345678-1234-5678-1234-567812345678"),
        "decimal": Decimal("123.4500"),
        "path": Path("artifacts/checkpoint.json"),
    }
    assert actual == expected
    assert str(actual["decimal"]) == "123.4500"
    assert {name: type(value) for name, value in actual.items()} == {
        name: type(value) for name, value in expected.items()
    }


def test_published_alpha_untagged_collections_remain_lists():
    assert datason.loads(_PAYLOADS["collections"]) == {"tuple": [1, "two"], "set": [3], "frozenset": [4]}


@pytest.mark.parametrize("name", ["unix", "unix_ms"])
def test_published_alpha_numeric_datetime(name):
    assert datason.loads(_PAYLOADS[name]) == dt.datetime(2026, 2, 7, tzinfo=dt.timezone.utc)


@pytest.mark.parametrize("name", ["hash", "hmac"])
def test_published_alpha_integrity_envelope(name):
    key = "public-test-fixture-key" if name == "hmac" else None
    valid, wire = verify_integrity(_PAYLOADS[name], key=key)
    assert valid
    assert json.loads(wire) == {"message": "café", "count": 7}


def test_published_alpha_numpy_payload():
    np = pytest.importorskip("numpy")
    actual = datason.loads(_PAYLOADS["numpy"])
    for name, dtype, value in (
        ("integer", np.int64, 7),
        ("floating", np.float64, 1.25),
        ("boolean", np.bool_, True),
        ("complex", np.complex128, 2 + 3j),
    ):
        assert type(actual[name]) is dtype
        assert actual[name] == value
    assert actual["array"].dtype == np.dtype("int32")
    np.testing.assert_array_equal(actual["array"], [[1, 2], [3, 4]])
    assert actual["empty_array"].dtype == np.dtype("float32")
    assert actual["empty_array"].shape == (2, 0, 4)


def test_published_alpha_pandas_payload():
    pd = pytest.importorskip("pandas")
    expected = pd.DataFrame(
        {"count": pd.Series([1, 2], dtype="int64"), "score": pd.Series([1.25, 2.5], dtype="float64")}
    )
    pd.testing.assert_frame_equal(datason.loads(_PAYLOADS["pandas"]), expected)
