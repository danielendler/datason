import json as std_json
from pathlib import Path
from typing import Any

import pytest

import datason.json as ds_json


def sample_data() -> dict[str, Any]:
    return {"a": 1, "b": [1, 2, 3], "c": {"d": "text"}}


def test_function_identity() -> None:
    """datason.json should expose the exact stdlib functions and errors."""
    assert ds_json.dumps is std_json.dumps
    assert ds_json.loads is std_json.loads
    assert ds_json.JSONDecodeError is std_json.JSONDecodeError


@pytest.mark.parametrize(
    "kwargs,data",
    [
        ({}, sample_data()),
        ({"indent": 2}, {"a": 1}),
        ({"separators": (",", ":")}, {"a": 1, "b": 2}),
        ({"sort_keys": True}, {"b": 2, "a": 1}),
        ({"ensure_ascii": False}, {"snowman": "☃"}),
    ],
)
def test_dumps_parity(kwargs: dict[str, Any], data: Any) -> None:
    """dumps should match stdlib json byte-for-byte across flags."""
    assert ds_json.dumps(data, **kwargs) == std_json.dumps(data, **kwargs)


def test_allow_nan_parity() -> None:
    data = {"nan": float("nan")}
    assert ds_json.dumps(data, allow_nan=True) == std_json.dumps(data, allow_nan=True)
    with pytest.raises(ValueError) as ds_err:
        ds_json.dumps(data, allow_nan=False)
    with pytest.raises(ValueError) as std_err:
        std_json.dumps(data, allow_nan=False)
    assert str(ds_err.value) == str(std_err.value)


def test_dump_and_load_parity(tmp_path: Path) -> None:
    data = sample_data()
    ds_file = tmp_path / "ds.json"
    std_file = tmp_path / "std.json"

    with ds_file.open("w") as f:
        ds_json.dump(data, f, sort_keys=True, indent=2)
    with std_file.open("w") as f:
        std_json.dump(data, f, sort_keys=True, indent=2)

    assert ds_file.read_text() == std_file.read_text()

    with ds_file.open() as f:
        ds_loaded = ds_json.load(f)
    with std_file.open() as f:
        std_loaded = std_json.load(f)
    assert ds_loaded == std_loaded


def test_dumps_non_serializable_error() -> None:
    class Foo:
        pass

    obj = {"foo": Foo()}
    with pytest.raises(TypeError) as ds_err:
        ds_json.dumps(obj)
    with pytest.raises(TypeError) as std_err:
        std_json.dumps(obj)
    assert str(ds_err.value) == str(std_err.value)


def test_loads_malformed_json_error() -> None:
    bad = '{"a": }'
    with pytest.raises(ds_json.JSONDecodeError) as ds_err:
        ds_json.loads(bad)
    with pytest.raises(std_json.JSONDecodeError) as std_err:
        std_json.loads(bad)
    assert str(ds_err.value) == str(std_err.value)
