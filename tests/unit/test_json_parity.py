import json as std_json

import pytest

import datason.json as ds_json


def sample_data():
    return {
        "a": 1,
        "b": [1, 2, 3],
        "c": {"d": "text"},
    }


def test_dumps_parity():
    data = sample_data()
    assert ds_json.dumps(data, sort_keys=True) == std_json.dumps(data, sort_keys=True)


def test_dump_parity(tmp_path):
    data = sample_data()
    ds_file = tmp_path / "ds.json"
    std_file = tmp_path / "std.json"
    with ds_file.open("w") as f:
        ds_json.dump(data, f, sort_keys=True)
    with std_file.open("w") as f:
        std_json.dump(data, f, sort_keys=True)
    assert ds_file.read_text() == std_file.read_text()


def test_loads_parity():
    data = sample_data()
    s = std_json.dumps(data)
    assert ds_json.loads(s) == std_json.loads(s)


def test_load_parity(tmp_path):
    data = sample_data()
    path = tmp_path / "data.json"
    with path.open("w") as f:
        std_json.dump(data, f)
    with path.open() as f1, path.open() as f2:
        assert ds_json.load(f1) == std_json.load(f2)


def test_error_parity():
    bad = '{"a": }'
    with pytest.raises(ds_json.JSONDecodeError) as ds_err:
        ds_json.loads(bad)
    with pytest.raises(std_json.JSONDecodeError) as std_err:
        std_json.loads(bad)
    assert str(ds_err.value) == str(std_err.value)
