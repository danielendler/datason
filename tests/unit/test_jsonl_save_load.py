import json
import gzip

import datason


def test_jsonl_roundtrip(tmp_path):
    records = [{"a": 1}, {"a": 2}]
    path = tmp_path / "data.jsonl"
    datason.save(records, path)
    loaded = list(datason.load(path))
    assert loaded == records


def test_jsonl_gzip_roundtrip(tmp_path):
    records = [{"b": 1}, {"b": 2}]
    path = tmp_path / "data.jsonl.gz"
    datason.save(records, path)
    loaded = list(datason.load(path))
    assert loaded == records


def test_ndjson_autodetect(tmp_path):
    records = [{"c": 3}]
    path = tmp_path / "data.ndjson"
    datason.save(records, path)
    with open(path, "r") as f:
        line = f.readline()
        assert json.loads(line) == records[0]
    loaded = list(datason.load(path))
    assert loaded == records
