"""Integration tests for file I/O (dump/load).

Tests dump() and load() with real temp files using pytest's
tmp_path fixture.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import uuid

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
import datason


class TestDumpLoad:
    """File-based dump/load roundtrips."""

    def test_dump_load_basic_dict(self, tmp_path: pathlib.Path) -> None:
        data = {"name": "test", "count": 42, "active": True}
        fp = tmp_path / "basic.json"

        with open(fp, "w") as f:
            datason.dump(data, f)
        with open(fp) as f:
            result = datason.load(f)

        assert result == data

    def test_dump_load_with_datetimes(self, tmp_path: pathlib.Path) -> None:
        data = {"ts": dt.datetime(2024, 6, 15, 12, 0), "date": dt.date(2024, 6, 15)}  # noqa: DTZ001
        fp = tmp_path / "datetimes.json"

        with open(fp, "w") as f:
            datason.dump(data, f, include_type_hints=True)
        with open(fp) as f:
            result = datason.load(f)

        assert isinstance(result["ts"], dt.datetime)
        assert isinstance(result["date"], dt.date)

    def test_dump_load_with_numpy(self, tmp_path: pathlib.Path) -> None:
        data = {"arr": np.array([1.0, 2.0, 3.0]), "scalar": np.int64(42)}
        fp = tmp_path / "numpy.json"

        with open(fp, "w") as f:
            datason.dump(data, f, include_type_hints=True)
        with open(fp) as f:
            result = datason.load(f)

        assert isinstance(result["arr"], np.ndarray)
        np.testing.assert_array_equal(result["arr"], [1.0, 2.0, 3.0])

    def test_dump_load_with_pandas(self, tmp_path: pathlib.Path) -> None:
        data = {"df": pd.DataFrame({"a": [1, 2], "b": [3, 4]})}
        fp = tmp_path / "pandas.json"

        with open(fp, "w") as f:
            datason.dump(data, f, include_type_hints=True)
        with open(fp) as f:
            result = datason.load(f)

        assert isinstance(result["df"], pd.DataFrame)
        assert list(result["df"].columns) == ["a", "b"]

    def test_dump_load_mixed_types(self, tmp_path: pathlib.Path) -> None:
        data = {
            "id": uuid.uuid4(),
            "ts": dt.datetime(2024, 1, 1),  # noqa: DTZ001
            "weights": np.array([0.5, 0.5]),
            "label": "model_v1",
        }
        fp = tmp_path / "mixed.json"

        with open(fp, "w") as f:
            datason.dump(data, f, include_type_hints=True)
        with open(fp) as f:
            result = datason.load(f)

        assert isinstance(result["id"], uuid.UUID)
        assert isinstance(result["ts"], dt.datetime)
        assert isinstance(result["weights"], np.ndarray)
        assert result["label"] == "model_v1"

    def test_dump_load_with_config(self, tmp_path: pathlib.Path) -> None:
        data = {"z": 1, "a": 2, "m": 3}
        fp = tmp_path / "sorted.json"

        with open(fp, "w") as f:
            datason.dump(data, f, sort_keys=True)
        with open(fp) as f:
            result = datason.load(f)

        assert result == data
        # Verify sorted in file
        with open(fp) as f:
            parsed = json.load(f)
        assert list(parsed.keys()) == ["a", "m", "z"]

    def test_dump_load_with_context_manager(self, tmp_path: pathlib.Path) -> None:
        data = {"ts": dt.datetime(2024, 1, 1)}  # noqa: DTZ001
        fp = tmp_path / "context.json"

        with datason.config(include_type_hints=True):
            with open(fp, "w") as f:
                datason.dump(data, f)
        with open(fp) as f:
            result = datason.load(f)

        assert isinstance(result["ts"], dt.datetime)


class TestFileEdgeCases:
    """Edge cases for file I/O."""

    def test_empty_dict_file_roundtrip(self, tmp_path: pathlib.Path) -> None:
        fp = tmp_path / "empty.json"
        with open(fp, "w") as f:
            datason.dump({}, f)
        with open(fp) as f:
            result = datason.load(f)
        assert result == {}

    def test_unicode_content(self, tmp_path: pathlib.Path) -> None:
        data = {"emoji": "Hello 🌍", "cjk": "你好世界", "arabic": "مرحبا"}
        fp = tmp_path / "unicode.json"

        with open(fp, "w") as f:
            datason.dump(data, f)
        with open(fp) as f:
            result = datason.load(f)

        assert result == data

    def test_large_data_file_roundtrip(self, tmp_path: pathlib.Path) -> None:
        data = {f"key_{i}": f"value_{i}" for i in range(10000)}
        fp = tmp_path / "large.json"

        with open(fp, "w") as f:
            datason.dump(data, f)
        with open(fp) as f:
            result = datason.load(f)

        assert len(result) == 10000  # noqa: PLR2004
        assert result["key_0"] == "value_0"
        assert result["key_9999"] == "value_9999"

    def test_load_invalid_json_raises(self, tmp_path: pathlib.Path) -> None:
        fp = tmp_path / "invalid.json"
        fp.write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            with open(fp) as f:
                datason.load(f)
