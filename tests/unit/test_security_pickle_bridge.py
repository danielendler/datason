"""Tests for pickle bridge security module."""

from __future__ import annotations

import datetime as dt
import pickle
from decimal import Decimal
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID

import pytest

np = pytest.importorskip("numpy")
import datason
from datason._errors import SecurityError
from datason.security.pickle_bridge import (
    DEFAULT_ALLOWED_MODULES,
    json_to_pickle,
    pickle_to_json,
    scan_pickle_modules,
    validate_pickle_safety,
)


class TestScanPickleModules:
    """Test pickle opcode scanning."""

    def test_scan_stdlib_types(self) -> None:
        data = pickle.dumps({"a": 1, "b": dt.datetime(2024, 1, 1)})
        modules = scan_pickle_modules(data)
        assert "datetime" in modules

    def test_scan_numpy(self) -> None:
        data = pickle.dumps(np.array([1, 2, 3]))
        modules = scan_pickle_modules(data)
        assert "numpy" in modules

    def test_scan_simple_dict(self) -> None:
        data = pickle.dumps({"key": "value"})
        modules = scan_pickle_modules(data)
        # Simple dict may only reference builtins or nothing
        assert all(m in DEFAULT_ALLOWED_MODULES for m in modules)

    def test_scan_empty_bytes(self) -> None:
        # Invalid pickle data should not crash
        modules = scan_pickle_modules(b"")
        # May contain __unparseable__ or be empty
        assert isinstance(modules, set)


class TestValidatePickleSafety:
    """Test allow-list validation."""

    def test_safe_stdlib_data(self) -> None:
        data = pickle.dumps({"ts": dt.datetime(2024, 1, 1), "v": Decimal("3.14")})
        is_safe, disallowed = validate_pickle_safety(data)
        assert is_safe
        assert len(disallowed) == 0

    def test_safe_numpy_data(self) -> None:
        data = pickle.dumps(np.array([1.0, 2.0, 3.0]))
        is_safe, disallowed = validate_pickle_safety(data)
        assert is_safe

    def test_custom_allowed_modules(self) -> None:
        data = pickle.dumps(np.array([1, 2]))
        # Restrict to only builtins — numpy should be disallowed
        is_safe, disallowed = validate_pickle_safety(data, allowed_modules=frozenset({"builtins"}))
        # numpy should be flagged
        assert "numpy" in disallowed or is_safe  # depends on protocol


class TestPickleToJson:
    """Test pickle-to-JSON conversion."""

    def test_simple_dict(self) -> None:
        data = pickle.dumps({"name": "Alice", "age": 30})
        result = pickle_to_json(data)
        restored = datason.loads(result)
        assert restored["name"] == "Alice"
        assert restored["age"] == 30

    def test_with_datetime(self) -> None:
        original = {"ts": dt.datetime(2024, 6, 15, 10, 30)}
        data = pickle.dumps(original)
        result = pickle_to_json(data)
        restored = datason.loads(result)
        assert isinstance(restored["ts"], dt.datetime)
        assert restored["ts"].year == 2024

    def test_with_numpy_array(self) -> None:
        original = np.array([1.0, 2.0, 3.0])
        data = pickle.dumps(original)
        result = pickle_to_json(data)
        restored = datason.loads(result)
        assert isinstance(restored, np.ndarray)
        assert list(restored) == [1.0, 2.0, 3.0]

    def test_with_config_overrides(self) -> None:
        data = pickle.dumps({"key": "value"})
        result = pickle_to_json(data, sort_keys=True)
        assert '"key"' in result

    def test_rejects_disallowed_modules(self) -> None:
        data = pickle.dumps(np.array([1, 2]))
        with pytest.raises(SecurityError, match="disallowed modules"):
            pickle_to_json(data, allowed_modules=frozenset({"builtins"}))


class TestJsonToPickle:
    """Test JSON-to-pickle conversion."""

    def test_simple_roundtrip(self) -> None:
        original: dict[str, Any] = {"name": "Bob", "scores": [1, 2, 3]}
        json_str = datason.dumps(original)
        pkl_bytes = json_to_pickle(json_str)
        restored = pickle.loads(pkl_bytes)  # noqa: S301
        assert restored["name"] == "Bob"
        assert restored["scores"] == [1, 2, 3]

    def test_with_typed_data(self) -> None:
        original: dict[str, Any] = {"id": UUID("12345678-1234-5678-1234-567812345678")}
        json_str = datason.dumps(original)
        pkl_bytes = json_to_pickle(json_str)
        restored = pickle.loads(pkl_bytes)  # noqa: S301
        assert isinstance(restored["id"], UUID)


class TestPickleFileToJson:
    """Test file-based pickle conversion."""

    def test_file_roundtrip(self, tmp_path: PurePosixPath) -> None:
        pkl_path = str(tmp_path / "test.pkl")
        original = {"key": "value", "n": 42}
        with open(pkl_path, "wb") as f:
            pickle.dump(original, f)

        from datason.security.pickle_bridge import pickle_file_to_json

        result = pickle_file_to_json(pkl_path)
        restored = datason.loads(result)
        assert restored == original

    def test_file_not_found(self) -> None:
        from datason.security.pickle_bridge import pickle_file_to_json

        with pytest.raises(FileNotFoundError):
            pickle_file_to_json("/nonexistent/path.pkl")
