import json

import pytest

import datason.api as api
from datason import _rustcore
from datason.core_new import SecurityError


class TestEligibleBasicTree:
    def test_supported_and_unsupported(self):
        assert _rustcore._eligible_basic_tree({"a": [1, 2, 3], "b": None})
        assert not _rustcore._eligible_basic_tree({"a": {1}})
        assert not _rustcore._eligible_basic_tree(float("inf"))


class TestRustCoreDumpsLoads:
    def test_success_paths(self, monkeypatch):
        def fake_dumps_core(obj, ensure_ascii, allow_nan, md, mtb, msl):
            return b"{}"

        def fake_loads_core(data, md, mtb, msl):
            assert data in (b"{}", "{}")
            return {"ok": True}

        monkeypatch.setattr(_rustcore, "_dumps_core", fake_dumps_core)
        monkeypatch.setattr(_rustcore, "_loads_core", fake_loads_core)
        monkeypatch.setattr(_rustcore, "AVAILABLE", True)

        assert _rustcore.dumps({}) == b"{}"
        assert _rustcore.loads(b"{}") == {"ok": True}

    def test_dumps_errors(self, monkeypatch):
        def bad_core(obj, *args):
            raise ValueError("boom")

        monkeypatch.setattr(_rustcore, "_dumps_core", bad_core)
        monkeypatch.setattr(_rustcore, "AVAILABLE", True)
        with pytest.raises(_rustcore.UnsupportedType):
            _rustcore.dumps({})

    def test_dumps_security_error(self, monkeypatch):
        def bad_core(obj, *args):
            raise SecurityError("limit")

        monkeypatch.setattr(_rustcore, "_dumps_core", bad_core)
        monkeypatch.setattr(_rustcore, "AVAILABLE", True)
        with pytest.raises(SecurityError):
            _rustcore.dumps({})

    def test_loads_unavailable(self, monkeypatch):
        monkeypatch.setattr(_rustcore, "AVAILABLE", False)
        with pytest.raises(_rustcore.UnsupportedType):
            _rustcore.loads(b"{}")

    def test_security_error_passthrough(self, monkeypatch):
        def bad_core(data, *args):
            raise SecurityError("limit")

        monkeypatch.setattr(_rustcore, "_loads_core", bad_core)
        monkeypatch.setattr(_rustcore, "AVAILABLE", True)
        with pytest.raises(SecurityError):
            _rustcore.loads(b"{}")


class TestApiIntegration:
    def test_load_basic_uses_rust(self, monkeypatch):
        class DummyRust:
            class UnsupportedType(Exception):
                pass

            def loads(self, data):
                return {"via": "rust"}

        monkeypatch.setattr(api, "_rustcore", DummyRust())
        assert api.load_basic("{}") == {"via": "rust"}

    def test_load_basic_fallback(self, monkeypatch):
        class DummyRust:
            class UnsupportedType(Exception):
                pass

            def loads(self, data):
                raise DummyRust.UnsupportedType()

        monkeypatch.setattr(api, "_rustcore", DummyRust())
        monkeypatch.setattr(api, "loads_json", lambda data, **kw: {"fallback": True})
        assert api.load_basic("{}") == {"fallback": True}

    def test_save_string_uses_rust(self, monkeypatch):
        class DummyRust:
            class UnsupportedType(Exception):
                pass

            @staticmethod
            def _eligible_basic_tree(obj):
                return True

            @staticmethod
            def dumps(obj, ensure_ascii=False, allow_nan=False):
                return b"1"

        monkeypatch.setattr(api, "_rustcore", DummyRust())
        assert api.save_string(1) == "1"

    def test_save_string_fallback(self, monkeypatch):
        class DummyRust:
            class UnsupportedType(Exception):
                pass

            @staticmethod
            def _eligible_basic_tree(obj):
                return False

            @staticmethod
            def dumps(obj, ensure_ascii=False, allow_nan=False):
                raise DummyRust.UnsupportedType()

        monkeypatch.setattr(api, "_rustcore", DummyRust())
        assert api.save_string({"a": 1}) == json.dumps({"a": 1})
