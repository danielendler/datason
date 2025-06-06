import builtins
import sys
import types

import pytest

from datason.config import SerializationConfig
from datason.deserializers import DeserializationSecurityError, deserialize_fast
from datason.ml_serializers import (
    _LAZY_IMPORTS,
    _lazy_import_jax,
    _lazy_import_pil,
    _lazy_import_scipy,
    _lazy_import_sklearn,
    _lazy_import_tensorflow,
    _lazy_import_torch,
    _lazy_import_transformers,
)
from datason.ml_serializers import (
    __getattr__ as ml_getattr,
)


class DummyTorch(types.ModuleType):
    pass


class DummyTensorFlow(types.ModuleType):
    pass


class DummySklearn(types.ModuleType):
    class BaseEstimator:
        pass


def test_lazy_import_torch_respects_patch(monkeypatch):
    dummy = DummyTorch("torch")
    _LAZY_IMPORTS["torch"] = None
    module = sys.modules["datason.ml_serializers"]
    monkeypatch.setattr(module, "torch", dummy, raising=False)
    assert _lazy_import_torch() is dummy
    # cached value should be reused even if attribute changes
    del module.__dict__["torch"]
    assert _lazy_import_torch() is dummy
    # __getattr__ should also return the cached module
    assert ml_getattr("torch") is dummy


def test_lazy_import_tensorflow_missing(monkeypatch):
    _LAZY_IMPORTS["tensorflow"] = None
    if "tensorflow" in sys.modules:
        monkeypatch.delitem(sys.modules, "tensorflow", raising=False)

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow" or name.startswith("tensorflow."):
            raise ImportError("no tf")
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fake_import)
    module = sys.modules["datason.ml_serializers"]
    module.__dict__.pop("tf", None)
    assert _lazy_import_tensorflow() is None


def test_lazy_import_sklearn_respects_patch(monkeypatch):
    dummy_sk = DummySklearn("sklearn")
    dummy_base = DummySklearn.BaseEstimator
    _LAZY_IMPORTS["sklearn"] = None
    _LAZY_IMPORTS["BaseEstimator"] = None
    module = sys.modules["datason.ml_serializers"]
    monkeypatch.setattr(module, "sklearn", dummy_sk, raising=False)
    monkeypatch.setattr(module, "BaseEstimator", dummy_base, raising=False)
    sk, be = _lazy_import_sklearn()
    assert sk is dummy_sk
    assert be is dummy_base
    # Cache should persist
    del module.__dict__["sklearn"]
    del module.__dict__["BaseEstimator"]
    sk_cached, be_cached = _lazy_import_sklearn()
    assert sk_cached is dummy_sk
    assert be_cached is dummy_base


def test_deserialize_fast_security_exceptions():
    config = SerializationConfig(max_size=3, max_depth=2)
    big_list = [1, 2, 3, 4]
    big_dict = {i: i for i in range(5)}
    with pytest.raises(DeserializationSecurityError, match="List size"):
        deserialize_fast(big_list, config)
    with pytest.raises(DeserializationSecurityError, match="Dictionary size"):
        deserialize_fast(big_dict, config)
    deep = {"a": {"b": {"c": {"d": 1}}}}
    with pytest.raises(DeserializationSecurityError, match="depth"):
        deserialize_fast(deep, config)


class DummyJax(types.ModuleType):
    def __init__(self, name="jax"):
        super().__init__(name)
        self.numpy = types.SimpleNamespace()


class DummyScipy(types.ModuleType):
    pass


class DummyPIL(types.ModuleType):
    def __init__(self):
        super().__init__("PIL")
        self.Image = object


def test_lazy_import_jax_respects_patch(monkeypatch):
    dummy = DummyJax()
    _LAZY_IMPORTS["jax"] = None
    _LAZY_IMPORTS["jnp"] = None
    module = sys.modules["datason.ml_serializers"]
    monkeypatch.setattr(module, "jax", dummy, raising=False)
    jax_mod, jnp_mod = _lazy_import_jax()
    assert jax_mod is dummy
    assert jnp_mod is dummy.numpy
    del module.__dict__["jax"]
    jax_cached, jnp_cached = _lazy_import_jax()
    assert jax_cached is dummy
    assert jnp_cached is dummy.numpy


def test_lazy_import_scipy_respects_patch(monkeypatch):
    dummy = DummyScipy("scipy")
    _LAZY_IMPORTS["scipy"] = None
    module = sys.modules["datason.ml_serializers"]
    monkeypatch.setattr(module, "scipy", dummy, raising=False)
    assert _lazy_import_scipy() is dummy
    del module.__dict__["scipy"]
    assert _lazy_import_scipy() is dummy


def test_lazy_import_pil_respects_patch(monkeypatch):
    dummy = DummyPIL()
    _LAZY_IMPORTS["PIL_Image"] = None
    module = sys.modules["datason.ml_serializers"]
    monkeypatch.setattr(module, "Image", dummy.Image, raising=False)
    assert _lazy_import_pil() is dummy.Image
    del module.__dict__["Image"]
    assert _lazy_import_pil() is dummy.Image


def test_lazy_import_transformers_respects_patch(monkeypatch):
    dummy = types.ModuleType("transformers")
    _LAZY_IMPORTS["transformers"] = None
    module = sys.modules["datason.ml_serializers"]
    monkeypatch.setattr(module, "transformers", dummy, raising=False)
    assert _lazy_import_transformers() is dummy
    del module.__dict__["transformers"]
    assert _lazy_import_transformers() is dummy
