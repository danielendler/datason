import builtins
import types
import sys
import pytest

from datason.ml_serializers import (
    _LAZY_IMPORTS,
    _lazy_import_torch,
    _lazy_import_tensorflow,
    _lazy_import_sklearn,
    __getattr__ as ml_getattr,
)
from datason.deserializers import deserialize_fast, DeserializationSecurityError
from datason.config import SerializationConfig


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
    module = sys.modules['datason.ml_serializers']
    monkeypatch.setattr(module, 'torch', dummy, raising=False)
    assert _lazy_import_torch() is dummy
    # cached value should be reused even if attribute changes
    del module.__dict__['torch']
    assert _lazy_import_torch() is dummy
    # __getattr__ should also return the cached module
    assert ml_getattr('torch') is dummy


def test_lazy_import_tensorflow_missing(monkeypatch):
    _LAZY_IMPORTS["tensorflow"] = None
    if 'tensorflow' in sys.modules:
        monkeypatch.delitem(sys.modules, 'tensorflow', raising=False)
    def fake_import(name, *args, **kwargs):
        if name == 'tensorflow' or name.startswith('tensorflow.'):
            raise ImportError('no tf')
        return original_import(name, *args, **kwargs)
    original_import = builtins.__import__
    monkeypatch.setattr(builtins, '__import__', fake_import)
    module = sys.modules['datason.ml_serializers']
    module.__dict__.pop('tf', None)
    assert _lazy_import_tensorflow() is None


def test_lazy_import_sklearn_respects_patch(monkeypatch):
    dummy_sk = DummySklearn('sklearn')
    dummy_base = DummySklearn.BaseEstimator
    _LAZY_IMPORTS['sklearn'] = None
    _LAZY_IMPORTS['BaseEstimator'] = None
    module = sys.modules['datason.ml_serializers']
    monkeypatch.setattr(module, 'sklearn', dummy_sk, raising=False)
    monkeypatch.setattr(module, 'BaseEstimator', dummy_base, raising=False)
    sk, be = _lazy_import_sklearn()
    assert sk is dummy_sk
    assert be is dummy_base
    # Cache should persist
    del module.__dict__['sklearn']
    del module.__dict__['BaseEstimator']
    sk_cached, be_cached = _lazy_import_sklearn()
    assert sk_cached is dummy_sk
    assert be_cached is dummy_base


def test_deserialize_fast_security_exceptions():
    config = SerializationConfig(max_size=3, max_depth=2)
    big_list = [1, 2, 3, 4]
    big_dict = {i: i for i in range(5)}
    with pytest.raises(DeserializationSecurityError, match='List size'):
        deserialize_fast(big_list, config)
    with pytest.raises(DeserializationSecurityError, match='Dictionary size'):
        deserialize_fast(big_dict, config)
    deep = {'a': {'b': {'c': {'d': 1}}}}
    with pytest.raises(DeserializationSecurityError, match='depth'):
        deserialize_fast(deep, config)
