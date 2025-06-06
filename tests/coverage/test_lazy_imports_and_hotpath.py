import builtins
import sys
import types

import pytest

from datason.config import SerializationConfig
from datason.deserializers import (
    DeserializationSecurityError,
    _clear_deserialization_caches,
    _get_pooled_dict,
    _is_homogeneous_basic_types,
    _return_dict_to_pool,
    deserialize_fast,
)
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
    # Clear caches to ensure clean state
    _clear_deserialization_caches()

    # Force garbage collection to clean up any lingering state
    import gc

    gc.collect()

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


def test_basic_homogeneous_list_optimization():
    """Test the homogeneous list optimization hot path."""
    # Clear caches to ensure clean state
    _clear_deserialization_caches()

    # Test with homogeneous integer list (should take fast path)
    homogeneous_ints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert _is_homogeneous_basic_types(homogeneous_ints)

    result = deserialize_fast(homogeneous_ints)
    assert result == homogeneous_ints
    assert isinstance(result, list)


def test_basic_homogeneous_list_strings():
    """Test homogeneous string list optimization."""
    # Clear caches to ensure clean state
    _clear_deserialization_caches()

    # Test with homogeneous string list
    homogeneous_strings = ["hello", "world", "test", "data"]
    assert _is_homogeneous_basic_types(homogeneous_strings)

    result = deserialize_fast(homogeneous_strings)
    assert result == homogeneous_strings


def test_mixed_type_list_no_optimization():
    """Test that mixed type lists don't trigger optimization."""
    # Clear caches to ensure clean state
    _clear_deserialization_caches()

    # Test with mixed types (should not take fast path)
    mixed_list = [1, "hello", 3.14, True, None]
    assert not _is_homogeneous_basic_types(mixed_list)

    result = deserialize_fast(mixed_list)
    assert result == mixed_list


def test_nested_structure_optimization():
    """Test optimization with nested structures."""
    # Clear caches to ensure clean state
    _clear_deserialization_caches()

    # Test with nested structure
    nested_data = {
        "simple_list": [1, 2, 3, 4, 5],
        "mixed_list": [1, "hello", True],
        "nested_dict": {"inner_list": ["a", "b", "c"], "inner_value": 42},
    }

    result = deserialize_fast(nested_data)
    assert result == nested_data


def test_object_pooling_usage():
    """Test the object pooling mechanisms."""
    # Clear caches to ensure clean state
    _clear_deserialization_caches()

    # Get a pooled dict
    pooled_dict = _get_pooled_dict()
    assert isinstance(pooled_dict, dict)
    assert len(pooled_dict) == 0

    # Use it
    pooled_dict["test"] = "value"
    assert pooled_dict["test"] == "value"

    # Return to pool
    _return_dict_to_pool(pooled_dict)

    # Get another one (might be the same, might be different)
    pooled_dict2 = _get_pooled_dict()
    assert isinstance(pooled_dict2, dict)


def test_large_homogeneous_list_performance():
    """Test performance optimization with large homogeneous lists."""
    # Clear caches to ensure clean state
    _clear_deserialization_caches()

    # Test with large homogeneous list
    large_list = [42] * 1000
    assert _is_homogeneous_basic_types(large_list)

    result = deserialize_fast(large_list)
    assert result == large_list
    assert len(result) == 1000


def test_empty_containers_optimization():
    """Test optimization with empty containers."""
    # Clear caches to ensure clean state
    _clear_deserialization_caches()

    # Test empty list
    empty_list = []
    assert _is_homogeneous_basic_types(empty_list)  # Empty lists are homogeneous

    result = deserialize_fast(empty_list)
    assert result == empty_list

    # Test empty dict
    empty_dict = {}
    result = deserialize_fast(empty_dict)
    assert result == empty_dict
