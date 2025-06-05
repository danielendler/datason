import sys
import types
from datetime import datetime
import uuid
from decimal import Decimal
from pathlib import Path

import pytest

from datason.deserializers import (
    _convert_string_keys_to_int_if_possible,
    _try_numpy_array_detection,
    _looks_like_numpy_array,
    _deserialize_with_type_metadata,
)


class DummyNumpy(types.ModuleType):
    class ndarray(list):
        def __init__(self, data):
            super().__init__(data)
            self.shape = (len(data),)
        def astype(self, dtype):
            self.dtype = dtype
            return self
        def reshape(self, shape):
            self.shape = tuple(shape)
            return self

    def array(self, data):
        return DummyNumpy.ndarray(data)


def test_convert_string_keys_basic():
    data = {"1": "a", "-2": "b", "3.5": "c", "x": 1}
    result = _convert_string_keys_to_int_if_possible(data)
    assert result == {1: "a", -2: "b", "3.5": "c", "x": 1}


def test_convert_string_keys_non_string():
    data = {1: "a", "two": 2}
    result = _convert_string_keys_to_int_if_possible(data)
    assert result == {1: "a", "two": 2}


def test_try_numpy_array_detection_without_numpy():
    assert _try_numpy_array_detection([1, 2, 3]) is None


def test_try_numpy_array_detection_with_numpy(monkeypatch):
    dummy = DummyNumpy("numpy")
    monkeypatch.setitem(sys.modules, "numpy", dummy)
    result = _try_numpy_array_detection([1, 2, 3, 4, 5, 6, 7, 8])
    assert isinstance(result, DummyNumpy.ndarray)
    assert list(result) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_looks_like_numpy_array(monkeypatch):
    dummy = DummyNumpy("numpy")
    monkeypatch.setitem(sys.modules, "numpy", dummy)
    arr1 = DummyNumpy.ndarray([1, 2])
    arr2 = DummyNumpy.ndarray([3, 4])
    assert _looks_like_numpy_array([arr1, arr2])
    assert _looks_like_numpy_array([arr1, [3, "a"]])


class DummyPandas(types.ModuleType):
    class CategoricalDtype:
        def __init__(self, categories=None, ordered=False):
            self.categories = categories
            self.ordered = ordered

    class Series(list):
        def __init__(self, data=None, index=None, name=None):
            if isinstance(data, dict):
                index = list(data.keys())
                data = list(data.values())
            super().__init__(data if data is not None else [])
            self.index = index
            self.name = name

        def astype(self, dtype):
            self.dtype = dtype
            return self

    class DataFrame:
        def __init__(self, data=None, index=None, columns=None, orient=None):
            self.data = data
            self.index = index
            self.columns = columns
            self.orient = orient

        @classmethod
        def from_dict(cls, data, orient="columns"):
            return cls(data=data, orient=orient)

    def __init__(self, name="pandas"):
        super().__init__(name)
        self.Series = DummyPandas.Series
        self.DataFrame = DummyPandas.DataFrame
        self.CategoricalDtype = DummyPandas.CategoricalDtype


def test_deserialize_with_type_metadata_basic(monkeypatch):
    dummy_pd = DummyPandas()
    monkeypatch.setattr(sys.modules["datason.deserializers"], "pd", dummy_pd, raising=False)
    dummy_np = DummyNumpy("numpy")
    monkeypatch.setitem(sys.modules, "numpy", dummy_np)

    dt_str = "2023-01-02T03:04:05"
    uuid_str = "12345678-1234-5678-1234-567812345678"
    assert _deserialize_with_type_metadata({"__datason_type__": "datetime", "__datason_value__": dt_str}) == datetime.fromisoformat(dt_str)
    assert _deserialize_with_type_metadata({"__datason_type__": "uuid.UUID", "__datason_value__": uuid_str}) == uuid.UUID(uuid_str)
    assert _deserialize_with_type_metadata({"__datason_type__": "complex", "__datason_value__": {"real": 1, "imag": 2}}) == 1 + 2j
    assert _deserialize_with_type_metadata({"__datason_type__": "decimal.Decimal", "__datason_value__": "3.14"}) == Decimal("3.14")
    assert _deserialize_with_type_metadata({"__datason_type__": "pathlib.Path", "__datason_value__": "/tmp"}) == Path("/tmp")
    assert _deserialize_with_type_metadata({"__datason_type__": "set", "__datason_value__": [1, 2]}) == {1, 2}
    assert _deserialize_with_type_metadata({"__datason_type__": "tuple", "__datason_value__": [1, 2]}) == (1, 2)

    df_meta = {"__datason_type__": "pandas.DataFrame", "__datason_value__": {"index": [0, 1], "columns": ["a"], "data": [[1], [2]]}}
    df_result = _deserialize_with_type_metadata(df_meta)
    assert isinstance(df_result, DummyPandas.DataFrame)
    assert df_result.data == [[1], [2]]
    assert df_result.index == [0, 1]
    assert df_result.columns == ["a"]

    series_meta = {"__datason_type__": "pandas.Series", "__datason_value__": [1, 2, 3]}
    series_result = _deserialize_with_type_metadata(series_meta)
    assert isinstance(series_result, DummyPandas.Series)
    assert list(series_result) == [1, 2, 3]


def test_deserialize_with_numpy_and_torch(monkeypatch):
    dummy_pd = DummyPandas()
    dummy_np = DummyNumpy("numpy")
    dummy_np.int32 = lambda x: f"int32({x})"
    dummy_np.int64 = lambda x: f"int64({x})"
    dummy_np.float32 = lambda x: f"float32({x})"
    dummy_np.float64 = lambda x: f"float64({x})"
    dummy_np.bool_ = bool
    dummy_np.complex128 = complex

    class DummyTensor(list):
        def __init__(self, data, device="cpu", requires_grad=False):
            super().__init__(data)
            self.device = device
            self.requires_grad = requires_grad
        def to(self, dtype):
            self.dtype = dtype
            return self
        def numel(self):
            return len(self)
        def reshape(self, shape):
            return self

    class DummyTorch(types.ModuleType):
        float32 = "float32"
        def __init__(self):
            super().__init__("torch")
        def tensor(self, data, device="cpu", requires_grad=False):
            return DummyTensor(data, device, requires_grad)

    dummy_torch = DummyTorch()

    sk_mod = types.ModuleType("sklearn.linear_model")
    class DummyModel:
        def __init__(self, **params):
            self.params = params
    sk_mod.LinearRegression = DummyModel

    monkeypatch.setattr(sys.modules["datason.deserializers"], "pd", dummy_pd, raising=False)
    monkeypatch.setattr(sys.modules["datason.deserializers"], "np", dummy_np, raising=False)
    monkeypatch.setitem(sys.modules, "numpy", dummy_np)
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", sk_mod)

    ndarray_meta = {"__datason_type__": "numpy.ndarray", "__datason_value__": {"data": [1, 2, 3, 4], "dtype": "float32", "shape": [2, 2]}}
    ndarray_result = _deserialize_with_type_metadata(ndarray_meta)
    assert isinstance(ndarray_result, DummyNumpy.ndarray)
    assert list(ndarray_result) == [1, 2, 3, 4]

    scalar_meta = {"__datason_type__": "numpy.int32", "__datason_value__": 5}
    assert _deserialize_with_type_metadata(scalar_meta) == "int32(5)"

    tensor_meta = {
        "__datason_type__": "torch.Tensor",
        "__datason_value__": {"_data": [1, 2], "_dtype": "float32", "_device": "cpu", "_requires_grad": True, "_shape": [2]},
    }
    tensor = _deserialize_with_type_metadata(tensor_meta)
    assert isinstance(tensor, DummyTensor)
    assert tensor.device == "cpu"
    assert tensor.requires_grad is True

    skl_meta = {"__datason_type__": "sklearn.linear_model.LinearRegression", "__datason_value__": {"_class": "sklearn.linear_model.LinearRegression", "_params": {"a": 1}}}
    model = _deserialize_with_type_metadata(skl_meta)
    assert isinstance(model, DummyModel)
    assert model.params == {"a": 1}
