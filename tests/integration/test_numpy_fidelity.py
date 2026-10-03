"""Public API regression tests for scientific values and reconstruction limits."""

import json
import warnings

import pytest

np = pytest.importorskip("numpy")

import datason
from datason._errors import DeserializationError, SecurityError, SerializationError


@pytest.mark.parametrize(
    "value",
    [
        np.int8(-1),
        np.int32(7),
        np.uint64(2**64 - 1),
        np.float32(1.25),
        np.float64(2.5),
        np.bool_(True),
        np.complex64(2 + 3j),
        np.datetime64("2026-10-03", "D"),
        np.timedelta64(123, "ns"),
    ],
)
def test_scalar_dtype_and_value(value):
    restored = datason.loads(datason.dumps(value))
    assert type(restored) is type(value)
    assert restored.dtype == value.dtype
    assert restored == value


@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0, 4), ()])
def test_empty_and_zero_dimensional_shapes(shape):
    original = np.zeros(shape, dtype=np.float32)
    restored = datason.loads(datason.dumps(original))
    assert restored.shape == original.shape
    assert restored.dtype == original.dtype
    np.testing.assert_array_equal(restored, original)


@pytest.mark.parametrize(
    "original",
    [
        np.array([[1 + 2j, 3 - 4j]], dtype=np.complex64),
        np.array(["2026-10-03", "NaT"], dtype="datetime64[ns]"),
        np.array([123, -456], dtype="timedelta64[ns]"),
    ],
)
def test_complex_and_temporal_arrays(original):
    restored = datason.loads(datason.dumps(original))
    assert restored.shape == original.shape
    assert restored.dtype == original.dtype
    np.testing.assert_array_equal(restored, original)


@pytest.mark.parametrize("shape", [[-1], [True], [1.5], [2]])
def test_invalid_shapes_raise(shape):
    payload = {
        "__datason_type__": "numpy.ndarray",
        "__datason_value__": {"data": [1], "shape": shape, "dtype": "int32"},
    }
    with pytest.raises(DeserializationError):
        datason.loads(json.dumps(payload))


def test_dtype_cannot_request_an_unbounded_allocation():
    payload = {
        "__datason_type__": "numpy.ndarray",
        "__datason_value__": {"data": ["a"], "shape": [1], "dtype": "U100000000"},
    }
    with pytest.raises(SecurityError):
        datason.loads(json.dumps(payload))


def test_legacy_scalar_without_dtype_is_readable():
    restored = datason.loads('{"__datason_type__":"numpy.integer","__datason_value__":7}')
    assert isinstance(restored, np.int64)


def test_structured_dtype_requires_explicit_plugin():
    value = np.array([(1, 2.0)], dtype=[("a", "i4"), ("b", "f8")])
    with pytest.raises(SerializationError, match="custom plugin"):
        datason.dumps(value)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("include_type_hints", [True, False])
def test_complex_scalar_dispatch_without_plugin_warnings(dtype, include_type_hints):
    original = dtype(2 + 3j)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = datason.loads(datason.dumps(original, include_type_hints=include_type_hints))
    if include_type_hints:
        assert type(restored) is type(original)
        assert restored.dtype == original.dtype
        assert restored == original
    else:
        assert restored == [2.0, 3.0]
