"""Batch array conversion preserves wire order, policies and allocation bounds."""

import json

import pytest

np = pytest.importorskip("numpy")

import datason
from datason._config import NanHandling, SerializationConfig
from datason._errors import SecurityError
from datason._protocols import DeserializeContext
from datason.plugins.numpy import NumpyPlugin


@pytest.mark.parametrize("dtype", ["complex64", "complex128", ">c8", np.clongdouble])
@pytest.mark.parametrize("layout", ["transposed", "reversed", "scalar", "empty"])
def test_complex_pairs_keep_row_order_and_shape(dtype, layout):
    original = (np.arange(64).reshape(8, 8) + 1j * np.arange(64, 128).reshape(8, 8)).astype(dtype)
    if layout == "transposed":
        original = original.T
    elif layout == "reversed":
        original = original[::-1, ::-2]
    elif layout == "scalar":
        original = original[:1, :1].reshape(())
    else:
        original = original[:0, :]
    wire = datason.dumps(original)
    payload = json.loads(wire)["__datason_value__"]
    assert payload["data"] == [[float(x.real), float(x.imag)] for x in original.flat]
    assert payload["encoding"] == "complex_pairs"
    restored = datason.loads(wire)
    assert restored.dtype == original.dtype
    assert restored.shape == original.shape
    np.testing.assert_array_equal(restored, original)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("policy", [NanHandling.NULL, NanHandling.STRING])
def test_batch_complex_components_still_apply_nonfinite_policy(dtype, policy):
    original = np.array([complex(float("nan"), 2), complex(3, float("inf"))] * 16, dtype=dtype)
    payload = json.loads(datason.dumps(original, nan_handling=policy))["__datason_value__"]
    expected = [[None, 2.0], [3.0, None]] if policy is NanHandling.NULL else [["NaN", 2.0], [3.0, "Infinity"]]
    assert payload["data"] == expected * 16


@pytest.mark.parametrize("include_type_hints", [False, True])
def test_array_string_leaves_still_apply_redaction(include_type_hints):
    wire = datason.dumps(
        np.array(["alice@example.com"]), include_type_hints=include_type_hints, redact_patterns=("email",)
    )
    assert "alice@example.com" not in wire
    restored = datason.loads(wire)
    assert restored[0] == "[REDACTED]"


@pytest.mark.parametrize("shape", [None, [1]])
@pytest.mark.parametrize("raw", [["a", "b"], [["a"], ["b"]]])
@pytest.mark.parametrize("dtype,budget", [("U2", 8), ("U0", 1)])
def test_actual_data_count_is_bounded_before_numpy_allocation(monkeypatch, shape, raw, dtype, budget):
    def forbidden(*args, **kwargs):
        pytest.fail("NumPy allocation happened before checking actual data size")

    monkeypatch.setattr(np, "array", forbidden)
    value = {"data": raw, "dtype": dtype}
    if shape is not None:
        value["shape"] = shape
    payload = {"__datason_type__": "numpy.ndarray", "__datason_value__": value}
    ctx = DeserializeContext(config=SerializationConfig(max_input_bytes=budget))
    with pytest.raises(SecurityError, match="byte budget"):
        NumpyPlugin().deserialize(payload, ctx)


def test_allocation_at_exact_byte_budget_is_allowed():
    payload = {
        "__datason_type__": "numpy.ndarray",
        "__datason_value__": {"data": [[1, 2]], "shape": [1, 2], "dtype": "int32"},
    }
    ctx = DeserializeContext(config=SerializationConfig(max_input_bytes=8))
    restored = NumpyPlugin().deserialize(payload, ctx)
    assert restored.nbytes == 8
    np.testing.assert_array_equal(restored, [[1, 2]])
