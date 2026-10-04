"""Reconstruction budgets, faithful empty shapes and explicit trust boundaries."""

import json

import pytest

import datason
from datason._errors import DeserializationError, SecurityError


def wire(tag, data, **metadata):
    return json.dumps({"__datason_type__": tag, "__datason_value__": data, **metadata})


@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0, 4), ()])
@pytest.mark.parametrize("library", ["torch", "tensorflow", "jax"])
def test_dense_empty_and_scalar_shapes(library, shape):
    np = pytest.importorskip("numpy")
    backend = pytest.importorskip(library)
    original = np.zeros(shape, dtype=np.float32)
    if library == "torch":
        value = backend.tensor(original)
    elif library == "tensorflow":
        value = backend.constant(original)
    else:
        import jax.numpy as jnp

        value = jnp.array(original)
    restored = datason.loads(datason.dumps(value))
    assert tuple(restored.shape) == shape
    actual = np.asarray(restored)
    assert actual.dtype == original.dtype
    np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize(
    "tag,library,constructor",
    [("torch.Tensor", "torch", "tensor"), ("tf.Tensor", "tensorflow", "constant"), ("jax.Array", "numpy", "array")],
)
@pytest.mark.parametrize(
    "shape,error",
    [
        ([1000], SecurityError),
        ([2], DeserializationError),
        ([True], DeserializationError),
        ([-1], DeserializationError),
    ],
)
def test_dense_rejection_precedes_constructor(monkeypatch, tag, library, constructor, shape, error):
    if tag == "jax.Array":
        pytest.importorskip("jax")
    backend = pytest.importorskip(library)
    payload = wire(tag, {"data": [1], "shape": shape, "dtype": "float32"})
    with monkeypatch.context() as patch:
        patch.setattr(backend, constructor, lambda *args, **kwargs: pytest.fail("constructor reached rejected payload"))
        with pytest.raises(error):
            datason.loads(payload, max_input_bytes=512)


def test_torch_reconstruction_uses_cpu_even_with_other_default_device():
    torch = pytest.importorskip("torch")
    with torch.device("meta"):
        restored = datason.loads(
            wire("torch.Tensor", {"data": [1], "shape": [1], "dtype": "float32", "device": "cuda:0"})
        )
    assert restored.device.type == "cpu"
    assert restored.tolist() == [1]


def test_tensorflow_variable_empty_shape_and_cpu():
    tf = pytest.importorskip("tensorflow")
    restored = datason.loads(datason.dumps(tf.Variable(tf.zeros([2, 0, 4]))))
    assert isinstance(restored, tf.Variable)
    assert tuple(restored.shape) == (2, 0, 4)
    assert "CPU" in restored.device


def test_jax_rejects_implicit_x64_loss(monkeypatch):
    jax = pytest.importorskip("jax")
    np = pytest.importorskip("numpy")
    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:
        enable_x64 = jax.experimental.enable_x64
    with enable_x64(False), monkeypatch.context() as patch:
        patch.setattr(np, "array", lambda *args, **kwargs: pytest.fail("unsupported dtype allocated"))
        with pytest.raises(DeserializationError, match="x64"):
            datason.loads(wire("jax.Array", {"data": [1], "shape": [1], "dtype": "int64"}))
    with enable_x64(True):
        assert str(datason.loads(wire("jax.Array", {"data": [1], "shape": [1], "dtype": "int64"})).dtype) == "int64"


@pytest.mark.parametrize("fmt,shape", [("csr", [10**9, 1]), ("csc", [1, 10**9])])
def test_sparse_pointer_budget_before_scipy_constructor(monkeypatch, fmt, shape):
    sp = pytest.importorskip("scipy.sparse")
    payload = wire(
        "scipy.sparse.matrix", {"row": [], "col": [], "data": [], "shape": shape, "format": fmt, "dtype": "float32"}
    )
    with monkeypatch.context() as patch:
        patch.setattr(sp, "coo_matrix", lambda *args, **kwargs: pytest.fail("unbounded sparse constructor reached"))
        with pytest.raises(SecurityError, match="byte budget"):
            datason.loads(payload)


def test_sparse_coo_large_logical_shape_is_not_dense_allocation():
    pytest.importorskip("scipy.sparse")
    restored = datason.loads(
        wire(
            "scipy.sparse.matrix",
            {"row": [0], "col": [0], "data": [1], "shape": [10**9, 10**9], "format": "coo", "dtype": "float32"},
        )
    )
    assert restored.shape == (10**9, 10**9)
    assert restored.nnz == 1


@pytest.mark.parametrize("indices", [[[2, 0]], [[0]], [[True, 0]]])
def test_tensorflow_sparse_coordinates_validated_before_allocation(monkeypatch, indices):
    tf = pytest.importorskip("tensorflow")
    payload = wire("tf.SparseTensor", {"indices": indices, "values": [1], "dense_shape": [2, 2], "dtype": "float32"})
    with monkeypatch.context() as patch:
        patch.setattr(tf, "constant", lambda *args, **kwargs: pytest.fail("invalid sparse allocation reached"))
        with pytest.raises(DeserializationError):
            datason.loads(payload)


@pytest.mark.parametrize(
    "tag",
    [
        "datetime",
        "numpy.ndarray",
        "pandas.DataFrame",
        "torch.Tensor",
        "tf.Variable",
        "scipy.sparse.matrix",
        "sklearn.estimator",
        "jax.Array",
        "polars.DataFrame",
        "plotly.Figure",
        "catboost.Model",
        "optuna.Study",
        "custom.model",
    ],
)
@pytest.mark.parametrize("strict", [True, False])
def test_disabling_plugins_prevents_all_dispatch(monkeypatch, tag, strict):
    from datason._registry import default_registry

    monkeypatch.setattr(
        default_registry, "find_deserializer", lambda *args: pytest.fail("plugin dispatched despite disabled policy")
    )
    with pytest.raises(DeserializationError, match="allow_plugin_deserialization"):
        datason.loads(wire("tuple", [json.loads(wire(tag, {}))]), allow_plugin_deserialization=False, strict=strict)


def test_non_estimator_class_never_receives_state(monkeypatch):
    from types import SimpleNamespace

    pytest.importorskip("sklearn")
    from datason.plugins import sklearn

    class OtherClass:
        def __setstate__(self, state):
            pytest.fail("non-estimator state hook executed")

    monkeypatch.setattr(sklearn.importlib, "import_module", lambda path: SimpleNamespace(OtherClass=OtherClass))
    payload = wire("sklearn.estimator", {"class": "sklearn.example.OtherClass", "state": {}})
    with pytest.warns(UserWarning, match="not a sklearn estimator"), pytest.raises(DeserializationError):
        datason.loads(payload)


def test_malformed_estimator_state_does_not_import(monkeypatch):
    pytest.importorskip("sklearn")
    from datason.plugins import sklearn

    monkeypatch.setattr(
        sklearn.importlib, "import_module", lambda path: pytest.fail("imported before state validation")
    )
    payload = wire("sklearn.estimator", {"class": "sklearn.linear_model.LinearRegression", "state": []})
    with pytest.warns(UserWarning, match="dictionary"), pytest.raises(DeserializationError):
        datason.loads(payload)


@pytest.mark.parametrize("shape", [[2, 3], [2, 0]])
def test_tensorflow_empty_sparse_indices_keep_rank(shape):
    tf = pytest.importorskip("tensorflow")
    np = pytest.importorskip("numpy")
    original = tf.SparseTensor(
        indices=np.empty((0, 2), dtype=np.int64), values=tf.constant([], dtype=tf.float32), dense_shape=shape
    )
    restored = datason.loads(datason.dumps(original))
    assert tuple(restored.indices.shape) == (0, 2)
    assert restored.dense_shape.numpy().tolist() == shape
    assert restored.dtype == tf.float32
