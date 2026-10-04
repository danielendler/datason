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


@pytest.mark.parametrize("library", ["torch", "tensorflow", "jax"])
def test_dense_bfloat16_preserves_dtype_and_values(library):
    backend = pytest.importorskip(library)
    if library == "torch":
        value = backend.tensor([1.0, 2.0], dtype=backend.bfloat16)
    elif library == "tensorflow":
        value = backend.constant([1.0, 2.0], dtype=backend.bfloat16)
    else:
        import jax.numpy as jnp

        value = jnp.array([1.0, 2.0], dtype=jnp.bfloat16)
    restored = datason.loads(datason.dumps(value))
    assert restored.dtype == value.dtype
    assert tuple(restored.shape) == (2,)
    if library == "tensorflow":
        assert restored.numpy().tolist() == [1.0, 2.0]
    else:
        assert restored.tolist() == [1.0, 2.0]


@pytest.mark.parametrize("dtype", ["object", "U1000", "V1000"])
def test_jax_rejects_non_numeric_dtype_before_allocation(monkeypatch, dtype):
    pytest.importorskip("jax")
    np = pytest.importorskip("numpy")
    with monkeypatch.context() as patch:
        patch.setattr(np, "array", lambda *args, **kwargs: pytest.fail("unsupported dtype allocated"))
        with pytest.raises(DeserializationError, match="numeric or boolean"):
            datason.loads(wire("jax.Array", {"data": [1], "shape": [1], "dtype": dtype}))


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


@pytest.mark.parametrize("library,tag", [("torch", "torch.Tensor"), ("tensorflow", "tf.Tensor"), ("jax", "jax.Array")])
def test_legacy_dense_payload_without_shape_uses_inference(library, tag):
    pytest.importorskip(library)
    np = pytest.importorskip("numpy")
    restored = datason.loads(wire(tag, {"data": [[1, 2]], "dtype": "float32"}))
    assert tuple(restored.shape) == (1, 2)
    assert np.asarray(restored).dtype == np.dtype("float32")
    np.testing.assert_array_equal(np.asarray(restored), [[1, 2]])


@pytest.mark.parametrize("dtype,width", [("bool", 1), ("uint8", 1), ("int64", 8), ("float16", 2), ("complex64", 8)])
def test_torch_dtype_budget_compatibility_without_itemsize(monkeypatch, dtype, width):
    torch = pytest.importorskip("torch")
    from datason.plugins import torch as plugin

    actual_getattr = getattr
    with monkeypatch.context() as patch:
        # Model the older supported Torch API: dtype.itemsize was absent.
        patch.setattr(
            plugin,
            "getattr",
            lambda obj, name, default=None: None if name == "itemsize" else actual_getattr(obj, name, default),
            raising=False,
        )
        expected = actual_getattr(torch, dtype)
        assert plugin._dtype_itemsize(expected) == width
        restored = datason.loads(wire("torch.Tensor", {"data": [1], "shape": [1], "dtype": dtype}))
    assert restored.dtype == expected
    assert restored.tolist() == [1]


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"shape": [-1, 2]}, "shape"),
        ({"shape": [2**63, 2]}, "shape"),
        ({"row": []}, "matching lengths"),
        ({"col": None}, "matching lengths"),
        ({"row": [2]}, "row index"),
        ({"col": [True]}, "column index"),
        ({"dtype": "object"}, "numeric or boolean"),
        ({"dtype": {"names": ["field"], "formats": ["f4"]}}, "numeric or boolean"),
    ],
)
def test_scipy_invalid_storage_is_rejected_before_constructor(monkeypatch, changes, message):
    sp = pytest.importorskip("scipy.sparse")
    value = {"row": [0], "col": [0], "data": [1], "shape": [2, 2], "format": "coo", "dtype": "float32"}
    value.update(changes)
    with monkeypatch.context() as patch:
        patch.setattr(sp, "coo_matrix", lambda *args, **kwargs: pytest.fail("invalid sparse allocation reached"))
        with pytest.raises(DeserializationError, match=message):
            datason.loads(wire("scipy.sparse.matrix", value))


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"dense_shape": [-1, 2]}, "shape"),
        ({"dense_shape": [2**63, 2]}, "shape"),
        ({"indices": None}, "matching lengths"),
        ({"values": []}, "matching lengths"),
    ],
)
def test_tensorflow_sparse_shape_and_lengths_precede_allocation(monkeypatch, changes, message):
    tf = pytest.importorskip("tensorflow")
    value = {"indices": [[0, 0]], "values": [1], "dense_shape": [2, 2], "dtype": "float32"}
    value.update(changes)
    with monkeypatch.context() as patch:
        patch.setattr(tf, "constant", lambda *args, **kwargs: pytest.fail("invalid sparse allocation reached"))
        with pytest.raises(DeserializationError, match=message):
            datason.loads(wire("tf.SparseTensor", value))


def test_tensorflow_sparse_buffer_budget_precedes_allocation(monkeypatch):
    tf = pytest.importorskip("tensorflow")
    payload = wire(
        "tf.SparseTensor", {"indices": [[0, 0]] * 40, "values": [1] * 40, "dense_shape": [1, 1], "dtype": "float64"}
    )
    with monkeypatch.context() as patch:
        patch.setattr(tf, "constant", lambda *args, **kwargs: pytest.fail("unbounded sparse allocation reached"))
        with pytest.raises(SecurityError, match="Sparse tensor.*byte budget"):
            datason.loads(payload, max_input_bytes=len(payload.encode()) + 1)


@pytest.mark.parametrize("missing", ["indices", "values", "dense_shape", "dtype"])
def test_tensorflow_incomplete_components_fail_before_eager_export(monkeypatch, missing):
    from types import SimpleNamespace

    pytest.importorskip("tensorflow")
    from datason._errors import PluginError
    from datason._protocols import SerializeContext
    from datason.plugins import tensorflow as plugin

    component = SimpleNamespace(dtype=SimpleNamespace(name="float32"))
    sparse = SimpleNamespace(indices=component, values=component, dense_shape=component)
    if missing == "dtype":
        sparse.values = SimpleNamespace(dtype=None)
    else:
        setattr(sparse, missing, None)
    monkeypatch.setattr(plugin, "_eager_list", lambda obj: pytest.fail("incomplete component exported"))
    with pytest.raises(PluginError, match="Incomplete"):
        plugin._serialize_sparse_tensor(sparse, SerializeContext(config=datason.strict_config()))


def test_tensorflow_symbolic_sparse_export_requires_eager_execution():
    tf = pytest.importorskip("tensorflow")
    from datason._errors import SerializationError

    with tf.Graph().as_default():
        sparse = tf.SparseTensor([[0, 0]], tf.constant([1.0]), [2, 2])
        with pytest.warns(UserWarning, match="requires eager tensors"), pytest.raises(SerializationError):
            datason.dumps(sparse)
