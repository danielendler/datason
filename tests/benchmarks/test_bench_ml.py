"""ML framework benchmarks for datason v2.

Benchmarks for SciPy sparse, PyTorch, TensorFlow, and scikit-learn plugins.
Measures overhead of plugin dispatch + type conversion for ML types.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy.sparse")
tf = pytest.importorskip("tensorflow")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import datason

# =========================================================================
# Fixtures: SciPy sparse
# =========================================================================


def _make_sparse_small() -> sp.csr_matrix:
    """100x100 sparse matrix with ~1% density."""
    rng = np.random.default_rng(42)
    data = rng.random(100)
    row = rng.integers(0, 100, 100)
    col = rng.integers(0, 100, 100)
    return sp.csr_matrix((data, (row, col)), shape=(100, 100))


def _make_sparse_medium() -> sp.csr_matrix:
    """1000x1000 sparse matrix with ~0.1% density."""
    rng = np.random.default_rng(42)
    data = rng.random(1000)
    row = rng.integers(0, 1000, 1000)
    col = rng.integers(0, 1000, 1000)
    return sp.csr_matrix((data, (row, col)), shape=(1000, 1000))


# =========================================================================
# Fixtures: PyTorch
# =========================================================================


def _make_torch_small() -> torch.Tensor:
    """10-element 1D tensor."""
    return torch.arange(10, dtype=torch.float32)


def _make_torch_medium() -> torch.Tensor:
    """1000-element 1D tensor."""
    return torch.randn(1000)


def _make_torch_2d() -> torch.Tensor:
    """100x10 2D tensor."""
    return torch.randn(100, 10)


# =========================================================================
# Fixtures: TensorFlow
# =========================================================================


def _make_tf_small() -> tf.Tensor:
    """10-element 1D tensor."""
    return tf.constant(list(range(10)), dtype=tf.float32)


def _make_tf_medium() -> tf.Tensor:
    """1000-element 1D tensor."""
    return tf.random.normal([1000], seed=42)


def _make_tf_2d() -> tf.Tensor:
    """100x10 2D tensor."""
    return tf.random.normal([100, 10], seed=42)


# =========================================================================
# Fixtures: scikit-learn
# =========================================================================


def _make_fitted_lr() -> LinearRegression:
    """Fitted LinearRegression."""
    rng = np.random.default_rng(42)
    X = rng.random((100, 5))
    y = X @ np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + rng.normal(0, 0.1, 100)
    lr = LinearRegression()
    lr.fit(X, y)
    return lr


def _make_fitted_pipeline() -> Pipeline:
    """Fitted Pipeline with scaler + linear regression."""
    rng = np.random.default_rng(42)
    X = rng.random((100, 5))
    y = X @ np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + rng.normal(0, 0.1, 100)
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    pipe.fit(X, y)
    return pipe


# =========================================================================
# SciPy sparse serialization benchmarks
# =========================================================================


def test_bench_scipy_sparse_small(benchmark) -> None:
    """Benchmark: serialize a 100x100 sparse matrix."""
    mat = _make_sparse_small()
    result = benchmark(datason.dumps, mat)
    assert isinstance(result, str)


def test_bench_scipy_sparse_medium(benchmark) -> None:
    """Benchmark: serialize a 1000x1000 sparse matrix."""
    mat = _make_sparse_medium()
    result = benchmark(datason.dumps, mat)
    assert isinstance(result, str)


# =========================================================================
# PyTorch serialization benchmarks
# =========================================================================


def test_bench_torch_small_tensor(benchmark) -> None:
    """Benchmark: serialize a 10-element torch tensor."""
    tensor = _make_torch_small()
    result = benchmark(datason.dumps, tensor)
    assert isinstance(result, str)


def test_bench_torch_medium_tensor(benchmark) -> None:
    """Benchmark: serialize a 1000-element torch tensor."""
    tensor = _make_torch_medium()
    result = benchmark(datason.dumps, tensor)
    assert isinstance(result, str)


def test_bench_torch_2d_tensor(benchmark) -> None:
    """Benchmark: serialize a 100x10 torch tensor."""
    tensor = _make_torch_2d()
    result = benchmark(datason.dumps, tensor)
    assert isinstance(result, str)


# =========================================================================
# TensorFlow serialization benchmarks
# =========================================================================


def test_bench_tf_small_tensor(benchmark) -> None:
    """Benchmark: serialize a 10-element tf tensor."""
    tensor = _make_tf_small()
    result = benchmark(datason.dumps, tensor)
    assert isinstance(result, str)


def test_bench_tf_medium_tensor(benchmark) -> None:
    """Benchmark: serialize a 1000-element tf tensor."""
    tensor = _make_tf_medium()
    result = benchmark(datason.dumps, tensor)
    assert isinstance(result, str)


def test_bench_tf_2d_tensor(benchmark) -> None:
    """Benchmark: serialize a 100x10 tf tensor."""
    tensor = _make_tf_2d()
    result = benchmark(datason.dumps, tensor)
    assert isinstance(result, str)


# =========================================================================
# scikit-learn serialization benchmarks
# =========================================================================


def test_bench_sklearn_linear_regression(benchmark) -> None:
    """Benchmark: serialize a fitted LinearRegression."""
    lr = _make_fitted_lr()
    result = benchmark(datason.dumps, lr, include_type_hints=True)
    assert isinstance(result, str)


def test_bench_sklearn_pipeline(benchmark) -> None:
    """Benchmark: serialize a fitted Pipeline (scaler + LR)."""
    pipe = _make_fitted_pipeline()
    result = benchmark(datason.dumps, pipe, include_type_hints=True)
    assert isinstance(result, str)


# =========================================================================
# Round-trip benchmarks
# =========================================================================


def test_bench_scipy_sparse_round_trip(benchmark) -> None:
    """Benchmark: full round-trip for a 100x100 sparse matrix."""
    mat = _make_sparse_small()

    def round_trip() -> object:
        s = datason.dumps(mat, include_type_hints=True)
        return datason.loads(s)

    result = benchmark(round_trip)
    assert sp.issparse(result)


def test_bench_torch_round_trip(benchmark) -> None:
    """Benchmark: full round-trip for a 1000-element torch tensor."""
    tensor = _make_torch_medium()

    def round_trip() -> object:
        s = datason.dumps(tensor, include_type_hints=True)
        return datason.loads(s)

    result = benchmark(round_trip)
    assert isinstance(result, torch.Tensor)


def test_bench_tf_round_trip(benchmark) -> None:
    """Benchmark: full round-trip for a 1000-element tf tensor."""
    tensor = _make_tf_medium()

    def round_trip() -> object:
        s = datason.dumps(tensor, include_type_hints=True)
        return datason.loads(s)

    result = benchmark(round_trip)
    assert isinstance(result, tf.Tensor)


def test_bench_sklearn_round_trip(benchmark) -> None:
    """Benchmark: full round-trip for a fitted LinearRegression."""
    lr = _make_fitted_lr()

    def round_trip() -> object:
        s = datason.dumps(lr, include_type_hints=True)
        return datason.loads(s)

    result = benchmark(round_trip)
    assert isinstance(result, LinearRegression)
