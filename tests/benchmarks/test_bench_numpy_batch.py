"""Regression benchmarks for the small and batched NumPy conversion paths."""

import pytest

np = pytest.importorskip("numpy")

import datason


@pytest.mark.parametrize("size", [4, 32, 4096])
def test_bench_complex_array_conversion(benchmark, size):
    values = np.arange(size, dtype=np.float32)
    original = (values + 1j * values).astype(np.complex64)
    wire = benchmark(datason.dumps, original)
    restored = datason.loads(wire)
    assert restored.dtype == original.dtype
    np.testing.assert_array_equal(restored, original)


def test_bench_untagged_numeric_array(benchmark):
    original = np.arange(4096, dtype=np.float32)
    wire = benchmark(datason.dumps, original, include_type_hints=False)
    assert datason.loads(wire) == original.tolist()


def test_bench_numeric_array_reconstruction(benchmark):
    original = np.arange(4096, dtype=np.float32)
    wire = datason.dumps(original)
    restored = benchmark(datason.loads, wire)
    assert restored.dtype == original.dtype
    np.testing.assert_array_equal(restored, original)
