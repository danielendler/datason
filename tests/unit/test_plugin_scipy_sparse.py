"""Tests for the SciPy sparse plugin."""

from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp

import datason
from datason._config import SerializationConfig
from datason._errors import PluginError
from datason._protocols import DeserializeContext, SerializeContext
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY
from datason.plugins.scipy_sparse import ScipySparsePlugin


@pytest.fixture()
def plugin() -> ScipySparsePlugin:
    return ScipySparsePlugin()


@pytest.fixture()
def ser_ctx() -> SerializeContext:
    return SerializeContext(config=SerializationConfig())


@pytest.fixture()
def deser_ctx() -> DeserializeContext:
    return DeserializeContext(config=SerializationConfig())


@pytest.fixture()
def sample_csr() -> sp.csr_matrix:
    return sp.csr_matrix(np.array([[1, 0, 2], [0, 0, 3], [4, 5, 0]]))


@pytest.fixture()
def sample_csc() -> sp.csc_matrix:
    return sp.csc_matrix(np.array([[1, 0, 2], [0, 0, 3], [4, 5, 0]]))


@pytest.fixture()
def sample_coo() -> sp.coo_matrix:
    return sp.coo_matrix(np.array([[1, 0, 2], [0, 0, 3], [4, 5, 0]]))


class TestScipySparsePluginProperties:
    def test_name(self, plugin: ScipySparsePlugin) -> None:
        assert plugin.name == "scipy_sparse"

    def test_priority(self, plugin: ScipySparsePlugin) -> None:
        assert plugin.priority == 250


class TestCanHandle:
    def test_csr_matrix(self, plugin: ScipySparsePlugin, sample_csr: sp.csr_matrix) -> None:
        assert plugin.can_handle(sample_csr)

    def test_csc_matrix(self, plugin: ScipySparsePlugin, sample_csc: sp.csc_matrix) -> None:
        assert plugin.can_handle(sample_csc)

    def test_coo_matrix(self, plugin: ScipySparsePlugin, sample_coo: sp.coo_matrix) -> None:
        assert plugin.can_handle(sample_coo)

    def test_rejects_dense_array(self, plugin: ScipySparsePlugin) -> None:
        assert not plugin.can_handle(np.array([1, 2, 3]))

    def test_rejects_python_list(self, plugin: ScipySparsePlugin) -> None:
        assert not plugin.can_handle([1, 2, 3])


class TestSerialize:
    def test_csr_with_hints(self, plugin: ScipySparsePlugin, sample_csr: sp.csr_matrix) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        result = plugin.serialize(sample_csr, ctx)
        assert result[TYPE_METADATA_KEY] == "scipy.sparse.matrix"
        assert result[VALUE_METADATA_KEY]["format"] == "csr"
        assert result[VALUE_METADATA_KEY]["shape"] == [3, 3]

    def test_csc_format_recorded(self, plugin: ScipySparsePlugin, sample_csc: sp.csc_matrix) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        result = plugin.serialize(sample_csc, ctx)
        assert result[VALUE_METADATA_KEY]["format"] == "csc"

    def test_coo_format_recorded(self, plugin: ScipySparsePlugin, sample_coo: sp.coo_matrix) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        result = plugin.serialize(sample_coo, ctx)
        assert result[VALUE_METADATA_KEY]["format"] == "coo"

    def test_without_hints(self, plugin: ScipySparsePlugin, sample_csr: sp.csr_matrix) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        result = plugin.serialize(sample_csr, ctx)
        assert isinstance(result, dict)
        assert "format" in result
        assert "row" in result
        assert "col" in result

    def test_dtype_preserved(self, plugin: ScipySparsePlugin) -> None:
        mat = sp.csr_matrix(np.array([[1.5, 0], [0, 2.5]], dtype=np.float32))
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        result = plugin.serialize(mat, ctx)
        assert result[VALUE_METADATA_KEY]["dtype"] == "float32"

    def test_data_values(self, plugin: ScipySparsePlugin, sample_csr: sp.csr_matrix) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        result = plugin.serialize(sample_csr, ctx)
        value = result[VALUE_METADATA_KEY]
        # Reconstruct and verify data integrity
        coo = sp.coo_matrix((value["data"], (value["row"], value["col"])), shape=value["shape"])
        np.testing.assert_array_equal(coo.toarray(), sample_csr.toarray())


class TestCanDeserialize:
    def test_sparse_matrix(self, plugin: ScipySparsePlugin) -> None:
        data = {TYPE_METADATA_KEY: "scipy.sparse.matrix", VALUE_METADATA_KEY: {}}
        assert plugin.can_deserialize(data)

    def test_rejects_non_scipy(self, plugin: ScipySparsePlugin) -> None:
        data = {TYPE_METADATA_KEY: "numpy.ndarray", VALUE_METADATA_KEY: {}}
        assert not plugin.can_deserialize(data)

    def test_rejects_missing_key(self, plugin: ScipySparsePlugin) -> None:
        assert not plugin.can_deserialize({"foo": "bar"})


class TestDeserialize:
    def test_csr_reconstructed(
        self, plugin: ScipySparsePlugin, deser_ctx: DeserializeContext, sample_csr: sp.csr_matrix
    ) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        serialized = plugin.serialize(sample_csr, ctx)
        result = plugin.deserialize(serialized, deser_ctx)
        assert sp.issparse(result)
        assert result.format == "csr"
        np.testing.assert_array_equal(result.toarray(), sample_csr.toarray())

    def test_csc_reconstructed(
        self, plugin: ScipySparsePlugin, deser_ctx: DeserializeContext, sample_csc: sp.csc_matrix
    ) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        serialized = plugin.serialize(sample_csc, ctx)
        result = plugin.deserialize(serialized, deser_ctx)
        assert result.format == "csc"

    def test_coo_reconstructed(
        self, plugin: ScipySparsePlugin, deser_ctx: DeserializeContext, sample_coo: sp.coo_matrix
    ) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        serialized = plugin.serialize(sample_coo, ctx)
        result = plugin.deserialize(serialized, deser_ctx)
        assert result.format == "coo"

    def test_bad_value_raises(self, plugin: ScipySparsePlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "scipy.sparse.matrix", VALUE_METADATA_KEY: "not a dict"}
        with pytest.raises(PluginError, match="Expected dict"):
            plugin.deserialize(data, deser_ctx)

    def test_unknown_type_raises(self, plugin: ScipySparsePlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "scipy.sparse.unknown", VALUE_METADATA_KEY: {}}
        with pytest.raises(PluginError, match="Unknown scipy.sparse type"):
            plugin.deserialize(data, deser_ctx)


class TestRoundTrip:
    def test_csr_matrix(self) -> None:
        mat = sp.csr_matrix(np.array([[1, 0, 2], [0, 0, 3]]))
        s = datason.dumps(mat, include_type_hints=True)
        result = datason.loads(s)
        assert sp.issparse(result)
        np.testing.assert_array_equal(result.toarray(), mat.toarray())

    def test_csc_matrix(self) -> None:
        mat = sp.csc_matrix(np.array([[1, 0], [0, 3], [4, 0]]))
        s = datason.dumps(mat, include_type_hints=True)
        result = datason.loads(s)
        assert result.format == "csc"

    def test_coo_matrix(self) -> None:
        mat = sp.coo_matrix(np.eye(3))
        s = datason.dumps(mat, include_type_hints=True)
        result = datason.loads(s)
        assert result.format == "coo"

    def test_sparse_in_dict(self) -> None:
        data = {"matrix": sp.csr_matrix(np.eye(3)), "label": "identity"}
        s = datason.dumps(data, include_type_hints=True)
        result = datason.loads(s)
        assert sp.issparse(result["matrix"])
        assert result["label"] == "identity"

    def test_json_valid(self) -> None:
        mat = sp.csr_matrix(np.array([[1.5, 0], [0, 2.5]]))
        s = datason.dumps(mat, include_type_hints=True)
        parsed = json.loads(s)
        assert isinstance(parsed, dict)

    def test_float32_dtype_preserved(self) -> None:
        mat = sp.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.float32))
        s = datason.dumps(mat, include_type_hints=True)
        result = datason.loads(s)
        assert result.dtype == np.float32
