"""Tests for the PyTorch plugin."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

import datason
from datason._config import SerializationConfig
from datason._errors import PluginError
from datason._protocols import DeserializeContext, SerializeContext
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY
from datason.plugins.torch import TorchPlugin


@pytest.fixture()
def plugin() -> TorchPlugin:
    return TorchPlugin()


@pytest.fixture()
def ser_ctx() -> SerializeContext:
    return SerializeContext(config=SerializationConfig())


@pytest.fixture()
def deser_ctx() -> DeserializeContext:
    return DeserializeContext(config=SerializationConfig())


class TestTorchPluginProperties:
    def test_name(self, plugin: TorchPlugin) -> None:
        assert plugin.name == "torch"

    def test_priority(self, plugin: TorchPlugin) -> None:
        assert plugin.priority == 300


class TestCanHandle:
    def test_tensor(self, plugin: TorchPlugin) -> None:
        assert plugin.can_handle(torch.tensor([1.0, 2.0, 3.0]))

    def test_device(self, plugin: TorchPlugin) -> None:
        assert plugin.can_handle(torch.device("cpu"))

    def test_dtype(self, plugin: TorchPlugin) -> None:
        assert plugin.can_handle(torch.float32)

    def test_size(self, plugin: TorchPlugin) -> None:
        assert plugin.can_handle(torch.Size([3, 4]))

    def test_rejects_python_list(self, plugin: TorchPlugin) -> None:
        assert not plugin.can_handle([1.0, 2.0])

    def test_rejects_numpy_array(self, plugin: TorchPlugin) -> None:
        import numpy as np

        assert not plugin.can_handle(np.array([1.0]))


class TestSerializeTensor:
    def test_1d_with_hints(self, plugin: TorchPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = plugin.serialize(tensor, ctx)
        assert result[TYPE_METADATA_KEY] == "torch.Tensor"
        assert result[VALUE_METADATA_KEY]["data"] == [1.0, 2.0, 3.0]
        assert result[VALUE_METADATA_KEY]["dtype"] == "float32"
        assert result[VALUE_METADATA_KEY]["shape"] == [3]

    def test_2d_with_hints(self, plugin: TorchPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        tensor = torch.tensor([[1, 2], [3, 4]])
        result = plugin.serialize(tensor, ctx)
        assert result[VALUE_METADATA_KEY]["shape"] == [2, 2]

    def test_dtype_preserved(self, plugin: TorchPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        for dtype_name, dtype in [("float64", torch.float64), ("int32", torch.int32)]:
            tensor = torch.tensor([1, 2, 3], dtype=dtype)
            result = plugin.serialize(tensor, ctx)
            assert result[VALUE_METADATA_KEY]["dtype"] == dtype_name

    def test_without_hints(self, plugin: TorchPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        tensor = torch.tensor([1.0, 2.0])
        result = plugin.serialize(tensor, ctx)
        assert result == [1.0, 2.0]

    def test_requires_grad_handled(self, plugin: TorchPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        result = plugin.serialize(tensor, ctx)
        assert result[VALUE_METADATA_KEY]["data"] == [1.0, 2.0]

    def test_device_recorded(self, plugin: TorchPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        tensor = torch.tensor([1.0])
        result = plugin.serialize(tensor, ctx)
        assert result[VALUE_METADATA_KEY]["device"] == "cpu"


class TestSerializeDevice:
    def test_cpu_with_hints(self, plugin: TorchPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        result = plugin.serialize(torch.device("cpu"), ctx)
        assert result[TYPE_METADATA_KEY] == "torch.device"
        assert result[VALUE_METADATA_KEY] == "cpu"

    def test_without_hints(self, plugin: TorchPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        result = plugin.serialize(torch.device("cpu"), ctx)
        assert result == "cpu"


class TestSerializeDtype:
    def test_float32_with_hints(self, plugin: TorchPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        result = plugin.serialize(torch.float32, ctx)
        assert result[TYPE_METADATA_KEY] == "torch.dtype"
        assert result[VALUE_METADATA_KEY] == "float32"

    def test_int64(self, plugin: TorchPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        result = plugin.serialize(torch.int64, ctx)
        assert result[VALUE_METADATA_KEY] == "int64"


class TestSerializeSize:
    def test_with_hints(self, plugin: TorchPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        result = plugin.serialize(torch.Size([3, 4, 5]), ctx)
        assert result[TYPE_METADATA_KEY] == "torch.Size"
        assert result[VALUE_METADATA_KEY] == [3, 4, 5]


class TestCanDeserialize:
    def test_tensor(self, plugin: TorchPlugin) -> None:
        data = {TYPE_METADATA_KEY: "torch.Tensor", VALUE_METADATA_KEY: {}}
        assert plugin.can_deserialize(data)

    def test_device(self, plugin: TorchPlugin) -> None:
        data = {TYPE_METADATA_KEY: "torch.device", VALUE_METADATA_KEY: "cpu"}
        assert plugin.can_deserialize(data)

    def test_dtype(self, plugin: TorchPlugin) -> None:
        data = {TYPE_METADATA_KEY: "torch.dtype", VALUE_METADATA_KEY: "float32"}
        assert plugin.can_deserialize(data)

    def test_size(self, plugin: TorchPlugin) -> None:
        data = {TYPE_METADATA_KEY: "torch.Size", VALUE_METADATA_KEY: [3, 4]}
        assert plugin.can_deserialize(data)

    def test_rejects_non_torch(self, plugin: TorchPlugin) -> None:
        data = {TYPE_METADATA_KEY: "numpy.ndarray", VALUE_METADATA_KEY: {}}
        assert not plugin.can_deserialize(data)

    def test_rejects_missing_key(self, plugin: TorchPlugin) -> None:
        assert not plugin.can_deserialize({"foo": "bar"})


class TestDeserialize:
    def test_tensor_1d(self, plugin: TorchPlugin, deser_ctx: DeserializeContext) -> None:
        data = {
            TYPE_METADATA_KEY: "torch.Tensor",
            VALUE_METADATA_KEY: {"data": [1.0, 2.0, 3.0], "dtype": "float32", "shape": [3]},
        }
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_tensor_2d(self, plugin: TorchPlugin, deser_ctx: DeserializeContext) -> None:
        data = {
            TYPE_METADATA_KEY: "torch.Tensor",
            VALUE_METADATA_KEY: {"data": [[1, 2], [3, 4]], "dtype": "int64", "shape": [2, 2]},
        }
        result = plugin.deserialize(data, deser_ctx)
        assert result.shape == torch.Size([2, 2])
        assert result.dtype == torch.int64

    def test_device(self, plugin: TorchPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "torch.device", VALUE_METADATA_KEY: "cpu"}
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, torch.device)
        assert str(result) == "cpu"

    def test_dtype(self, plugin: TorchPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "torch.dtype", VALUE_METADATA_KEY: "float32"}
        result = plugin.deserialize(data, deser_ctx)
        assert result == torch.float32

    def test_size(self, plugin: TorchPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "torch.Size", VALUE_METADATA_KEY: [3, 4, 5]}
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, torch.Size)
        assert list(result) == [3, 4, 5]

    def test_tensor_bad_value_raises(self, plugin: TorchPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "torch.Tensor", VALUE_METADATA_KEY: "not a dict"}
        with pytest.raises(PluginError, match="Expected dict"):
            plugin.deserialize(data, deser_ctx)

    def test_unknown_type_raises(self, plugin: TorchPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "torch.unknown", VALUE_METADATA_KEY: {}}
        with pytest.raises(PluginError, match="Unknown torch type"):
            plugin.deserialize(data, deser_ctx)


class TestRoundTrip:
    def test_tensor_1d(self) -> None:
        tensor = torch.tensor([1.0, 2.0, 3.0])
        s = datason.dumps(tensor, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, tensor)

    def test_tensor_2d_float64(self) -> None:
        tensor = torch.randn(3, 4, dtype=torch.float64)
        s = datason.dumps(tensor, include_type_hints=True)
        result = datason.loads(s)
        assert result.shape == torch.Size([3, 4])
        assert result.dtype == torch.float64
        assert torch.allclose(result, tensor)

    def test_device_roundtrip(self) -> None:
        device = torch.device("cpu")
        s = datason.dumps(device, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result, torch.device)

    def test_tensor_in_dict(self) -> None:
        data = {"weights": torch.tensor([0.1, 0.9]), "label": "model"}
        s = datason.dumps(data, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result["weights"], torch.Tensor)
        assert result["label"] == "model"

    def test_json_valid(self) -> None:
        tensor = torch.randn(5)
        s = datason.dumps(tensor, include_type_hints=True)
        parsed = json.loads(s)
        assert isinstance(parsed, dict)

    def test_mixed_torch_types(self) -> None:
        data = {
            "tensor": torch.tensor([1.0]),
            "device": torch.device("cpu"),
        }
        s = datason.dumps(data, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result["tensor"], torch.Tensor)
        assert isinstance(result["device"], torch.device)

    def test_size_via_plugin_direct(self) -> None:
        """torch.Size inherits from tuple, so core handles it as a sequence.
        Plugin serialization works directly but not via dumps (tuple fast-path)."""
        from datason.plugins.torch import TorchPlugin

        plugin = TorchPlugin()
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        result = plugin.serialize(torch.Size([2, 3]), ctx)
        assert result[TYPE_METADATA_KEY] == "torch.Size"

    def test_bool_tensor(self) -> None:
        tensor = torch.tensor([True, False, True])
        s = datason.dumps(tensor, include_type_hints=True)
        result = datason.loads(s)
        assert result.dtype == torch.bool
