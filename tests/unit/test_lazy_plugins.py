"""Deferred dispatch, cached failures and simultaneous first use."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import pytest

from datason._config import SerializationConfig
from datason._errors import PluginError
from datason._protocols import DeserializeContext, SerializeContext
from datason._registry import PluginRegistry
from datason._types import TYPE_METADATA_KEY
from datason.plugins import _lazy


class Foreign:
    __module__ = "torch.test"


class ApplicationSubclass(Foreign):
    pass


class Delegate:
    name = "torch"
    priority = 300

    def can_handle(self, obj):
        return isinstance(obj, Foreign)

    def serialize(self, obj, ctx):
        return {"value": 7}

    def can_deserialize(self, data):
        return data[TYPE_METADATA_KEY] == "torch.test"

    def deserialize(self, data, ctx):
        return 7


def deferred():
    return _lazy.LazyPlugin("torch", 300, "Delegate", ("torch",), ("torch.",))


def test_unrelated_objects_and_tags_do_not_import(monkeypatch):
    monkeypatch.setattr(_lazy.importlib, "import_module", lambda _: pytest.fail("unexpected import"))
    plugin = deferred()
    assert not plugin.can_handle(object())
    for tag in (None, [], "subprocess.Popen", "tensorflow.test", "torchish.test"):
        assert not plugin.can_deserialize({TYPE_METADATA_KEY: tag})
    assert not _lazy.matches_family(type("Odd", (), {"__module__": None})(), ("torch",))


def test_subclass_dispatch_and_wire_namespace_use_only_fixed_target(monkeypatch):
    imports = []
    monkeypatch.setattr(
        _lazy.importlib, "import_module", lambda target: imports.append(target) or SimpleNamespace(Delegate=Delegate)
    )
    plugin = deferred()
    assert plugin.name == "torch" and plugin.priority == 300
    assert plugin.can_handle(ApplicationSubclass())
    assert not plugin.can_handle(object())
    assert plugin.serialize(Foreign(), SerializeContext(SerializationConfig())) == {"value": 7}
    assert plugin.can_deserialize({TYPE_METADATA_KEY: "torch.test"})
    assert not plugin.can_deserialize({TYPE_METADATA_KEY: "torch.unknown"})
    assert plugin.deserialize({TYPE_METADATA_KEY: "torch.test"}, DeserializeContext(SerializationConfig())) == 7
    assert imports == ["datason.plugins.torch"]


def test_deserialization_can_be_first_use(monkeypatch):
    monkeypatch.setattr(_lazy.importlib, "import_module", lambda _: SimpleNamespace(Delegate=Delegate))
    plugin = deferred()
    assert plugin.can_deserialize({TYPE_METADATA_KEY: "torch.test"})
    assert plugin.deserialize({TYPE_METADATA_KEY: "torch.test"}, DeserializeContext(SerializationConfig())) == 7


@pytest.mark.parametrize("first", ["can_handle", "can_deserialize", "serialize", "deserialize"])
def test_successful_activation_bypasses_loading_checks_on_warmed_operations(monkeypatch, first):
    monkeypatch.setattr(_lazy.importlib, "import_module", lambda _: SimpleNamespace(Delegate=Delegate))
    plugin = deferred()
    obj, wire = Foreign(), {TYPE_METADATA_KEY: "torch.test"}
    calls = {
        "can_handle": (obj,),
        "can_deserialize": (wire,),
        "serialize": (obj, SerializeContext(SerializationConfig())),
        "deserialize": (wire, DeserializeContext(SerializationConfig())),
    }
    getattr(plugin, first)(*calls[first])
    monkeypatch.setattr(plugin, "_load", lambda: pytest.fail("warmed operation used the loader"))
    monkeypatch.setattr(_lazy, "matches_family", lambda *_: pytest.fail("warmed operation inspected the family"))
    assert plugin.can_handle(obj) and not plugin.can_handle(object())
    assert plugin.serialize(*calls["serialize"]) == {"value": 7}
    assert plugin.deserialize(*calls["deserialize"]) == 7
    assert plugin.can_deserialize(wire)
    for invalid in ({}, {TYPE_METADATA_KEY: []}, {TYPE_METADATA_KEY: "untrusted.module.Class"}):
        assert not plugin.can_deserialize(invalid)


def test_callbacks_captured_before_activation_remain_valid(monkeypatch):
    monkeypatch.setattr(_lazy.importlib, "import_module", lambda _: SimpleNamespace(Delegate=Delegate))
    plugin = deferred()
    callbacks = (plugin.can_handle, plugin.serialize, plugin.deserialize)
    assert plugin.can_deserialize({TYPE_METADATA_KEY: "torch.test"})
    assert callbacks[0](Foreign())
    assert callbacks[1](Foreign(), SerializeContext(SerializationConfig())) == {"value": 7}
    assert callbacks[2]({TYPE_METADATA_KEY: "torch.test"}, DeserializeContext(SerializationConfig())) == 7


def test_warmed_delegate_errors_keep_registry_warning_and_fallback(monkeypatch):
    class Failing(Delegate):
        def serialize(self, obj, ctx):
            raise PluginError("conversion failed")

    monkeypatch.setattr(_lazy.importlib, "import_module", lambda _: SimpleNamespace(Delegate=Failing))
    plugin, fallback = deferred(), Delegate()
    fallback.name, fallback.priority = "fallback", 400
    registry = PluginRegistry()
    registry.register(plugin)
    registry.register(fallback)
    assert plugin.can_handle(Foreign())
    with pytest.warns(UserWarning, match="Plugin 'torch' failed.*conversion failed"):
        result = registry.find_serializer(Foreign(), SerializeContext(SerializationConfig()))
    assert result == (fallback, {"value": 7})
    assert registry.plugin_count == 2


@pytest.mark.parametrize(
    "missing", [ModuleNotFoundError("torch missing"), ImportError("transitive dependency missing")]
)
def test_unavailable_plugin_is_cached_and_direct_use_reports_failure(monkeypatch, missing):
    imports = []

    def unavailable(target):
        imports.append(target)
        raise missing

    monkeypatch.setattr(_lazy.importlib, "import_module", unavailable)
    plugin = deferred()
    assert not plugin.can_handle(Foreign())
    assert not plugin.can_handle(Foreign())
    assert not plugin.can_deserialize({TYPE_METADATA_KEY: "torch.test"})
    with pytest.raises(PluginError, match="unavailable"):
        plugin.serialize(Foreign(), SerializeContext(SerializationConfig()))
    with pytest.raises(PluginError, match="unavailable"):
        plugin.deserialize({TYPE_METADATA_KEY: "torch.test"}, DeserializeContext(SerializationConfig()))
    assert len(imports) == 1


def test_unexpected_import_error_is_visible_and_can_be_retried(monkeypatch):
    plugin = deferred()

    def broken(_):
        raise RuntimeError("library initialization failed")

    monkeypatch.setattr(_lazy.importlib, "import_module", broken)
    with pytest.raises(RuntimeError, match="initialization failed"):
        plugin.can_handle(Foreign())
    monkeypatch.setattr(_lazy.importlib, "import_module", lambda _: SimpleNamespace(Delegate=Delegate))
    assert plugin.can_handle(Foreign())


def test_concurrent_first_serialization_and_loading_construct_once(monkeypatch):
    plugin = deferred()
    gate = Barrier(8)
    entered = Event()
    release = Event()
    queued = Event()
    count = []
    waiters = []
    count_lock = Lock()

    class ObservedLock:
        def __init__(self):
            self.lock = Lock()

        def __enter__(self):
            with count_lock:
                waiters.append(True)
                if len(waiters) == 8:
                    queued.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    monkeypatch.setattr(plugin, "_lock", ObservedLock())
    cold_can_handle, cold_can_deserialize = plugin.can_handle, plugin.can_deserialize

    def slow_import(target):
        with count_lock:
            count.append(target)
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(Delegate=Delegate)

    monkeypatch.setattr(_lazy.importlib, "import_module", slow_import)

    def dispatch(index):
        gate.wait(timeout=5)
        if index % 2:
            assert cold_can_handle(Foreign())
            return plugin.serialize(Foreign(), SerializeContext(SerializationConfig()))["value"]
        assert cold_can_deserialize({TYPE_METADATA_KEY: "torch.test"})
        return plugin.deserialize({TYPE_METADATA_KEY: "torch.test"}, DeserializeContext(SerializationConfig()))

    with ThreadPoolExecutor(max_workers=8) as workers:
        pending = [workers.submit(dispatch, index) for index in range(8)]
        assert entered.wait(5)
        assert queued.wait(5)
        release.set()
        assert [item.result(timeout=5) for item in pending] == [7] * 8
    assert count == ["datason.plugins.torch"]


def test_priority_and_registration_count_survive_activation(monkeypatch):
    monkeypatch.setattr(_lazy.importlib, "import_module", lambda _: SimpleNamespace(Delegate=Delegate))
    registry = PluginRegistry()
    lazy = deferred()
    registry.register(lazy)
    override = Delegate()
    override.name = "application"
    override.priority = 50
    registry.register(override)
    assert registry.find_serializer(Foreign(), SerializeContext(SerializationConfig()))[0] is override
    assert not lazy._attempted
    assert (
        registry.find_deserializer({TYPE_METADATA_KEY: "torch.test"}, DeserializeContext(SerializationConfig()))[0]
        is override
    )
    assert registry.plugin_count == 2
    assert lazy.can_handle(Foreign())
    assert registry.plugin_count == 2


def test_registration_probes_only_available_top_level_dependencies(monkeypatch):
    import datason.plugins as plugins

    registry = PluginRegistry()
    seen = []
    monkeypatch.setattr(plugins, "default_registry", registry)
    monkeypatch.setattr(plugins, "find_spec", lambda root: seen.append(root) or (object() if root == "scipy" else None))
    plugins._register_builtins()
    assert registry.plugin_count == 7  # five stdlib, SciPy, metadata-only ml_misc
    assert seen == ["numpy", "pandas", "scipy", "torch", "tensorflow", "sklearn", "pydantic"]
