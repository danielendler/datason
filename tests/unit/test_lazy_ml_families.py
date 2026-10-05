"""Load only the requested miscellaneous ML family, including direct codec use."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest

from datason._errors import PluginError
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY
from datason.plugins import ml_misc as misc


@pytest.fixture()
def isolated(monkeypatch):
    monkeypatch.setattr(misc, "_loaded", set())
    for name in ("POLARS", "JAX", "CATBOOST", "OPTUNA", "PLOTLY"):
        monkeypatch.setattr(misc, "_HAS_" + name, False)
    for name in ("pl", "jax", "jnp", "catboost", "optuna", "go"):
        monkeypatch.setattr(misc, name, None)


@pytest.mark.parametrize("family", ["polars", "jax", "catboost", "optuna", "plotly"])
def test_only_selected_family_imported_and_cached(monkeypatch, isolated, family):
    imported = []
    monkeypatch.setattr(misc.importlib, "import_module", lambda name: imported.append(name) or object())
    misc._load_framework(family)
    misc._load_framework(family)
    expected = ["jax", "jax.numpy"] if family == "jax" else ["plotly.graph_objects" if family == "plotly" else family]
    assert imported == expected
    assert getattr(misc, "_HAS_" + family.upper())
    assert misc._loaded == {family}


@pytest.mark.parametrize("family", ["polars", "jax", "catboost", "optuna", "plotly"])
def test_missing_family_is_cached_without_importing_other_families(monkeypatch, isolated, family):
    imported = []

    def unavailable(name):
        imported.append(name)
        raise ImportError("optional library missing")

    monkeypatch.setattr(misc.importlib, "import_module", unavailable)
    misc._load_framework(family)
    misc._load_framework(family)
    assert len(imported) == 1
    assert not getattr(misc, "_HAS_" + family.upper())


def test_unknown_family_fails_without_importing_or_caching(monkeypatch, isolated):
    monkeypatch.setattr(misc.importlib, "import_module", lambda _: pytest.fail("unknown family imported"))
    with pytest.raises(ValueError, match="Unknown optional ML family: unknown"):
        misc._load_framework("unknown")
    assert misc._loaded == set()
    assert not any(getattr(misc, "_HAS_" + name.upper()) for name in misc._FRAMEWORKS)


def test_initialization_failure_remains_visible_and_can_be_retried(monkeypatch, isolated):
    imported = []

    def broken(name):
        imported.append(name)
        raise RuntimeError("library initialization failed")

    monkeypatch.setattr(misc.importlib, "import_module", broken)
    with pytest.raises(RuntimeError, match="library initialization failed"):
        misc._load_framework("polars")
    assert not misc._HAS_POLARS and misc.pl is None
    assert misc._loaded == set()
    module = object()
    monkeypatch.setattr(misc.importlib, "import_module", lambda name: imported.append(name) or module)
    misc._load_framework("polars")
    misc._load_framework("polars")
    assert imported == ["polars", "polars"]
    assert misc.pl is module and misc._HAS_POLARS
    assert misc._loaded == {"polars"}


def test_missing_jax_numpy_does_not_publish_partial_initialization(monkeypatch, isolated):
    imported = []

    def partial(name):
        imported.append(name)
        if name == "jax.numpy":
            raise ImportError("transitive dependency missing")
        return object()

    monkeypatch.setattr(misc.importlib, "import_module", partial)
    misc._load_framework("jax")
    misc._load_framework("jax")
    assert imported == ["jax", "jax.numpy"]
    assert not misc._HAS_JAX and misc.jnp is None
    assert misc._loaded == {"jax"}


def test_jaxlib_and_application_subclasses_route_to_jax(monkeypatch, isolated):
    selected = []
    monkeypatch.setattr(misc, "_load_framework", selected.append)
    base = type("ArrayImpl", (), {"__module__": "jaxlib._jax"})
    child = type("AppArray", (base,), {"__module__": "application"})
    misc._load_for_object(child())
    assert selected == ["jax"]
    misc._load_for_object(object())
    assert selected == ["jax"]
    # An alias on an application subclass must not hide its actual base family.
    alias = type("AliasArray", (base,), {"__module__": "polars"})
    misc._load_for_object(alias())
    assert selected == ["jax", "polars", "jax"]


def test_cached_families_skip_class_scans_including_unavailable_results(monkeypatch, isolated):
    misc._loaded.update(misc._FRAMEWORKS)
    monkeypatch.setattr(misc, "matches_family", lambda *_: pytest.fail("cached family rescanned"))
    monkeypatch.setattr(misc, "_load_framework", lambda _: pytest.fail("cached family reloaded"))
    assert not misc.MlMiscPlugin().can_handle(object())


def test_cached_family_does_not_hide_a_new_family_on_an_application_subclass(monkeypatch, isolated):
    misc._loaded.add("polars")
    selected = []
    monkeypatch.setattr(misc, "_load_framework", selected.append)
    base = type("ArrayImpl", (), {"__module__": "jaxlib._jax"})
    alias = type("AppArray", (base,), {"__module__": "polars"})
    misc._load_for_object(alias())
    assert selected == ["jax"]


@pytest.mark.parametrize("tag", ["catboost.Model", "optuna.Study"])
def test_metadata_only_tags_do_not_import_frameworks(monkeypatch, isolated, tag):
    monkeypatch.setattr(misc.importlib, "import_module", lambda _: pytest.fail("metadata needs no framework"))
    result = misc.MlMiscPlugin().deserialize({TYPE_METADATA_KEY: tag, VALUE_METADATA_KEY: {"params": {}}}, None)
    assert result == {"params": {}}


@pytest.mark.parametrize(
    "restore,args",
    [
        (misc._reconstruct_polars_df, ({},)),
        (misc._reconstruct_polars_series, ({},)),
        (misc._reconstruct_jax_array, ({}, None)),
        (misc._reconstruct_plotly_figure, ({},)),
    ],
)
def test_direct_reconstruction_reports_missing_library(monkeypatch, isolated, restore, args):
    def unavailable(_):
        raise ImportError("missing")

    monkeypatch.setattr(misc.importlib, "import_module", unavailable)
    with pytest.raises(PluginError, match="not installed"):
        restore(*args)


def test_concurrent_misc_first_import_runs_once(monkeypatch, isolated):
    entered, release = Event(), Event()
    gate = Barrier(8)
    imported = []

    def slow_import(name):
        imported.append(name)
        entered.set()
        assert release.wait(5)
        return object()

    monkeypatch.setattr(misc.importlib, "import_module", slow_import)

    def load(_):
        gate.wait(timeout=5)
        misc._load_framework("polars")

    with ThreadPoolExecutor(max_workers=8) as workers:
        pending = [workers.submit(load, n) for n in range(8)]
        assert entered.wait(5)
        release.set()
        for item in pending:
            item.result(timeout=5)
    assert imported == ["polars"]
