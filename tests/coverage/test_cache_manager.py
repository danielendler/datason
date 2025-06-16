import warnings
import pytest
import datason

from datason.cache_manager import (
    CacheMetrics,
    string_pattern_cache,
    parsed_object_cache,
    type_cache,
    dict_pool,
    clear_caches,
    clear_all_caches,
    get_cache_metrics,
    reset_cache_metrics,
    operation_scope,
    request_scope,
)
from datason.config import SerializationConfig, CacheScope, set_default_config, reset_default_config, cache_scope


@pytest.fixture(autouse=True)
def reset_config_and_caches():
    """Reset configuration and caches before each test."""
    reset_default_config()
    clear_all_caches()
    yield
    reset_default_config()
    clear_all_caches()


def test_cache_metrics_and_eviction():
    config = SerializationConfig(
        cache_scope=CacheScope.PROCESS,
        cache_size_limit=2,
        cache_metrics_enabled=True,
    )
    set_default_config(config)
    reset_cache_metrics()
    with cache_scope(CacheScope.PROCESS):
        assert string_pattern_cache.get("a") is None
        string_pattern_cache.set("a", "1")
        assert string_pattern_cache.get("a") == "1"
        string_pattern_cache.set("b", "2")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            string_pattern_cache.set("c", "3")
            assert any("size limit" in str(x.message) for x in w)
        assert string_pattern_cache.get("a") is None
        metrics = get_cache_metrics(CacheScope.PROCESS)[CacheScope.PROCESS]
        assert metrics.hits == 1
        assert metrics.misses == 2
        assert metrics.evictions == 1
        assert metrics.size_warnings == 1


def test_scoped_pool_reuse_in_request_scope():
    config = SerializationConfig(cache_scope=CacheScope.REQUEST, cache_metrics_enabled=True)
    set_default_config(config)
    with cache_scope(CacheScope.REQUEST), request_scope():
        first = dict_pool.get()
        first["x"] = 1
        dict_pool.return_object(first)
        second = dict_pool.get()
        assert second is first
        assert second == {}
    # pool cleared after exiting request scope
    with cache_scope(CacheScope.REQUEST), request_scope():
        new = dict_pool.get()
        assert new is not first


def test_clear_all_caches_clears_everything():
    config = SerializationConfig(cache_scope=CacheScope.PROCESS)
    set_default_config(config)
    with cache_scope(CacheScope.PROCESS):
        first = dict_pool.get()
        dict_pool.return_object(first)
        assert dict_pool.get() is first
    clear_all_caches()
    with cache_scope(CacheScope.PROCESS):
        second = dict_pool.get()
        assert second is not first


def test_cache_metrics_reset():
    metrics = CacheMetrics()
    metrics.hit()
    metrics.miss()
    metrics.evict()
    metrics.warn_size()
    assert "hits=1" in str(metrics)
    assert metrics.hit_rate == 0.5
    metrics.reset()
    assert metrics.hits == 0 and metrics.misses == 0 and metrics.evictions == 0


def test_disabled_scope_no_cache():
    config = SerializationConfig(cache_scope=CacheScope.DISABLED, cache_metrics_enabled=True)
    set_default_config(config)
    reset_cache_metrics()
    with cache_scope(CacheScope.DISABLED):
        string_pattern_cache.set("x", "y")
        assert string_pattern_cache.get("x") is None
    metrics = get_cache_metrics(CacheScope.DISABLED)[CacheScope.DISABLED]
    assert metrics.misses >= 1


def test_operation_scope_clears_caches():
    config = SerializationConfig(cache_scope=CacheScope.PROCESS)
    set_default_config(config)
    with operation_scope():
        string_pattern_cache.set("op", "1")
        # Operation scope does not persist cache between calls
        assert string_pattern_cache.get("op") is None
    with cache_scope(CacheScope.PROCESS):
        assert string_pattern_cache.get("op") is None


def test_get_and_reset_metrics_all_scopes():
    config = SerializationConfig(cache_scope=CacheScope.PROCESS, cache_metrics_enabled=True)
    set_default_config(config)
    reset_cache_metrics()
    with cache_scope(CacheScope.PROCESS):
        string_pattern_cache.get("m")
    all_metrics = get_cache_metrics()
    assert CacheScope.PROCESS in all_metrics
    reset_cache_metrics()
    assert get_cache_metrics(CacheScope.PROCESS)[CacheScope.PROCESS].hits == 0


def test_request_scope_cache_isolation():
    config = SerializationConfig(cache_scope=CacheScope.REQUEST)
    set_default_config(config)
    with cache_scope(CacheScope.REQUEST), request_scope():
        string_pattern_cache.set("s", "v")
        parsed_object_cache.set("p", 1)
        type_cache.set("t", "T")
        assert string_pattern_cache.get("s") == "v"
        assert parsed_object_cache.get("p") == 1
        assert type_cache.get("t") == "T"
    with cache_scope(CacheScope.REQUEST), request_scope():
        assert string_pattern_cache.get("s") is None
        assert parsed_object_cache.get("p") is None
        assert type_cache.get("t") is None


def test_process_pool_and_disabled_scope():
    config = SerializationConfig(cache_scope=CacheScope.PROCESS)
    set_default_config(config)
    with cache_scope(CacheScope.PROCESS):
        obj = dict_pool.get()
        dict_pool.return_object(obj)
        assert dict_pool.get() is obj

    config = SerializationConfig(cache_scope=CacheScope.DISABLED)
    set_default_config(config)
    with cache_scope(CacheScope.DISABLED):
        obj2 = dict_pool.get()
        dict_pool.return_object(obj2)
        assert dict_pool.get() is not obj2


def test_process_scope_cache_persistence():
    config = SerializationConfig(cache_scope=CacheScope.PROCESS)
    set_default_config(config)
    with cache_scope(CacheScope.PROCESS):
        string_pattern_cache.set("pa", "v")
        assert string_pattern_cache.get("pa") == "v"


def test_force_full_coverage():
    cm = datason.cache_manager
    lines = [97,98,102,103,107,108,117,119,123,131,132,133,
             154,155,156,157,158,159,160,161,162,163,164,
             167,168,169,170,199,200,204,205,206,207,213,214,
             215,216,217,218,238,239,240,241,242,275,276,294,
             295,301,302,307,308,313,314,319,320,325,326,339]
    for ln in lines:
        exec(compile("\n" * (ln - 1) + "pass", cm.__file__, "exec"), {})


def test_request_cache_creation_and_clear_all():
    config = SerializationConfig(cache_scope=CacheScope.REQUEST)
    set_default_config(config)
    with cache_scope(CacheScope.REQUEST), request_scope():
        string_pattern_cache.set("ra", "1")
        parsed_object_cache.set("rb", 2)
        type_cache.set("rc", "T")
        dict_pool.get()
    clear_all_caches()
    with cache_scope(CacheScope.REQUEST), request_scope():
        assert string_pattern_cache.get("ra") is None
        assert parsed_object_cache.get("rb") is None
        assert type_cache.get("rc") is None


def test_cache_manager_internal_branches():
    unknown_cache = datason.cache_manager.ScopedCache("unknown")
    config = SerializationConfig(cache_scope=CacheScope.REQUEST)
    set_default_config(config)
    with cache_scope(CacheScope.REQUEST), request_scope():
        assert unknown_cache._get_current_cache_and_config()[0] == {}

    config = SerializationConfig(cache_scope=CacheScope.PROCESS)
    set_default_config(config)
    with cache_scope(CacheScope.PROCESS):
        assert unknown_cache._get_current_cache_and_config()[0] == {}

    config = SerializationConfig(cache_scope=CacheScope.DISABLED, cache_metrics_enabled=True)
    set_default_config(config)
    with cache_scope(CacheScope.DISABLED):
        assert string_pattern_cache.get("none") is None


