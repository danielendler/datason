# Scaffold Datason v2 Project Structure

Create the v2 project structure from scratch. This command sets up the entire directory layout based on the architecture in CLAUDE.md and the learnings from LEARNINGS_AND_STRATEGY.md.

## Step 1: Create directory structure
```
datason/
  __init__.py
  _core.py
  _deserialize.py
  _config.py
  _types.py
  _protocols.py
  _registry.py
  _cache.py
  _errors.py
  py.typed              (PEP 561 marker)
  plugins/
    __init__.py
    datetime.py
    uuid.py
    decimal.py
    path.py
    numpy.py
    pandas.py
    ml_torch.py
    ml_tensorflow.py
    ml_sklearn.py
    ml_misc.py
  security/
    __init__.py
    redaction.py
    integrity.py
    pickle_bridge.py
tests/
  __init__.py
  conftest.py
  unit/
    __init__.py
    test_core.py
    test_deserialize.py
    test_config.py
    test_registry.py
    test_cache.py
    test_errors.py
    test_plugin_datetime.py
    test_plugin_uuid.py
    test_plugin_numpy.py
    test_plugin_pandas.py
  integration/
    __init__.py
    test_round_trip.py
    test_ml_workflows.py
  benchmarks/
    __init__.py
    test_bench_core.py
    conftest.py
```

## Step 2: Write _errors.py
Define the error hierarchy:
- DatasonError (base)
- SecurityError
- SerializationError
- DeserializationError
- PluginError

## Step 3: Write _types.py
Define type constants:
- JSON_BASIC_TYPES tuple
- TYPE_METADATA_KEY = "__datason_type__"
- VALUE_METADATA_KEY = "__datason_value__"

## Step 4: Write _protocols.py
Define the TypePlugin Protocol with:
- name, priority attributes
- can_handle(), serialize(), can_deserialize(), deserialize() methods
- SerializeContext and DeserializeContext dataclasses

## Step 5: Write _config.py
Port the SerializationConfig dataclass from v1 config.py, but simplified.
Include preset factory functions.

## Step 6: Write _registry.py
Implement PluginRegistry class with:
- register(plugin) method
- dispatch_serialize(obj) -> plugin
- dispatch_deserialize(data) -> plugin
- Priority-sorted plugin list

## Step 7: Write _cache.py
Thread-safe caching with threading.Lock on all mutable state.

## Step 8: Write _core.py and _deserialize.py
Minimal recursive engines that delegate to plugins via the registry.
Must be < 300 lines each.

## Step 9: Write __init__.py
Exports: dumps, loads, dump, load, config, SerializationConfig
Must be < 50 lines.

## Step 10: Write tests/conftest.py
Single conftest with explicit fixtures for cache clearing and state reset.

## Step 11: Verify
Run `/quality-check` to validate the scaffold passes all gates.
