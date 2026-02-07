# Changelog

All notable changes to datason are documented here. This project uses [Semantic Versioning](https://semver.org/).

## 2.0.0a1 (Unreleased)

Complete ground-up rewrite with plugin-based architecture.

### Added

- **Plugin architecture**: Every non-JSON type handled by a `TypePlugin` with priority-based dispatch
- **5-function API**: `dumps`, `loads`, `dump`, `load`, `config` -- matches `json` module interface
- **10 built-in plugins**: datetime, UUID, Decimal, Path, NumPy, Pandas, PyTorch, TensorFlow, scikit-learn, SciPy sparse
- **SerializationConfig**: Frozen dataclass with `DateFormat`, `NanHandling`, `DataFrameOrient` enums
- **Config presets**: `ml_config()`, `api_config()`, `strict_config()`, `performance_config()`
- **Context manager**: `datason.config()` for thread-safe scoped configuration via ContextVar
- **Security features**: PII redaction (field + pattern), integrity verification (hash/HMAC), depth/size/circular reference limits
- **Custom plugin support**: Implement `TypePlugin` protocol and register with `default_registry.register()`
- **Thread safety**: `threading.Lock` on global registry, `contextvars.ContextVar` for config scoping
- **AI agent support**: `llms.txt` and `llms-full.txt` for machine-readable API documentation
- **MkDocs documentation**: Full documentation site with getting started, API reference, configuration, security, and plugin guides
- **542 tests**: Unit tests, integration tests, property-based tests (Hypothesis), benchmarks
- **93% code coverage** with branch coverage enabled

### Changed (vs v1)

- Python 3.10+ minimum (was 3.8+)
- Plugin-based dispatch replaces monolithic if/elif type chains
- API reduced from 100+ functions to 5
- `__init__.py` reduced from 678 lines to 36 lines
- All modules under 500-line hard limit
- All functions under 50-line hard limit

### Architecture

```
datason/
  __init__.py          # 36 lines, 13 exports
  _core.py             # Serialization engine
  _deserialize.py      # Deserialization engine
  _config.py           # Config dataclass + presets
  _registry.py         # Plugin dispatch
  _protocols.py        # TypePlugin protocol
  _cache.py            # Thread-safe caching
  _errors.py           # Error hierarchy
  _types.py            # Type constants
  plugins/             # 10 type handler plugins
  security/            # Redaction + integrity
```
