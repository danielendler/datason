# Changelog

All notable changes to datason are documented here. This project uses [Semantic Versioning](https://semver.org/).

## 2.0.0a1 (2026-02-07)

Complete ground-up rewrite with plugin-based architecture.

### Added

- **Plugin architecture**: Every non-JSON type handled by a `TypePlugin` with priority-based dispatch
- **5-function API**: `dumps`, `loads`, `dump`, `load`, `config` -- matches `json` module interface
- **13 built-in plugins**: datetime, UUID, Decimal, Path, NumPy, Pandas, SciPy sparse, PyTorch, TensorFlow, scikit-learn, + misc (Polars, JAX, CatBoost, Optuna, Plotly)
- **SerializationConfig**: Frozen dataclass with `DateFormat`, `NanHandling`, `DataFrameOrient` enums
- **Config presets**: `ml_config()`, `api_config()`, `strict_config()`, `performance_config()`
- **Context manager**: `datason.config()` for thread-safe scoped configuration via ContextVar
- **Security features**: PII redaction (field + pattern), integrity verification (hash/HMAC), pickle bridge (safe pickle-to-JSON conversion), depth/size/circular reference limits
- **Custom plugin support**: Implement `TypePlugin` protocol and register with `default_registry.register()`
- **Thread safety**: `threading.Lock` on global registry, `contextvars.ContextVar` for config scoping
- **AI agent support**: `llms.txt` and `llms-full.txt` for machine-readable API documentation
- **MkDocs documentation**: Full documentation site deployed to GitHub Pages
- **612 tests**: Unit, integration, property-based (Hypothesis), fuzz, and snapshot (syrupy) tests
- **90%+ code coverage** with branch coverage enabled
- **CI pipeline**: Consolidated 4-workflow GitHub Actions (CI, docs, release, publish) with Python 3.10-3.13 matrix
- **Benchmarks**: 39 pytest-benchmark tests covering core, data science, and ML serialization

### Changed (vs v1)

- Python 3.10+ minimum (was 3.8+)
- Plugin-based dispatch replaces monolithic if/elif type chains
- API reduced from 100+ functions to 5
- `__init__.py` reduced from 678 lines to <60 lines
- All modules under 500-line hard limit (enforced by CI)
- All functions under 50-line hard limit (enforced by CI)
- Dev dependencies moved to `[dependency-groups]` (PEP 735)
- CI consolidated from 12 workflows to 4
