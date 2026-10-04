# Changelog

All notable changes to datason are documented here. This project uses [Semantic Versioning](https://semver.org/).

## 2.0.0a2 (Unreleased)

This alpha keeps the pure Python, zero-required-dependency core and Python 3.10+
minimum. See [release notes](docs/releases/2.0.0a2.md) for upgrade guidance.

### Documentation

- Reorganize onboarding and navigation around API responses, diagnostics, and typed storage.
- Clarify development-source versus published-alpha installation, optional extras,
  reconstruction contracts, JSON compatibility, configuration scope, and trust controls.
- Add recipes, troubleshooting, supported-type guidance, and runnable advanced examples.
- Validate documentation snippets in CI and publish synchronized AI reference files.

### Added

- Opt-in LangGraph checkpoint serializer with SQLite resume compatibility CI (#118).
- Structured dataclass/Pydantic/enum data and typed binary values (#116).
- Serialization error field paths, preserving exception identity and causes (#123).
- Input-byte/node budgets and a plugin reconstruction policy (#113).
- 24 persisted payload fixtures captured from actual a1 source and an explicit
  migration contract (#127).
- LangGraph Interrupt/Send runtime codecs, a two-generation SDK matrix and
  offline Agents snapshot/context/approval restoration tests.

### Fixed

- NumPy scalar dtype/unsigned range, complex and temporal values, empty array shapes;
  supported Pandas labels, indexes, dtypes and timestamp precision (#115).
- Empty ML tensor shapes, portable CPU restoration, sparse allocation checks,
  JAX x64 loss rejection and sklearn estimator-state validation (#128).
- Preserve numeric bfloat16 across Torch/TF/JAX reconstruction.
- NumPy complex128 dispatch under warnings-as-errors (#122).
- JSON options/callback compatibility, collection reconstruction, mapping-key
  validation and shared non-finite-value policies (#97, #101, #113).
- Integrity canonicalization and keyed hash-downgrade rejection (#111).
- Dependency lock security refresh and expanded CI/package validation (#106, #117).

### Performance

- Avoid unnecessary primitive cycle bookkeeping, duplicate NumPy materialization
  and item-by-item conversion for supported complex arrays (#124).
- Skip impossible stock email regex matches on strings without `@` (#125).
- Focused CI NumPy benchmarks improved approximately 7% encoding / 11% decoding
  against the immediate pre-optimization baseline; no general speed claim.

### Upgrade considerations

- Pickle conversion requires explicit `trusted=True`; untrusted pickle remains unsafe (#112).
- Reserved metadata keys and normalized key collisions are rejected. Metadata and
  plugin output now count toward configured limits.
- Application models normalize to structured dictionaries rather than reconstructing
  arbitrary classes. Legacy scalar tags cannot recover absent dtype information;
  large unsigned values need an authoritative dtype. Legacy untagged collections
  remain lists. Typed ML snapshots require trusted application-owned state.
- The LangGraph adapter uses a distinct checkpoint format; existing native checkpoints
  require an explicit migration. This is still an alpha, not a stable 2.0 release.

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
