# Datason Project: Learnings, Approaches & Strategy for v2

A comprehensive post-mortem and forward-looking guide based on building datason v0.1-v0.13.

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [What Went Well](#2-what-went-well)
3. [What Went Wrong](#3-what-went-wrong)
4. [Architecture Learnings](#4-architecture-learnings)
5. [Code Quality & Complexity Learnings](#5-code-quality--complexity-learnings)
6. [Testing Learnings](#6-testing-learnings)
7. [CI/CD & DevOps Learnings](#7-cicd--devops-learnings)
8. [API Design Learnings](#8-api-design-learnings)
9. [Performance Learnings](#9-performance-learnings)
10. [Dependency Management Learnings](#10-dependency-management-learnings)
11. [Strategy for v2 (New Project)](#11-strategy-for-v2-new-project)
12. [Recommended Modern Tech Stack](#12-recommended-modern-tech-stack)
13. [Migration Checklist](#13-migration-checklist)

---

## 1. Project Summary

**What datason is:** A Python serialization library that acts as a drop-in replacement for `json` with intelligent handling of complex types (datetime, UUID, numpy, pandas, PyTorch, TensorFlow, scikit-learn, etc.).

**Scale:**
- 22 Python modules, ~14,360 lines of source code
- 100+ exported public API functions
- 10+ ML framework integrations
- Python 3.8-3.13 support
- Zero required dependencies (all optional)
- ~84% test coverage across 76+ unit test files

**Core value proposition:** Serialize anything Python can produce (especially ML/data science objects) to JSON and back with perfect round-trip fidelity.

---

## 2. What Went Well

### 2.1 Zero-Dependency Core
The decision to have zero required runtime dependencies was excellent. It made the library easy to install, reduced supply chain risk, and forced clean optional import patterns. **Keep this in v2.**

### 2.2 Comprehensive Type Coverage
Supporting 50+ types including all major ML frameworks gave the library real differentiation. The lazy import pattern prevented startup overhead.

### 2.3 Security-First Serialization
Built-in depth limits (50), size limits (100K objects), string length limits (1M chars), and circular reference detection from early on prevented common attack vectors. The PII redaction engine and pickle bridge were valuable enterprise features.

### 2.4 Configuration System
The `SerializationConfig` dataclass with enums (DateFormat, DataFrameOrient, NanHandling, TypeCoercion, CacheScope) was well-designed. Preset configs (ml_config, api_config, strict_config, performance_config) provided good developer UX.

### 2.5 Multi-Level Caching
The four-tier cache scope system (operation/request/process/disabled) with context managers was a strong architectural choice that allowed safe caching in web servers vs. scripts.

### 2.6 CI/CD Pipeline
The GitHub Actions setup with strategic Python version x dependency matrix testing (12 concurrent jobs) was efficient. OIDC trusted publishing to PyPI was modern and secure.

### 2.7 Test Infrastructure
Autouse fixtures for cache clearing, ML state restoration, and config reset prevented test pollution. Dependency-aware skip markers (147 skipif decorators) handled the optional dependency matrix well.

---

## 3. What Went Wrong

### 3.1 God Modules Emerged
Two files became massive and hard to maintain:
- `core_new.py`: 2,919 lines - serialization engine
- `deserializers_new.py`: 2,946 lines - deserialization engine

These files had functions 100-500 lines long with 6+ levels of nesting. The `_serialize_full_path()` function alone was 508 lines with 15+ type category branches.

**Lesson:** Establish a hard module size limit (e.g., 500 lines) from day one. Use dispatch tables or the visitor pattern instead of growing if/elif chains.

### 3.2 Naming Suggests Rewrites That Never Completed
Files named `core_new.py` and `deserializers_new.py` imply they replaced `core.py` and `deserializers.py`, but the "new" suffix was never cleaned up. This signals incomplete refactoring.

**Lesson:** Never leave "new" or "v2" in filenames. Complete the migration or don't start it.

### 3.3 Duplicate Logic Proliferated
Four near-identical "is this a basic JSON type?" functions existed:
- `_is_json_basic_type()`
- `_is_json_basic_type_with_config()`
- `_is_json_basic_type_safe()`
- `_is_json_serializable_basic_type()`

Similarly, 40+ lazy import functions in `ml_serializers.py` each had ~15 lines of identical boilerplate.

**Lesson:** Aggressively DRY utility functions. Use factory functions or decorators to generate repetitive code. Code review should block duplication.

### 3.4 API Surface Exploded
100+ public functions is too many. The API grew organically through phases:
- Phase 1: `serialize()`/`deserialize()`
- Phase 2: `dump_ml()`/`dump_api()`/`dump_secure()`/`load_basic()`/`load_smart()`/`load_perfect()`
- Phase 3: `dump()`/`dumps()`/`load()`/`loads()` + JSON compat + file I/O + streaming

Each phase added functions without consolidating the previous ones, leading to multiple ways to do the same thing.

**Lesson:** Design the final API upfront. Deprecate old APIs aggressively. A smaller, consistent API is better than a large one with overlapping functions.

### 3.5 Thread Safety Was Ignored
Multiple global mutable dictionaries (`_TYPE_CACHE`, `_DYNAMIC_STRING_CACHE`, `_RESULT_DICT_POOL`) without locks. The ContextVar-based caching helped for request scope, but process-level caches are unsafe.

**Lesson:** Design for thread safety from the start. Use `threading.Lock` or lock-free data structures for global caches.

### 3.6 Premature/Scattered Optimization
Performance optimizations were added reactively across the codebase:
- String interning pools
- Homogeneity detection with sampling
- Memory pooling for dicts/lists
- Multiple fast-path functions (`_serialize_hot_path`, `deserialize_fast`)
- Inline type checking with pre-computed constants

These optimizations made the code significantly harder to read and maintain, and some (like the homogeneity check) introduced security vulnerabilities (depth bomb attacks) that required additional defensive code.

**Lesson:** Profile first, optimize second. Keep optimizations isolated behind clean interfaces. Document performance-critical sections.

### 3.7 Mixed Concerns in Core Functions
The `serialize()` function mixed:
- Core serialization logic
- Redaction logic
- Profiling instrumentation
- Depth analysis
- Error handling

**Lesson:** Use decorator patterns or middleware chains to compose cross-cutting concerns. Keep the core function purely about serialization.

### 3.8 Inconsistent Error Handling
Three different error strategies coexisted:
1. Catch + warn + continue (for type conversion failures)
2. Silently ignore (for some edge cases)
3. Raise SecurityError (for limit violations)

No clear policy on when each was appropriate.

**Lesson:** Define an explicit error handling strategy document. Categorize errors into fatal, recoverable, and ignorable with clear rules for each.

### 3.9 The `__init__.py` Became a Mega-Exporter
At 678 lines, `__init__.py` imports from every module, handles feature detection, defines helper functions, and manages deprecation warnings. It's fragile and hard to maintain.

**Lesson:** Keep `__init__.py` minimal. Use explicit submodule imports or a `__all__` list generated from module-level exports.

### 3.10 Documentation Bloat
The CHANGELOG alone is 103KB. SECURITY.md is 16.7KB. While comprehensive docs are good, the volume suggests scope creep and feature churn. Multiple PRD documents for performance optimization suggest planning happened in docs rather than code.

**Lesson:** Keep changelogs concise. Use git history for detailed change tracking. PRDs belong in a wiki or project management tool, not the repo.

---

## 4. Architecture Learnings

### 4.1 What Worked: Layered Architecture
```
Public API (api.py) -> Core Engine (core_new.py) -> Type Handlers -> ML Serializers
```
This layering was sound. The API layer provided intention-revealing functions while the core engine handled the recursive processing.

### 4.2 What Worked: Registry Pattern for Types
The `TypeRegistry` in `type_registry.py` that paired serialize + deserialize handlers together prevented the "split-brain" problem where serialization and deserialization drift apart.

### 4.3 What Failed: No Clear Module Boundaries
Modules had circular awareness of each other. `core_new.py` knew about ML types, redaction, profiling, and caching. This made it impossible to understand one module without understanding all of them.

### 4.4 Recommendation for v2: Plugin Architecture
```
Core (serialize/deserialize primitives)
  |
  +-- Plugin: datetime handler
  +-- Plugin: numpy handler
  +-- Plugin: pandas handler
  +-- Plugin: ML frameworks handler
  +-- Plugin: security/redaction
  +-- Plugin: integrity/signing
```
Each plugin registers itself with the core via a clean interface. Core knows nothing about specific types.

### 4.5 Recommendation for v2: Use Protocol Classes
Instead of isinstance chains, define `Serializable` and `Deserializable` protocols:
```python
class TypePlugin(Protocol):
    def can_handle(self, obj: Any) -> bool: ...
    def serialize(self, obj: Any, context: Context) -> Any: ...
    def deserialize(self, data: Any, context: Context) -> Any: ...
    def priority(self) -> int: ...
```

---

## 5. Code Quality & Complexity Learnings

### 5.1 Functions That Were Too Long
| Function | Lines | Problem |
|----------|-------|---------|
| `_serialize_full_path()` | 508 | 15+ type branches in one function |
| `_deserialize_with_type_metadata()` | 378 | 50+ type dispatch cases |
| `serialize()` | 118 | Mixed serialization + redaction + profiling |
| `_is_homogeneous_collection()` | 95 | Caching + detection + security in one |

**Rule for v2:** No function > 50 lines. No module > 500 lines. Use dispatch tables for type routing.

### 5.2 Magic Numbers Were Everywhere
Depth limits (50, 100, 2, 8, 10), size limits (100K, 10K, 5, 20), cache sizes (1000, 500, 200, 100, 20), sample sizes (10, 20) -- all hardcoded with inconsistent values.

**Rule for v2:** All limits as named constants in a single `constants.py`. All configurable limits exposed via config.

### 5.3 Type Checking Was Ad-Hoc
Pre-computed type tuples scattered across the module:
```python
_JSON_BASIC_TYPES = (str, int, float, bool, type(None))
_NUMERIC_TYPES = (int, float)
_TYPE_STR = str  # For identity checks
```

**Rule for v2:** Centralize all type categorization in one place. Use an enum or registry, not scattered tuples.

### 5.4 Comments and TODOs
Only 11 TODOs (good discipline) but one disabled feature:
```python
# TODO: Re-enable custom handler when linter issue is resolved
```
This blocked extensibility for unknown duration.

**Rule for v2:** TODOs get issues. Blocked features get fixed or removed within one release cycle.

---

## 6. Testing Learnings

### 6.1 What Worked Well
- **Autouse fixtures** (`ensure_clean_test_state`) preventing cache/state leaks between tests
- **Dependency-aware skipping** with 147 `skipif` decorators for optional packages
- **Strategic CI matrix** testing Python 3.8 (core only) through 3.13 (core only) with full suite on 3.11/3.12
- **Edge case test directory** dedicated to boundary conditions
- **Performance regression tests** with explicit thresholds

### 6.2 What Was Missing
- **No property-based testing** (Hypothesis) -- would have caught edge cases in type detection
- **No mutation testing** -- would have validated test quality
- **No snapshot testing** -- serialization output changes would be caught automatically
- **No fuzz testing** -- important for a library that parses arbitrary input
- **Parametrize was underused** -- only 8 uses across the entire test suite, despite many tests following similar patterns with different data types

### 6.3 Test Organization Problem
Test files at multiple levels:
```
tests/unit/           (76 files)
tests/integration/    (20+ files)
tests/edge_cases/     (16 files)
tests/coverage/       (coverage boost)
tests/performance/
tests/test_*.py       (root-level feature tests)
```
Root-level test files (`test_modern_api.py`, `test_configurable_caching.py`, etc.) broke the organizational structure.

**Rule for v2:** All tests in subdirectories. No root-level test files. Clear separation: `tests/unit/`, `tests/integration/`, `tests/benchmarks/`.

### 6.4 Fixture Over-Engineering
Three conftest.py files with overlapping autouse fixtures that all do cache clearing + state restoration. Some tests needed `@pytest.mark.no_autofixture` to opt out.

**Rule for v2:** One conftest.py at the test root. Simpler fixtures. Prefer explicit setup/teardown over global autouse when possible.

### 6.5 Coverage Targets
80% overall / 85% core / 75% features / 70% utilities was reasonable and achievable. The Codecov integration with patch coverage (70% on new code) prevented regression.

**Keep this approach in v2** but add mutation testing to validate test quality, not just quantity.

---

## 7. CI/CD & DevOps Learnings

### 7.1 What Worked
- **12 GitHub workflows** covering every aspect (test, build, release, publish, docs, performance, compatibility, version bumping, auto-merge)
- **OIDC trusted publishing** to PyPI (no stored API tokens)
- **Smart matrix exclusions** reducing CI cost (e.g., Python 3.8 only runs core tests)
- **External benchmark repo** (`datason-benchmarks`) keeping performance history separate
- **Pre-commit hooks** catching json import violations and doc sync issues

### 7.2 What Could Be Better
- **12 workflows is too many** to maintain. Some could be consolidated.
- **No local CI reproduction** -- no `Makefile`, `justfile`, or `taskfile.yml` for running CI steps locally
- **No dependency lock file** -- `pip-tools` is listed but no `requirements.txt` lock file committed
- **No container-based development** -- no `Dockerfile` or `devcontainer.json`

### 7.3 Recommendations for v2

**Use a task runner** (just, make, or task) with targets mirroring CI:
```
just test          # pytest
just lint          # ruff check + ruff format --check
just typecheck     # mypy/pyright
just security      # bandit + safety
just bench         # pytest-benchmark
just ci            # all of the above
just release       # build + publish
```

**Use dependency locking** via `uv` or `pip-compile` for reproducible CI builds.

**Use devcontainers** for consistent development environments.

---

## 8. API Design Learnings

### 8.1 The API Evolution Problem
The API grew through 3 phases without deprecating previous phases:

| Phase | Functions | Problem |
|-------|-----------|---------|
| 1 | `serialize()`, `deserialize()` | Too generic |
| 2 | `dump_ml()`, `dump_api()`, `load_smart()`, `load_perfect()` | Good intent-revealing names |
| 3 | `dump()`, `dumps()`, `load()`, `loads()` + JSON compat | Overlaps with Phase 2 |

Users now face: Should I use `serialize()`, `dump()`, `dumps()`, `dump_ml()`, or `save_ml()`?

### 8.2 Recommendation for v2: Minimal API
```python
import datason

# Core (4 functions)
datason.dumps(obj)          # -> JSON string
datason.loads(s)            # -> Python object
datason.dump(obj, fp)       # -> write to file
datason.load(fp)            # -> read from file

# Configuration via context
with datason.config(preset="ml"):
    datason.dumps(model)

# Or inline
datason.dumps(model, preset="ml")

# Explicit presets as module-level shortcuts (optional)
datason.ml.dumps(model)
datason.api.dumps(response)
```

Total public API: ~10 functions instead of 100+.

### 8.3 The `__init__.py` Export Problem
Exporting 100+ symbols from `__init__.py` means:
- IDE autocomplete is overwhelming
- Import time increases
- Any module failure breaks everything
- Circular import risk increases

**Rule for v2:** `__init__.py` exports < 20 symbols. Advanced features accessed via submodules.

---

## 9. Performance Learnings

### 9.1 Achieved Performance (Worth Preserving)
- Simple data: 0.6ms (target: <1ms)
- Complex data: 2.1ms (target: <5ms)
- Throughput: 272K items/sec (target: >250K)
- NumPy: 5.5M elements/sec (target: >5M)
- Pandas: 195K rows/sec (target: >150K)

### 9.2 Optimizations That Were Worth It
- **Lazy imports** for ML libraries (10-50ms startup savings)
- **Type caching** via `_TYPE_CACHE` (avoids repeated isinstance)
- **Early exit** for already-serialized data
- **Pre-computed type tuples** for fast membership testing

### 9.3 Optimizations That Weren't Worth It
- **String interning pool** (~40 common strings) -- micro-optimization that added complexity
- **Memory pooling for dicts/lists** -- marginal benefit, complex cleanup code, hard to debug
- **Homogeneity detection with sampling** -- introduced a security vulnerability (depth bomb), required defensive code that negated the performance gain
- **Multiple fast-path functions** (`_serialize_hot_path`, `deserialize_fast`) -- code duplication for marginal speedup

### 9.4 Recommendations for v2
- Start with a clean, readable implementation
- Profile with real-world workloads before optimizing
- Keep optimizations behind clean interfaces (strategy pattern)
- Consider Rust extension for genuinely hot paths instead of Python micro-optimizations
- Benchmark suite from day 1, run on every PR

---

## 10. Dependency Management Learnings

### 10.1 What Worked
- Zero required dependencies
- Optional extras organized by use case (`ml`, `crypto`, `dev`, `docs`)
- Graceful fallback with feature detection flags
- `hatchling` as modern build backend

### 10.2 What Could Be Better
- **No lock file** for reproducible builds
- **Broad version ranges** (e.g., `pandas>=1.3.0`) can cause subtle compatibility issues
- **No dependency update automation** (Dependabot/Renovate)
- **Runtime version parsing** from `pyproject.toml` via regex is fragile

### 10.3 Recommendations for v2

**Use `uv`** for dependency management:
```bash
uv init datason-v2
uv add --optional ml numpy pandas torch
uv lock  # Creates lockfile
uv sync  # Install from lockfile
```

**Use `importlib.metadata`** for version at runtime:
```python
from importlib.metadata import version
__version__ = version("datason")
```

**Set up Renovate/Dependabot** for automated dependency updates with CI validation.

**Pin minimum versions based on actual tested versions**, not guesses.

---

## 11. Strategy for v2 (New Project)

### 11.1 Core Principles (Refined)

1. **Zero dependencies, plugin architecture** -- core handles JSON primitives only; everything else is a plugin
2. **Small API surface** -- 4 core functions (`dumps`/`loads`/`dump`/`load`) + configuration
3. **Type safety first** -- use `typing.Protocol` for all interfaces, run `pyright` in strict mode
4. **Thread safe by default** -- no global mutable state without locks
5. **Profile before optimize** -- clean code first, optimized code second with benchmarks proving the need

### 11.2 Module Structure

```
datason/
  __init__.py          (<50 lines, minimal exports)
  _core.py             (recursive serialization, <300 lines)
  _deserialize.py      (recursive deserialization, <300 lines)
  _config.py           (configuration dataclass + presets)
  _types.py            (type constants + categories)
  _protocols.py        (Protocol classes for plugins)
  _registry.py         (plugin registration + dispatch)
  _cache.py            (thread-safe caching)
  _errors.py           (error hierarchy)

  plugins/
    __init__.py
    datetime.py         (datetime, date, time, timedelta)
    uuid.py             (UUID handling)
    decimal.py          (Decimal, complex)
    path.py             (Path objects)
    numpy.py            (ndarray, scalar types)
    pandas.py           (DataFrame, Series, Timestamp)
    ml_torch.py         (PyTorch tensors, models)
    ml_tensorflow.py    (TensorFlow tensors)
    ml_sklearn.py       (scikit-learn models)
    ml_misc.py          (CatBoost, Optuna, Plotly, Polars, JAX)

  security/
    __init__.py
    redaction.py        (PII redaction)
    integrity.py        (hash, sign, verify)
    pickle_bridge.py    (safe pickle conversion)

tests/
  unit/
  integration/
  benchmarks/
  conftest.py           (single conftest)
```

**Rules:**
- No module > 500 lines
- No function > 50 lines
- No more than 3 levels of nesting
- All type handlers are plugins with a standard interface

### 11.3 Plugin Interface

```python
# _protocols.py
from typing import Protocol, Any

class TypePlugin(Protocol):
    """Interface all type handler plugins must implement."""

    name: str
    priority: int  # Lower = checked first

    def can_handle(self, obj: Any) -> bool:
        """Return True if this plugin can serialize the object."""
        ...

    def serialize(self, obj: Any, context: SerializeContext) -> Any:
        """Convert obj to a JSON-serializable representation."""
        ...

    def can_deserialize(self, data: Any) -> bool:
        """Return True if this plugin can reconstruct from data."""
        ...

    def deserialize(self, data: Any, context: DeserializeContext) -> Any:
        """Reconstruct the original object from serialized data."""
        ...
```

### 11.4 Error Handling Strategy

```python
# _errors.py
class DatasonError(Exception):
    """Base class for all datason errors."""

class SecurityError(DatasonError):
    """Raised when security limits are exceeded. Always fatal."""

class SerializationError(DatasonError):
    """Raised when an object cannot be serialized. Fatal by default."""

class DeserializationError(DatasonError):
    """Raised when data cannot be deserialized. Fatal by default."""

class PluginError(DatasonError):
    """Raised when a plugin fails. Logged, falls back to next plugin."""
```

**Policy:**
- Security violations: always raise, never swallow
- Type handling failures: raise by default, configurable fallback to string representation
- Plugin failures: log warning, try next plugin
- No silent `except: pass` anywhere

### 11.5 Testing Strategy

- **pytest** with **hypothesis** for property-based testing
- **pytest-benchmark** from day 1
- **Snapshot testing** (syrupy or inline-snapshot) for serialization output
- **Mutation testing** (mutmut) to validate test quality
- **Fuzz testing** for deserialization (important for security)
- **Parametrize heavily** -- one test function per behavior, parametrized across types
- Single `conftest.py` with explicit fixtures (no autouse for cleanup -- use proper isolation instead)

### 11.6 CI/CD Strategy

Consolidate to 3-4 workflows:
1. **ci.yml** -- lint + typecheck + test + build (on every push/PR)
2. **release.yml** -- publish to PyPI (on tag)
3. **docs.yml** -- build and deploy docs
4. **benchmark.yml** -- performance tracking (on PR)

Use `just` as task runner with targets matching CI steps.

---

## 12. Recommended Modern Tech Stack

### Build & Package Management
| Current (v1) | Recommended (v2) | Why |
|---------------|-------------------|-----|
| hatchling | **uv** + hatchling | uv for deps/venv, hatchling for building |
| pip install -e | `uv sync` | 10-100x faster, lockfile support |
| No lockfile | `uv.lock` | Reproducible builds |
| pip-tools | **uv** | Built-in lock/sync |

### Code Quality
| Current (v1) | Recommended (v2) | Why |
|---------------|-------------------|-----|
| ruff (lint + format) | **ruff** (keep) | Already best-in-class |
| mypy (strict) | **pyright** (strict) | Faster, better inference, LSP native |
| bandit | **ruff** rules (S/B) | Bandit rules available in ruff |
| safety | **pip-audit** (keep) | Already using pip-audit |

### Testing
| Current (v1) | Recommended (v2) | Why |
|---------------|-------------------|-----|
| pytest | **pytest** (keep) | Standard |
| pytest-cov | **pytest-cov** (keep) | Standard |
| No property testing | **hypothesis** | Catches edge cases in type handling |
| No snapshot testing | **syrupy** or **inline-snapshot** | Catches serialization format changes |
| No mutation testing | **mutmut** | Validates test effectiveness |
| No fuzz testing | **atheris** or **pythonfuzz** | Security-critical for deserialization |

### CI/CD
| Current (v1) | Recommended (v2) | Why |
|---------------|-------------------|-----|
| 12 GitHub workflows | **3-4 workflows** | Simpler to maintain |
| No task runner | **just** or **task** | Local CI reproduction |
| No devcontainer | **devcontainer.json** | Consistent environments |
| No Dependabot | **Renovate** | Automated dependency updates |

### Documentation
| Current (v1) | Recommended (v2) | Why |
|---------------|-------------------|-----|
| mkdocs-material | **mkdocs-material** (keep) | Already good |
| 103KB CHANGELOG | **Auto-generated from git** | Use `git-cliff` or similar |
| PRDs in repo | **GitHub Discussions/Wiki** | Keep repo focused on code |

### Python Version Support
| Current (v1) | Recommended (v2) | Why |
|---------------|-------------------|-----|
| Python 3.8+ | **Python 3.10+** | 3.8/3.9 are EOL. 3.10 gives match/case, PEP 604 unions, better typing |

### Optional: Rust Extension
| Current (v1) | Recommended (v2) | Why |
|---------------|-------------------|-----|
| maturin + PyO3 | **maturin + PyO3** (keep) | Already good if needed |
| Optional/incomplete | **Core hot path only** | Profile first, only accelerate proven bottlenecks |

---

## 13. Migration Checklist

### Phase 1: Project Setup
- [ ] Initialize with `uv init`
- [ ] Set up `pyproject.toml` with hatchling + Python >=3.10
- [ ] Create `justfile` with targets: test, lint, typecheck, bench, ci
- [ ] Set up `.devcontainer/devcontainer.json`
- [ ] Create 3 GitHub workflows (ci, release, docs)
- [ ] Set up Renovate for dependency updates
- [ ] Configure pyright in strict mode
- [ ] Configure ruff with comprehensive rules (including bandit S rules)
- [ ] Set up pre-commit hooks (ruff + pyright + trailing whitespace)

### Phase 2: Core Architecture
- [ ] Define `TypePlugin` protocol in `_protocols.py`
- [ ] Implement plugin registry with priority-based dispatch
- [ ] Implement core `_serialize()` with plugin dispatch (<300 lines)
- [ ] Implement core `_deserialize()` with plugin dispatch (<300 lines)
- [ ] Implement `SerializationConfig` dataclass with presets
- [ ] Implement thread-safe caching with proper locks
- [ ] Define error hierarchy in `_errors.py`
- [ ] Write comprehensive unit tests with hypothesis

### Phase 3: Built-in Plugins
- [ ] datetime plugin (datetime, date, time, timedelta)
- [ ] uuid plugin
- [ ] decimal/complex plugin
- [ ] path plugin
- [ ] numpy plugin (with lazy import)
- [ ] pandas plugin (with lazy import)
- [ ] Tests for each plugin including property-based tests

### Phase 4: ML Framework Plugins
- [ ] PyTorch plugin
- [ ] TensorFlow plugin
- [ ] scikit-learn plugin
- [ ] Miscellaneous ML plugin (CatBoost, Optuna, Plotly, Polars, JAX)
- [ ] Integration tests with real libraries

### Phase 5: Security Features
- [ ] PII redaction as plugin/middleware
- [ ] Integrity (hash/sign/verify)
- [ ] Pickle bridge
- [ ] Security-focused fuzz testing

### Phase 6: Performance
- [ ] Benchmark suite covering all major paths
- [ ] Profile with real-world workloads
- [ ] Optimize only proven bottlenecks
- [ ] Consider Rust extension for hot paths if Python isn't fast enough
- [ ] Set up benchmark tracking on PRs

### Phase 7: Polish
- [ ] API documentation with mkdocs-material
- [ ] Migration guide from v1
- [ ] Examples directory
- [ ] PyPI publishing via OIDC
- [ ] CHANGELOG via git-cliff

---

## Key Takeaways (TL;DR)

1. **The zero-dependency core was the best decision.** Keep it.
2. **The API surface grew 10x too large.** Design the final API first, keep it to ~10 functions.
3. **God modules (3000+ lines) are unmaintainable.** Hard limit: 500 lines per module, 50 lines per function.
4. **Plugin architecture > if/elif chains.** Use Protocol classes for type dispatch.
5. **Thread safety must be designed in from day 1,** not retrofitted.
6. **Profile before you optimize.** Half the micro-optimizations weren't worth their complexity cost.
7. **Property-based testing + fuzz testing are essential** for a serialization library.
8. **Modern tooling (uv, pyright, just)** reduces friction significantly.
9. **3 CI workflows > 12 CI workflows.** Consolidate aggressively.
10. **Python 3.10+** as minimum drops compatibility baggage and enables cleaner code.
