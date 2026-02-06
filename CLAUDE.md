# Datason v2 - Python Serialization Library

## Project Overview
Datason is a zero-dependency Python serialization library that acts as a drop-in replacement for `json` with intelligent handling of complex types (datetime, UUID, numpy, pandas, PyTorch, TensorFlow, scikit-learn, etc.). We are building v2 from scratch based on the learnings documented in `LEARNINGS_AND_STRATEGY.md`.

## Architecture

### Core Principle: Plugin-Based Type Dispatch
Every type beyond JSON primitives (str, int, float, bool, None, dict, list) is handled by a **TypePlugin**. The core serializer knows nothing about specific types - it dispatches to registered plugins via a priority-sorted registry.

```
datason/
  __init__.py          # <50 lines, exports: dumps, loads, dump, load, config
  _core.py             # Recursive serialization engine (<300 lines)
  _deserialize.py      # Recursive deserialization engine (<300 lines)
  _config.py           # SerializationConfig dataclass + presets
  _types.py            # Type constants, categories, JSON_BASIC_TYPES
  _protocols.py        # TypePlugin Protocol class
  _registry.py         # Plugin registration + priority dispatch
  _cache.py            # Thread-safe caching (threading.Lock on all global state)
  _errors.py           # Error hierarchy: DatasonError > SecurityError, SerializationError, etc.
  plugins/             # Each file is a self-contained type handler
  security/            # Redaction, integrity, pickle bridge
tests/
  unit/                # One test file per source module
  integration/         # Cross-module tests
  benchmarks/          # pytest-benchmark tests
  conftest.py          # Single conftest, explicit fixtures
```

### Hard Limits (Enforced by CI)
- No module > 500 lines
- No function > 50 lines
- No nesting > 3 levels
- No more than 20 exports from `__init__.py`
- Python 3.10+ only (use match/case, PEP 604 unions)

### Public API (Complete)
```python
datason.dumps(obj, **config)    # -> JSON string
datason.loads(s, **config)      # -> Python object
datason.dump(obj, fp, **config) # -> write to file
datason.load(fp, **config)      # -> read from file
datason.config(preset="ml")     # -> context manager for config
```
That's it. No `dump_ml`, `load_smart`, `save_string`, `load_perfect`, etc.

## Development Commands

### Using just (preferred)
```bash
just test              # pytest with coverage
just lint              # ruff check + ruff format --check
just typecheck         # pyright --project .
just bench             # pytest tests/benchmarks/ --benchmark-only
just security          # ruff check --select S + pip-audit
just ci                # all of the above
just release           # build + twine check
```

### Direct commands
```bash
uv sync                                        # Install all deps from lockfile
uv run pytest tests/unit/ -x --tb=short        # Run unit tests
uv run pytest tests/unit/ --cov=datason        # With coverage
uv run ruff check datason/                     # Lint
uv run ruff format datason/ tests/             # Format
uv run pyright                                 # Type check
```

## Coding Standards

### Error Handling Policy
- **Security violations** (depth/size limits): ALWAYS raise `SecurityError`, never swallow
- **Type handling failures**: Raise `SerializationError` by default; configurable fallback to `str(obj)`
- **Plugin failures**: Log warning via `warnings.warn()`, try next plugin in priority order
- **NEVER** use bare `except: pass` or `except Exception: pass`

### Plugin Interface
Every type handler implements this Protocol:
```python
class TypePlugin(Protocol):
    name: str
    priority: int  # Lower = checked first (0-99: builtins, 100-199: stdlib, 200+: third-party)

    def can_handle(self, obj: Any) -> bool: ...
    def serialize(self, obj: Any, ctx: SerializeContext) -> Any: ...
    def can_deserialize(self, data: Any) -> bool: ...
    def deserialize(self, data: Any, ctx: DeserializeContext) -> Any: ...
```

### Type Metadata Format
```python
{"__datason_type__": "typename", "__datason_value__": <serialized_value>}
```

### Thread Safety
- ALL global mutable state MUST use `threading.Lock`
- Prefer `contextvars.ContextVar` for request-scoped state
- No module-level mutable dicts/lists without locks

### Imports
- Core modules: only stdlib imports
- Plugin modules: lazy imports for third-party libraries
- Pattern: `try: import X; except ImportError: X = None`

### Testing
- Use `hypothesis` for property-based tests on all type handlers
- Use `pytest.mark.parametrize` for type variations (not copy-paste tests)
- Use `syrupy` for snapshot testing serialization output
- One test file per source module: `test_core.py`, `test_registry.py`, etc.
- Benchmarks go in `tests/benchmarks/`, not mixed with unit tests

## Key Files for Context
- `LEARNINGS_AND_STRATEGY.md` - Full post-mortem from v1 with detailed anti-patterns to avoid
- `pyproject.toml` - Build config, dependencies, tool settings
- `justfile` - All development task definitions
- `.claude/commands/` - Custom Claude Code workflows

## Important Gotchas
1. NumPy/Pandas/PyTorch are OPTIONAL. Always check `if np is not None:` before using.
2. The `__datason_type__` key in dicts is reserved for type metadata. Never serialize user data that conflicts.
3. Circular references must be detected via `id()` tracking in a set, not by value comparison.
4. NaN and Infinity in floats are not valid JSON. Handle via `NanHandling` config enum.
5. Python 3.10's `match/case` should be used for type dispatch in plugins instead of if/elif chains.
