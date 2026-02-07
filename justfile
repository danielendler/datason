# Datason v2 - Development Task Runner
# Usage: just <task>
# Install just: brew install just (macOS) or cargo install just

# Default: show available tasks
default:
    @just --list

# ============================================================
# CORE DEVELOPMENT
# ============================================================

# Install all dependencies from lockfile
install:
    uv sync --all-extras

# Run unit tests with coverage
test *ARGS:
    uv run pytest tests/unit/ --cov=datason --cov-report=term-missing -x --tb=short {{ARGS}}

# Run a specific test file or pattern
test-file FILE:
    uv run pytest {{FILE}} -v --tb=short

# Run replay/perf harness tests
test-perf:
    uv run pytest tests/perf/ -v

# Run tests matching a keyword
test-k KEYWORD:
    uv run pytest tests/unit/ -k "{{KEYWORD}}" -v --tb=short

# ============================================================
# CODE QUALITY
# ============================================================

# Lint and auto-fix
lint:
    uv run ruff check datason/ tests/ --fix
    uv run ruff format datason/ tests/

# Lint check only (no fixes, for CI)
lint-check:
    uv run ruff check datason/ tests/
    uv run ruff format datason/ tests/ --check

# Type checking with pyright
typecheck:
    uv run pyright --project .

# Security scanning
security:
    uv run ruff check datason/ --select S
    uv run pip-audit

# ============================================================
# ARCHITECTURE ENFORCEMENT
# ============================================================

# Check module line counts (limit: 500)
check-lines:
    @echo "=== Module Line Counts (limit: 500) ==="
    @find datason -name "*.py" -exec wc -l {} + | sort -rn | head -20
    @echo ""
    @echo "=== Violations (>500 lines) ==="
    @find datason -name "*.py" -exec wc -l {} + | sort -rn | awk '$1 > 500 && !/total/ {print "VIOLATION: " $0}'

# Check function lengths (limit: 50)
check-functions:
    @echo "=== Functions >50 lines ==="
    @uv run python -c "\
    import ast, sys, pathlib; \
    violations = []; \
    [violations.append(f'{f}: {n.name} ({n.end_lineno - n.lineno + 1} lines)') \
     for f in sorted(pathlib.Path('datason').rglob('*.py')) \
     for n in ast.walk(ast.parse(f.read_text())) \
     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
     and n.end_lineno and n.end_lineno - n.lineno + 1 > 50]; \
    [print(f'VIOLATION: {v}') for v in violations] or print('All functions within limit')"

# Check __init__.py export count (limit: 20)
check-exports:
    @echo "=== __init__.py Export Count (limit: 20) ==="
    @uv run python -c "\
    import ast; \
    tree = ast.parse(open('datason/__init__.py').read()); \
    all_node = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) \
                and any(isinstance(t, ast.Name) and t.id == '__all__' for t in n.targets)]; \
    count = len(all_node[0].value.elts) if all_node else 'No __all__ defined'; \
    print(f'Exports: {count}')"

# Run ALL architecture checks
check-arch: check-lines check-functions check-exports

# ============================================================
# BENCHMARKS
# ============================================================

# Run benchmarks
bench *ARGS:
    uv run pytest tests/benchmarks/ --benchmark-only --benchmark-sort=mean {{ARGS}}

# Run benchmarks and save results
bench-save:
    uv run pytest tests/benchmarks/ --benchmark-only --benchmark-save=baseline --benchmark-sort=mean

# Compare benchmarks against saved baseline
bench-compare:
    uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline --benchmark-sort=mean

# Run real-data replay benchmark suite (sample corpus)
perf-real *ARGS:
    uv run python scripts/perf/replay_benchmark.py --input perf/workloads/sample --output perf/results/replay-summary.json {{ARGS}}

# Compare replay suite against an existing baseline summary
perf-real-compare BASELINE *ARGS:
    uv run python scripts/perf/replay_benchmark.py --input perf/workloads/sample --output perf/results/replay-summary.json --baseline {{BASELINE}} --fail-on-regression {{ARGS}}

# Run mutation testing (manual/targeted use)
mutmut *ARGS:
    uvx --from mutmut mutmut run {{ARGS}}

# ============================================================
# FULL CI PIPELINE
# ============================================================

# Run everything (what CI runs)
ci: lint-check typecheck test check-arch security

# Quick check (lint + test, no typecheck/security)
quick: lint test

# ============================================================
# BUILD & RELEASE
# ============================================================

# Build the package
build:
    uv run python -m build

# Check the built package
check-build: build
    uv run twine check dist/*

# Clean build artifacts
clean:
    rm -rf dist/ build/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ============================================================
# UTILITIES
# ============================================================

# Show project stats
stats:
    @echo "=== Project Stats ==="
    @echo "Source files:"
    @find datason -name "*.py" | wc -l | tr -d ' '
    @echo "Source lines:"
    @find datason -name "*.py" -exec cat {} + | wc -l | tr -d ' '
    @echo "Test files:"
    @find tests -name "*.py" | wc -l | tr -d ' '
    @echo "Test lines:"
    @find tests -name "*.py" -exec cat {} + | wc -l | tr -d ' '
