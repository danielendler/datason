# Full Quality Check

Run ALL quality gates. This is the command to run before any PR or commit. Follow this exact sequence:

## Step 1: Lint + Format
Run `just lint` or:
```
uv run ruff check datason/ tests/ --fix
uv run ruff format datason/ tests/
```

## Step 2: Type Check
Run `just typecheck` or:
```
uv run pyright --project .
```

## Step 3: Tests with Coverage
Run `just test` or:
```
uv run pytest tests/unit/ --cov=datason --cov-report=term-missing -x --tb=short
```

## Step 4: Architecture Enforcement
Check these hard limits and report violations:
- Any module > 500 lines? Run: `wc -l datason/*.py datason/**/*.py | sort -rn | head -20`
- Any function > 50 lines? Run: `python -c "import ast, sys; [print(f'{n.name}: {n.end_lineno - n.lineno + 1} lines') for f in sys.argv[1:] for n in ast.walk(ast.parse(open(f).read())) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.end_lineno - n.lineno + 1 > 50]" datason/*.py`
- `__init__.py` exports > 20? Count them and report.

## Step 5: Security Scan
Run `just security` or:
```
uv run ruff check datason/ --select S
uv run pip-audit
```

## Report
Provide a pass/fail summary for each step. All 5 must pass before merging.
