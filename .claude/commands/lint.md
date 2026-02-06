# Lint and Format

Run the full code quality pipeline. Follow this exact workflow:

1. Run `just lint` (or the individual commands below if just is not available):
   ```
   uv run ruff check datason/ tests/ --fix
   uv run ruff format datason/ tests/
   uv run pyright --project .
   ```
2. If ruff reports unfixable errors:
   - Fix them manually
   - Re-run ruff to verify
3. If pyright reports type errors:
   - Fix type annotations
   - Prefer `X | None` over `Optional[X]` (Python 3.10+)
   - Use `typing.Protocol` for interfaces
   - Re-run pyright to verify
4. Report a clean summary of what was fixed and what remains
