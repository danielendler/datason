# Test Runner

Run the test suite and report results. Follow this exact workflow:

1. Run `just test` (or `uv run pytest tests/unit/ -x --tb=short` if just is not available)
2. If tests fail:
   - Analyze the failure output
   - Identify the root cause
   - Fix the issue
   - Re-run only the failing tests to verify the fix
   - Then run the full suite again to check for regressions
3. If tests pass:
   - Report the pass count and coverage percentage
   - Flag any coverage drops below 80%

Always run tests before committing code.
