# Review Pull Request

Review the current branch's changes against main. Argument (optional): PR number $ARGUMENTS

## Review Checklist

### 1. Understand the Changes
```
git log main..HEAD --oneline
git diff main...HEAD --stat
```
Read the diff thoroughly.

### 2. Architecture Compliance
- [ ] No module > 500 lines
- [ ] No function > 50 lines
- [ ] No nesting > 3 levels
- [ ] New types use the plugin interface (not if/elif in core)
- [ ] Thread-safe global state (uses Lock)
- [ ] Proper error handling (no bare except)

### 3. API Surface
- [ ] No new public exports from `__init__.py` without justification
- [ ] Public functions have docstrings
- [ ] Type annotations on all public functions

### 4. Testing
- [ ] New code has unit tests
- [ ] Edge cases tested (None, empty, malformed, very large)
- [ ] Hypothesis property tests for type handlers
- [ ] Snapshot tests for serialization format changes
- [ ] No test pollution (no global state leaks between tests)

### 5. Performance
- [ ] No unnecessary object creation in hot paths
- [ ] Lazy imports for third-party libraries
- [ ] No O(n^2) algorithms on potentially large inputs
- [ ] Benchmarks updated if performance-critical code changed

### 6. Security
- [ ] Depth limits enforced on recursive operations
- [ ] Size limits enforced on collections
- [ ] No pickle usage without explicit opt-in
- [ ] No eval/exec on untrusted input

Provide a structured review with PASS/FAIL per section and specific file:line references for issues.
