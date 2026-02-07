# Rust Feasibility Gate (Investigate-Only)

Rust is a candidate acceleration path, not a default.

## Entry Criteria

A candidate hot kernel should satisfy all of:

1. Accounts for >=15% cumulative runtime on real workloads.
2. Stable semantics and low coupling to Python object-model edge cases.
3. Clear fallback path to pure Python without API change.

## Likely Candidate Areas

- Primitive dict/list serialization fast path
- Datetime encode/decode helper kernels

## Boundary Rules

- Public API remains unchanged (`datason.dumps/loads` etc.).
- Rust extension is optional.
- Pure-Python fallback must remain first-class and tested.

## Packaging Expectations

- Use `PyO3` + `maturin` for wheel builds.
- Provide platform wheel matrix and source fallback behavior.
- Keep failure mode explicit when extension is unavailable.

## ROI Template

Before implementation, capture:

- baseline and projected p95/p99 improvement by workload class
- throughput and memory impact
- wheel/build complexity cost
- maintenance and platform compatibility risk

Only proceed with implementation if ROI is clear and maintainability cost is acceptable.
