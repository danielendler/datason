# Performance Strategy (Real-Data First)

datason v2 optimizes for **real workload latency**, not synthetic benchmark scores alone.

## Goals

- Primary metric: `p95` latency by workload class and operation.
- Secondary metrics: `p99`, throughput, and peak memory.
- Keep synthetic pytest-benchmark suite as a regression guardrail.

## Workload Taxonomy

- `api`: nested request/response payloads
- `event_log`: high-cardinality and sparse event records
- `data_science`: dataframe-like and feature-batch payloads
- `ml_serving`: model request/response envelopes

Sample corpus is stored at:

- `perf/workloads/sample/*.ndjson`
- schema: `perf/workloads/schema.json`
- anonymization contract: `perf/workloads/ANONYMIZATION.md`

## Running Real-Data Replay Benchmarks

```bash
just perf-real
```

This writes:

- `perf/results/replay-summary.json`

Compare against baseline:

```bash
just perf-real-compare perf/results/replay-summary-base.json
```

## CI Policy

- Real-data replay comparison runs in CI benchmark job.
- Initial policy is **non-blocking** warnings at `>10%` p95 regression.
- Promote to blocking only after stable baseline runs.

## Optimization Policy

1. Profile first (`cProfile`/trace evidence).
2. Optimize targeted hotspots only.
3. Re-measure real-data p95/p99 before and after each change.
4. Consider Rust only when Python-level optimizations are exhausted.
