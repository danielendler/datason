# Replay Workload Anonymization Contract

All replay workload records committed to this repository must follow these rules:

1. No raw production PII, secrets, tenant identifiers, or customer content.
2. Use deterministic placeholder identifiers for stable comparisons.
3. Truncate free-text fields to representative length; avoid real messages.
4. Keep payload shape and type distributions representative of production.
5. Record provenance in `source_tag` (e.g., `sample:api:v1`).

Recommended redaction strategy for exported datasets:

- Keys that identify users/tenants: replace with deterministic hashes.
- Large text fields: keep first N chars + marker suffix.
- Arrays: preserve shape, optionally perturb values while keeping ranges.
- Timestamps: shift by a constant offset to preserve intervals.

Only small anonymized samples should be committed to `perf/workloads/sample/`.
Larger datasets should be provided externally (CI artifact input or local ignored paths).
