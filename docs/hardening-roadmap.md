# Datason hardening and AI integration roadmap

Review date: October 3, 2026. This is a proposal and implementation record for
open PRs, not a claim that these changes are available in the published alpha.

## Product direction

The useful opportunity is a reliable Python data boundary: normalize supported
values for tool/API responses, export diagnostics under explicit redaction
policies, and preserve supported types in internal stored state. A general Swiss
army knife is too broad a contract to validate or explain well. Keep the core
small, make policies explicit, and attach integrations to demonstrated workloads.

The recent framework review found demand around checkpoint serialization and
durable state, but some adjacent problems need orchestration, migration, storage,
or schema validation. Those are separate product responsibilities. For example,
[LangGraph #8689](https://github.com/langchain-ai/langgraph/issues/8689),
[#8705](https://github.com/langchain-ai/langgraph/issues/8705), and
[Pydantic AI #6675](https://github.com/pydantic/pydantic-ai/issues/6675) are useful
investigation leads; issue reports establish symptoms, not datason adoption.
Pydantic AI's [message-history documentation](https://ai.pydantic.dev/message-history/)
also describes deliberate JSON normalization. Preserve types only where the
consumer's contract requires it.

Older AI models may explain how weaknesses entered the project. Release readiness
should depend on reproducible behavior, security boundaries, and compatibility
fixtures rather than which model wrote the code.

## First implementation batch

| Work | PR | Evidence and limits |
| --- | --- | --- |
| Authenticated integrity envelopes | [#111](https://github.com/danielendler/datason/pull/111) | 51 focused tests; keyed verification rejects hash downgrade |
| Explicit pickle trust | [#112](https://github.com/danielendler/datason/pull/112) | 23 focused tests; default rejection happens before file reads/unpickling |
| Shared policies and input budgets | [#113](https://github.com/danielendler/datason/pull/113) | 557 tests passed; validate the full JSON representation before reconstruction |
| Independent regression CI | [#114](https://github.com/danielendler/datason/pull/114) | Python 3.10–3.13 test jobs passed; quality/security gates remain enforced |
| Scientific fidelity | [#115](https://github.com/danielendler/datason/pull/115) | 611 tests passed; dtype, shape, Pandas labels/dtypes, timestamp units |
| Structured agent data | [#116](https://github.com/danielendler/datason/pull/116) | 570 tests passed; application models normalize without dynamic reconstruction |
| Security lock refresh | [#117](https://github.com/danielendler/datason/pull/117) | No known advisories in 161 applicable pinned dependencies; CI quality/tests/build passed |
| LangGraph checkpoint adapter | [#118](https://github.com/danielendler/datason/pull/118) | SQLite close/reopen/resume verified; pinned framework CI passed on Python 3.11 and 3.13 |

Local suites use Python 3.12, NumPy 2.3.5, Pandas 2.2.3, and Pydantic 2.13.5.
Three optional ML test modules were skipped locally. Counts are branch-specific
and must not be added together. See each PR for validation scope and API changes.
A local combined merge with the latest main branch passed 658 tests and all 30
snapshots, plus lint, formatting, and module/function limits. The refreshed lock
was reconciled with the Pydantic extra during that combined validation.
The dependency PR's non-blocking replay benchmark reported a small-load p95
increase (0.020 ms to 0.026 ms); investigate repeatability before claiming a
performance improvement. This batch makes no throughput claim.

## Merge and release sequence

1. Review #117's major library upgrades and #111/#112's security behavior. They
   target main independently. Existing single-package dependency PRs overlap with
   #117 and should be reconciled after it lands.
2. Resolve the overlap between existing compatibility PRs #97 and #101. This batch
   builds on #101. Merge #101, then the CI follow-up #114 and policy PR #113,
   retargeting each stacked PR to main after its dependency lands.
3. Merge #115 and #116 after #113. Merge #118 after #116. When the lock refresh and
   the Pydantic extra meet, regenerate `uv.lock` from the refreshed versions and
   preserve all security upgrades. A local combined merge verified that resolution.
4. Re-run the full CI matrix on the resulting main branch and retain serialized
   fixtures from previous alpha releases. Document the intentional changes: pickle
   trust opt-in, metadata-key/key-collision rejection, metadata-inclusive budgets,
   exact scalar dispatch, scientific metadata, and model normalization.
5. Cut the next alpha only after required checks pass. PR creation does not
   authorize merging or publishing a release.

## Next validation milestones

| Priority | Work | Completion criterion |
| --- | --- | --- |
| P1 | Persisted format fixtures and migration contract | Read fixtures from each supported prior alpha; document which representations require migration |
| P1 | ML reconstruction review | Enumerate constructors, file/import behavior, and allocation paths for every plugin before recommending untrusted typed loads |
| P1 | Framework compatibility matrix | Exercise actual supported SDK versions, model hydration, interrupts, and checkpoint schema upgrades; avoid implicit pickle fallbacks |
| P2 | API/tool schema validation example | Produce ordinary JSON for an actual tool result and validate it against its declared schema, including binary field encoding |
| P2 | Performance and optional imports | Measure cold import, allocations, representative dump/load p50/p95, and standard JSON baselines using the merged replay framework |
| P2 | Adoption evidence | Reproduce reported framework failures, validate adapter usefulness with maintainers/users, and observe continued use before expanding integrations |

Do not add more type families merely to increase a supported-type count. Broaden
the contract only when a concrete workload, fidelity fixture, and integration test
justify the maintenance cost. Security/resource checks are not a sandbox for
trusted plugins, parser hooks, model serializers, or arbitrary Python callbacks.
