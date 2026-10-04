# Datason hardening and AI integration roadmap

Review date: October 4, 2026. Earlier hardening batches and release preparation
are merged. The P1 follow-up below is implemented in review branches; its PRs
must merge before those changes are part of main. The published alpha has not
been replaced.

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
| Scientific fidelity | [#115](https://github.com/danielendler/datason/pull/115) | 616 tests passed with Pandas 2.2 and 3.0; dtype, shape, labels, timestamp units |
| Structured agent data | [#116](https://github.com/danielendler/datason/pull/116) | 570 tests passed; application models normalize without dynamic reconstruction |
| Security lock refresh | [#117](https://github.com/danielendler/datason/pull/117) | No known advisories in 161 applicable pinned dependencies; CI quality/tests/build passed |
| LangGraph checkpoint adapter | [#118](https://github.com/danielendler/datason/pull/118) | SQLite close/reopen/resume verified; pinned framework CI passed on Python 3.11 and 3.13 |

Local suites use Python 3.12, NumPy 2.3.5, Pandas 2.2.3/3.0.1, and Pydantic 2.13.5.
Three optional ML test modules were skipped locally. Counts are branch-specific
and must not be added together. See each PR for validation scope and API changes.
The fully integrated branch passed 660 tests and all 30 snapshots, plus lint,
formatting, and module/function limits. The refreshed lock was reconciled with
the Pydantic extra, and the final lockfile check passed.
The dependency PR's non-blocking replay benchmark reported a small-load p95
increase (0.020 ms to 0.026 ms); investigate repeatability before claiming a
performance improvement. This batch makes no throughput claim.

## Integration and release status

The dependency refresh #117 and security fixes #111/#112 landed first. CI follow-up
#114 was merged into compatibility PR #101 so that its strict typing job installed
the optional libraries it checks. #101 then landed on main, followed by policy
PR #113, structured-data PR #116, scientific PR #115, and the checkpoint adapter
#118. Stacked PRs were retargeted to main as their dependencies landed.

The lockfile conflict was resolved by preserving the refreshed dependency versions
and regenerating the lock with the four additional Pydantic packages. Broader CI
also exposed Pandas 3's different string missing-value convention; metadata now
preserves that convention, and snapshot inputs use explicit stable column dtypes.
The adapter's byte-input typing was aligned with the supported runtime contract.

The callback compatibility and packaging follow-up #97 is now merged, as are
complex scalar dispatch #122, failure paths #123, NumPy/traversal optimizations
#124 and email-redaction fast-path #125. The final main code tree matches the
combined local run: 695 tests passed, five optional skips and all 30 snapshots.
Main CI and LangGraph compatibility checks are green.

Release preparation targets **2.0.0a2**, retaining alpha status. Eight persisted
payload fixtures captured from the actual a1 release tag now cover standard
values, legacy collections, NumPy, a basic DataFrame, numeric datetimes and
legacy default-formatted integrity envelopes. This is initial compatibility
coverage, not a comprehensive migration guarantee. The
[release notes](releases/2.0.0a2.md) document trust opt-in, key rejection,
metadata-inclusive budgets, scalar metadata and model normalization. Preparing
or merging release metadata does not tag, publish or upload the package.

## P1 follow-up prepared for review

These are fidelity, reconstruction and integration milestones for the existing
five-function API, not a renewed API migration. The combined candidate passes
**944 tests without skips and all 30 snapshots** locally with actual optional
libraries and current framework pins. Branch-specific counts are in each PR.

| Work | Implementation and evidence | Remaining boundary |
| --- | --- | --- |
| Persisted formats ([#127](https://github.com/danielendler/datason/pull/127)) | 24 actual a1 payload fixtures, including fitted ML predictions, custom codecs, ambiguous units, unsigned overflow and legacy signatures; explicit migration decisions | Missing producer information cannot be guessed; fixture coverage is bounded to recorded library versions |
| Reconstruction review ([#128](https://github.com/danielendler/datason/pull/128)) | Inventory every built-in loader; 69 regression cases for pre-allocation checks, empty shapes, bfloat16, CPU restoration, JAX x64 policy, estimator hydration and dispatch-off | Typed loads remain for trusted application-owned state; buffer estimates and namespace checks are not a sandbox |
| [Framework matrix](framework-compatibility.md) ([#129](https://github.com/danielendler/datason/pull/129)) | Real SQLite reopen/resume, closed Interrupt/Send codecs, application hydration/schema upgrade, old checkpoint replay and offline Agents approval/context restoration; two pinned SDK generations with Python 3.11/3.13 CI jobs | Native-format migrations, arbitrary message classes, Send timeout policies and replay authorization remain application responsibilities |

Release preparation remains **2.0.0a2 (unreleased)**. This follow-up expands the
original eight-fixture sample and the initial checkpoint adapter; it does not
tag or publish the package. The framework matrix describes the exact supported
versions and evidence rather than claiming compatibility with every SDK release.

## Remaining validation milestones

| Priority | Work | Completion criterion |
| --- | --- | --- |
| P2 | API/tool schema validation example | Produce ordinary JSON for an actual tool result and validate it against its declared schema, including binary field encoding |
| P2 | Performance and optional imports | Measure cold import, allocations, representative dump/load p50/p95, and standard JSON baselines using the merged replay framework |
| P2 | Adoption evidence | Reproduce reported framework failures, validate adapter usefulness with maintainers/users, and observe continued use before expanding integrations |

Do not add more type families merely to increase a supported-type count. Broaden
the contract only when a concrete workload, fidelity fixture, and integration test
justify the maintenance cost. Security/resource checks are not a sandbox for
trusted plugins, parser hooks, model serializers, or arbitrary Python callbacks.
