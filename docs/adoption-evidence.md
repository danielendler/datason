# Application and adoption evidence

Checked October 4, 2026, after P1 PRs
[#127](https://github.com/danielendler/datason/pull/127),
[#128](https://github.com/danielendler/datason/pull/128) and
[#129](https://github.com/danielendler/datason/pull/129) merged.

## What the evidence supports

Datason has a concrete technical role for scientific Python values crossing
checkpoint and JSON boundaries. These local checks establish compatibility and
fidelity for bounded cases. They do not establish independent adoption,
production retention or willingness to pay. The author reports personal use in
financialModel02 and no known external use; that application is an author-owned
case, rather than an independent customer.

## Native framework comparison

The reproducible corpus compares the default LangGraph `JsonPlusSerializer`
with `DatasonSerializer`, without pickle fallback. Each success checks values,
dtype and shape, including scalar dtype and empty dimensions.

| Case | Native older/current | Datason older/current |
| --- | --- | --- |
| NumPy int32 scalar | TypeError | Preserved |
| NumPy float32 scalar | TypeError | Preserved |
| Contiguous datetime64[D] array | TypeError | Preserved |
| Contiguous timedelta64[ms] array | TypeError | Preserved |
| Empty float32 array, shape (0, 3) | Preserved | Preserved |

Older pins are LangGraph 1.0.0/checkpoint 2.1.2; current pins are
1.2.12/4.2.0. These are synthetic local reproducers of the scientific boundaries
in [#8689](https://github.com/langchain-ai/langgraph/issues/8689) and
[#8705](https://github.com/langchain-ai/langgraph/issues/8705), not a verbatim
execution of every issue attachment. Native success on the empty numeric array
is useful counterevidence: Datason is not necessary for every NumPy checkpoint.

From a checkout, select the desired pins:

```bash
uv run --locked --extra numpy \
  --with langgraph==1.2.12 --with langgraph-checkpoint==4.2.0 \
  python -m scripts.validate_framework_boundaries
```

The report separates installed distribution metadata from runtime source
hashes, which matters when testing a checkout with `PYTHONPATH` against an older
editable installation. The recorded [current report](evidence/2026-10-04-langgraph-current.json) and
[older report](evidence/2026-10-04-langgraph-older.json) retain the exact local
results. Native fixes change the report to success; regression
tests require Datason fidelity without requiring upstream to remain broken.
The [framework matrix](framework-compatibility.md) separately tests actual
SQLite reopen/resume, interrupts, Send packets and Agents hooks. Passing this
small serializer corpus is not a replacement for those lifecycle tests.

CrewAI's Pydantic persistence fix and the Agents structured diagnostic fix were
already merged upstream. They are evidence of the underlying problem, not
current reasons to replace native serialization. Pydantic JSON mode remains a
reasonable first choice when the application already has an adequate wire
model. MCP schema generation and nullable/binary representation require an
application-owned schema contract; normalization alone cannot resolve them.

## financialModel02 source review

The reviewed application revision is
[`6011e2d`](https://github.com/danielendler/financialModel02/tree/6011e2d0100160992e85aa8dad939df630175990).
Its main commit is dated June 13, 2025, outside the opportunity review's
July–October 2026 window. Repository push dates and source comments do not
establish when an application was deployed.

| Source at that revision | Observed use |
| --- | --- |
| [shared/utils/enhanced_datason.py](https://github.com/danielendler/financialModel02/blob/6011e2d0100160992e85aa8dad939df630175990/shared/utils/enhanced_datason.py) | Separate API, typed database/cache and financial analytics policies; DataFrame orientations; restoration helpers |
| [backend/app/database/enhanced_queries.py](https://github.com/danielendler/financialModel02/blob/6011e2d0100160992e85aa8dad939df630175990/backend/app/database/enhanced_queries.py) | JSON storage and cache serialization/restoration paths |
| [backend/app/api/v1/recurring/router.py](https://github.com/danielendler/financialModel02/blob/6011e2d0100160992e85aa8dad939df630175990/backend/app/api/v1/recurring/router.py) | Enhanced recurring-candidate endpoint calls response formatting |
| [backend/app/utils/datason_utils.py](https://github.com/danielendler/financialModel02/blob/6011e2d0100160992e85aa8dad939df630175990/backend/app/utils/datason_utils.py) | DataFrame/Series pre-conversion and recursive timestamp post-processing around Datason |
| [backend/app/utils/recursive_serialize.py](https://github.com/danielendler/financialModel02/blob/6011e2d0100160992e85aa8dad939df630175990/backend/app/utils/recursive_serialize.py) | Parallel custom conversion implementation remains |
| [backend/requirements.lock](https://github.com/danielendler/financialModel02/blob/6011e2d0100160992e85aa8dad939df630175990/backend/requirements.lock) | Datason pinned to 0.8.0; unpinned requirements use >=0.8.0 |

The wrappers call old functions such as `serialize`, `auto_deserialize` and
`get_api_config`, and use `check_if_serialized`. Those are absent from the
current five-function API/configuration. This source does not establish v2
application compatibility. Updating the dependency alone is insufficient; an
application upgrade needs explicit call-site changes and persisted-data
fixtures. The retained custom serializer also means this is not proof that all
conversion work was eliminated. No production application, database or customer
payload was executed during this review.

## Bounded application-shaped validation

The [financial example](https://github.com/danielendler/datason/blob/main/examples/financial_snapshot.py)
uses new synthetic values shaped like recurring-candidate data. It exercises
the existing v2 API, without importing or modifying financialModel02:

```bash
uv run --locked --extra numpy --extra pandas python -m examples.financial_snapshot
```

The ordinary API projection emits UUID/datetime strings, decimal text, numeric
feature lists and record-oriented transactions with nullable fields. A separate
trusted snapshot restores Decimal, UUID, timezone-aware datetime, float32 scalar
and matrix dtype/shape, and a DataFrame with nullable Int64 and timezone-aware
columns. Tests assert these contracts and isolate storage from active diagnostic
redaction. This demonstrates a useful replacement for some manual conversion,
not a completed application migration or a claim to restore every financial
object. Currency, rounding and monetary schema rules remain application-owned.

## Adoption milestone remains open

Before expanding integrations, obtain independent application snapshots and
record: which native alternative was tried, repeated custom conversion removed,
loss caught, upgrade effort, and whether the team keeps the adapter. The earlier
suggested threshold is three independent teams choosing continued use. There
are currently no such confirmed teams or retention measurements. No maintainer
outreach or production performance claim was made in this work.

The author-owned financial case can provide a first upgrade validation once its
actual stored payloads and application tests are available. Keep that result
separate from independent adoption, and preserve old fixtures before changing
storage formats. Do not broaden the runtime API merely to satisfy old wrapper
names; use the existing documented contracts.
