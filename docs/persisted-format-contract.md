# Persisted formats and migration contract

The supported historical producer is **v2.0.0a1**, source commit
`88b110f79922ceef7e4ae3bc92392c0bdbdfa2ba`. The core reader recognizes supported
`__datason_type__` / `__datason_value__` records; this envelope does not carry an
application schema version or promise arbitrary historical plugin compatibility.
Unknown tags fail under the default strict policy. No dotted application class
names are imported automatically to hydrate models.

## Compatibility evidence

Tests retain the original eight payloads and 15 additional payloads captured from
that exact source: fitted LinearRegression and a fitted Pipeline, SciPy CSR,
Torch/TF/JAX float32 arrays, a TF sparse tensor, a numeric Polars frame, a Plotly
figure, CatBoost and Optuna diagnostic metadata, a custom plugin record, two
ambiguous numeric timestamps and a compact legacy HMAC envelope. Capture records
Python and every optional library version. These are historical **Datason**
fixtures produced with the recorded libraries, not artifacts from old releases
of every ML framework.

The capture script verifies the producer commit and unmodified source. Preserve
the stored strings rather than regenerating them with a candidate writer:

```bash
git worktree add --detach /tmp/datason-a1 v2.0.0a1
PYTHONPATH=/tmp/datason-a1 python scripts/capture_alpha1_compat.py \
  --output /tmp/v2.0.0a1-extended.json
pytest tests/integration/test_published_alpha_fixtures.py \
  tests/integration/test_extended_alpha_fixtures.py
```

Use an isolated environment containing Datason distribution metadata, the
optional packages at the fixture's versions, and a CPU-capable runtime. The
producer is established from the imported source, not distribution metadata.
Tests skip a family when its optional library is absent; optional compatibility
CI must install the libraries to exercise those cases.

## Reader and migration decisions

| Stored representation | Contract and action |
| --- | --- |
| Standard tagged values and supported NumPy arrays | Read with the current loader; validate required values, dtype and shape |
| Legacy scalar tag without dtype | Reads as int64/float64/complex128; original width is absent and cannot be inferred |
| Legacy untagged tuple/set/frozenset | Reads as a list; restore a collection only from an authoritative application schema |
| Supported fitted sklearn fixture | Predictions match in the tested library versions; pin dependencies and verify predictions before migrating other estimators |
| CatBoost/Optuna export | Diagnostic metadata, not model weights or resumable study storage; use framework-native artifacts for executable state |
| Custom plugin record | Register reviewed code explicitly; version the application's tags and support old tags deliberately |
| Ambiguous numeric datetime | Supply the producer's seconds/milliseconds policy and timezone semantics; never infer them from magnitude |
| Default-formatted legacy integrity envelope | Supported by the legacy verification path; keyed verification still requires an HMAC |
| Custom-formatted legacy signature | Requires the original signed bytes; never sign a payload after verification failed |
| Native LangGraph checkpoint | Different serializer format; export validated application state to a new Datason-backed thread rather than silently rewriting framework history |
| v1-era Datason formats | No general v1 reader guarantee; use a pinned legacy reader in a trusted isolated environment, validate, then write the current format |

For owned numeric-date records missing `timestamp_unit`, the fixture tests show
adding the **known** unit before loading. They intentionally prove that the old
magnitude heuristic gives a different date in the ambiguous cases. Legacy naive
local timestamps additionally require the producer's timezone; keep that
conversion in application migration code and write an explicit ISO datetime.
Do not edit a signed payload before authentication; authenticate and unwrap it
first, then validate and convert.

For a custom-formatted legacy HMAC, if the exact original JSON is retained:

```python
import json
from datason.security.integrity import verify_hmac, wrap_with_integrity

signature = json.loads(legacy_envelope)["__datason_hmac__"]
if not verify_hmac(original_signed_json, key, signature):
    raise ValueError("Original snapshot authentication failed")
# Use only the authenticated original; ignore unsigned envelope data.
upgraded_envelope = wrap_with_integrity(original_signed_json, key=key)
```

If the signed bytes are gone, do not guess formatting or bypass verification.
Recover from a trusted backup/source. A hash alone authenticates no owner.

## Application schema upgrades

Persist an application `schema_version` alongside the data. Authenticate the
complete snapshot where needed, check supported versions, apply explicit
application migrations, validate the target schema, then write a new snapshot.
Keep originals until resume/prediction checks pass. Never mutate a checkpoint DB
in place while an execution may still use it. Datason's type envelope and
LangGraph's internal checkpoint version do not replace this application contract.
