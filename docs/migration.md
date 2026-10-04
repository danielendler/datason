# Migration from v1

v2 is a rewrite with different entry points, configuration, and metadata.
Choose a v2 installation explicitly using [Getting started](getting-started.md#installation).
An unqualified PyPI install currently selects stable v1.

## Map the operations

| v1 intent | v2 operation |
| --- | --- |
| Serialize to an intermediate Python representation | `dumps(obj)` returns JSON text; use a JSON reader if you need primitives |
| Restore an intermediate representation | `loads(text)` accepts JSON text/bytes, not an already parsed dict |
| ML-oriented export | `dumps(obj, **asdict(ml_config()))` |
| Read/write files | `load(file)` / `dump(obj, file)` on open file-like objects |
| Specialized smart/perfect loaders | Choose explicit config and reconstruction contracts |
| Python 3.8+ | Python 3.10+ |

Do not replace `serialize` with `dumps` without checking the expected return
type. For a consumer expecting ordinary JSON objects, disable tags and parse the
result; for a framework expecting a JSON string, pass it directly.

## Replace specialized helpers with explicit settings

```python
from dataclasses import asdict

import datason
from datason import ml_config

output = {"loss": 0.25}
text = datason.dumps(output, **asdict(ml_config()))
assert datason.loads(text) == output
```

The ML preset enables string fallback, which loses unsupported type information.
Use strict defaults and workload-specific validation for stored state.
See [Configuration](configuration.md) and [Supported types](supported-types.md).

## Inspect existing stored data

v2 uses `__datason_type__` / `__datason_value__` tags. Historical v1 formats are
not covered by a blanket compatibility guarantee. Keep the old decoder available,
read representative records, normalize or convert them under an explicit schema,
and validate the v2 output before switching readers or replacing files.

The current source tests the original eight fixture payloads and 16 additional
payloads captured from `2.0.0a1`; that sample does
not establish compatibility with every v1 or alpha payload. Old untagged
collections remain lists, missing scalar dtype information cannot be recovered,
and legacy numeric timestamps may have ambiguous units. See
[Release notes](releases/2.0.0a2.md) for the covered cases and the [persisted-format contract](persisted-format-contract.md)
for explicit migration decisions.

## Review policy changes

- NaN/Infinity normalize to null by default; select a documented policy.
- Reserved metadata keys and normalized key collisions raise errors.
- Representation metadata and plugin output count toward traversal budgets.
- Application models normalize to fields; explicitly validate or hydrate classes.
- Pickle migration requires explicit trust; see [Trusted pickle migration](pickle-migration.md).
- A LangGraph serializer change does not migrate existing checkpoints; see [LangGraph](langgraph-checkpoints.md).

Run round-trip assertions against real fixtures and dependency versions before
an upgrade. [Troubleshooting](troubleshooting.md) covers missing plugins, limits,
and output that differs from your expected schema.
