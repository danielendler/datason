# datason

**Serialize Python data to JSON for APIs, diagnostics, and stored state.**
datason handles datetime, UUID, Decimal, paths, and collections, with optional
plugins for NumPy, Pandas, and ML libraries. The core has no runtime dependencies.
Python 3.10+ is required.

!!! note "Match your installation to these docs"
    These docs follow development `main`, currently versioned `2.0.0a2`.
    As of October 4, 2026, PyPI's stable release is v1 (`0.13.0`) and its
    published v2 alpha is `2.0.0a1`. Some features here are newer than that alpha.
    Start with the [installation guide](getting-started.md#installation).

## Your first round trip

```python
import datetime as dt
from decimal import Decimal

import datason

event = {"observed": dt.datetime(2026, 10, 4, tzinfo=dt.timezone.utc),
         "price": Decimal("19.99")}
text = datason.dumps(event)
restored = datason.loads(text)
assert restored == event
```

The JSON contains type metadata so datason can restore supported Python types.
For example, `Decimal("19.99")` becomes:

```json
{"__datason_type__": "decimal.Decimal", "__datason_value__": "19.99"}
```

An ordinary JSON reader sees that object; it does not reconstruct a Decimal.
For a consumer expecting a plain string, use `include_type_hints=False`.

## Choose the output your consumer needs

| Your task | Start with | What to expect |
| --- | --- | --- |
| Return an API or tool response | [API recipe](recipes.md#api-and-tool-responses) | Plain JSON with tags disabled; validate against the consumer's schema |
| Export logs and diagnostics | [Redaction recipe](recipes.md#redacted-diagnostics) | Explicit field/pattern redaction; keep the original state separately |
| Save Python data for later | [Stored-data recipe](recipes.md#typed-stored-data) | Type tags enabled; supported types reconstructed by datason |
| Preserve arrays and DataFrames | [Scientific fidelity](scientific-fidelity.md) | Defined dtype, shape, and index contracts, with examples |
| Resume a LangGraph workflow | [Checkpoint guide](langgraph-checkpoints.md) | An opt-in serializer and complete SQLite pause/resume example |
| Handle your own Python type | [Custom plugins](plugins.md) | A runnable plugin and round-trip check |

## Familiar interface, explicit policies

The five everyday operations are `dumps`, `loads`, `dump`, `load`, and `config`.
[Configuration enums and preset factories](configuration.md) are also exported.
Common stdlib JSON arguments such as `indent` and `parse_float` work, but defaults
differ: datason emits Unicode directly, converts non-finite numbers to `null`,
and includes type metadata. It also enforces traversal and input budgets.
See [JSON compatibility](api.md#compatibility-with-stdlib-json).

Type preservation depends on the type and policy: application models normalize
to fields, redaction changes data, and some ML plugins export metadata only.
The [supported-types table](supported-types.md) explains what comes back.

Continue with [Getting started](getting-started.md) for installation and complete
examples. Use [Troubleshooting](troubleshooting.md) when an error or unexpected
output gets in the way, and the [API reference](api.md) for exact call behavior.
