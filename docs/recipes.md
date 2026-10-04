# Recipes

Each example is complete and can run independently on the source version
covered by these docs. See [Installation](getting-started.md#installation) first.

## API and tool responses

Use plain JSON when the consumer does not understand datason metadata. This
example chooses ISO dates, sorted keys, and no tags through `api_config`:

```python
import datetime as dt
import json
from dataclasses import asdict
from decimal import Decimal

import datason
from datason import api_config

response = {"created": dt.datetime(2026, 10, 4, tzinfo=dt.timezone.utc),
            "price": Decimal("19.99"), "score": float("nan")}
with datason.config(**asdict(api_config())):
    text = datason.dumps(response)
assert json.loads(text) == {
    "created": "2026-10-04T00:00:00+00:00", "price": "19.99", "score": None,
}
```

Decide whether your schema permits a Decimal string and a nullable score before
returning the result. Serialization converts values; it does not validate a tool
schema. For an ordinary JSON request, use
`datason.loads(text, allow_plugin_deserialization=False)` to reject plugin tags.
See [Serialization boundaries](serialization-boundaries.md) for the collection-tag
exception and trusted callback behavior.

## Redacted diagnostics

Choose field names and patterns for the records you actually export:

```python
import json

import datason

record = {"user": "alice", "api_token": "private-token",
          "message": "Contact alice@example.com", "latency_ms": 12}
text = datason.dumps(record, include_type_hints=False,
                     redact_fields=("token",), redact_patterns=("email",))
assert json.loads(text) == {
    "user": "alice", "api_token": "[REDACTED]",
    "message": "Contact [REDACTED]", "latency_ms": 12,
}
assert record["api_token"] == "private-token"
```

Field matching uses case-insensitive substrings: `key` also matches `monkey`.
Pattern matching runs on string values. Test your rules against representative
records; matching rules do not establish that every sensitive field was removed.
Keep redacted exports separate from resumable state. See [Security](security.md).

## Typed stored data

Use tags and test the restored values that matter to your application. This
example requires `numpy,pandas` and uses finite array values:

```python
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import datason

state = {"weights": np.array([0.25, 0.75], dtype=np.float32),
         "metrics": pd.DataFrame({"count": pd.array([1, None], dtype="Int64")}),
         "budget": Decimal("10.00"), "payload": b"data"}
with TemporaryDirectory() as directory:
    path = Path(directory) / "state.json"
    with path.open("w", encoding="utf-8") as file:
        datason.dump(state, file)
    with path.open(encoding="utf-8") as file:
        restored = datason.load(file)
np.testing.assert_array_equal(restored["weights"], state["weights"])
assert restored["weights"].dtype == np.dtype("float32")
pd.testing.assert_frame_equal(restored["metrics"], state["metrics"])
assert restored["budget"] == state["budget"]
assert restored["payload"] == b"data"
```

Retain representative stored JSON as upgrade fixtures. JSON tags are a
version-sensitive format, and disabling them changes the storage contract.
For graph workflows, see the [LangGraph checkpoint example](langgraph-checkpoints.md).

## Find the unsupported field

An unsupported value raises `SerializationError`. Use its `path` to find the
field instead of enabling a blanket string fallback:

```python
import datason
from datason._errors import SerializationError

class Unsupported:
    pass

try:
    datason.dumps({"results": [{"value": Unsupported()}]})
except SerializationError as error:
    assert error.path == "$.results[0].value"
else:
    raise AssertionError("Expected an unsupported-value error")
```

Normalize that field explicitly, use `default=` for a one-way JSON conversion,
or write a [custom plugin](plugins.md) when reconstruction is required.
`fallback_to_string=True` is a deliberate lossy export policy.
