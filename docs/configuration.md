# Configuration

Choose configuration around the output your consumer expects. The defaults
include type tags for Python reconstruction; `api_config()` disables tags for
ordinary JSON responses. See [Recipes](recipes.md) for complete workflows.

## Scope and precedence

Inline options override the active scope. Without a scope, they override
`SerializationConfig` defaults. The context manager yields the active frozen
config and restores the previous scope on exit, including after an exception.
It uses `ContextVar` for scoped thread/async-task state.

```python
import datason
from datason import NanHandling

with datason.config(sort_keys=True, nan_handling=NanHandling.STRING) as active:
    assert active.sort_keys
    assert datason.dumps({"v": float("NaN")}) == '{"v": "NaN"}'
    assert datason.dumps({"v": float("NaN")}, nan_handling=NanHandling.NULL) == '{"v": null}'
    # A new scope starts from defaults, not unspecified settings in the outer scope.
    with datason.config(sort_keys=True):
        assert datason.dumps({"v": float("NaN")}) == '{"v": null}'
    assert datason.dumps({"v": float("NaN")}) == '{"v": "NaN"}'
assert datason.dumps({"v": float("NaN")}) == '{"v": null}'
```

## All options

Use enum members, not their string spellings, for enum-valued options. Limits
must be nonnegative integers; negative values and booleans raise `ValueError`.
Not every formatting option affects loading: for example, type tags already
record the representation needed by the corresponding deserializer.

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `date_format` | `DateFormat` | `ISO` | Datetime: `ISO`, `UNIX`, `UNIX_MS`, `STRING` |
| `dataframe_orient` | `DataFrameOrient` | `RECORDS` | Frame: `RECORDS`, `SPLIT`, `DICT`, `LIST`, `VALUES` |
| `nan_handling` | `NanHandling` | `NULL` | Non-finite output: `NULL`, `STRING`, `KEEP`, `DROP` |
| `include_type_hints` | `bool` | `True` | Write type metadata when the handler supports it |
| `sort_keys` | `bool` | `False` | Sort output dictionary keys |
| `max_depth` | `int` | `50` | Representation traversal depth, including metadata |
| `max_size` | `int` | `100_000` | Entries per representation container |
| `max_string_length` | `int` | `1_000_000` | Characters per string/key |
| `max_input_bytes` | `int` | `16_777_216` | Incoming encoded JSON budget; also used for supported NumPy allocation estimates |
| `max_nodes` | `int` | `1_000_000` | Traversal work, including plugin conversion |
| `fallback_to_string` | `bool` | `False` | Stringify unsupported values; loses their original type |
| `strict` | `bool` | `True` | Raise for unknown type metadata during loading |
| `allow_plugin_deserialization` | `bool` | `True` | Permit typed plugin reconstruction during loading |
| `redact_fields` | `tuple[str, ...]` | `()` | Case-insensitive field-name substring matches |
| `redact_patterns` | `tuple[str, ...]` | `()` | Named patterns or custom regexes on string values |

`strict=False` leaves unknown tags as dictionaries; it does not bypass limits
or all malformed-payload errors. `allow_plugin_deserialization=False` rejects
plugin tags even with `strict=False`. Built-in collection tags can still restore
collections. See [Serialization boundaries](serialization-boundaries.md).

## Date formats

With tags disabled, the format determines the plain JSON value. This example
uses an aware UTC datetime so numeric output does not depend on local time:

```python
import datetime as dt
import json

import datason
from datason import DateFormat

stamp = dt.datetime(1970, 1, 1, 0, 0, 1, tzinfo=dt.timezone.utc)
assert json.loads(datason.dumps(stamp, include_type_hints=False,
                               date_format=DateFormat.ISO)) == "1970-01-01T00:00:01+00:00"
assert json.loads(datason.dumps(stamp, include_type_hints=False,
                               date_format=DateFormat.UNIX)) == 1.0
assert json.loads(datason.dumps(stamp, include_type_hints=False,
                               date_format=DateFormat.UNIX_MS)) == 1000.0
assert json.loads(datason.dumps(stamp, include_type_hints=False,
                               date_format=DateFormat.STRING)) == "1970-01-01 00:00:01+00:00"
```

Tagged numeric datetime records include explicit units and an ISO representation
for reconstruction. Naive numeric encoding uses UTC. Named timezone identity is
not preserved, although the offset is. See [Scientific fidelity](scientific-fidelity.md).

## Non-finite numbers

These policies apply to supported float leaves, including plugin output.
`allow_nan=False` is a final JSON encoder check: `KEEP` plus that argument raises
`ValueError`, while default normalization to null succeeds.

```python
import datason
from datason import NanHandling

assert datason.dumps({"v": float("NaN")}) == '{"v": null}'
assert datason.dumps({"v": float("NaN")}, nan_handling=NanHandling.STRING) == '{"v": "NaN"}'
assert datason.dumps({"v": float("NaN")}, nan_handling=NanHandling.KEEP) == '{"v": NaN}'
assert datason.dumps({"v": float("NaN")}, nan_handling=NanHandling.DROP) == '{"v": null}'
```

`KEEP` can produce tokens outside the JSON standard. `DROP` currently replaces
with null rather than removing a field or array element. These are output
policies; ordinary incoming JSON NaN tokens follow the stdlib decoder unless
you provide a `parse_constant` callback. Choose a policy that matches your schema
and stored-data requirements.

## DataFrame orientation

For API output without tags:

```python
import json

import pandas as pd

import datason
from datason import DataFrameOrient

frame = pd.DataFrame({"a": [1, 2]})
assert json.loads(datason.dumps(frame, include_type_hints=False,
                               dataframe_orient=DataFrameOrient.RECORDS)) == [{"a": 1}, {"a": 2}]
assert json.loads(datason.dumps(frame, include_type_hints=False,
                               dataframe_orient=DataFrameOrient.SPLIT)) == {
    "columns": ["a"], "index": [0, 1], "data": [[1], [2]],
}
```

`DICT` maps columns to index/value mappings, `LIST` maps columns to value lists,
and `VALUES` emits rows only. Tagged frames carry separate metadata; empty
frames, duplicate labels, and non-string columns may force split orientation
for fidelity. See [Scientific fidelity](scientific-fidelity.md).

## Presets

Factories return a `SerializationConfig`, not a context manager or a dict.
Use `dataclasses.asdict` to pass it as options. Factory overrides are supported:

```python
from dataclasses import asdict

import datason
from datason import api_config, NanHandling

preset = api_config(nan_handling=NanHandling.STRING)
with datason.config(**asdict(preset)):
    assert datason.dumps({"score": float("NaN")}) == '{"score": "NaN"}'
```

| Factory | Differences from defaults | Tradeoff |
| --- | --- | --- |
| `api_config()` | Sorted keys; no type tags | Ordinary JSON, no exact Python reconstruction |
| `ml_config()` | UNIX_MS dates; string fallback | Convenient export; unsupported objects lose type information |
| `strict_config()` | Explicit strict loading, tags, no string fallback | Same current defaults; not a requirement that all input contain tags |
| `performance_config()` | No tags, no sorting, KEEP non-finite values, string fallback | Lossy; may emit non-standard JSON; benchmark your workload |

For JSON encoder/decoder kwargs and their interaction with plugins, see the
[API reference](api.md#compatibility-with-stdlib-json). For redaction and budgets,
see [Security](security.md).
