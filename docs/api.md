# API reference

The five everyday operations mirror the stdlib JSON interface. Configuration
enums, `SerializationConfig`, and preset factories are also exported; see
[Configuration](configuration.md). These signatures describe the current source.

## `datason.dumps`

```text
datason.dumps(obj: Any, **kwargs: Any) -> str
```

Convert a supported Python value to a JSON string. Type metadata is enabled by
default. Accepts configuration fields and the encoder options listed below.
Raises `SerializationError` for unsupported values and `SecurityError` for
budget or circular-reference violations.

```python
from decimal import Decimal

import datason

assert datason.dumps({"price": Decimal("19.99")}, include_type_hints=False) == '{"price": "19.99"}'
assert datason.dumps({"z": 1, "a": 2}, sort_keys=True) == '{"a": 2, "z": 1}'
```

## `datason.loads`

```text
datason.loads(s: str | bytes | bytearray, **kwargs: Any) -> Any
```

Parse JSON, validate representation budgets, then reconstruct supported tagged
values. Untagged strings remain strings. Accepts configuration fields and the
decoder options below. Missing libraries, unknown tags in strict mode, and invalid
typed payloads can raise `DeserializationError`. Invalid JSON raises stdlib
`json.JSONDecodeError`; parser recursion failures become `SecurityError`.

```python
import datetime as dt

import datason

value = {"observed": dt.datetime(2026, 10, 4, tzinfo=dt.timezone.utc)}
assert datason.loads(datason.dumps(value)) == value
assert datason.loads(b'{"value": 1}') == {"value": 1}
```

Use `allow_plugin_deserialization=False` for ordinary incoming JSON when typed
plugin reconstruction should be rejected. Built-in collection tags may still
restore collections; callbacks remain trusted code. See
[Serialization boundaries](serialization-boundaries.md).

## `datason.dump` and `datason.load`

```text
datason.dump(obj: Any, fp: IOBase, **kwargs: Any) -> None
datason.load(fp: IOBase, **kwargs: Any) -> Any
```

`dump` writes JSON text to a writable file-like object, and `load` reads from an
object with `read`. Use a text stream for writing. Reading supports JSON text or
bytes under the input budget. Options match `dumps` and `loads` respectively.
These operations do not provide incremental dataset streaming or atomic writes.

```python
import io
from decimal import Decimal

import datason

buffer = io.StringIO()
data = {"price": Decimal("19.99")}
assert datason.dump(data, buffer, indent=2) is None
buffer.seek(0)
assert datason.load(buffer) == data
```

## `datason.config`

```text
datason.config(**kwargs: Any) -> context manager yielding SerializationConfig
```

Select configuration temporarily. Inline call options override the active scope.
A new scope starts from defaults, and the previous scope is restored on exit.
Only configuration fields are accepted here; encoder options such as `indent`
belong on `dumps`/`dump`. See [Scope and precedence](configuration.md#scope-and-precedence).

```python
import datason
from datason import NanHandling

with datason.config(nan_handling=NanHandling.STRING) as active:
    assert active.nan_handling is NanHandling.STRING
    assert datason.dumps([float("nan")]) == '["NaN"]'
assert datason.dumps([float("nan")]) == '[null]'
```

## Compatibility with stdlib `json`

| Aspect | datason behavior |
| --- | --- |
| Interface | `dumps`/`loads`/`dump`/`load` with JSON kwargs and additional config fields |
| Default Unicode | `ensure_ascii=False`; set `True` for stdlib-style escaping |
| Default non-finite values | Normalize to `null`; `NanHandling.KEEP` preserves non-standard tokens |
| Supported Python types | Plugins and collection tags add representations beyond stdlib |
| Unknown values | `SerializationError` by default; callbacks or string fallback can normalize |
| Object keys | Normalize to strings; reject reserved tags and normalization collisions |
| Limits | Input and traversal budgets enforced, including metadata |

`dumps`/`dump` accept `indent`, `ensure_ascii`, `separators`, `allow_nan`,
`skipkeys`, `check_circular`, `default`, and `cls`. `sort_keys` is a configuration
field. Registered plugins take priority over `default` or a `json.JSONEncoder`
subclass. Callback results are traversed under datason's policies and budgets.
`check_circular=False` affects the final JSON encoder while datason still checks
circular references. `skipkeys=True` omits keys outside stdlib's accepted key types.

`loads`/`load` accept `parse_float`, `parse_int`, `parse_constant`, `object_hook`,
`object_pairs_hook`, and `cls`. These callbacks run during JSON parsing, before
datason's full-tree validation and reconstruction. They can change what plugins
receive. They are trusted application code and are not sandboxed.

```python
from decimal import Decimal

import datason

assert datason.loads('{"price": 19.99}', parse_float=Decimal) == {"price": Decimal("19.99")}
assert datason.dumps({"name": "café"}, ensure_ascii=False) == '{"name": "café"}'
```

Unknown keyword arguments raise `TypeError` naming the receiving function.
`strict=False` controls unknown type tags, not JSON syntax or all plugin errors.
See [Troubleshooting](troubleshooting.md) for common output surprises.

## Error types

Import the error classes from `datason._errors`. This underscored module is the
current error import path and is subject to alpha API changes.

| Error | Meaning |
| --- | --- |
| `DatasonError` | Base class for datason errors |
| `SecurityError` | Budget or circular-reference violation; always propagates |
| `SerializationError` | Unsupported value, reserved key, key collision, or normalization failure; `path` identifies a field when available |
| `DeserializationError` | Unknown, disallowed, or malformed typed representation |
| `PluginError` | A plugin signals it cannot complete the operation; registry warns and tries another plugin |

Other errors from stdlib parsing, user callbacks, and third-party libraries may
propagate. String fallback does not suppress security failures. Inspect
`SerializationError.path` with the [error recipe](recipes.md#find-the-unsupported-field).
