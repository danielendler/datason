# Serialization policies and trust boundaries

Datason applies redaction and non-finite-number policies to every string and
numeric leaf, including values returned by type plugins. Metadata type names are
preserved so field redaction does not rewrite the dispatch label. Redaction can
make typed data impossible to reconstruct; use a diagnostic copy when exporting
redacted records rather than resuming execution from them.

The following limits apply to JSON representations, including type metadata:

| Configuration | Default | Meaning |
| --- | --- | --- |
| `max_depth` | 50 | Maximum traversal depth |
| `max_size` | 100,000 | Maximum entries per container |
| `max_string_length` | 1,000,000 | Maximum characters per key or string |
| `max_nodes` | 1,000,000 | Maximum traversal work, including plugin conversion |
| `max_input_bytes` | 16,777,216 | Maximum encoded JSON input before parsing |

`load` reads at most the input budget plus one character or byte. `loads` checks
the byte budget before parsing, then validates the entire parsed representation
before a reconstruction plugin executes. Parser recursion failures become
`SecurityError`. Limits are not a sandbox for trusted Python callbacks or plugins.

For incoming data that should remain ordinary JSON, disable plugin execution:

```python
import datason

data = datason.loads(incoming_json, allow_plugin_deserialization=False)
```

Typed plugin records then raise `DeserializationError`, even with `strict=False`.
Built-in collection tags may still restore tuples, sets, and frozensets without
importing application code. Unknown plugin records remain an error in this mode.

User dictionaries containing `__datason_type__` are rejected during serialization
because that key is reserved. Keys that collide after conversion to strings are
also rejected rather than silently overwriting data. Use explicit string keys for
API data; generic mapping-key preservation is outside the current contract.

`SerializationError` identifies the responsible field in its message and its
`path` attribute, for example `$.diagnostics.unhandled` or `$.tools[0].result`.
Keys containing punctuation use JSON-quoted brackets, such as `$["a.b"]`.
Container-level failures identify the containing mapping or sequence. Plugin
output is traversed too, so its generated representation may appear in the path.
Paths identify fields and indexes without including the unsupported value's
representation. Exceptions raised directly outside Datason traversal may have
`path=None`. This diagnostic does not report successful normalization or guarantee
lossless restoration; non-finite and application-model policies still apply.
