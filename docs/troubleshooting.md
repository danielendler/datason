# Troubleshooting

## The documented API is missing

Check the installed version and interpreter:

```bash
python -c 'import datason; print(datason.__version__, datason.__file__)'
python -m pip show datason
```

An unqualified install currently selects stable v1, whose API differs from v2.
Follow [Installation](getting-started.md#installation) for a published alpha or
source matching these development docs. Avoid a local file named `datason.py`
that shadows the package.

## My API response contains `__datason_type__`

Tags are enabled by default for Python reconstruction. Use
`include_type_hints=False` or `api_config()` for plain JSON consumers.
Decimal and dates then become strings, arrays become JSON lists, and binary
values become base64 text. Confirm those choices against the API schema.
See the [API recipe](recipes.md#api-and-tool-responses).

## A datetime or model comes back as a string or dict

`loads` reconstructs recognized tagged values; it does not guess types from
ordinary strings. Preserve tags when storing supported typed data. Dataclasses,
Pydantic models, and Enums normalize even with tags enabled; explicitly validate
or hydrate application classes. See [Structured agent data](agent-data.md).

## `SerializationError`: unsupported type

Use `error.path` to locate the field. Normalize it, supply `default=`, or register
a custom plugin. Confirm that the optional library is installed in the same
interpreter, then restart the process after installation because registration
occurs when datason is imported. String fallback loses the original type.
See the [error recipe](recipes.md#find-the-unsupported-field).

## `DeserializationError`: missing or unknown plugin

Install the writer's relevant dependencies and register its custom plugins in
the reader. For inspection only, `strict=False` leaves unknown tags as
ordinary dictionaries; it does not reconstruct them or suppress every malformed
payload error. With `allow_plugin_deserialization=False`, plugin tags are rejected
regardless of `strict`. See [Serialization boundaries](serialization-boundaries.md).

## NaN or Infinity changed

The default policy converts non-finite numbers to `null`, including plugin
leaves. `STRING` emits strings; `KEEP` can emit the non-standard tokens `NaN` and
`Infinity`, which strict JSON consumers reject. `DROP` currently also emits
`null`; it does not remove an element. See [Configuration](configuration.md#non-finite-numbers).

## `SecurityError`: a limit was reached

Read the limit named by the error. Metadata counts toward representation depth
and size, so tagged data can reach a limit before an equivalent plain dict does.
`max_size` is per container, `max_nodes` bounds traversal, and `max_input_bytes`
bounds incoming JSON and supported NumPy allocation estimates. They are not a
process-wide memory quota. Increase only the relevant budget for a known workload.
See [Serialization boundaries](serialization-boundaries.md).

## Reserved or colliding dictionary keys

Rename `__datason_type__`, which is reserved for type dispatch. Convert keys
explicitly before serialization: `1` and `"1"` can collide after JSON key
normalization. datason raises rather than overwrite a value.

## Configuration seems to disappear in a nested scope

Inline kwargs merge with the active config. A new `datason.config(...)` scope
starts from defaults rather than inheriting unspecified settings from the outer
scope, then restores the outer config on exit. See
[Scope and precedence](configuration.md#scope-and-precedence).

## Need to report a problem?

Include the installed version, Python/library versions, a small nonsensitive
input, the call/options, expected behavior, and the full error. Open an
[issue](https://github.com/danielendler/datason/issues). For stored-data failures,
include a sanitized fixture that still reproduces the behavior.
