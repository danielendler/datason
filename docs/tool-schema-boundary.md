# Validating a tool output contract

Datason can normalize Python scientific values for a tool response. The
application still owns the wire schema. Tagged snapshots, API responses and
redacted diagnostics have different contracts; choose the policy explicitly.

The [runnable MCP example](https://github.com/danielendler/datason/blob/main/examples/mcp_schema_boundary.py)
starts an in-memory MCP server and client without credentials or model calls.
It discovers the SDK-advertised output schema, calls the tool, and validates the
actual `structured_content` against that schema. CI exercises MCP 2.2.0 and
2.3.0 on Python 3.11 and 3.13. These pins are a bounded compatibility claim.

From a checkout:

```bash
uv run --locked --extra numpy --extra pydantic \
  --with mcp==2.3.0 --with jsonschema==4.26.0 \
  python -m examples.mcp_schema_boundary
```

The tool's Pydantic return model contains only wire types, so its validation and
serialization schemas describe the same shape. It forbids unexpected fields.
Datason normalizes a datetime, UUID, NumPy scalar, array and binary preview with
an explicit `api_config()` policy before model validation. The client checks the
SDK's actual advertised schema, rather than assuming it matches a separately
generated local schema.

| Input | Wire value | Required contract |
| --- | --- | --- |
| UTC datetime | ISO string | `string` with `date-time` format |
| UUID | String | `string` with `uuid` format |
| Float32 scalar | Number | `number` |
| Non-finite scalar | `null` | `number` or `null` |
| Float32 array | List of numbers | Array of numbers; dtype is intentionally normalized |
| Bytes | Base64 string | Explicit decoding rule and validation |

JSON Schema `contentEncoding` is an annotation; it does not automatically
validate base64. The example also validates the encoding and tests that decoding
reproduces the original bytes. `FormatChecker` enforces the date/UUID formats.
Neither check establishes that binary content is a safe file to execute or open.

The regression tests reject a nullable result under a number-only schema,
malformed base64, and a tagged datetime under the string wire schema. They also
prove that an active diagnostic configuration cannot accidentally redact the
tool's binary field or emit a tagged/string non-finite representation.

This recipe does not repair arbitrary SDK schema generation, aliases, computed
fields or provider protocols. For models with different validation and
serialization shapes, select a suitable wire model or fix the SDK schema at its
documented hook. Do not inject Datason tags into a provider's ordinary-JSON
contract. Keep large model/media artifacts in binary storage with references;
base64 is appropriate only for bounded payloads.
