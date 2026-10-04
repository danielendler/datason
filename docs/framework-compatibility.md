# Framework compatibility matrix

The supported integration contract is deliberately bounded to the installed SDK
pins below. Tests execute framework hooks and real SQLite persistence; they do
not replace SDK state machines, rely on pickle fallbacks, call paid model APIs
or contact a provider. Core Datason has no new required runtime dependencies.

## Pinned matrix and evidence

| Framework | SDK / checkpoint / storage pins | Python CI | Covered behavior |
| --- | --- | --- | --- |
| LangGraph older | 1.0.0 / 2.1.2 / SQLite 2.0.11 | 3.11, 3.13 | Reopen/resume, dynamic interrupts by id, Send fan-out, explicit application hydration/schema upgrades |
| LangGraph current | 1.2.12 / 4.2.0 / SQLite 3.1.1 | 3.11, 3.13 | Same cases; response-schema record preservation when supported |
| OpenAI Agents older | 0.22.0 | 3.11, 3.13 | Actual custom-context hooks, tool approval pause/resume and rebound model/tool dependencies |
| OpenAI Agents current | 0.23.1 | 3.11, 3.13 | Same cases, including an actual snapshot captured by 0.22.0 |

A committed SQLite dump captured under the older LangGraph/checkpoint/storage
pins is also loaded and resumed by the current matrix. A committed Agents
snapshot captured under 0.22.0 is restored and resumed by both SDK versions.
These are owned repository fixtures; SQL execution is test setup, not an API for
accepting untrusted SQL. Source SDK versions, format and codec hash are retained.
The capture script requires the older pins and disables tracing/provider calls.

Local verification uses Python 3.12.14. The CI jobs validate the other listed
Python versions and upload actual codec execution coverage. Changed-code
coverage requires 90%, including partially covered branches. This table states explicit test scope, not support for all
SDK releases, arbitrary historical checkpoint schemas or provider transports.

## LangGraph runtime records

Constructing `DatasonSerializer` enables a reviewed integration plugin **when
LangGraph is installed**. Registration is atomic and idempotent by plugin name.
The plugin is shared by Datason's normal registry; ordinary application model
classes continue to normalize rather than being imported from their names.
Importing the core package does not install or import LangGraph.

The codec preserves only two closed, versioned tags:

- `langgraph.Interrupt.v1`: interrupt id, value and optional JSON response schema.
- `langgraph.Send.v1`: target node and argument.

Payload fields never select an arbitrary Python class or module to import.
Nested scientific/binary values use Datason's existing traversal and policies.
`allow_plugin_deserialization=False` also blocks these runtime tags. Interrupt
response schemas must be data, not arbitrary Python classes. Older SDKs that
cannot represent a non-null response schema reject it. Send timeout policies
require a reviewed application codec and are deliberately rejected here.
Other framework object families and LangChain message-class hydration remain
outside this integration contract.

Dynamic interrupts previously normalized to dictionaries, so resumed execution
failed when LangGraph accessed `interrupt.id`. The regression tests close/reopen
SQLite before resuming the exact interrupt. Fan-out tests additionally exercise
pending Send packets. The outer wire label remains `datason-json-v1`; plain
existing Datason checkpoints remain readable. Older dictionary-normalized
interrupt writes cannot safely regain a runtime type just by guessing fields:
export/recover them deliberately from an owned snapshot and application schema.

Native JsonPlus/MessagePack checkpoints have another format. Tests generate a
real native payload and confirm explicit rejection rather than a pickle fallback.
Do not silently change the serializer on a live native-checkpoint database. Export
validated application state and start a new Datason-backed thread; checkpoint
history and interrupted operations need a separate framework-aware migration.

## Application schema upgrade and hydration

The test persists a version-1 record, pauses before consumption, reopens SQLite,
then applies an explicit migration and `Reading.model_validate` before the node
uses the record. The node writes schema version 2 and validated fields back.
An application owns the version check, migrations and validation; Datason's
wire label and LangGraph's internal schema version do not replace that contract.
Keep old snapshots and check resume behavior before retiring them.

## Agents SDK context recipe

Use the documented SDK hooks for a **non-mapping** context and a dedicated
configuration rather than active diagnostic redaction/string-fallback settings. The SDK intentionally
handles mapping contexts directly; its custom serializer hook does not rewrite
arbitrary mapping leaves.

The context codec itself can be checked independently of a provider call:

```python
import datetime as dt
from dataclasses import dataclass, asdict

import datason

@dataclass
class AppContext:
    observed: dt.datetime
    payload: bytes

def serialize_context(context):
    return {"datason_context": datason.dumps(context, **asdict(datason.strict_config()))}

def deserialize_context(mapping):
    return AppContext(**datason.loads(mapping["datason_context"], **asdict(datason.strict_config())))

context = AppContext(dt.datetime(2026, 10, 4, tzinfo=dt.timezone.utc), b"hello")
assert deserialize_context(serialize_context(context)) == context
```

Attach these hooks to the SDK-owned result and your reviewed agent. This
integration sketch assumes `result` and `agent` are supplied by your application
and runs inside an async function:

```text
snapshot = result.to_state().to_json(
    context_serializer=serialize_context, strict_context=True,
)
# Store the SDK-owned snapshot using the application's storage policy.
state = await RunState.from_json(
    agent, snapshot,
    context_deserializer=deserialize_context, strict_context=True,
)
```

The test uses datetime, float32 NumPy values and bytes in a dataclass context,
then verifies a paused tool executes once after the test's explicit approval.
Runtime clients, callables and live connections are supplied by application code,
not serialized. These tests do not establish replay prevention or exactly-once
execution across concurrently restored copies.

Only restore server-owned snapshots or state whose integrity **and ownership**
the application has verified. SDK snapshots contain pending calls and approvals;
Datason's encoding and HMAC helpers do not grant authorization to resume them.
