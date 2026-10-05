# Custom Plugins

datason uses a plugin-based architecture. Every type beyond JSON primitives is handled by a `TypePlugin`. You can add your own plugins to handle custom types.

## TypePlugin Protocol

```text
class TypePlugin(Protocol):
    name: str         # Unique plugin name
    priority: int     # Lower = checked first (400+ for user plugins)

    def can_handle(self, obj: Any) -> bool: ...
    def serialize(self, obj: Any, ctx: SerializeContext) -> Any: ...
    def can_deserialize(self, data: dict[str, Any]) -> bool: ...
    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any: ...
```

### Priority Ranges

| Range | Category |
|-------|----------|
| 0-99 | Reserved (built-in overrides) |
| 100-199 | Stdlib types (datetime, UUID, Decimal, Path) |
| 200-299 | Data science (NumPy, Pandas) |
| 300-399 | ML frameworks (PyTorch, TensorFlow, scikit-learn) |
| 400+ | User-defined plugins |

## Example: Money Type

```python
from decimal import Decimal
from typing import Any

from datason._protocols import SerializeContext, DeserializeContext
from datason._registry import default_registry
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY


class Money:
    """Simple money type for demonstration."""
    def __init__(self, amount: Decimal, currency: str):
        self.amount = amount
        self.currency = currency


class MoneyPlugin:
    """Plugin to serialize Money objects."""

    @property
    def name(self) -> str:
        return "money"

    @property
    def priority(self) -> int:
        return 400

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, Money)

    def serialize(self, obj: Any, ctx: SerializeContext) -> dict[str, Any]:
        value = {"amount": str(obj.amount), "currency": obj.currency}
        if not ctx.config.include_type_hints:
            return value
        return {TYPE_METADATA_KEY: "Money", VALUE_METADATA_KEY: value}

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        return data.get(TYPE_METADATA_KEY) == "Money"

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Money:
        value = data.get(VALUE_METADATA_KEY)
        if (not isinstance(value, dict)
                or not isinstance(value.get("amount"), str)
                or not isinstance(value.get("currency"), str)):
            from datason._errors import DeserializationError
            raise DeserializationError("Money requires string amount and currency")
        return Money(
            amount=Decimal(value["amount"]),
            currency=value["currency"],
        )


# Register
default_registry.register(MoneyPlugin())

# Use
import datason

invoice = {"total": Money(Decimal("99.95"), "USD")}
json_str = datason.dumps(invoice)
restored = datason.loads(json_str)
assert isinstance(restored["total"], Money)
assert restored["total"].amount == Decimal("99.95")
assert restored["total"].currency == "USD"

import json
assert json.loads(datason.dumps(invoice, include_type_hints=False)) == {
    "total": {"amount": "99.95", "currency": "USD"},
}
```

## Registration and policy behavior

Register once during application startup, in both the writer and reader process.
The default registry is global: registration affects subsequent operations in
that process. Lower priority values run first; the first matching serializer is
used. The stdlib/scientific/ML priority bands above are conventions, while the
actual built-in priorities vary within them. Structured application normalizers
run at 10,000+ so a user plugin at 400 gets the first opportunity to handle a model.

The example uses underscored extension modules because that is the current
extension interface. It is subject to alpha API changes. You implement the
protocol structurally; inheriting from `TypePlugin` is not required.

A plugin should respect `ctx.config.include_type_hints`: return an ordinary
JSON representation for API consumers and tagged values for reconstruction.
Returned values pass through shared redaction, non-finite-number policies,
circular checks, and traversal budgets. Return native fields for traversal
rather than calling public `dumps` and accidentally creating a second JSON string.

`PluginError` warns and lets the registry try another plugin. `SecurityError`
and other exceptions propagate. Treat plugins and their reconstruction code as
trusted Python; representation limits do not sandbox them. Validate payload
structure in a custom deserializer before using it. For more detail, see
[Serialization boundaries](serialization-boundaries.md).

## Built-in handlers

See [Supported types](supported-types.md) for libraries, installation extras,
normalization behavior, and reconstruction limits. The core directly handles
JSON primitives and collection tags; registered plugins handle other supported
values. Application-model normalization is deliberately distinct from
reconstructing their classes.


## Optional libraries load on first use

`import datason` registers stdlib handlers and lightweight optional descriptors.
It probes only top-level module specifications, without executing installed
NumPy, Pandas, SciPy, Torch, TensorFlow, sklearn or Pydantic libraries. The
registered descriptor keeps the same name and priority when it activates.
Registry counts describe registered candidates, not imported frameworks.

A matching Python object activates its plugin. Matching considers base classes,
so application subclasses still work. A known tagged snapshot can also be the
first use: its fixed tag namespace activates the appropriate reconstruction
plugin without requiring the application to import that library first. Import
targets come from a reviewed table; payloads and class module strings do not
choose arbitrary modules to import. Reconstruction still follows the existing
trusted-state policies, limits and `allow_plugin_deserialization` setting.

The miscellaneous ML plugin loads Polars, JAX, CatBoost, Optuna or Plotly
individually. CatBoost/Optuna metadata-only exports still load without importing
those frameworks. Libraries may import their own dependencies, and explicitly
importing a plugin module still imports its associated library.

Initialization is synchronized once per plugin/family, outside the registry
lock. Successful and unavailable dependency results are cached; subsequent
object checks, serialization and reconstruction call the active handler directly.
The descriptor retains its identity, name and priority. Tagged dispatch keeps
its known-namespace guard and reads the cached handler without re-entering the
loader. Miscellaneous ML families with a cached initialization result skip
repeated class-family scans; uninitialized families still activate on demand.
Unexpected initialization failures remain
visible at first use rather than being mistaken for an absent library.

This reduces startup cost when installed libraries are unused. Required imports
move to first use: warming a known typed workload during application startup
may be appropriate for latency-sensitive services. It does not accelerate native
library initialization or promise arbitrary model reconstruction.

### Bounded local startup verification

Measured October 4, 2026 against the eager runtime on main `e09b698`, using
Python 3.12.14 in a shared Linux container. Three fresh processes per stage and
environment record import intervals and resident-memory growth. Sources are
asserted inside each child; OS caches are not cleared.

| Installed environment | Eager median import | Deferred median import | Eager / deferred resident growth |
| --- | --- | --- | --- |
| Core only | 35 ms | 30 ms | 4.9 / 4.9 MiB |
| NumPy/Pandas | 306 ms | 27 ms | 58.4 / 4.9 MiB |
| Full ML validation | 4,692 ms | 26 ms | 1,083.0 / 4.8 MiB |

The core-only difference is small compared with shared-container variation.
The installed-ML result avoids importing unused native libraries; it is not a
universal serialization speedup. In a separate fresh-process sample, first
Torch reconstruction paid about 1.18 seconds after a 67 ms Datason import, versus
a 5.48-second eager import followed by a sub-millisecond load. The required
library cost is deferred, and an application needing every framework will still
pay their initialization costs.

Five warmed rounds of plain JSON, 64-element float32 NumPy snapshots and
64-element Torch snapshots showed comparable medians: NumPy dumps 0.172 →
0.167 ms and loads 0.095 → 0.097 ms; Torch dumps 0.133 → 0.129 ms and loads
0.079 → 0.080 ms. These small local differences do not establish a throughput
improvement or an end-to-end application result. Values, dtype and shape were
checked before sampling.

[Raw observations and environment/source manifests](evidence/2026-10-04-lazy-imports.json)
retain each sample/round. Fresh-process regression tests verify unloaded
optional libraries, first tagged loads, application subclasses, descriptor
priority, disabled dispatch and budgets. Unit regressions cover concurrent
initialization, unavailable dependencies, retries after unexpected failures
and metadata-only exports.

### Warm dispatch validation

A draft optimization measured October 5, 2026 replaces initialized descriptor
callbacks with the same handler's bound methods and skips miscellaneous family
scans once their initialization result is cached. First use, descriptor priority,
tag guards, unavailable dependencies and later family activation retain their
existing behavior.

The comparison uses the earlier lazy implementation from PR #136 (`b8ce1cf`),
one process with the same library versions and unchanged converters, five
shuffled pairs per operation, one library thread and CPU JAX. All optional
handlers/families are warmed first. Values and supported dtype/shape semantics
are checked, and each mode must emit identical bytes before timing. There are
36 small/medium fixtures across 12 optional integrations, JSON and five stdlib
or structured controls: 72 end-to-end operations, plus 12 dispatch probes.

The table records small-fixture medians of round medians in microseconds,
**earlier lazy implementation → draft**. `can_handle` measures the individual
handler check, not a complete serialization request.

| Integration | dumps (µs) | loads (µs) | can_handle (µs) |
| --- | ---: | ---: | ---: |
| NumPy | 36.92 → 35.83 | 20.78 → 19.90 | 0.29 → 0.24 |
| Pandas | 323.34 → 362.58 | 410.20 → 451.32 | 0.41 → 0.35 |
| SciPy | 100.44 → 96.59 | 105.00 → 103.93 | 0.39 → 0.33 |
| PyTorch | 38.95 → 36.90 | 25.18 → 24.30 | 0.43 → 0.37 |
| TensorFlow | 40.55 → 36.98 | 85.83 → 87.97 | 0.32 → 0.26 |
| sklearn | 196.06 → 176.98 | 71.31 → 67.92 | 0.20 → 0.14 |
| Polars | 53.06 → 45.17 | 36.24 → 35.16 | 2.02 → 0.35 |
| JAX | 52.23 → 43.42 | 72.20 → 69.37 | 2.67 → 0.53 |
| CatBoost | 58.17 → 48.31 | 18.81 → 18.11 | 3.46 → 0.58 |
| Optuna | 1,223.51 → 1,322.42 | 62.26 → 61.48 | 2.41 → 0.61 |
| Plotly | 2,304.80 → 2,580.32 | 10,255.19 → 11,106.50 | 6.34 → 3.71 |
| Pydantic | 43.20 → 36.36 | 13.84 → 13.72 | 0.24 → 0.18 |

All 12 dispatch probes improved in every paired round. Complete operations
do **not** improve uniformly: small Pandas/Optuna/Plotly operations were slower
in this sample, and medium Optuna loads were 51.7% slower. Other medium cases and
unchanged JSON/stdlib controls varied substantially too. These results do not
isolate the cause of the slower observations or justify a universal speed claim;
the draft needs further representative performance review. Conversion and
library construction/copying dominate several workloads.

CatBoost/Optuna loads are metadata-only. Pydantic and dataclass loads produce
normalized fields, with explicit model validation checked separately. Polars
uses an Int64 fixture because the existing loader infers dtypes; its Float32
schema does not currently restore Float32. The fixtures do not establish fidelity
for every type supported by these libraries. sklearn's small/medium fitted models
have 2/64 features, while the other fixture scales are generally 10/1,000 elements
or metadata entries; actual output sizes are recorded.

[Raw observations, source hashes and versions](evidence/2026-10-05-warm-dispatch.json)
include all slower observations, controls and paired rounds. Reproduce with
`scripts/perf/warm_dispatch_study.py --baseline-dir PATH --output REPORT.json`
using an application-owned earlier checkout and installed optional libraries.
