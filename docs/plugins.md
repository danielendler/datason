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
operations retain the active handler. Unexpected initialization failures remain
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
