# Custom Plugins

datason uses a plugin-based architecture. Every type beyond JSON primitives is handled by a `TypePlugin`. You can add your own plugins to handle custom types.

## TypePlugin Protocol

```python
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
        return {
            TYPE_METADATA_KEY: "Money",
            VALUE_METADATA_KEY: {
                "amount": str(obj.amount),
                "currency": obj.currency,
            },
        }

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        return data.get(TYPE_METADATA_KEY) == "Money"

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Money:
        value = data[VALUE_METADATA_KEY]
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
```

## Built-in Plugins

datason ships with 10 built-in plugins:

| Plugin | Types Handled | Priority |
|--------|--------------|----------|
| `datetime` | datetime, date, time, timedelta | 100 |
| `uuid` | UUID | 110 |
| `decimal` | Decimal, complex | 120 |
| `path` | Path, PurePath | 130 |
| `numpy` | ndarray, integer, floating, bool_ | 200 |
| `pandas` | DataFrame, Series, Timestamp, Timedelta | 210 |
| `scipy_sparse` | csr_matrix, csc_matrix, coo_matrix, etc. | 300 |
| `torch` | Tensor | 310 |
| `tensorflow` | Tensor, EagerTensor | 320 |
| `sklearn` | BaseEstimator subclasses | 330 |
