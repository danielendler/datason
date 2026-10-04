# datason

[![CI](https://github.com/danielendler/datason/actions/workflows/ci.yml/badge.svg)](https://github.com/danielendler/datason/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/danielendler/datason/graph/badge.svg?token=UYL9LvVb8O)](https://codecov.io/gh/danielendler/datason)
[![PyPI version](https://img.shields.io/pypi/v/datason.svg)](https://pypi.org/project/datason/)
[![Python versions](https://img.shields.io/pypi/pyversions/datason.svg)](https://pypi.org/project/datason/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://danielendler.github.io/datason/)

**JSON serialization for Python APIs, diagnostics, and stored state.**
Handle datetime, UUID, Decimal, paths, and collections, with optional plugins for
NumPy, Pandas, and ML libraries. The core has no runtime dependencies. Python 3.10+.

[Get started](https://danielendler.github.io/datason/getting-started/) ·
[Recipes](https://danielendler.github.io/datason/recipes/) ·
[Supported types](https://danielendler.github.io/datason/supported-types/) ·
[API reference](https://danielendler.github.io/datason/api/)

## Install the version you intend to use

This README and the documentation follow development `main` (versioned
`2.0.0a2`). As of October 4, 2026, PyPI's stable release is v1 (`0.13.0`), and
its published v2 alpha is `2.0.0a1`. An unqualified `pip install datason`
installs v1, whose API differs. v2 remains an alpha.

```bash
# Published v2 alpha; some documented features landed after this release
python -m pip install 'datason==2.0.0a1'

# Source matching these development docs (requires Git)
python -m pip install 'datason @ git+https://github.com/danielendler/datason.git@main'

# Add NumPy and Pandas support to current source
python -m pip install 'datason[numpy,pandas] @ git+https://github.com/danielendler/datason.git@main'
```

Pin a reviewed commit instead of `main` for reproducible deployments.
The [installation guide](docs/getting-started.md#installation) explains all
extras, including `pydantic`, `ml`, and `ml-extra`. `all` includes NumPy, Pandas,
ML and crypto dependencies; it excludes `pydantic` and `ml-extra`.

## First round trip

This example uses only the core package:

```python
import datetime as dt
from decimal import Decimal

import datason

event = {"observed": dt.datetime(2026, 10, 4, tzinfo=dt.timezone.utc),
         "price": Decimal("19.99")}
text = datason.dumps(event)
restored = datason.loads(text)
assert restored == event
assert isinstance(restored["price"], Decimal)
```

`dumps` writes a JSON string, including metadata for supported Python types.
A Decimal is represented as
`{"__datason_type__": "decimal.Decimal", "__datason_value__": "19.99"}`.
`loads` uses those tags to reconstruct it; an ordinary JSON reader sees a dict.

## Choose the right JSON for your task

| Task | Settings and behavior | Guide |
| --- | --- | --- |
| API or tool response | Disable tags; check normalized values against the consumer's schema | [Recipes](docs/recipes.md#api-and-tool-responses) |
| Logs and diagnostics | Select redaction fields/patterns; keep original state separately | [Security](docs/security.md) |
| Internal stored data | Keep tags; install the corresponding libraries when restoring | [Scientific fidelity](docs/scientific-fidelity.md) |
| Application models | Dataclass/Pydantic fields and Enum values normalize; validate models explicitly | [Structured data](docs/agent-data.md) |
| LangGraph checkpoint | Opt-in serializer with a SQLite pause/resume example | [LangGraph](docs/langgraph-checkpoints.md) |

For ordinary API JSON:

```python
import json
from decimal import Decimal

import datason

text = datason.dumps({"price": Decimal("19.99")}, include_type_hints=False)
assert json.loads(text) == {"price": "19.99"}
```

For scientific stored data (install `numpy,pandas`):

```python
import numpy as np
import pandas as pd

import datason

array = np.array([[1, 2], [3, 4]], dtype=np.int16)
restored = datason.loads(datason.dumps(array))
np.testing.assert_array_equal(restored, array)
assert restored.dtype == array.dtype

frame = pd.DataFrame({"score": pd.array([95, None], dtype="Int64")})
pd.testing.assert_frame_equal(datason.loads(datason.dumps(frame)), frame)
```

Type tags support defined round-trip contracts, not every Python object.
Non-finite-number policies and redaction can change values. Some ML plugins
export metadata only; tensors do not preserve every runtime property.
See [Supported types](docs/supported-types.md) before choosing a storage format.

## Everyday API

| Operation | Result |
| --- | --- |
| `datason.dumps(obj, **options)` | JSON string |
| `datason.loads(text, **options)` | Python values; supported tagged types reconstructed |
| `datason.dump(obj, file, **options)` | Write JSON to an open text file |
| `datason.load(file, **options)` | Read and deserialize within input budgets |
| `datason.config(**options)` | Temporarily select configuration in a context |

Configuration enums, `SerializationConfig`, and four preset factories are also
exported. Common JSON arguments such as `indent`, `default`, and `parse_float`
are supported. Defaults differ from stdlib `json`: Unicode is emitted directly,
NaN/Infinity become `null`, tags are enabled, and traversal budgets are enforced.
See [API compatibility](docs/api.md#compatibility-with-stdlib-json).

```python
from dataclasses import asdict

import datason
from datason import api_config

with datason.config(**asdict(api_config())):
    text = datason.dumps({"status": "ok", "value": float("nan")})
assert text == '{"status": "ok", "value": null}'
```

Inline options override the active scope. Entering a new `config` scope starts
from defaults; see [Configuration](docs/configuration.md#scope-and-precedence).

## Explore and contribute

- [Troubleshooting](docs/troubleshooting.md): versions, missing plugins, tags, NaN, and limits.
- [Custom plugins](docs/plugins.md): complete Money type example and registration.
- [Serialization boundaries](docs/serialization-boundaries.md): reserved keys, budgets, and incoming data.
- [Migration from v1](docs/migration.md): API and persisted-data differences.
- [Contributing](CONTRIBUTING.md): development setup and documentation checks.
- [Examples](examples/): runnable basic, type, security, and checkpoint examples.
- [Hardening roadmap](docs/hardening-roadmap.md): tested scope and remaining validation.
- [For AI agents](docs/ai-agents.md): [llms.txt](llms.txt) and [llms-full.txt](llms-full.txt).

For local development, run `uv sync --locked --group docs`, then `uv run pytest`
and `uv run mkdocs build --strict`. Report reproducible problems in
[Issues](https://github.com/danielendler/datason/issues) or discuss usage in
[Discussions](https://github.com/danielendler/datason/discussions).

## License

MIT
