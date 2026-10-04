# Getting started

This guide takes you from installation to a typed round trip, plain API JSON,
and file storage. You need Python 3.10+.

## Installation

These docs describe development `main` (versioned `2.0.0a2`). As of October 4,
2026, `pip install datason` selects stable v1 (`0.13.0`), and the published v2
alpha is `2.0.0a1`. Choose deliberately:

```bash
# Published v2 alpha; it predates some features described in these docs
python -m pip install 'datason==2.0.0a1'

# Current source matching the development docs (requires Git)
python -m pip install 'datason @ git+https://github.com/danielendler/datason.git@main'
```

The following optional extras install the libraries used by their plugins.
For current source, select extras in the package name:

```bash
python -m pip install 'datason[numpy,pandas] @ git+https://github.com/danielendler/datason.git@main'
```

| Extra | Installs |
| --- | --- |
| `numpy` | NumPy |
| `pandas` | Pandas and its dependencies |
| `pydantic` | Pydantic v2 for model-field normalization |
| `ml` | PyTorch, scikit-learn, SciPy; TensorFlow on Python 3.11+ |
| `ml-extra` | Polars, JAX, Plotly, CatBoost, Optuna, Pillow, Transformers |
| `crypto` | Cryptography; hash/HMAC helpers themselves use only stdlib |
| `all` | `numpy`, `pandas`, `ml`, `crypto`; excludes `pydantic` and `ml-extra` |

Install only what you use; `ml` and `ml-extra` can be large. Existing installed
libraries are detected at import time. The [supported-types guide](supported-types.md)
lists each handler's behavior; installing an extra does not guarantee fidelity
for every object in that library. For reproducible development deployments,
replace `main` in the install URL with a reviewed commit SHA.

Check the version and interpreter if the API looks different:

```bash
python -c 'import datason; print(datason.__version__, datason.__file__)'
```

## Restore standard Python types

This example uses only the core package:

```python
import datetime as dt
import uuid
from decimal import Decimal
from pathlib import Path

import datason

original = {
    "observed": dt.datetime(2026, 10, 4, 10, 30, tzinfo=dt.timezone.utc),
    "id": uuid.UUID("12345678-1234-5678-1234-567812345678"),
    "price": Decimal("19.99"),
    "path": Path("models"),
}
text = datason.dumps(original)
restored = datason.loads(text)
assert restored == original
assert isinstance(restored["price"], Decimal)
```

Type hints are enabled by default. `loads` reads the metadata written by `dumps`;
it does not guess that an arbitrary ISO string is a datetime. You need the same
optional libraries or custom plugins when restoring their tagged values.

## Produce ordinary API JSON

Disable type hints when an API consumer expects JSON primitives:

```python
import datetime as dt
import json
from decimal import Decimal

import datason

response = {"created": dt.datetime(2026, 10, 4, tzinfo=dt.timezone.utc),
            "price": Decimal("19.99")}
text = datason.dumps(response, include_type_hints=False, sort_keys=True)
assert json.loads(text) == {"created": "2026-10-04T00:00:00+00:00", "price": "19.99"}
```

Decimal becomes a string to retain its precision. The consumer's schema must
permit that representation. Loading this untagged output returns strings.

## Preserve NumPy and Pandas data

Install the `numpy,pandas` extras before running this example:

```python
import numpy as np
import pandas as pd

import datason

array = np.array([[1, 2], [3, 4]], dtype=np.int16)
restored_array = datason.loads(datason.dumps(array))
np.testing.assert_array_equal(restored_array, array)
assert restored_array.dtype == array.dtype

frame = pd.DataFrame({"score": pd.array([95, None], dtype="Int64")})
restored_frame = datason.loads(datason.dumps(frame))
pd.testing.assert_frame_equal(restored_frame, frame)
```

See [Scientific fidelity](scientific-fidelity.md) for indexes, empty shapes,
timestamp units, and unsupported dtypes. Default non-finite-number policies can
change values even when tags are enabled.

## Write and read a file

Use a text file with an explicit encoding. `dump` writes JSON to an open file;
`load` reads it under the configured input budget. Neither streams a dataset
incrementally.

```python
from decimal import Decimal
from tempfile import TemporaryDirectory
from pathlib import Path

import datason

data = {"price": Decimal("19.99")}
with TemporaryDirectory() as directory:
    path = Path(directory) / "data.json"
    with path.open("w", encoding="utf-8") as file:
        datason.dump(data, file, indent=2)
    with path.open(encoding="utf-8") as file:
        assert datason.load(file) == data
```

## Go further

- [Recipes](recipes.md): API responses, redacted logs, stored scientific data, and errors.
- [Configuration](configuration.md): defaults, scope, presets, and output policies.
- [Supported types](supported-types.md): normalization versus reconstruction.
- [Serialization boundaries](serialization-boundaries.md): incoming data and budgets.
- [Migration from v1](migration.md): changed entry points and persisted data.
