# Trusted pickle migration

Pickle can execute arbitrary Python code while loading. Datason's module scanner
does not make an untrusted pickle safe: allowed modules also contain executable
functions. `validate_pickle_safety` keeps its historical name, but returns only
diagnostic information about detected module references.

Migration now requires an explicit trust declaration:

```python
import pickle
from decimal import Decimal

import datason
from datason.security.pickle_bridge import pickle_to_json

# This example creates its own trusted bytes; do not apply it to unknown files.
trusted_pickle_bytes = pickle.dumps({"price": Decimal("19.99")})
json_text = pickle_to_json(trusted_pickle_bytes, trusted=True)
assert datason.loads(json_text) == {"price": Decimal("19.99")}
```

Omitting `trusted=True` raises `SecurityError` before scanning, opening a file,
or loading the pickle. This is an intentional safety change for the alpha API.
Passing a custom `allowed_modules` filter does not replace the trust declaration.
Use JSON or a separately isolated migration process for untrusted inputs.

For a file from a source you already trust to execute code, call
`pickle_file_to_json(path, trusted=True)` from the same module. Confirm the JSON
representation and restore it in a separate check before replacing stored data.
See [Supported types](supported-types.md) for normalization limitations and
[Migration from v1](migration.md) for historical JSON compatibility.
