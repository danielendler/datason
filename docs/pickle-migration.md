# Trusted pickle migration

Pickle can execute arbitrary Python code while loading. Datason's module scanner
does not make an untrusted pickle safe: allowed modules also contain executable
functions. `validate_pickle_safety` keeps its historical name, but returns only
diagnostic information about detected module references.

Migration now requires an explicit trust declaration:

```python
from datason.security.pickle_bridge import pickle_to_json, pickle_file_to_json

# Only for data from a source you already trust to execute Python code.
json_text = pickle_to_json(trusted_pickle_bytes, trusted=True)
json_text = pickle_file_to_json("trusted-model.pkl", trusted=True)
```

Omitting `trusted=True` raises `SecurityError` before scanning, opening a file,
or loading the pickle. This is an intentional safety change for the alpha API.
Passing a custom `allowed_modules` filter does not replace the trust declaration.
Use JSON or a separately isolated migration process for untrusted inputs.
