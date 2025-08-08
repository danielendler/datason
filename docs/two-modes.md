# 🔀 Two Modes: When to Use Which

DataSON offers two distinct ways to work with JSON data:

1. **Compat Mode** – a strict drop-in replacement for Python's `json` module.
2. **Enhanced Mode** – smarter serialization with type inference and modern helpers.

Use Compat Mode when you need guaranteed parity with the standard library.
Switch to Enhanced Mode when you want DataSON's advanced features.

## Compat Mode

```diff
-import json
+import datason.json as json

data = json.loads('{"created": "2024-01-01"}')
assert isinstance(data["created"], str)
```

## Enhanced Mode

```diff
-import json
+import datason

# Smart loading with type inference
item = datason.load_smart('{"created": "2024-01-01"}')
assert item["created"].year == 2024
```

## Choosing a Mode

| Use case | Recommended mode |
|----------|-----------------|
| Legacy code expecting `json` behavior | Compat Mode |
| Need datetime parsing or complex types | Enhanced Mode |
| Migrating gradually | Start with Compat, upgrade to Enhanced |

The two modes can coexist in the same project, letting you migrate at your own pace.
