# For AI coding agents

Use this documentation to write datason code with explicit output policies and
version assumptions. For serializing application/agent models, use the
[structured-data guide](agent-data.md).

## Machine-readable documentation

- [llms.txt](https://danielendler.github.io/datason/llms.txt): concise overview and links.
- [llms-full.txt](https://danielendler.github.io/datason/llms-full.txt): assembled guides and reference with complete examples.
- [Source llms.txt](https://github.com/danielendler/datason/blob/main/llms.txt) and
  [source full reference](https://github.com/danielendler/datason/blob/main/llms-full.txt).

The full reference is generated from the user documentation with
`python scripts/sync_docs.py`. CI checks it for drift. GitHub Pages serves both
files at the site root after a docs deployment.

## Before generating code

1. Check the version: these docs follow development `main`, ahead of the published
   alpha as of October 4, 2026. Use [Installation](getting-started.md#installation).
2. Choose plain JSON for a tool/API consumer or tagged data for supported Python
   reconstruction. Validate the consumer's schema separately.
3. Check [Supported types](supported-types.md), including metadata-only plugins
   and application-model normalization. Do not assume every object is supported.
4. Keep callbacks/plugins trusted; use the documented reconstruction controls
   for ordinary incoming JSON. Redaction belongs in diagnostic exports.
5. Test restored properties and retain stored-data fixtures before upgrades.

## A small working example

```python
import json
from decimal import Decimal

import datason

result = {"cost": Decimal("19.99")}
text = datason.dumps(result, include_type_hints=False)
assert json.loads(text) == {"cost": "19.99"}
```

Use [Recipes](recipes.md) for APIs, redaction, and persistence; [API](api.md) and
[Configuration](configuration.md) for exact options. A string fallback loses
unknown type information and should be an explicit policy, not a universal fix.
