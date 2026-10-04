# Supported types

Start by choosing whether the consumer needs plain JSON or reconstructed Python
types. `include_type_hints=True` is the default; it writes tags for supported
handlers. With tags disabled, values normalize for JSON consumers. Neither mode
promises to preserve every property of every object.

## Core types

No optional libraries are needed for these handlers.

| Input | Tagged `loads(dumps(value))` | Without tags |
| --- | --- | --- |
| JSON primitives, dict, list | JSON-compatible values | Same structure; non-finite policies still apply |
| datetime, date, time, timedelta | Corresponding stdlib type | Date/time string or configured timestamp; duration in seconds |
| UUID | UUID | UUID string |
| Decimal | Decimal | Decimal string, preserving decimal precision |
| complex | complex | List `[real, imag]` |
| Path, PurePath | Path value | Path string; concrete path flavor is not a portability guarantee |
| tuple, set, frozenset | Corresponding collection | List; set elements ordered by `repr`, not natural numeric order |
| bytes, bytearray | Corresponding binary type | Base64 string; document the encoding in your schema |
| Dataclass instance | Field dictionary | Field dictionary |
| Enum member | Member value | Member value |

Dataclasses and Enums do not regain their application class. Dict keys become strings; use explicit string
keys when the consumer's contract matters. Reserved metadata keys and collisions
raise errors. See [Serialization boundaries](serialization-boundaries.md).

## Optional libraries

Install extras using the [version-aware installation guide](getting-started.md#installation).
Both writer and reader need the relevant library for reconstruction.

| Library / extra | Input | Reconstruction and limits |
| --- | --- | --- |
| NumPy / `numpy` | Arrays and supported numeric, complex, temporal scalars | Supported dtype widths, shape, temporal units; see the [scientific contract](scientific-fidelity.md) |
| Pandas / `pandas` | DataFrame, Series, Timestamp, Timedelta | Supported labels, indexes, dtypes, categories; exclusions listed in the [scientific contract](scientific-fidelity.md) |
| Pydantic v2 / `pydantic` | BaseModel instances | Alias field dictionaries; call `model_validate` yourself |
| PyTorch / `ml` | Tensor, device, Size | Supported numeric values/dtype/shape; CPU tensors, no autograd history |
| TensorFlow / `ml` (Python 3.11+) | Tensor / EagerTensor, Variable, SparseTensor | Supported numeric values/dtype/shape; CPU placement, no graph or training-state guarantee |
| scikit-learn / `ml` | BaseEstimator and Pipeline | Supported state reconstruction; custom objects within state may need plugins; verify predictions and versions |
| SciPy / `ml` | Sparse matrices | CSR, CSC, COO representations; other formats restore as COO; sparse arrays restore as matrices |
| Polars / `ml-extra` | DataFrame, Series | Reconstructed from values; stored schema is not fully reapplied |
| JAX / `ml-extra` | Array | Supported numeric/bool dtype and shape; x64 narrowing rejected when x64 is disabled |
| Plotly / `ml-extra` | Figure | Reconstructed from figure data |
| CatBoost / `ml-extra` | Model | Metadata dictionary; fitted trees are not restored |
| Optuna / `ml-extra` | Study | Metadata/trial summaries; study storage is not restored |

`ml-extra` also installs Pillow and Transformers, but the current plugin registry
has no dedicated handlers for their object types. `all` excludes `ml-extra` and
`pydantic`. A package being installed does not imply every object it exposes is
supported.

ML reconstruction calls library code and can be version-sensitive. Use reviewed,
trusted typed records and workload-specific validation; see
[Security](security.md) and the [hardening roadmap](hardening-roadmap.md).

## What changes values?

- `NanHandling.NULL` (default) turns non-finite leaves into `null`.
- Redaction changes strings and selected fields, including plugin representations.
- `include_type_hints=False` removes information needed for reconstruction.
- `fallback_to_string=True` converts unsupported objects to strings.
- Application models normalize to fields; datason does not dynamically import them.

For concrete output and comparisons, use [Getting started](getting-started.md)
and [Recipes](recipes.md). For another type, choose a JSON `default` callback
for normalization or a [custom plugin](plugins.md) for a typed contract.

See the [built-in reconstruction review](ml-reconstruction-review.md) for current
ML constructor paths, supported numeric cases, buffer checks, and exclusions.
