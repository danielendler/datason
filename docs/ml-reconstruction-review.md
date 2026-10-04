# Built-in reconstruction review

Review date: October 4, 2026. Scope: every current built-in loader and its
constructors, imports, file behavior, resource expansion and fidelity limits.
This is a source/control-flow review backed by the regression tests below,
not a security assessment of the installed native libraries themselves.

## Trust contract

Typed loads are intended for **trusted application-owned snapshots and reviewed
plugins**. Registered code, installed library imports and estimator state hooks
are trusted Python. Checking a module namespace does not make model state safe.
Datason does not provide a safe-unpickling or arbitrary-code sandbox.

For a boundary that must not dispatch reconstruction plugins:

```python
fields = datason.loads(raw_json, allow_plugin_deserialization=False)
# Validate ordinary fields against the application's schema before use.
```

This rejects typed plugin tags before dispatch, even with `strict=False` and
when they are nested inside tagged collections. Basic tuple/set/frozenset
reconstruction remains available. Do not provide unreviewed parser hooks or
custom JSON decoder classes: those execute during parsing, before this policy.
Optional installed libraries may already be imported when Datason itself loads;
the policy controls reconstruction, not module initialization.

## Constructor and fidelity inventory

| Family / implementation | Reconstruction and possible effects | Fidelity and resource boundary |
| --- | --- | --- |
| datetime | `fromisoformat`, `fromtimestamp`, timedelta constructors; no selected module/file loading | Current numeric records include units/ISO; legacy timezone/unit ambiguity needs an explicit migration |
| UUID, Decimal, Path | UUID/Decimal/Path constructors; Path does not open the named file | Values restore under input/string limits; Path is a value, not an authorized filesystem capability |
| structured / Pydantic | Validated base64 decode for binary; application models, dataclasses and enums normalize on write | No automatic model-class import/hydration; application explicitly validates loaded fields |
| NumPy | dtype parsing and array/scalar constructors; no pickle or user-selected module import | Dtype/shape/allocation checks; structured/void rejected; object arrays and extended precision outside the guaranteed contract |
| Pandas | DataFrame/Series/index/category constructors and dtype conversions | Supported dtype/index metadata restored; budgets cover JSON and underlying NumPy paths, not every possible extension dtype or aggregate allocation |
| SciPy sparse | COO constructor, optional CSR/CSC conversion | Check shape/coordinates/dtype and estimated COO/pointer buffers before construction; retain huge logical COO shapes with few stored values. Sparse arrays normalize to matrices; other formats normalize to COO |
| Torch | dtype lookup, CPU tensor/reshape, device/Size value constructors | Dense shape and represented-buffer budgets checked; empty/scalar shapes restored; tested numeric tensors always CPU; complex/quantized tensors need a separate contract. Device metadata does not allocate on that device; autograd/storage alias identity is not retained |
| TensorFlow | dtype parsing; CPU constant/reshape/Variable/SparseTensor constructors | Dense shape and buffer checks; sparse coordinate/index-buffer checks. Numeric dense/sparse cases are tested; complex/string/resource/variant tensors are outside the guaranteed round-trip contract. Variable name/trainability and original device identity are not guaranteed |
| sklearn | Import installed `sklearn.*` module; require a BaseEstimator class; `object.__new__` and `__setstate__`; Pipeline constructor | Reject non-estimator classes and malformed state before hydration. State hooks can execute trusted installed code; no safe-untrusted-model claim. Predictions verified for retained examples; library compatibility and Pipeline options remain bounded |
| Polars | DataFrame/Series constructors from exported values; no payload-selected import/file API | Inferred types; stored schema is not a general dtype restoration contract. Empty/narrow/temporal/extension columns need application validation |
| JAX | NumPy buffer then configured JAX backend array | Numeric/bool dtypes, shape and buffer checks. Reject x64 narrowing when application x64 is disabled; application chooses backend/x64 policy; device identity is not retained |
| CatBoost | Return diagnostic dictionary; no model constructor, weight import or training on load | Metadata only; use native artifacts to retain fitted model weights |
| Optuna | Return diagnostic dictionary; no study/storage constructor on load | Metadata only; trial summaries are not resumable storage |
| Plotly | Figure constructor and its schema validation | Restores supported figure data/layout; no file/image fetch API invoked by Datason. External asset references may be used later by an application renderer |
| user plugins | Application-defined methods | Code and allocation/I/O behavior must be reviewed separately; the built-in inventory cannot certify them |

The core imports available optional plugin modules by availability. Only sklearn
reconstruction selects a dotted module/class from a payload, and that selection
is restricted to sklearn estimator classes. All libraries listed above remain
optional; the shared dense allocation helper uses only the Python standard
library.

## Allocation checks and intentional changes

`max_input_bytes` caps both the parsed input and an estimated **represented
buffer** for dense ML values. A shape must agree with the supplied element
count, use non-negative integer dimensions, and respect `max_size`. Validate
before expensive constructors, then apply the stored shape, including zero-size
multidimensional arrays. Arrays with omitted legacy shape still use inference.

Sparse storage is proportional to stored entries and sometimes a pointer array,
not the full logical dense product. CSR checks a row pointer budget, CSC a column
pointer budget, and COO permits very large logical dimensions when its buffers
fit. Sparse coordinates must be within their shape. TF sparse checks its index
and value buffers without treating the logical dense product as an allocation.

These estimates do **not** cap peak memory, total allocations across a snapshot,
JIT compilation, native-library workspaces or application callbacks. For untrusted
input, use the dispatch-off policy, schema validation and application/process
resource controls rather than inferring a sandbox from individual checks.

JAX snapshots requiring 64-bit values fail clearly when x64 is disabled instead
of silently narrowing. Enable x64 deliberately in the application if its runtime
supports it, or perform an explicit schema-approved conversion before writing.
CPU reconstruction of Torch/TF tensors is intentional and portable; no arbitrary
device from the payload is selected for tensor allocation.

## Regression evidence

`tests/integration/test_ml_reconstruction.py` exercises actual Torch, TF, JAX,
SciPy and sklearn libraries: empty/scalar dimensions, malformed shapes,
constructor-not-reached assertions, huge CSR/CSC pointer rejection, large sparse
COO acceptance, TF sparse coordinate checks, CPU restoration, JAX x64 policy,
non-estimator state-hook rejection, and malformed state rejected before import.
It also checks dispatch-off behavior for every family and nested typed records.

Pinned local runtime: Torch 2.10.0+cpu, TF 2.20.0, JAX 0.9.0.1, SciPy 1.17.1,
sklearn 1.8.0, NumPy 2.4.2 and Pandas 3.0.1. Optional compatibility CI covers core
CPU ML and miscellaneous ML separately when the persisted-format PR is present.
A broader supported-library/version policy remains a future expansion, rather
than evidence supplied by this source audit.
