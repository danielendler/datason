# Scientific round-trip contract

With `include_type_hints=True`, datason preserves numeric NumPy scalar dtype
widths, unsigned integers, booleans, and complex scalars. Array shape metadata is
used during reconstruction, including empty dimensions and zero-dimensional
arrays. Complex arrays use pairs of real/imaginary components; datetime and
timedelta arrays use signed integer ticks plus their dtype/unit. Temporal scalars
use the same tick representation. Old scalar payloads without dtype metadata
remain readable using their previous default dtype.

Array dtype size, dimensions, and estimated storage are checked before NumPy
allocation. The reconstruction storage budget uses `max_input_bytes`. This is a
per-array estimate, not a process memory limit. Structured and void dtypes require
an explicit custom plugin rather than silently losing their field definitions.
Arbitrary object arrays and extended-precision scalars are outside the guaranteed
fidelity contract. Non-finite-number and redaction policies may intentionally
change values, so a diagnostic export is not always a resumable checkpoint.

Typed Pandas frames preserve indexes, columns, names, and column dtypes. Nullable
integers/strings (including Pandas 3's missing-value convention), categorical
domains/order, timestamps, and nanosecond timedeltas
are supported. Series preserve their index and dtype. Common indexes include
RangeIndex, ordinary Index, DatetimeIndex, TimedeltaIndex, CategoricalIndex, and
MultiIndex. Empty frames, duplicate labels, and non-string columns use a split
representation when type hints are enabled to avoid lossy dictionary orientations.
PeriodIndex and IntervalIndex require a custom plugin. DataFrame attrs, arbitrary
extension arrays, and every third-party dtype are outside this contract.

Numeric Python datetime records include explicit seconds/milliseconds units and
an ISO representation that preserves naive status, UTC offset, microseconds, and
fold. Naive numeric encoding uses UTC independently of the machine timezone.
Legacy records without explicit units retain the old magnitude heuristic; their
unit ambiguity cannot be repaired after the fact. Named timezone identity is not
preserved; the serialized UTC offset is.

Without type hints, these types normalize to JSON values according to the chosen
API policies. Such JSON is intended for consumers that do not understand datason
metadata and does not promise exact type reconstruction. Persisted typed payloads
remain version-sensitive: retain fixture files and test them before upgrading.
