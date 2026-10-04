# Boundary performance baseline

Measured October 4, 2026 on merged runtime source `030b3bb`, Python 3.12.14,
Linux x86_64, in a shared development container. This is a bounded local study,
not a production capacity claim or a universal serializer ranking. No runtime
optimization or regression threshold was changed.

The [raw report](https://github.com/danielendler/datason/blob/main/perf/evidence/2026-10-04-boundary-study.json)
and [complete environment manifest](https://github.com/danielendler/datason/blob/main/perf/evidence/2026-10-04-environments.json)
contain source/tool/input hashes, package versions, five individual rounds,
pooled p50/p95/p99, encoded UTF-8 byte sizes, Python-managed allocation peaks,
fresh-process import samples and a cProfile call summary. Each operation has
500 samples; shuffled interleaving avoids timing each library in a separate
long block. Tracemalloc and profiling run outside latency sampling. Shared
hardware and scheduling still affect tails; these are observations, not gates.

## Ordinary JSON

Eight existing NDJSON samples are synthetic JSON-only objects, approximately
0.3–0.6 KB. Names such as `medium` and `data_science` do not turn these fixtures
into large real DataFrames or customer traffic. All codecs are checked for equal
JSON values before timing. Datason and stdlib JSON sort keys and emit default
whitespace; orjson sorts keys and emits compact bytes; Pydantic object emits compact
bytes in insertion order. Encoding sizes and these native output differences
are retained rather than hidden through another conversion.

For `api-small-nested` (308 bytes with Datason/stdlib, 274 compact bytes):

| Codec | Dump p50 / p95 (ms) | Load p50 / p95 (ms) | Dump Python peak (bytes) |
| --- | --- | --- | --- |
| Datason API policy | 0.062 / 0.098 | 0.039 / 0.054 | 2,966 |
| stdlib json | 0.007 / 0.009 | 0.005 / 0.007 | 1,302 |
| orjson 3.12.0 | 0.002 / 0.003 | 0.003 / 0.004 | 4,281 |
| Pydantic 2.13.5 object adapter | 0.003 / 0.005 | 0.005 / 0.007 | 307 |

Across these eight inputs, Datason dump medians are about 8–9 times stdlib JSON
and 34–43 times orjson. The absolute extra median time versus stdlib is roughly
0.05–0.08 ms per dump. Datason also applies normalization and resource policies;
the baselines do not implement matching depth/node/string budgets. Already-JSON
hot endpoints should normally use their existing JSON codec when they do not
need those policies. Pydantic object here is a generic codec (`TypeAdapter(object)`), not a benchmark of
validation against a full application schema.

## Typed scientific data

Two synthetic payloads contain datetime, UUID, float32 scalar, binary bytes and
float32 matrices of 64 or 4,096 elements. Tagged snapshot round trips verify
values, scalar/array dtype, shape and binary fidelity before timing. These are
small/medium diagnostic or checkpoint payloads, not model weights or video.

For the 4,096-element matrix (shape 1,024 × 4):

| Operation | p50 / p95 (ms) | Python peak (bytes) | Input contract |
| --- | --- | --- | --- |
| Datason API dump | 5.389 / 8.548 | 293,123 | Typed Python values → ordinary JSON |
| stdlib JSON dump | 0.660 / 1.053 | 35,145 | Already-normalized JSON values |
| orjson dump | 0.137 / 0.227 | 259,737 | Already-normalized JSON values |
| Pydantic object dump | 0.235 / 0.408 | 29,785 | Already-normalized JSON values |
| Datason snapshot dump | 5.512 / 8.708 | 293,393 | Typed values → tagged snapshot |
| Datason snapshot load | 2.551 / 4.105 | 242,044 | Tagged snapshot → verified typed values |

Projection happens before the baseline codec measurements; those rows exclude
conversion and cannot be called equivalent replacements for the Datason typed
pipeline. API output is 33,856 bytes with Datason and 29,752 compact bytes;
the tagged snapshot is 34,210 bytes. Tracemalloc reports Python-managed peaks,
not total native allocator use or process RSS. A Rust codec can reduce the final
JSON encoding cost, but does not remove Datason's Python policy/traversal work.

## Fresh-process import

Three new interpreter processes per environment measure the import interval,
process wall time, and Linux `/proc/self/status` RSS. OS file caches are not
cleared. RSS growth is measured inside each child process before and after
import; inherited `getrusage` high-water marks would give misleading deltas.

| Installed environment | Median import | Median resident growth | Observed imports |
| --- | --- | --- | --- |
| Core only | 22 ms | 4.9 MiB | No optional libraries |
| NumPy/Pandas | 272 ms | 58.5 MiB | NumPy, Pandas |
| Full ML validation environment | 4,602 ms | 1,080.9 MiB | NumPy, Pandas, Torch, TensorFlow, JAX, sklearn |

Core-only has zero required dependencies. When optional libraries are installed,
plugin registration currently imports them eagerly. Installing more packages
therefore changes startup cost even if a process only emits a small API payload.
The report records exact environment versions; these environments also differ
in NumPy/Pandas versions. RSS includes imported libraries, not just Datason's
own Python objects. This cost is especially relevant to workers, CLIs and short
jobs; an already-running ML process may have paid much of it anyway.

## Bottlenecks and next decisions

The 20 tagged dump/load pairs for the 4,096-element case generated about
2.05 million profiled calls. Tree budget checks used 0.268 seconds inclusive
of a 0.644-second profiled run (about 42%). Sequence serialization used 0.284
seconds inclusive. Those cumulative totals overlap other function costs and
must not be added into a speedup forecast. Repeated `isinstance`, list-stack
operations, finite-float handling and NumPy allocation validation are visible.
These profiled times are not the uninstrumented table timings.

The focused next candidate is lazy optional-plugin imports: keep plugin names,
priorities and tagged loading behavior while avoiding unrelated ML imports on
ordinary JSON calls. Require fresh-process tests, installed/uninstalled cases,
concurrent first use, reconstruction coverage and before/after import evidence.
Do not move this cost unnoticed to every hot operation.

For typed throughput, inspect the budget and sequence traversals before changing
loops. NumPy already projects arrays through `tolist()`; wrapping a Python
callback in `map` still performs per-element work. A homogeneous-array fast path
could help only if it preserves non-finite policies, limits, redaction semantics
and dtype/shape tests. Combining traversals needs equivalent pre-allocation and
metadata-inclusive checks; deleting validation is not a performance fix.

At these small API sizes, network/model latency may dominate, but no end-to-end
application timing was measured. At batch/export sizes the milliseconds and
allocations can matter. Profile the actual financial endpoint before choosing
an optimization. Cache an unchanged projection when application semantics
permit, and use binary artifacts/references for large tensors. There is no
current evidence requiring a Rust rewrite or broad new optimization framework.

## Reproduce

From a checkout with NumPy, Pydantic and orjson installed:

```bash
python -m scripts.perf.boundary_study --output perf/results/boundary-study.json
```

Add `--import-python core=/path/to/core-venv/bin/python` (repeat for other
installed environments) to measure fresh imports on Linux. Use the raw manifest
to recreate package versions, and set CPU library thread counts to one as in the
validation run. The evidence workflow exercises the harness and uploads a fresh
report; it has no timing threshold. The existing replay regression workflow and
90% runtime patch-coverage gate remain intact.
