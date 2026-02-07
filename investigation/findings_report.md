# Datason v2.0.0a1 Evaluation Report

## Executive Summary
Datason is a powerful zero-dependency serialization library that successfully handles many complex types (datetime, numpy, pandas, etc.). However, it currently falls short of its "drop-in replacement" claim for the standard `json` library and has some fidelity and security concerns that should be addressed before a stable release.

## 1. Drop-in Compatibility Issues
Despite the README claiming it's a "Drop-in replacement for json.dumps/json.loads", the API signatures are not fully compatible:
- **Missing Arguments**: `datason.dumps` raises `TypeError` if passed standard `json` arguments like `indent`, `skipkeys`, or `ensure_ascii`.
- **Missing Arguments in loads**: `datason.loads` does not support `parse_float`, `parse_int`, etc.
- **Strict Configuration**: Datason uses a `SerializationConfig` object internally that doesn't accept all `json` parameters.

**Recommendation**: Support common `json` arguments in `dumps`/`loads` and pass them through to the underlying `json` calls where appropriate.

## 2. Type Fidelity & Round-trip Concerns
- **Collections**: `tuple`, `set`, and `frozenset` are serialized as standard JSON lists. On `loads()`, they are restored as `list`, losing their original type identity even when `include_type_hints=True`.
- **Timezone Mismatch**: When using `date_format=DateFormat.UNIX_MS`, timezone-naive datetimes are restored as UTC-aware datetimes. This changes the meaning of the data on round-trip.
- **Float Special Values**: `NanHandling.STRING` serializes `NaN` as `"nan"` (lowercase), but the test expected `"NaN"`. Consistency with standard JSON or other libraries (like `simplejson`) should be checked.

## 3. Security Considerations
- **Resource Exhaustion**: The `max_depth` and `max_size` limits are effective and work as advertised.
- **Arbitrary Code Execution (ACE)**: The plugin system allows for arbitrary code execution during `loads()` if a malicious plugin is registered. While plugin registration is usually controlled by the developer, any third-party library that registers a Datason plugin could potentially introduce vulnerabilities.
- **Untrusted Type Hints**: Deserializing payloads with type hints (like `pathlib.Path`) automatically instantiates those types. While not inherently ACE, it could lead to unexpected behavior if an application doesn't expect objects of certain types to appear in its data structures.

## 4. Performance & Robustness
- **Concurrancy**: `datason.config` context manager is correctly implemented using `ContextVar`, making it thread-safe.
- **Large Inputs**: Handled well up to the defined security limits. Note that very large integers are limited by Python's `sys.set_int_max_str_digits()`.

## 5. Detailed Bug List
1. `TypeError` on `datason.dumps(data, indent=4)`.
2. `tuple` -> `list` (fidelity loss).
3. `set` -> `list` (fidelity loss).
4. `datetime` (naive) -> `datetime` (UTC) when using UNIX timestamps.
5. `NanHandling.STRING` produces `"nan"` instead of `"NaN"`.

## Conclusion
Datason is a very promising library with a clean architecture. Fixing the API compatibility and collection fidelity would make it a much stronger contender for replacing `json` in complex Python projects.
