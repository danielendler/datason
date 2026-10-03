# Testing Plan for Datason v2.0.0a1

## 1. Goal
Thoroughly evaluate the `datason` package for bugs, security vulnerabilities, and performance issues. As an expert blackbox tester, I will attempt to break the library using various creative scenarios.

## 2. Test Scenarios

### 2.1. Drop-in Replacement Fidelity
*   **Argument Compatibility**: Verify if `datason.dumps` and `datason.loads` support all standard `json` arguments (e.g., `indent`, `separators`, `skipkeys`, `ensure_ascii`, `check_circular`, `allow_nan`, `cls`, `default`).
*   **Exception Types**: Compare exception types raised by `datason` vs `json` for same invalid inputs.

### 2.2. Security & Resource Exhaustion
*   **Recursion & Depth**: Test `max_depth` limit. Try to bypass it with custom objects or plugins.
*   **Memory Exhaustion**: Test `max_size` limit with extremely large dictionaries or lists.
*   **Circular References**: Verify detection of circular references in complex graphs.
*   **Malicious Type Hints**: Craft JSON strings with `__datason_type__` pointing to internal classes or unexpected types to see if it triggers arbitrary code execution or unexpected state changes.
*   **Path Traversal**: Since `Path` objects are supported, test if deserializing a `Path` can be used to leak info or access restricted files (though it's just a Path object, check how it's used).
*   **Redaction Bypassing**: Attempt to obfuscate keys or use nested structures to avoid `redact_fields`. Test regex redaction performance and potential ReDoS (Regular Expression Denial of Service).
*   **Integrity Verification**: Attempt to tamper with a payload and its signature/hash. Test with weak keys or common hashing vulnerabilities.

### 2.3. Edge Case Data Types
*   **Floating Point**: `NaN`, `inf`, `-inf` in various configurations and `nan_handling` settings.
*   **Empty Containers**: Empty sets, frozensets, tensors, dataframes.
*   **Extreme Values**: Max/min dates, huge Decimals, very long strings, complex numbers with extreme components.
*   **Nested Mixed Types**: A list of tensors containing dictionaries of datetimes.
*   **Type Fidelity**: Ensure `tuple` vs `list` vs `set` are correctly restored (if `include_type_hints=True`).

### 2.4. Plugin System & Customization
*   **Conflict Resolution**: Register two plugins for the same type.
*   **Malicious Plugins**: Create a plugin that performs side effects during `serialize` or `deserialize`.
*   **Stateful Plugins**: Check if plugins can leak state between different `dumps`/`loads` calls.

### 2.5. Environment & Dependencies
*   **Missing Dependencies**: Run tests in an environment where `numpy`, `torch`, or `pandas` are NOT installed to ensure graceful degradation.
*   **Version Mismatch**: Test with older/newer versions of supported libraries if possible.

### 2.6. Concurrent & Multi-threaded Usage
*   **Thread Safety**: Test if `datason.config` (context manager) is thread-local and doesn't leak settings across threads.
*   **Race Conditions**: Concurrent `dumps`/`loads` calls sharing the same registry.

## 3. Execution Strategy
1.  **Exploratory Testing**: Manual scripts to probe identified scenarios.
2.  **Automated Scripting**: Write a suite of `pytest` scripts specifically targeting these scenarios.
3.  **Fuzzing**: Use `hypothesis` or simple random mutation fuzzing on the input strings for `loads`.
4.  **Reporting**: Document every failure, unexpected behavior, or security concern.

## 4. Deliverables
- `testing_plan.md` (This document)
- `test_break_datason.py` (Script containing breaking test cases)
- `findings_report.md` (Summary of results for the maker)
