"""Trusted pickle-to-JSON migration for datason.

Converts pickle-serialized Python objects to datason JSON format,
providing a migration path for pickle data whose source the caller trusts.
Module scanning is a diagnostic compatibility check, not a security sandbox.

Security model:
- Pickle loading can execute arbitrary code, including from allowed modules.
- Conversion requires explicit trusted=True; untrusted pickle is never loaded.
- The optional module filter cannot establish that a pickle is safe.
"""

from __future__ import annotations

import io
import pickle  # nosec B403 — intentional: caller must explicitly trust pickle input
import pickletools  # nosec B403
from typing import Any

import datason

from .._errors import SecurityError

# Default module filter for migration diagnostics; this does not confer safety.
DEFAULT_ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "builtins",
        "collections",
        "datetime",
        "decimal",
        "fractions",
        "pathlib",
        "uuid",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "torch",
        "tensorflow",
    }
)


def scan_pickle_modules(data: bytes) -> set[str]:
    """Scan pickle bytes and extract all module names referenced.

    Uses pickletools to inspect opcodes without executing the pickle.
    Handles both protocol 0-1 (GLOBAL "module\\nclass") and protocol
    2+ (SHORT_BINUNICODE module, SHORT_BINUNICODE class, STACK_GLOBAL).

    Args:
        data: Raw pickle bytes to scan.

    Returns:
        Set of top-level module names (e.g. {"numpy", "sklearn"}).
    """
    modules: set[str] = set()
    string_stack: list[str] = []
    try:
        for opcode, arg, _pos in pickletools.genops(data):
            if opcode.name in ("GLOBAL", "INST") and isinstance(arg, str):
                # Protocol 0-1: "module.submod\nclass"
                mod_line = arg.split("\n")[0]
                modules.add(mod_line.split(".")[0])
            elif opcode.name in ("SHORT_BINUNICODE", "BINUNICODE") and isinstance(arg, str):
                string_stack.append(arg)
            elif opcode.name == "STACK_GLOBAL":
                # Protocol 2+: pops (module_name, class_name)
                if len(string_stack) >= 2:
                    module_name = string_stack[-2]
                    modules.add(module_name.split(".")[0])
                string_stack.clear()
            else:
                # Non-string opcodes reset the tracking
                if opcode.name not in ("MEMOIZE",):
                    string_stack.clear()
    except Exception:  # noqa: BLE001
        modules.add("__unparseable__")
    return modules


def validate_pickle_safety(
    data: bytes,
    allowed_modules: frozenset[str] = DEFAULT_ALLOWED_MODULES,
) -> tuple[bool, set[str]]:
    """Check detected module references, without establishing pickle safety.

    The historical function name is retained for compatibility. A True result
    only means the scanner found no disallowed module; it does not authorize
    execution or make untrusted pickle safe.

    Args:
        data: Raw pickle bytes.
        allowed_modules: Frozenset of allowed top-level module names.

    Returns:
        Tuple of (modules_allowed, disallowed_modules). A True result is only
        a diagnostic result and never establishes that executing pickle is safe.
    """
    found = scan_pickle_modules(data)
    disallowed = found - allowed_modules
    return (len(disallowed) == 0, disallowed)


def pickle_to_json(
    data: bytes,
    allowed_modules: frozenset[str] = DEFAULT_ALLOWED_MODULES,
    *,
    trusted: bool = False,
    **kwargs: Any,
) -> str:
    """Convert pickle bytes to a datason JSON string.

    Requires trusted=True before scanning or loading. The module filter is
    supplementary; allowed modules can contain arbitrary executable callables.

    Args:
        data: Raw pickle bytes to convert.
        allowed_modules: Frozenset of allowed module prefixes.
        trusted: Explicit assertion that the pickle source is trusted.
        **kwargs: Passed to datason.dumps (config overrides).

    Returns:
        JSON string from datason.dumps.

    Raises:
        SecurityError: If pickle references disallowed modules.
    """
    _require_trusted(trusted)
    is_safe, disallowed = validate_pickle_safety(data, allowed_modules)
    if not is_safe:
        raise SecurityError(
            f"Pickle references disallowed modules: {sorted(disallowed)}. Add them to allowed_modules if trusted."
        )
    obj = pickle.loads(data)  # noqa: S301  # nosec B301 — caller explicitly trusts the input
    return datason.dumps(obj, **kwargs)


def pickle_file_to_json(
    pickle_path: str,
    allowed_modules: frozenset[str] = DEFAULT_ALLOWED_MODULES,
    *,
    trusted: bool = False,
    **kwargs: Any,
) -> str:
    """Read a pickle file and convert to datason JSON string.

    Args:
        pickle_path: Path to the pickle file.
        allowed_modules: Frozenset of allowed module prefixes.
        trusted: Explicit assertion that the file source is trusted.
        **kwargs: Passed to datason.dumps.

    Returns:
        JSON string.

    Raises:
        SecurityError: If pickle references disallowed modules.
        FileNotFoundError: If the file doesn't exist.
    """
    _require_trusted(trusted)
    with open(pickle_path, "rb") as f:
        data = f.read()
    return pickle_to_json(data, allowed_modules, trusted=True, **kwargs)


def _require_trusted(trusted: bool) -> None:
    """Fail before opening files or executing pickle code without explicit trust."""
    if trusted is not True:
        raise SecurityError("Pickle can execute arbitrary code. Convert only trusted input with trusted=True.")


def json_to_pickle(json_str: str, **kwargs: Any) -> bytes:
    """Convert a datason JSON string to pickle bytes.

    Deserializes via datason.loads, then pickle-serializes the result.

    Args:
        json_str: JSON string (possibly with datason type metadata).
        **kwargs: Passed to datason.loads.

    Returns:
        Pickle bytes.
    """
    obj = datason.loads(json_str, **kwargs)
    buf = io.BytesIO()
    pickle.dump(obj, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.getvalue()
