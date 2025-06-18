"""
JSON Compatibility Module - DataSON configured for stdlib json compatibility.

This module uses DataSON's core functionality but configured to behave exactly
like Python's built-in json module. This proves DataSON can be configured for
perfect compatibility while using our unified codebase.

For enhanced features (smart datetime parsing, dict output, ML types),
use the main datason API instead.

Examples:
    # Drop-in replacement for json module (using DataSON core)
    import datason.json as json
    json_str = json.dumps(data)  # Returns string (like stdlib json)
    obj = json.load(file)        # Basic parsing (like stdlib json)

    # Enhanced DataSON features
    import datason
    result = datason.dumps(data) # Returns dict with smart features
"""

import json as _json
from typing import Any

from .config import OutputType, SerializationConfig, TypeCoercion

# Import DataSON's core functionality
from .core_new import serialize as _core_serialize
from .deserializers_new import deserialize as _basic_deserialize

# Re-export json module constants and exceptions for compatibility
JSONDecodeError = _json.JSONDecodeError
JSONEncoder = _json.JSONEncoder
JSONDecoder = _json.JSONDecoder

# Configuration for JSON compatibility - disable all DataSON enhancements
_JSON_COMPAT_CONFIG = SerializationConfig(
    # Disable smart features to match stdlib json exactly
    uuid_format="string",  # UUIDs as strings, not objects
    parse_uuids=False,  # Don't auto-convert UUID strings
    datetime_output=OutputType.JSON_SAFE,  # Keep datetimes as strings
    series_output=OutputType.JSON_SAFE,  # Basic pandas Series output
    dataframe_output=OutputType.JSON_SAFE,  # Basic DataFrame output
    numpy_output=OutputType.JSON_SAFE,  # Basic numpy array output
    type_coercion=TypeCoercion.SAFE,  # Safe type handling
    include_type_hints=False,  # No type metadata
    auto_detect_types=False,  # No auto-detection of complex types
    check_if_serialized=False,  # Always process
    # Basic serialization only
    sort_keys=False,
    ensure_ascii=False,  # Let JSON handle this
)


def dumps(
    obj: Any,
    *,
    skipkeys: bool = False,
    ensure_ascii: bool = True,
    check_circular: bool = True,
    allow_nan: bool = True,
    cls=None,
    indent=None,
    separators=None,
    default=None,
    sort_keys: bool = False,
    **kwargs: Any,
) -> str:
    """
    Serialize obj to JSON string using DataSON core (stdlib json compatible).

    This uses DataSON's serialization engine but configured to behave exactly
    like stdlib json.dumps() - proving DataSON can be configured for compatibility.
    """
    # Use DataSON's core serialization with compatibility config
    serialized = _core_serialize(obj, config=_JSON_COMPAT_CONFIG)

    # Convert to JSON string using stdlib json with exact same parameters
    json_params = {
        "skipkeys": skipkeys,
        "ensure_ascii": ensure_ascii,
        "check_circular": check_circular,
        "allow_nan": allow_nan,
        "cls": cls,
        "indent": indent,
        "separators": separators,
        "default": default,
        "sort_keys": sort_keys,
    }

    return _json.dumps(serialized, **json_params)


def loads(s: str, **kwargs: Any) -> Any:
    """
    Parse JSON string using DataSON core (stdlib json compatible).

    This uses DataSON's deserialization engine but configured to behave exactly
    like stdlib json.loads() - no smart parsing, just basic JSON.
    """
    # First parse with stdlib json to get basic structure
    parsed = _json.loads(s, **kwargs)

    # Then process with DataSON's basic deserializer (no smart features)
    return _basic_deserialize(parsed, parse_dates=False, parse_uuids=False)


def dump(
    obj: Any,
    fp,
    *,
    skipkeys: bool = False,
    ensure_ascii: bool = True,
    check_circular: bool = True,
    allow_nan: bool = True,
    cls=None,
    indent=None,
    separators=None,
    default=None,
    sort_keys: bool = False,
    **kwargs: Any,
) -> None:
    """
    Serialize obj to JSON and write to file using DataSON core (stdlib compatible).
    """
    # Use DataSON's core serialization with compatibility config
    serialized = _core_serialize(obj, config=_JSON_COMPAT_CONFIG)

    # Write to file using stdlib json with exact same parameters
    json_params = {
        "skipkeys": skipkeys,
        "ensure_ascii": ensure_ascii,
        "check_circular": check_circular,
        "allow_nan": allow_nan,
        "cls": cls,
        "indent": indent,
        "separators": separators,
        "default": default,
        "sort_keys": sort_keys,
    }

    _json.dump(serialized, fp, **json_params)


def load(fp, **kwargs: Any) -> Any:
    """
    Parse JSON from file using DataSON core (stdlib json compatible).
    """
    # First parse with stdlib json
    parsed = _json.load(fp, **kwargs)

    # Then process with DataSON's basic deserializer (no smart features)
    return _basic_deserialize(parsed, parse_dates=False, parse_uuids=False)


# Module-level documentation
__doc__ = """
DataSON JSON Compatibility Module

This module uses DataSON's core functionality configured to provide 100% API
compatibility with Python's built-in json module. This proves DataSON can be
configured for perfect compatibility using our unified codebase.

Functions:
    dumps(obj, **kwargs) -> str: Serialize using DataSON core (stdlib compatible)
    loads(s, **kwargs) -> Any: Parse using DataSON core (stdlib compatible)
    dump(obj, fp, **kwargs) -> None: Write using DataSON core (stdlib compatible)
    load(fp, **kwargs) -> Any: Read using DataSON core (stdlib compatible)

For enhanced DataSON features, use the main datason module instead:
    import datason  # Enhanced API with smart features
"""

__all__ = ["dumps", "loads", "dump", "load", "JSONDecodeError", "JSONEncoder", "JSONDecoder"]
