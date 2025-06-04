"""Data Transformation Utilities for datason.

This module exposes useful data transformation tools directly, without requiring
users to go through the full serialization process. Perfect for data analysis,
comparison, enhancement, and transformation workflows.

SECURITY: This module applies the same security patterns as core.py to prevent
resource exhaustion and infinite recursion attacks.
"""

import re
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# SECURITY: Import same constants as core.py for consistency
try:
    from .core import MAX_OBJECT_SIZE, MAX_SERIALIZATION_DEPTH, MAX_STRING_LENGTH
except ImportError:
    # Fallback constants if core import fails - SECURITY FIX: Use secure values
    MAX_SERIALIZATION_DEPTH = 50
    MAX_OBJECT_SIZE = 100_000  # SECURITY FIX: Reduced from 10_000_000 to 100_000 to prevent size bomb attacks
    MAX_STRING_LENGTH = 1_000_000


class UtilitySecurityError(Exception):
    """Raised when security limits are exceeded during utility operations."""


class UtilityConfig:
    """Configuration class for utility operations with security limits."""

    def __init__(
        self,
        max_depth: int = MAX_SERIALIZATION_DEPTH,
        max_object_size: int = MAX_OBJECT_SIZE,
        max_string_length: int = MAX_STRING_LENGTH,
        max_collection_size: int = 10_000,
        enable_circular_reference_detection: bool = True,
        timeout_seconds: Optional[float] = None,
    ):
        """Initialize utility configuration.

        Args:
            max_depth: Maximum recursion depth for nested operations
            max_object_size: Maximum size for objects being processed
            max_string_length: Maximum string length to process
            max_collection_size: Maximum collection size for anomaly detection
            enable_circular_reference_detection: Whether to detect circular references
            timeout_seconds: Optional timeout for operations (not implemented yet)
        """
        self.max_depth = max_depth
        self.max_object_size = max_object_size
        self.max_string_length = max_string_length
        self.max_collection_size = max_collection_size
        self.enable_circular_reference_detection = enable_circular_reference_detection
        self.timeout_seconds = timeout_seconds


def get_default_utility_config() -> UtilityConfig:
    """Get default utility configuration."""
    return UtilityConfig()


def deep_compare(
    obj1: Any, obj2: Any, tolerance: float = 1e-10, config: Optional[UtilityConfig] = None
) -> Dict[str, Any]:
    """Deep comparison of two objects with tolerance for numeric values.

    Args:
        obj1: First object to compare
        obj2: Second object to compare
        tolerance: Tolerance for floating point comparisons
        config: Optional utility configuration with security limits

    Returns:
        Dictionary with comparison results including differences and summary

    Raises:
        UtilitySecurityError: If security limits are exceeded
    """
    if config is None:
        config = get_default_utility_config()

    result: Dict[str, Any] = {
        "differences": [],
        "summary": {
            "total_differences": 0,
            "type_mismatches": 0,
            "value_differences": 0,
        },
    }

    # SECURITY: Track visited objects and depth to prevent attacks
    visited1: set = set() if config.enable_circular_reference_detection else set()
    visited2: set = set() if config.enable_circular_reference_detection else set()

    _compare_recursive(obj1, obj2, result, "", tolerance, visited1, visited2, 0, config)

    result["are_equal"] = len(result["differences"]) == 0
    result["summary"]["total_differences"] = len(result["differences"])

    return result


def _compare_recursive(
    obj1: Any,
    obj2: Any,
    result: Dict[str, Any],
    path: str,
    tolerance: float,
    visited1: set,
    visited2: set,
    depth: int,
    config: UtilityConfig,
) -> None:
    """Recursively compare objects with security checks."""
    # SECURITY: Depth limit enforcement
    if depth > config.max_depth:
        raise UtilitySecurityError(
            f"Maximum comparison depth ({config.max_depth}) exceeded at path '{path}'. "
            "This may indicate circular references or extremely nested data."
        )

    # SECURITY: Object size checks
    if hasattr(obj1, "__len__") and len(obj1) > config.max_object_size:
        raise UtilitySecurityError(f"Object size exceeds maximum ({config.max_object_size}) at path '{path}'")
    if hasattr(obj2, "__len__") and len(obj2) > config.max_object_size:
        raise UtilitySecurityError(f"Object size exceeds maximum ({config.max_object_size}) at path '{path}'")

    # SECURITY: Circular reference detection
    if config.enable_circular_reference_detection:
        obj1_id = id(obj1)
        obj2_id = id(obj2)

        if obj1_id in visited1 or obj2_id in visited2:
            # Already visited, assume equal to avoid infinite recursion
            return

        # Add to visited sets for complex objects
        if isinstance(obj1, (dict, list, tuple)) and isinstance(obj2, (dict, list, tuple)):
            visited1.add(obj1_id)
            visited2.add(obj2_id)

    try:
        if not isinstance(obj1, type(obj2)) or not isinstance(obj2, type(obj1)):
            result["differences"].append(
                f"Type mismatch at {path or 'root'}: {type(obj1).__name__} vs {type(obj2).__name__}"
            )
            result["summary"]["type_mismatches"] += 1
        elif isinstance(obj1, dict):
            _compare_dicts(obj1, obj2, result, path, tolerance, visited1, visited2, depth, config)
        elif isinstance(obj1, (list, tuple)):
            _compare_lists(obj1, obj2, result, path, tolerance, visited1, visited2, depth, config)
        elif isinstance(obj1, (int, float)):
            _compare_numbers(obj1, obj2, result, path, tolerance)
        elif isinstance(obj1, str):
            _compare_strings(obj1, obj2, result, path, config)
        else:
            if obj1 != obj2:
                result["differences"].append(f"Value difference at {path or 'root'}: {obj1} != {obj2}")
                result["summary"]["value_differences"] += 1
    finally:
        # Clean up: remove from visited sets when done
        if (
            config.enable_circular_reference_detection
            and isinstance(obj1, (dict, list, tuple))
            and isinstance(obj2, (dict, list, tuple))
        ):
            visited1.discard(id(obj1))
            visited2.discard(id(obj2))


def _compare_dicts(
    dict1: dict,
    dict2: dict,
    result: Dict[str, Any],
    path: str,
    tolerance: float,
    visited1: set,
    visited2: set,
    depth: int,
    config: UtilityConfig,
) -> None:
    """Compare two dictionaries with security checks."""
    # SECURITY: Size limits
    if len(dict1) > config.max_collection_size or len(dict2) > config.max_collection_size:
        warnings.warn(f"Large dictionary detected at {path} (sizes: {len(dict1)}, {len(dict2)})", stacklevel=2)

    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        new_path = f"{path}.{key}" if path else str(key)

        if key not in dict1:
            result["differences"].append(f"Missing key in first object at {new_path}")
            result["summary"]["value_differences"] += 1
        elif key not in dict2:
            result["differences"].append(f"Missing key in second object at {new_path}")
            result["summary"]["value_differences"] += 1
        else:
            _compare_recursive(
                dict1[key], dict2[key], result, new_path, tolerance, visited1, visited2, depth + 1, config
            )


def _compare_lists(
    list1: Union[list, tuple],
    list2: Union[list, tuple],
    result: Dict[str, Any],
    path: str,
    tolerance: float,
    visited1: set,
    visited2: set,
    depth: int,
    config: UtilityConfig,
) -> None:
    """Compare two lists or tuples with security checks."""
    # SECURITY: Size limits
    if len(list1) > config.max_collection_size or len(list2) > config.max_collection_size:
        warnings.warn(f"Large list detected at {path} (sizes: {len(list1)}, {len(list2)})", stacklevel=2)

    if len(list1) != len(list2):
        result["differences"].append(f"Length mismatch at {path or 'root'}: {len(list1)} vs {len(list2)}")
        result["summary"]["value_differences"] += 1

    for i in range(min(len(list1), len(list2))):
        new_path = f"{path}[{i}]" if path else f"[{i}]"
        _compare_recursive(list1[i], list2[i], result, new_path, tolerance, visited1, visited2, depth + 1, config)


def _compare_numbers(
    num1: Union[int, float], num2: Union[int, float], result: Dict[str, Any], path: str, tolerance: float
) -> None:
    """Compare two numbers with tolerance."""
    if abs(num1 - num2) > tolerance:
        result["differences"].append(f"Numeric difference at {path}: {num1} vs {num2} (diff: {abs(num1 - num2)})")
        result["summary"]["value_differences"] += 1


def _compare_strings(str1: str, str2: str, result: Dict[str, Any], path: str, config: UtilityConfig) -> None:
    """Compare two strings with security checks."""
    # SECURITY: String length limits
    if len(str1) > config.max_string_length or len(str2) > config.max_string_length:
        warnings.warn(f"Large string detected at {path} (lengths: {len(str1)}, {len(str2)})", stacklevel=2)
        # Truncate for comparison
        str1_truncated = str1[: config.max_string_length] if len(str1) > config.max_string_length else str1
        str2_truncated = str2[: config.max_string_length] if len(str2) > config.max_string_length else str2
        if str1_truncated != str2_truncated:
            result["differences"].append(f"String difference at {path}: truncated strings differ")
            result["summary"]["value_differences"] += 1
    elif str1 != str2:
        result["differences"].append(f"String difference at {path}: {repr(str1)} != {repr(str2)}")
        result["summary"]["value_differences"] += 1


def find_data_anomalies(
    data: Any, rules: Optional[Dict[str, Any]] = None, config: Optional[UtilityConfig] = None
) -> Dict[str, Any]:
    """Find potential anomalies in data structures with security limits.

    Args:
        data: Data structure to analyze
        rules: Optional rules for anomaly detection
        config: Optional utility configuration with security limits

    Returns:
        Dictionary containing found anomalies

    Raises:
        UtilitySecurityError: If security limits are exceeded
    """
    if config is None:
        config = get_default_utility_config()

    if rules is None:
        rules = {
            "max_string_length": 10000,  # Anomaly detection limit, not security limit
            "max_collection_size": 1000,  # Anomaly detection limit
            "max_dict_size": 1000,  # Anomaly detection limit
            "max_list_length": 1000,  # Anomaly detection limit
            "suspicious_patterns": [r"<script", r"javascript:", r"eval\("],
            "detect_suspicious_patterns": True,
        }

    anomalies: Dict[str, List[Any]] = {
        "large_strings": [],
        "large_collections": [],
        "suspicious_patterns": [],
        "security_violations": [],
    }

    # SECURITY: Track visited objects and depth
    visited: set = set() if config.enable_circular_reference_detection else set()

    _detect_anomalies_recursive(data, anomalies, "", rules, visited, 0, config)
    return anomalies


def _detect_anomalies_recursive(
    obj: Any,
    anomalies: Dict[str, List[Any]],
    path: str,
    rules: Dict[str, Any],
    visited: set,
    depth: int,
    config: UtilityConfig,
) -> None:
    """Recursively detect anomalies with security checks."""
    # SECURITY: Depth limit enforcement
    if depth > config.max_depth:
        anomalies["security_violations"].append(
            {
                "path": path,
                "violation": "max_depth_exceeded",
                "details": f"Depth {depth} exceeds maximum {config.max_depth}",
            }
        )
        return

    # SECURITY: Circular reference detection
    if config.enable_circular_reference_detection:
        obj_id = id(obj)
        if obj_id in visited:
            anomalies["security_violations"].append(
                {"path": path, "violation": "circular_reference", "details": "Circular reference detected"}
            )
            return

        if isinstance(obj, (dict, list, tuple)):
            visited.add(obj_id)

    try:
        if isinstance(obj, str):
            # SECURITY: String length check
            if len(obj) > config.max_string_length:
                anomalies["security_violations"].append(
                    {
                        "path": path,
                        "violation": "string_too_long",
                        "details": f"String length {len(obj)} exceeds maximum {config.max_string_length}",
                    }
                )
                return  # Don't process overly long strings

            if len(obj) > rules.get("max_string_length", 10000):
                anomalies["large_strings"].append({"path": path, "length": len(obj)})

            if rules.get("detect_suspicious_patterns"):
                suspicious = rules.get("suspicious_patterns", [])
                for pattern in suspicious:
                    try:
                        if re.search(pattern, obj, re.IGNORECASE):
                            anomalies["suspicious_patterns"].append({"path": path, "pattern": pattern})
                    except re.error:
                        # Invalid regex pattern, skip
                        pass

        elif isinstance(obj, dict):
            # SECURITY: Size check
            if len(obj) > config.max_object_size:
                anomalies["security_violations"].append(
                    {
                        "path": path,
                        "violation": "object_too_large",
                        "details": f"Dict size {len(obj)} exceeds maximum {config.max_object_size}",
                    }
                )
                return

            if len(obj) > rules.get("max_dict_size", 1000):
                anomalies["large_collections"].append({"path": path, "type": "dict", "size": len(obj)})

            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else str(key)
                _detect_anomalies_recursive(value, anomalies, new_path, rules, visited, depth + 1, config)

        elif isinstance(obj, (list, tuple)):
            # SECURITY: Size check
            if len(obj) > config.max_object_size:
                anomalies["security_violations"].append(
                    {
                        "path": path,
                        "violation": "object_too_large",
                        "details": f"List size {len(obj)} exceeds maximum {config.max_object_size}",
                    }
                )
                return

            if len(obj) > rules.get("max_list_length", 1000):
                anomalies["large_collections"].append({"path": path, "type": type(obj).__name__, "size": len(obj)})

            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                _detect_anomalies_recursive(item, anomalies, new_path, rules, visited, depth + 1, config)

    finally:
        # Clean up: remove from visited set when done
        if config.enable_circular_reference_detection and isinstance(obj, (dict, list, tuple)):
            visited.discard(id(obj))


def enhance_data_types(
    data: Any, enhancement_rules: Optional[Dict[str, Any]] = None, config: Optional[UtilityConfig] = None
) -> Tuple[Any, Dict[str, Any]]:
    """Enhance data types by applying smart type inference and conversion.

    Args:
        data: Data structure to enhance
        enhancement_rules: Optional rules for enhancement
        config: Optional utility configuration with security limits

    Returns:
        Tuple of (enhanced_data, enhancement_report)

    Raises:
        UtilitySecurityError: If security limits are exceeded
    """
    if config is None:
        config = get_default_utility_config()

    if enhancement_rules is None:
        enhancement_rules = {
            "convert_numeric_strings": True,
            "parse_dates": True,
            "clean_whitespace": True,
            "normalize_booleans": True,
            "parse_numbers": True,
        }

    report: Dict[str, List[Dict[str, Any]]] = {
        "enhancements_applied": [],
        "type_conversions": [],
        "cleaned_values": [],
        "security_warnings": [],
    }

    # SECURITY: Track visited objects and depth
    visited: set = set() if config.enable_circular_reference_detection else set()

    enhanced = _enhance_recursive(data, enhancement_rules, report, "", visited, 0, config)
    return enhanced, report


def _enhance_recursive(
    obj: Any,
    rules: Dict[str, Any],
    report: Dict[str, List[Dict[str, Any]]],
    path: str,
    visited: set,
    depth: int,
    config: UtilityConfig,
) -> Any:
    """Recursively enhance data structures with security checks."""
    # SECURITY: Depth limit enforcement
    if depth > config.max_depth:
        raise UtilitySecurityError(f"Maximum enhancement depth ({config.max_depth}) exceeded at path '{path}'")

    # SECURITY: Object size checks
    if hasattr(obj, "__len__") and len(obj) > config.max_object_size:
        raise UtilitySecurityError(f"Object size exceeds maximum ({config.max_object_size}) at path '{path}'")

    # SECURITY: Circular reference detection
    if config.enable_circular_reference_detection:
        obj_id = id(obj)
        if obj_id in visited:
            raise UtilitySecurityError(f"Circular reference detected at path '{path}'")

        if isinstance(obj, (dict, list, tuple)):
            visited.add(obj_id)

    try:
        if isinstance(obj, str):
            return _enhance_string(obj, rules, report, path)
        elif isinstance(obj, dict):
            return {
                key: _enhance_recursive(
                    value, rules, report, f"{path}.{key}" if path else str(key), visited, depth + 1, config
                )
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [
                _enhance_recursive(
                    item, rules, report, f"{path}[{i}]" if path else f"[{i}]", visited, depth + 1, config
                )
                for i, item in enumerate(obj)
            ]
        elif isinstance(obj, tuple):
            return tuple(
                _enhance_recursive(
                    item, rules, report, f"{path}[{i}]" if path else f"[{i}]", visited, depth + 1, config
                )
                for i, item in enumerate(obj)
            )
        else:
            return obj
    finally:
        # Clean up: remove from visited set when done
        if config.enable_circular_reference_detection and isinstance(obj, (dict, list, tuple)):
            visited.discard(id(obj))


def _enhance_string(s: str, rules: Dict[str, Any], report: Dict[str, List[Dict[str, Any]]], path: str) -> Any:
    """Enhance a string by parsing it to appropriate types."""
    original = s

    # Handle nulls
    if rules.get("handle_nulls") and s.lower() in ["null", "none", "nil", "", "n/a", "nan"]:
        report["cleaned_values"].append({"path": path, "original": original, "new": None, "reason": "null_value"})
        return None

    # Normalize strings
    if rules.get("normalize_strings"):
        s = s.strip()
        if s != original:
            report["cleaned_values"].append({"path": path, "original": original, "new": s, "reason": "normalized"})

    # Try to parse as number
    if rules.get("parse_numbers"):
        try:
            if "." not in s and "e" not in s.lower():
                num_int = int(s)
                report["type_conversions"].append({"path": path, "from": "string", "to": "int", "value": num_int})
                return num_int
        except ValueError:
            pass

        try:
            num_float = float(s)
            report["type_conversions"].append({"path": path, "from": "string", "to": "float", "value": num_float})
            return num_float
        except ValueError:
            pass

    # Try to parse as date
    if rules.get("parse_dates"):
        parsed_date = _try_parse_date(s)
        if parsed_date:
            report["type_conversions"].append({"path": path, "from": "string", "to": "datetime", "value": parsed_date})
            return parsed_date

    return s


def _try_parse_date(s: str) -> Optional[datetime]:
    """Try to parse a string as a date."""
    common_formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
    ]

    for fmt in common_formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    return None


def normalize_data_structure(
    data: Any, target_structure: Optional[str] = None, config: Optional[UtilityConfig] = None
) -> Any:
    """Normalize data structure to a consistent format with security limits.

    Args:
        data: Data structure to normalize
        target_structure: Target structure type ("flat", "records", etc.)
        config: Optional utility configuration with security limits

    Returns:
        Normalized data structure

    Raises:
        UtilitySecurityError: If security limits are exceeded
    """
    if config is None:
        config = get_default_utility_config()

    # SECURITY: Size check before processing
    if hasattr(data, "__len__") and len(data) > config.max_object_size:
        raise UtilitySecurityError(f"Object size ({len(data)}) exceeds maximum ({config.max_object_size})")

    if target_structure == "flat":
        return _flatten_structure(data, config=config)
    elif target_structure == "records" and isinstance(data, dict):
        return _dict_to_records(data, config)
    else:
        return data


def _flatten_structure(
    data: Any,
    prefix: str = "",
    sep: str = ".",
    config: Optional[UtilityConfig] = None,
    visited: Optional[set] = None,
    depth: int = 0,
) -> Dict[str, Any]:
    """Flatten nested data structure with security checks."""
    if config is None:
        config = get_default_utility_config()

    if visited is None:
        visited = set() if config.enable_circular_reference_detection else set()

    # SECURITY: Depth limit enforcement
    if depth > config.max_depth:
        raise UtilitySecurityError(f"Maximum flattening depth ({config.max_depth}) exceeded")

    # SECURITY: Circular reference detection
    if config.enable_circular_reference_detection and isinstance(data, (dict, list)):
        obj_id = id(data)
        if obj_id in visited:
            return {prefix or "circular_ref": "<CIRCULAR_REFERENCE>"}
        visited.add(obj_id)

    result = {}

    try:
        if isinstance(data, dict):
            # SECURITY: Size check
            if len(data) > config.max_object_size:
                raise UtilitySecurityError(f"Dict size ({len(data)}) exceeds maximum ({config.max_object_size})")

            for key, value in data.items():
                new_key = f"{prefix}{sep}{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    result.update(_flatten_structure(value, new_key, sep, config, visited, depth + 1))
                else:
                    result[new_key] = value
        elif isinstance(data, list):
            # SECURITY: Size check
            if len(data) > config.max_object_size:
                raise UtilitySecurityError(f"List size ({len(data)}) exceeds maximum ({config.max_object_size})")

            for i, item in enumerate(data):
                new_key = f"{prefix}{sep}{i}" if prefix else str(i)
                if isinstance(item, (dict, list)):
                    result.update(_flatten_structure(item, new_key, sep, config, visited, depth + 1))
                else:
                    result[new_key] = item
        else:
            result[prefix] = data
    finally:
        # Clean up: remove from visited set when done
        if config.enable_circular_reference_detection and isinstance(data, (dict, list)):
            visited.discard(id(data))

    return result


def _dict_to_records(data: dict, config: UtilityConfig) -> List[Dict[str, Any]]:
    """Convert dict to records format with security checks."""
    if not data:
        return []

    # SECURITY: Size check
    if len(data) > config.max_object_size:
        raise UtilitySecurityError(f"Dict size ({len(data)}) exceeds maximum ({config.max_object_size})")

    keys = list(data.keys())
    first_value = data[keys[0]]

    if not isinstance(first_value, list):
        return [data]

    # SECURITY: Check list length
    if len(first_value) > config.max_object_size:
        raise UtilitySecurityError(f"List length ({len(first_value)}) exceeds maximum ({config.max_object_size})")

    return [{key: data[key][i] for key in keys} for i in range(len(first_value))]


def standardize_datetime_formats(
    data: Any, target_format: str = "iso", config: Optional[UtilityConfig] = None
) -> Tuple[Any, List[str]]:
    """Standardize datetime formats throughout a data structure with security limits.

    Args:
        data: Data structure to process
        target_format: Target datetime format ("iso", "unix", "unix_ms", or custom format)
        config: Optional utility configuration with security limits

    Returns:
        Tuple of (converted_data, conversion_log)

    Raises:
        UtilitySecurityError: If security limits are exceeded
    """
    if config is None:
        config = get_default_utility_config()

    conversion_log = []
    visited: set = set() if config.enable_circular_reference_detection else set()

    def convert_datetime(dt: datetime, path: str) -> Any:
        conversion_log.append(f"Converted datetime at {path}")

        if target_format == "iso":
            return dt.isoformat()
        elif target_format == "unix":
            return dt.timestamp()
        elif target_format == "unix_ms":
            return int(dt.timestamp() * 1000)
        else:
            return dt.strftime(target_format)

    converted = _convert_datetimes_recursive(data, convert_datetime, "", visited, 0, config)
    return converted, conversion_log


def _convert_datetimes_recursive(
    obj: Any, converter: Callable[[datetime, str], Any], path: str, visited: set, depth: int, config: UtilityConfig
) -> Any:
    """Recursively convert datetime objects in data structures with security checks."""
    # SECURITY: Depth limit enforcement
    if depth > config.max_depth:
        raise UtilitySecurityError(f"Maximum datetime conversion depth ({config.max_depth}) exceeded at path '{path}'")

    # SECURITY: Circular reference detection
    if config.enable_circular_reference_detection and isinstance(obj, (dict, list, tuple)):
        obj_id = id(obj)
        if obj_id in visited:
            return "<CIRCULAR_REFERENCE>"
        visited.add(obj_id)

    try:
        if isinstance(obj, datetime):
            return converter(obj, path)
        elif isinstance(obj, dict):
            # SECURITY: Size check
            if len(obj) > config.max_object_size:
                raise UtilitySecurityError(
                    f"Dict size ({len(obj)}) exceeds maximum ({config.max_object_size}) at path '{path}'"
                )

            return {
                key: _convert_datetimes_recursive(
                    value, converter, f"{path}.{key}" if path else str(key), visited, depth + 1, config
                )
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            # SECURITY: Size check
            if len(obj) > config.max_object_size:
                raise UtilitySecurityError(
                    f"List size ({len(obj)}) exceeds maximum ({config.max_object_size}) at path '{path}'"
                )

            return [
                _convert_datetimes_recursive(
                    item, converter, f"{path}[{i}]" if path else f"[{i}]", visited, depth + 1, config
                )
                for i, item in enumerate(obj)
            ]
        elif isinstance(obj, tuple):
            return tuple(
                _convert_datetimes_recursive(
                    item, converter, f"{path}[{i}]" if path else f"[{i}]", visited, depth + 1, config
                )
                for i, item in enumerate(obj)
            )
        else:
            return obj
    finally:
        # Clean up: remove from visited set when done
        if config.enable_circular_reference_detection and isinstance(obj, (dict, list, tuple)):
            visited.discard(id(obj))


def extract_temporal_features(data: Any, config: Optional[UtilityConfig] = None) -> Dict[str, Any]:
    """Extract temporal features from datetime fields in data with security limits.

    Args:
        data: Data structure to analyze
        config: Optional utility configuration with security limits

    Returns:
        Dictionary with temporal features

    Raises:
        UtilitySecurityError: If security limits are exceeded
    """
    if config is None:
        config = get_default_utility_config()

    features: Dict[str, Any] = {
        "datetime_fields": [],
        "date_ranges": {},
        "timezones": set(),
    }

    visited: set = set() if config.enable_circular_reference_detection else set()

    _extract_temporal_recursive(data, features, "", visited, 0, config)
    features["timezones"] = list(features["timezones"])

    return features


def _extract_temporal_recursive(
    obj: Any, features: Dict[str, Any], path: str, visited: set, depth: int, config: UtilityConfig
) -> None:
    """Recursively extract temporal features with security checks."""
    # SECURITY: Depth limit enforcement
    if depth > config.max_depth:
        raise UtilitySecurityError(f"Maximum temporal extraction depth ({config.max_depth}) exceeded at path '{path}'")

    # SECURITY: Circular reference detection
    if config.enable_circular_reference_detection and isinstance(obj, (dict, list, tuple)):
        obj_id = id(obj)
        if obj_id in visited:
            return  # Skip circular references
        visited.add(obj_id)

    try:
        if isinstance(obj, datetime):
            features["datetime_fields"].append(path)

            if obj.tzinfo:
                features["timezones"].add(str(obj.tzinfo))

            if path not in features["date_ranges"]:
                features["date_ranges"][path] = {"min": obj, "max": obj}
            else:
                features["date_ranges"][path]["min"] = min(features["date_ranges"][path]["min"], obj)
                features["date_ranges"][path]["max"] = max(features["date_ranges"][path]["max"], obj)

        elif isinstance(obj, dict):
            # SECURITY: Size check
            if len(obj) > config.max_object_size:
                raise UtilitySecurityError(
                    f"Dict size ({len(obj)}) exceeds maximum ({config.max_object_size}) at path '{path}'"
                )

            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else str(key)
                _extract_temporal_recursive(value, features, new_path, visited, depth + 1, config)

        elif isinstance(obj, (list, tuple)):
            # SECURITY: Size check
            if len(obj) > config.max_object_size:
                raise UtilitySecurityError(
                    f"List size ({len(obj)}) exceeds maximum ({config.max_object_size}) at path '{path}'"
                )

            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                _extract_temporal_recursive(item, features, new_path, visited, depth + 1, config)
    finally:
        # Clean up: remove from visited set when done
        if config.enable_circular_reference_detection and isinstance(obj, (dict, list, tuple)):
            visited.discard(id(obj))


def get_available_utilities() -> Dict[str, List[str]]:
    """Get list of all available utility functions organized by category."""
    return {
        "data_comparison": ["deep_compare", "find_data_anomalies"],
        "data_enhancement": ["enhance_data_types", "normalize_data_structure"],
        "datetime_utilities": ["standardize_datetime_formats", "extract_temporal_features"],
        "configuration": ["get_default_utility_config", "UtilityConfig"],
        "security": ["UtilitySecurityError"],
    }


# PANDAS/NUMPY SPECIFIC UTILITIES
def enhance_pandas_dataframe(
    df: Any,  # pandas.DataFrame
    enhancement_rules: Optional[Dict[str, Any]] = None,
    config: Optional[UtilityConfig] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Enhanced pandas DataFrame processing with security limits.

    Args:
        df: pandas DataFrame to enhance
        enhancement_rules: Optional enhancement rules
        config: Optional utility configuration with security limits

    Returns:
        Tuple of (enhanced_dataframe, enhancement_report)

    Raises:
        UtilitySecurityError: If security limits are exceeded
        ImportError: If pandas is not available
    """
    try:
        import pandas as pd
    except ImportError as err:
        raise ImportError("pandas is required for DataFrame enhancement") from err

    if config is None:
        config = get_default_utility_config()

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # SECURITY: Size checks
    if len(df) > config.max_object_size:
        raise UtilitySecurityError(f"DataFrame row count ({len(df)}) exceeds maximum ({config.max_object_size})")

    if len(df.columns) > config.max_collection_size:
        raise UtilitySecurityError(
            f"DataFrame column count ({len(df.columns)}) exceeds maximum ({config.max_collection_size})"
        )

    report: Dict[str, Any] = {
        "columns_processed": [],
        "type_conversions": [],
        "memory_usage_before": df.memory_usage(deep=True).sum(),
    }

    if enhancement_rules is None:
        enhancement_rules = {
            "infer_object_types": True,
            "convert_categories": True,
            "parse_dates": True,
            "downcast_integers": True,
            "downcast_floats": True,
        }

    enhanced_df = df.copy()

    # Process each column
    for col in enhanced_df.columns:
        original_dtype = enhanced_df[col].dtype

        if enhancement_rules.get("infer_object_types") and enhanced_df[col].dtype == "object":
            # Try to infer better types for object columns
            try:
                enhanced_df[col] = pd.to_numeric(enhanced_df[col])
                if enhanced_df[col].dtype != original_dtype:
                    report["type_conversions"].append(
                        {"column": col, "from": str(original_dtype), "to": str(enhanced_df[col].dtype)}
                    )
            except (ValueError, TypeError):
                # If conversion fails, keep original column
                pass

        if (
            enhancement_rules.get("convert_categories")
            and enhanced_df[col].dtype == "object"
            and enhanced_df[col].nunique() < len(enhanced_df) * 0.5
        ):
            enhanced_df[col] = enhanced_df[col].astype("category")
            report["type_conversions"].append({"column": col, "from": "object", "to": "category"})

        report["columns_processed"].append(col)

    report["memory_usage_after"] = enhanced_df.memory_usage(deep=True).sum()
    report["memory_saved"] = report["memory_usage_before"] - report["memory_usage_after"]

    return enhanced_df, report


def enhance_numpy_array(
    arr: Any,  # numpy.ndarray
    enhancement_rules: Optional[Dict[str, Any]] = None,
    config: Optional[UtilityConfig] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Enhanced numpy array processing with security limits.

    Args:
        arr: numpy array to enhance
        enhancement_rules: Optional enhancement rules
        config: Optional utility configuration with security limits

    Returns:
        Tuple of (enhanced_array, enhancement_report)

    Raises:
        UtilitySecurityError: If security limits are exceeded
        ImportError: If numpy is not available
    """
    try:
        import numpy as np
    except ImportError as err:
        raise ImportError("numpy is required for array enhancement") from err

    if config is None:
        config = get_default_utility_config()

    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy array")

    # SECURITY: Size checks
    if arr.size > config.max_object_size:
        raise UtilitySecurityError(f"Array size ({arr.size}) exceeds maximum ({config.max_object_size})")

    report: Dict[str, Any] = {
        "original_shape": arr.shape,
        "original_dtype": str(arr.dtype),
        "memory_usage_before": arr.nbytes,
        "optimizations_applied": [],
    }

    if enhancement_rules is None:
        enhancement_rules = {
            "optimize_dtype": True,
            "remove_inf": True,
            "remove_nan": True,
            "normalize_range": False,
        }

    enhanced_arr = arr.copy()

    # Optimize data type
    if enhancement_rules.get("optimize_dtype"):
        if np.issubdtype(enhanced_arr.dtype, np.integer):
            # Try to downcast integers
            for dtype in [np.int8, np.int16, np.int32, np.int64]:
                if np.can_cast(enhanced_arr, dtype):
                    enhanced_arr = enhanced_arr.astype(dtype)
                    report["optimizations_applied"].append(f"downcasted to {dtype}")
                    break
        elif np.issubdtype(enhanced_arr.dtype, np.floating) and np.can_cast(enhanced_arr, np.float32):
            enhanced_arr = enhanced_arr.astype(np.float32)
            report["optimizations_applied"].append("downcasted to float32")

    # Handle infinite values
    if enhancement_rules.get("remove_inf") and np.issubdtype(enhanced_arr.dtype, np.floating):
        inf_mask = np.isinf(enhanced_arr)
        if np.any(inf_mask):
            enhanced_arr[inf_mask] = np.nan
            report["optimizations_applied"].append("converted inf to nan")

    # Handle NaN values
    if enhancement_rules.get("remove_nan") and np.issubdtype(enhanced_arr.dtype, np.floating):
        nan_mask = np.isnan(enhanced_arr)
        if np.any(nan_mask):
            enhanced_arr = enhanced_arr[~nan_mask]
            report["optimizations_applied"].append("removed NaN values")

    # Normalize range
    if enhancement_rules.get("normalize_range") and np.issubdtype(enhanced_arr.dtype, np.number):
        arr_min, arr_max = enhanced_arr.min(), enhanced_arr.max()
        if arr_max > arr_min:  # Avoid division by zero
            enhanced_arr = (enhanced_arr - arr_min) / (arr_max - arr_min)
            report["optimizations_applied"].append("normalized to [0,1] range")

    report["final_shape"] = enhanced_arr.shape
    report["final_dtype"] = str(enhanced_arr.dtype)
    report["memory_usage_after"] = enhanced_arr.nbytes
    report["memory_saved"] = report["memory_usage_before"] - report["memory_usage_after"]

    return enhanced_arr, report
