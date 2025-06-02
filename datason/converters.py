"""Safe conversion utilities for datason.

This module provides robust type conversion functions that handle edge cases
like NaN, None, and invalid values gracefully.
"""

import math
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float, handling NaN, None, and Inf values safely.

    This function is particularly useful when working with pandas DataFrames
    that may contain NaN values or when processing data from external sources
    that may have None values.

    Args:
        value: Value to convert to float
        default: Default value to return if conversion fails or value is NaN/None/Inf

    Returns:
        Float value or default if conversion fails

    Examples:
        >>> safe_float(42.5)
        42.5
        >>> safe_float(None)
        0.0
        >>> safe_float(float('nan'))
        0.0
        >>> safe_float(float('inf'))
        0.0
        >>> safe_float("invalid", 10.0)
        10.0
    """
    if value is None:
        return default
    try:
        float_val = float(value)
        return default if (math.isnan(float_val) or math.isinf(float_val)) else float_val
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Convert value to int, handling NaN and None values safely.

    This function is particularly useful when working with pandas DataFrames
    that may contain NaN values or when processing data from external sources
    that may have None values.

    Args:
        value: Value to convert to int
        default: Default value to return if conversion fails or value is NaN/None

    Returns:
        Integer value or default if conversion fails

    Examples:
        >>> safe_int(42)
        42
        >>> safe_int(42.7)
        42
        >>> safe_int(None)
        0
        >>> safe_int(float('nan'))
        0
        >>> safe_int("invalid", 10)
        10
    """
    if value is None:
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        # Handle string representations of floats
        if isinstance(value, str):
            try:
                float_val = float(value)
                if math.isnan(float_val) or math.isinf(float_val):
                    return default
                return int(float_val)
            except (ValueError, TypeError):
                return default
        return int(value)
    except (ValueError, TypeError):
        return default
