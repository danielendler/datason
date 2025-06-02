"""Specialized serializers for datason.

This module provides specialized serialization functions for specific use cases
and data structures that require custom handling.
"""

import math
from datetime import datetime
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None


def serialize_detection_details(detection_details: Any) -> Any:
    """Serialize detection_details for JSON storage.

    Converts pandas Timestamps and other non-JSON-serializable objects
    to JSON-compatible formats.

    Args:
        detection_details: Dictionary containing detection details

    Returns:
        Serialized dictionary safe for JSON storage
    """
    if not isinstance(detection_details, dict):
        return detection_details

    def _serialize_value(value: Any) -> Any:
        """Helper function to serialize individual values."""
        if value is None:
            return None
        if isinstance(value, list):
            return [_serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: _serialize_value(v) for k, v in value.items()}
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, float):
            # Handle Python float NaN/Inf
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        # Handle numpy and pandas types if available
        if np is not None:
            if hasattr(np, "ndarray") and isinstance(value, np.ndarray):
                return [_serialize_value(item) for item in value]
            if hasattr(np, "integer") and isinstance(value, np.integer):
                return int(value)
            if hasattr(np, "floating") and isinstance(value, np.floating):
                if np.isnan(value) or np.isinf(value):
                    return None
                return float(value)

        if pd is not None:
            if hasattr(pd, "Series") and isinstance(value, pd.Series):
                return [_serialize_value(item) for item in value]
            if hasattr(pd, "Timestamp") and isinstance(value, pd.Timestamp):
                return value.isoformat()
            # Check pd.isna for scalar values
            try:
                if pd.isna(value):
                    return None
            except (ValueError, TypeError):
                pass

        return value

    serialized = {}
    for method_name, method_details in detection_details.items():
        serialized[method_name] = _serialize_value(method_details)

    return serialized
