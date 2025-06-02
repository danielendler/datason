"""Data conversion utilities for datason.

This module provides functions for converting and cleaning various data formats,
particularly for handling string representations of complex data types.
"""

import ast
from typing import Any, Dict, List, Union


def convert_string_method_votes(
    transactions: Union[Dict[str, Any], List[Dict[str, Any]], None],
) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
    """Convert string method_votes to lists.

    This function handles conversion of method_votes from strings to lists
    when they are improperly stored in the database.

    Args:
        transactions: Either a single transaction dict or a list of transaction dicts

    Returns:
        Single transaction or list of transactions with properly formatted method_votes
    """
    # Handle None case
    if transactions is None:
        return None

    # Handle single transaction case
    if isinstance(transactions, dict):
        # Process a single transaction
        if "method_votes" in transactions:
            # Convert string to list if needed
            if isinstance(transactions["method_votes"], str):
                if transactions["method_votes"].startswith("[") and transactions["method_votes"].endswith("]"):
                    # This is a string representation of a list, safely parse it to convert to an actual list
                    try:
                        transactions["method_votes"] = ast.literal_eval(transactions["method_votes"])
                    except (SyntaxError, ValueError):
                        # If eval fails, set to empty list as fallback
                        transactions["method_votes"] = []
                else:
                    # Plain string, convert to single-item list
                    transactions["method_votes"] = [transactions["method_votes"]]

            # Handle None or empty values
            if transactions["method_votes"] is None or (
                isinstance(transactions["method_votes"], list) and len(transactions["method_votes"]) == 0
            ):
                transactions["method_votes"] = []

        return transactions

    # Handle list of transactions case
    result = []
    for tx in transactions:
        if tx is not None:
            converted = convert_string_method_votes(tx)
            if isinstance(converted, dict):
                result.append(converted)
    return result
