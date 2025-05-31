#!/usr/bin/env python3
"""Basic usage examples for datason.

This script demonstrates the key features and capabilities of the datason package.
"""

from datetime import datetime, timezone
import json
from typing import Any, Dict
import uuid

import datason as ds


def example_basic_serialization() -> None:
    """Demonstrate basic serialization of common Python types."""
    print("=== Basic Serialization ===")

    # Basic data types
    data = {
        "string": "Hello, World!",
        "integer": 42,
        "float": 3.14159,
        "boolean": True,
        "null": None,
        "list": [1, 2, 3, "four"],
        "nested_dict": {"inner_key": "inner_value", "numbers": [1, 2, 3]},
    }

    serialized = ds.serialize(data)
    print("Original data structure converted to JSON-ready format:")
    print(json.dumps(serialized, indent=2))
    print()


def example_datetime_handling() -> None:
    """Demonstrate datetime and timestamp handling."""
    print("=== DateTime Handling ===")

    data = {
        "created_at": datetime.now(),
        "birth_date": datetime(1990, 5, 15, 14, 30, 0),
        "timestamps": [
            datetime(2023, 1, 1),
            datetime(2023, 6, 15),
            datetime(2023, 12, 31),
        ],
    }

    serialized = ds.serialize(data)
    print("DateTime objects converted to ISO format:")
    print(json.dumps(serialized, indent=2))
    print()


def example_edge_cases() -> None:
    """Demonstrate handling of edge cases and problematic values."""
    print("=== Edge Cases ===")

    data = {
        "nan_value": float("nan"),
        "infinity": float("inf"),
        "negative_infinity": float("-inf"),
        "mixed_list": [1, float("nan"), "text", None, float("inf")],
        "uuid": uuid.uuid4(),
    }

    serialized = ds.serialize(data)
    print("Edge cases handled gracefully:")
    print(json.dumps(serialized, indent=2))
    print()


def example_safe_conversions() -> None:
    """Demonstrate safe conversion utilities."""
    print("=== Safe Conversions ===")

    # Test various inputs that might cause issues
    test_values = [
        "42.5",  # Valid string number
        "invalid",  # Invalid string
        None,  # None value
        float("nan"),  # NaN
        "",  # Empty string
        [1, 2, 3],  # Wrong type
    ]

    print("Safe float conversions:")
    for value in test_values:
        safe_result = ds.safe_float(value, default=0.0)
        print(f"  {value!r:15} -> {safe_result}")

    print("\nSafe int conversions:")
    for value in test_values:
        safe_result = ds.safe_int(value, default=0)
        print(f"  {value!r:15} -> {safe_result}")
    print()


def example_nested_complex_data() -> None:
    """Demonstrate serialization of deeply nested and complex data structures."""
    print("=== Complex Nested Data ===")

    # Simulate a real-world data structure
    user_data = {
        "users": [
            {
                "id": 1,
                "name": "Alice Johnson",
                "email": "alice@example.com",
                "created_at": datetime(2023, 1, 15, 10, 30, 0),
                "profile": {
                    "age": 28,
                    "preferences": {
                        "theme": "dark",
                        "notifications": True,
                        "score": 89.5,
                    },
                    "tags": ("developer", "python", "json"),
                    "metadata": {
                        "last_login": datetime(2023, 12, 1, 15, 45, 0),
                        "session_id": uuid.uuid4(),
                        "stats": {
                            "points": 1250,
                            "level": 5,
                            "accuracy": 0.95,
                            "invalid_metric": float("nan"),
                        },
                    },
                },
            },
            {
                "id": 2,
                "name": "Bob Smith",
                "email": "bob@example.com",
                "created_at": datetime(2023, 3, 20, 14, 15, 0),
                "profile": {
                    "age": 35,
                    "preferences": {
                        "theme": "light",
                        "notifications": False,
                        "score": float("inf"),  # Edge case
                    },
                    "tags": ("manager", "finance"),
                    "metadata": {
                        "last_login": datetime(2023, 11, 28, 9, 20, 0),
                        "session_id": uuid.uuid4(),
                        "stats": {
                            "points": 890,
                            "level": 3,
                            "accuracy": 0.87,
                            "invalid_metric": None,
                        },
                    },
                },
            },
        ],
        "summary": {
            "total_users": 2,
            "created_at": datetime.now(),
            "version": "1.0.0",
            "config": {"max_retries": 3, "timeout": 30.0, "enabled": True},
        },
    }

    serialized = ds.serialize(user_data)
    print("Complex nested structure with various data types:")
    print(json.dumps(serialized, indent=2))
    print()


def example_custom_objects() -> None:
    """Demonstrate serialization of custom objects."""
    print("=== Custom Objects ===")

    class User:
        def __init__(self, name: str, age: int):
            self.name = name
            self.age = age
            self.created_at = datetime.now()
            self.id = uuid.uuid4()

    class Product:
        def __init__(self, name: str, price: float):
            self.name = name
            self.price = price
            self.in_stock = True

        def dict(self) -> Dict[str, Any]:
            """Custom serialization method."""
            return {
                "product_name": self.name,
                "price_usd": self.price,
                "availability": "available" if self.in_stock else "out_of_stock",
            }

    # Create test objects
    user = User("Charlie Brown", 30)
    product = Product("Laptop", 999.99)

    data = {
        "user": user,  # Will use __dict__
        "product": product,  # Will use .dict() method
        "timestamp": datetime.now(),
    }

    serialized = ds.serialize(data)
    print("Custom objects serialized:")
    print(json.dumps(serialized, indent=2))
    print()


def main() -> None:
    """Run all examples."""
    print("datason Examples")
    print("=" * 50)
    print()

    example_basic_serialization()
    example_datetime_handling()
    example_edge_cases()
    example_safe_conversions()
    example_nested_complex_data()
    example_custom_objects()

    print("All examples completed successfully! 🎉")


if __name__ == "__main__":
    main()
