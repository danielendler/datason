"""SerialPy Bidirectional Serialization Example.

This example demonstrates the complete round-trip serialization and
deserialization capabilities of SerialPy, showing how complex Python
objects can be serialized to JSON-compatible formats and then restored
back to their original types.
"""

from datetime import datetime
import json
import uuid

import serialpy as sp


def demonstrate_basic_round_trip() -> None:
    """Demonstrate basic round-trip serialization."""
    print("🔄 Basic Round-Trip Serialization")
    print("=" * 50)

    # Original complex data
    original_data = {
        "user_id": uuid.uuid4(),
        "created_at": datetime.now(),
        "metadata": {
            "last_login": datetime(2023, 12, 25, 14, 30, 0),
            "session_id": uuid.uuid4(),
            "preferences": {"theme": "dark", "notifications": True, "score": 95.5},
        },
        "tags": ["python", "json", "serialization"],
    }

    print("Original data types:")
    print(f"  user_id: {type(original_data['user_id']).__name__}")
    print(f"  created_at: {type(original_data['created_at']).__name__}")
    print(f"  last_login: {type(original_data['metadata']['last_login']).__name__}")
    print(f"  session_id: {type(original_data['metadata']['session_id']).__name__}")
    print()

    # Step 1: Serialize to JSON-compatible format
    serialized = sp.serialize(original_data)
    print("✅ Serialized to JSON-compatible format")
    print(
        f"  user_id: {type(serialized['user_id']).__name__} = {serialized['user_id']}"
    )
    print(f"  created_at: {type(serialized['created_at']).__name__}")
    print()

    # Step 2: Convert to actual JSON string (simulate storage/transmission)
    json_string = json.dumps(serialized, indent=2)
    print("✅ Converted to JSON string (ready for storage/transmission)")
    print(f"JSON size: {len(json_string)} characters")
    print()

    # Step 3: Parse JSON string back to Python objects
    parsed_data = json.loads(json_string)
    print("✅ Parsed JSON string back to Python dict")
    print(
        f"  user_id: {type(parsed_data['user_id']).__name__} = {parsed_data['user_id']}"
    )
    print()

    # Step 4: Deserialize back to original types
    deserialized = sp.deserialize(parsed_data)
    print("✅ Deserialized back to original types")
    print(f"  user_id: {type(deserialized['user_id']).__name__}")
    print(f"  created_at: {type(deserialized['created_at']).__name__}")
    print(f"  last_login: {type(deserialized['metadata']['last_login']).__name__}")
    print(f"  session_id: {type(deserialized['metadata']['session_id']).__name__}")
    print()

    # Verify data integrity
    print("🔍 Data Integrity Check:")
    print(f"  UUIDs match: {original_data['user_id'] == deserialized['user_id']}")
    print(
        f"  Datetime values preserved: {original_data['created_at'].replace(microsecond=0) == deserialized['created_at'].replace(microsecond=0)}"
    )
    print(
        f"  Nested data preserved: {original_data['metadata']['preferences'] == deserialized['metadata']['preferences']}"
    )
    print()


def demonstrate_safe_deserialization() -> None:
    """Demonstrate safe deserialization with error handling."""
    print("🛡️  Safe Deserialization")
    print("=" * 50)

    # Valid JSON
    valid_json = '{"timestamp": "2023-12-25T10:30:00", "id": "12345678-1234-5678-9012-123456789abc"}'
    result = sp.safe_deserialize(valid_json)
    print("✅ Valid JSON deserialized successfully:")
    print(f"  timestamp: {type(result['timestamp']).__name__}")
    print(f"  id: {type(result['id']).__name__}")
    print()

    # Invalid JSON - graceful handling
    invalid_json = '{"invalid": json syntax}'
    result = sp.safe_deserialize(invalid_json)
    print("⚠️  Invalid JSON handled gracefully:")
    print(f"  Result: {result}")
    print(f"  Type: {type(result).__name__}")
    print()


def demonstrate_selective_parsing() -> None:
    """Demonstrate selective parsing options."""
    print("⚙️  Selective Parsing Options")
    print("=" * 50)

    test_data = {
        "timestamp": "2023-12-25T10:30:00",
        "user_id": "12345678-1234-5678-9012-123456789abc",
        "status": "active",
    }

    # Parse with all options enabled (default)
    full_parse = sp.deserialize(test_data)
    print("🔧 Full parsing (default):")
    print(f"  timestamp: {type(full_parse['timestamp']).__name__}")
    print(f"  user_id: {type(full_parse['user_id']).__name__}")
    print()

    # Parse with datetime disabled
    no_dates = sp.deserialize(test_data, parse_dates=False)
    print("🔧 Datetime parsing disabled:")
    print(f"  timestamp: {type(no_dates['timestamp']).__name__}")
    print(f"  user_id: {type(no_dates['user_id']).__name__}")
    print()

    # Parse with UUID disabled
    no_uuids = sp.deserialize(test_data, parse_uuids=False)
    print("🔧 UUID parsing disabled:")
    print(f"  timestamp: {type(no_uuids['timestamp']).__name__}")
    print(f"  user_id: {type(no_uuids['user_id']).__name__}")
    print()


def demonstrate_complex_data_structures() -> None:
    """Demonstrate with complex nested data structures."""
    print("🏗️  Complex Data Structures")
    print("=" * 50)

    # Simulate a complex application data structure
    app_data = {
        "application": {
            "name": "MyApp",
            "version": "2.1.0",
            "deployed_at": datetime.now(),
            "instance_id": uuid.uuid4(),
        },
        "users": [
            {
                "id": uuid.uuid4(),
                "username": "alice",
                "created_at": datetime(2023, 1, 15, 9, 30, 0),
                "last_seen": datetime(2023, 12, 20, 16, 45, 0),
                "session": {
                    "id": uuid.uuid4(),
                    "started_at": datetime(2023, 12, 20, 16, 0, 0),
                    "expires_at": datetime(2023, 12, 21, 16, 0, 0),
                },
            },
            {
                "id": uuid.uuid4(),
                "username": "bob",
                "created_at": datetime(2023, 3, 10, 11, 0, 0),
                "last_seen": datetime(2023, 12, 19, 14, 20, 0),
                "session": {
                    "id": uuid.uuid4(),
                    "started_at": datetime(2023, 12, 19, 14, 0, 0),
                    "expires_at": datetime(2023, 12, 20, 14, 0, 0),
                },
            },
        ],
        "metrics": {
            "total_users": 2,
            "active_sessions": 2,
            "last_updated": datetime.now(),
        },
    }

    print(f"Original structure: {len(app_data['users'])} users with nested sessions")

    # Full round trip
    serialized = sp.serialize(app_data)
    json_str = json.dumps(serialized, indent=2)
    parsed = json.loads(json_str)
    deserialized = sp.deserialize(parsed)

    print("✅ Round-trip completed successfully")
    print(
        f"  Application ID type: {type(deserialized['application']['instance_id']).__name__}"
    )
    print(
        f"  Deployment time type: {type(deserialized['application']['deployed_at']).__name__}"
    )
    print(f"  User count preserved: {len(deserialized['users'])}")

    # Check nested data integrity
    for i, user in enumerate(deserialized["users"]):
        print(f"  User {i+1}:")
        print(f"    ID type: {type(user['id']).__name__}")
        print(f"    Session ID type: {type(user['session']['id']).__name__}")
        print(f"    Timestamps preserved: {isinstance(user['created_at'], datetime)}")
    print()


def demonstrate_performance_awareness() -> None:
    """Demonstrate performance-conscious usage."""
    print("⚡ Performance-Conscious Usage")
    print("=" * 50)

    # Create data that's already JSON-compatible (should be fast)
    simple_data = {
        "name": "test",
        "values": [1, 2, 3, 4, 5],
        "active": True,
        "nested": {"status": "ok", "count": 42},
    }

    print("Simple JSON-compatible data (optimized path):")
    serialized_simple = sp.serialize(simple_data)
    print(f"  Serialized successfully: {serialized_simple is simple_data}")
    print()

    # Create data that needs processing
    complex_data = {
        "timestamp": datetime.now(),
        "id": uuid.uuid4(),
        "values": [1, 2, 3, 4, 5],
    }

    print("Complex data requiring processing:")
    serialized_complex = sp.serialize(complex_data)
    print(f"  Types converted: {isinstance(serialized_complex['timestamp'], str)}")
    print(f"  UUID converted: {isinstance(serialized_complex['id'], str)}")
    print()


def main() -> None:
    """Run all demonstrations."""
    print("🚀 SerialPy Bidirectional Serialization Demo")
    print("=" * 70)
    print()

    demonstrate_basic_round_trip()
    print("\n" + "─" * 70 + "\n")

    demonstrate_safe_deserialization()
    print("\n" + "─" * 70 + "\n")

    demonstrate_selective_parsing()
    print("\n" + "─" * 70 + "\n")

    demonstrate_complex_data_structures()
    print("\n" + "─" * 70 + "\n")

    demonstrate_performance_awareness()

    print("✨ Demo completed! SerialPy provides complete bidirectional")
    print("   serialization with intelligent type restoration.")


if __name__ == "__main__":
    main()
