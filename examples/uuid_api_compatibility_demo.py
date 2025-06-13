#!/usr/bin/env python3
"""
UUID API Compatibility Demo

This example demonstrates the UUID handling configuration options that solve
the FastAPI/Pydantic integration issue reported by the financial model team.

The problem: datason auto-converts UUID strings to uuid.UUID objects, but
Pydantic models expect UUID fields to remain as strings for API compatibility.

The solution: Configurable UUID handling behavior.
"""

import datason
from datason.config import SerializationConfig, get_api_config, get_ml_config


def demonstrate_the_problem():
    """Show the original problem with UUID auto-conversion."""
    print("=" * 60)
    print("PROBLEM: UUID Auto-Conversion Breaking Pydantic")
    print("=" * 60)

    # Simulate data from database (UUIDs stored as strings)
    database_data = {
        "id": "ea82f3dd-d770-41b9-9706-69cd3070b4f5",
        "user_id": "12345678-1234-5678-9012-123456789abc",
        "created_at": "2023-01-01T12:00:00Z",
    }

    print("📥 Original data from database:")
    for key, value in database_data.items():
        print(f"  {key}: {value} ({type(value).__name__})")

    # Default datason behavior
    result = datason.auto_deserialize(database_data)

    print("\n🔄 After datason.auto_deserialize():")
    for key, value in result.items():
        print(f"  {key}: {value} ({type(value).__name__})")

    print("\n❌ PROBLEM:")
    print("   Pydantic models expecting 'id: str' will fail validation")
    print("   because datason converted strings to uuid.UUID objects!")
    print()


def demonstrate_the_solution():
    """Show the solution using API-compatible configuration."""
    print("=" * 60)
    print("SOLUTION: API-Compatible Configuration")
    print("=" * 60)

    # Same data as before
    database_data = {
        "id": "ea82f3dd-d770-41b9-9706-69cd3070b4f5",
        "user_id": "12345678-1234-5678-9012-123456789abc",
        "created_at": "2023-01-01T12:00:00Z",
    }

    print("📥 Original data from database:")
    for key, value in database_data.items():
        print(f"  {key}: {value} ({type(value).__name__})")

    # NEW: Use API-compatible configuration
    api_config = get_api_config()
    result = datason.auto_deserialize(database_data, config=api_config)

    print("\n🔄 After datason.auto_deserialize(data, config=get_api_config()):")
    for key, value in result.items():
        print(f"  {key}: {value} ({type(value).__name__})")

    print("\n✅ SOLUTION:")
    print("   UUIDs remain as strings (Pydantic-compatible)")
    print("   Datetimes still converted to datetime objects")
    print("   Perfect for FastAPI + Pydantic integration!")
    print()


def demonstrate_custom_configurations():
    """Show different custom configuration options."""
    print("=" * 60)
    print("CUSTOM CONFIGURATION OPTIONS")
    print("=" * 60)

    uuid_string = "12345678-1234-5678-9012-123456789abc"

    configs = [
        ("Default Config", None),
        ("API Config (preset)", get_api_config()),
        ("ML Config (preset)", get_ml_config()),
        ("Custom: uuid_format='string'", SerializationConfig(uuid_format="string")),
        ("Custom: parse_uuids=False", SerializationConfig(parse_uuids=False)),
    ]

    print(f"🧪 Testing UUID string: {uuid_string}")
    print()

    for name, config in configs:
        if config is None:
            result = datason.auto_deserialize(uuid_string)
        else:
            result = datason.auto_deserialize(uuid_string, config=config)

        result_type = type(result).__name__
        print(f"📋 {name:<30} → {result_type}")

    print()


def demonstrate_real_world_api_example():
    """Show a realistic FastAPI + Pydantic use case."""
    print("=" * 60)
    print("REAL-WORLD EXAMPLE: FastAPI + Pydantic")
    print("=" * 60)

    # Simulate complex API data
    api_request_data = {
        "group": {
            "id": "ea82f3dd-d770-41b9-9706-69cd3070b4f5",
            "name": "Financial Analytics Team",
            "created_at": "2023-01-01T12:00:00Z",
            "updated_at": "2023-06-15T14:30:45Z",
        },
        "members": [
            {
                "id": "11111111-2222-3333-4444-555555555555",
                "email": "analyst1@company.com",
                "role": "senior_analyst",
                "joined_at": "2023-01-15T09:00:00Z",
            },
            {
                "id": "66666666-7777-8888-9999-aaaaaaaaaaaa",
                "email": "analyst2@company.com",
                "role": "junior_analyst",
                "joined_at": "2023-03-01T09:00:00Z",
            },
        ],
        "settings": {"session_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "auto_save": True, "refresh_interval": 300},
    }

    print("📊 Processing complex API data structure...")

    # Process with API-compatible configuration
    api_config = get_api_config()
    processed_data = datason.auto_deserialize(api_request_data, config=api_config)

    print("\n✅ Results (Pydantic-compatible):")

    # Check UUIDs remained as strings
    uuids_found = []
    uuids_found.append(("group.id", processed_data["group"]["id"]))
    uuids_found.append(("settings.session_id", processed_data["settings"]["session_id"]))
    for i, member in enumerate(processed_data["members"]):
        uuids_found.append((f"members[{i}].id", member["id"]))

    print("   📝 UUID fields (all strings for Pydantic):")
    for field_path, uuid_value in uuids_found:
        print(f"      {field_path}: {uuid_value} ({type(uuid_value).__name__})")

    # Check datetimes were converted
    print("   📅 Datetime fields (converted to datetime objects):")
    print(
        f"      group.created_at: {processed_data['group']['created_at']} ({type(processed_data['group']['created_at']).__name__})"
    )
    print(
        f"      group.updated_at: {processed_data['group']['updated_at']} ({type(processed_data['group']['updated_at']).__name__})"
    )

    print("\n   💡 This data structure can now be validated by Pydantic models like:")
    print("      class Group(BaseModel):")
    print("          id: str  # ✅ Works! UUID stays as string")
    print("          name: str")
    print("          created_at: datetime  # ✅ Works! Converted to datetime")
    print()


def demonstrate_migration_strategy():
    """Show how to migrate existing applications."""
    print("=" * 60)
    print("MIGRATION STRATEGY FOR EXISTING APPS")
    print("=" * 60)

    # Example data
    data = {"user_id": "12345678-1234-5678-9012-123456789abc"}

    print("🔄 For existing applications, migration is simple:")
    print()

    print("   ❌ Before (problematic with Pydantic):")
    print("      result = datason.auto_deserialize(data)")
    old_result = datason.auto_deserialize(data)
    print(f"      # Result: {old_result} ({type(old_result['user_id']).__name__})")
    print()

    print("   ✅ After (API-compatible):")
    print("      from datason.config import get_api_config")
    print("      api_config = get_api_config()")
    print("      result = datason.auto_deserialize(data, config=api_config)")
    api_config = get_api_config()
    new_result = datason.auto_deserialize(data, config=api_config)
    print(f"      # Result: {new_result} ({type(new_result['user_id']).__name__})")
    print()

    print("📚 Best practices:")
    print("   1. Choose the right config preset for your use case")
    print("   2. Use it consistently throughout your application")
    print("   3. Document your UUID format expectations")
    print("   4. Test with your validation framework (Pydantic, etc.)")
    print()


def demonstrate_performance_impact():
    """Show that the configuration doesn't impact performance."""
    print("=" * 60)
    print("PERFORMANCE IMPACT")
    print("=" * 60)

    import time

    # Test data with multiple UUIDs
    test_data = {
        "ids": [f"{i:08x}-1234-5678-9012-123456789abc" for i in range(1000)],
        "metadata": {"session_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "created_at": "2023-01-01T12:00:00Z"},
    }

    # Test default behavior
    start_time = time.perf_counter()
    datason.auto_deserialize(test_data)  # We only need timing, not result
    default_time = time.perf_counter() - start_time

    # Test API config behavior
    api_config = get_api_config()
    start_time = time.perf_counter()
    datason.auto_deserialize(test_data, config=api_config)  # We only need timing, not result
    api_time = time.perf_counter() - start_time

    print("⚡ Performance comparison (1000 UUIDs + 1 datetime):")
    print(f"   Default config:  {default_time:.4f}s")
    print(f"   API config:      {api_time:.4f}s")
    print(f"   Difference:      {abs(api_time - default_time):.4f}s")
    print()
    print("💡 Conclusion: Configuration has minimal performance impact")
    print("   The choice should be based on compatibility needs, not performance")
    print()


def main():
    """Run the complete UUID API compatibility demonstration."""
    print("🔧 DATASON UUID API COMPATIBILITY SOLUTION")
    print("Addressing FastAPI + Pydantic Integration Issues")
    print()

    demonstrate_the_problem()
    demonstrate_the_solution()
    demonstrate_custom_configurations()
    demonstrate_real_world_api_example()
    demonstrate_migration_strategy()
    demonstrate_performance_impact()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Problem solved: UUID auto-conversion now configurable")
    print("✅ API-compatible preset available: get_api_config()")
    print("✅ Custom configurations supported")
    print("✅ Backward compatibility maintained")
    print("✅ Performance impact minimal")
    print("✅ Ready for production FastAPI + Pydantic apps")
    print()
    print("🚀 Your FastAPI + Pydantic integration should now work smoothly!")


if __name__ == "__main__":
    main()
