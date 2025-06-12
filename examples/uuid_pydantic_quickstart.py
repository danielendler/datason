#!/usr/bin/env python3
"""
UUID + Pydantic Quick Start Guide

This is the fastest way to solve UUID compatibility issues with Pydantic and FastAPI.
Run this script to see the problem and solution in action.

Installation:
    pip install datason pydantic fastapi

Run:
    python uuid_pydantic_quickstart.py
"""

from typing import Any, Dict

# For the demo, we'll handle missing dependencies gracefully
try:
    from pydantic import BaseModel

    HAS_PYDANTIC = True
except ImportError:
    print("⚠️  Pydantic not installed. Install with: pip install pydantic")
    HAS_PYDANTIC = False

    # Mock BaseModel for demo
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


import datason
from datason.config import get_api_config

# Define User model if Pydantic is available
if HAS_PYDANTIC:

    class User(BaseModel):
        user_id: str
        session_id: str
        email: str
        created_at: str  # Using string to avoid datetime parsing issues in demo
        profile: Dict[str, Any]
else:
    User = BaseModel  # Fallback to mock

# =============================================================================
# THE PROBLEM DEMONSTRATION
# =============================================================================


def demonstrate_problem():
    """Show the UUID compatibility problem with Pydantic."""

    print("❌ THE PROBLEM: UUID Auto-Conversion Breaking Pydantic")
    print("=" * 60)

    # Typical data from database/API with UUID strings
    api_data = {
        "user_id": "12345678-1234-5678-9012-123456789abc",
        "session_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "email": "user@example.com",
        "created_at": "2023-01-01T12:00:00Z",
        "profile": {"theme": "dark", "device_id": "ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb"},
    }

    print("📥 Original data from API/database:")
    print(f"   user_id type: {type(api_data['user_id'])} = {api_data['user_id']}")
    print(f"   session_id type: {type(api_data['session_id'])}")
    print(f"   nested device_id type: {type(api_data['profile']['device_id'])}")

    # Default datason behavior - converts UUIDs to objects
    print("\n🔄 After datason.auto_deserialize() with default config:")
    result_default = datason.auto_deserialize(api_data)
    print(f"   user_id type: {type(result_default['user_id'])} = {result_default['user_id']}")
    print(f"   session_id type: {type(result_default['session_id'])}")
    print(f"   nested device_id type: {type(result_default['profile']['device_id'])}")

    # Try to create Pydantic model with UUID object (this should fail)
    print("\n   Attempting to create Pydantic model with UUID object...")
    try:
        # This will fail: ValidationError
        User(**result_default)  # Don't assign to variable since we expect this to fail
        print("   ✅ Success (unexpected)")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        print("   🎯 This is the problem UUID auto-conversion causes!")

    return result_default


# =============================================================================
# THE SOLUTION DEMONSTRATION
# =============================================================================


def demonstrate_solution(api_data: Dict[str, Any]) -> Dict[str, Any]:
    """Demonstrate the solution using API config."""
    print("\n" + "=" * 50)
    print("2. THE SOLUTION: Using get_api_config()")
    print("=" * 50)

    # Get API-optimized config (keeps UUIDs as strings)
    api_config = get_api_config()
    print(f"API Config: {api_config}")

    # Process the same data with API config
    result_api = datason.auto_deserialize(api_data, config=api_config)
    print(f"\nWith API config: {result_api}")
    print(f"UUID type: {type(result_api['user_id'])}")

    # Try to create Pydantic model with string UUID (this should succeed)
    print("\n   Attempting to create Pydantic model with string UUID...")
    try:
        user = User(**result_api)
        print(f"   ✅ Success! Created: {user}")
        print(f"   🎯 UUID preserved as string: {user.user_id} (type: {type(user.user_id)})")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    return result_api


# =============================================================================
# FASTAPI INTEGRATION EXAMPLE
# =============================================================================


def demonstrate_fastapi_integration():
    """Show realistic FastAPI integration patterns."""
    print("\n" + "=" * 50)
    print("3. FASTAPI INTEGRATION EXAMPLE")
    print("=" * 50)

    # Simulate database returning string UUIDs
    database_data = {
        "id": "ea82f3dd-d770-41b9-9706-69cd3070b4f5",
        "name": "Alice Smith",
        "email": "alice@example.com",
        "created_at": "2024-01-15T10:30:00Z",
    }

    print(f"Database data: {database_data}")

    # This is what you'd do in your FastAPI endpoint
    def process_user_for_api(raw_data):
        """Process database data for API response."""
        config = get_api_config()
        return datason.auto_deserialize(raw_data, config=config)

    # Process and return
    processed_data = process_user_for_api(database_data)
    user_response = User(**processed_data)

    print(f"API Response: {user_response}")
    print(f"JSON: {user_response.json()}")


# =============================================================================
# PERFORMANCE DEMONSTRATION
# =============================================================================


def demonstrate_performance():
    """Show performance with different dataset sizes."""

    print("\n⚡ PERFORMANCE DEMONSTRATION")
    print("=" * 60)

    import time

    # Generate test dataset
    test_sizes = [10, 100, 1000]
    api_config = get_api_config()

    for size in test_sizes:
        dataset = [
            {
                "id": f"{i:08x}-1234-5678-9012-123456789abc",
                "user_id": f"{i:08x}-4321-8765-2109-cba987654321",
                "created_at": "2023-01-01T12:00:00Z",
                "data": {"value": i},
            }
            for i in range(size)
        ]

        # Time the processing
        start_time = time.perf_counter()
        result = datason.auto_deserialize(dataset, config=api_config)
        duration = time.perf_counter() - start_time

        print(f"📊 {size:4d} records: {duration:.4f}s ({size / duration:.0f} records/sec)")

        # Verify UUIDs stayed as strings
        sample = result[0] if result else {}
        print(f"     Sample ID type: {type(sample.get('id', 'N/A'))}")


# =============================================================================
# COMMON PATTERNS & BEST PRACTICES
# =============================================================================


def demonstrate_common_patterns():
    """Show common patterns and best practices."""

    print("\n🎯 COMMON PATTERNS & BEST PRACTICES")
    print("=" * 60)

    api_config = get_api_config()

    # Pattern 1: Complex nested data
    print("\n1️⃣ Complex nested data with multiple UUIDs:")
    nested_data = {
        "user_id": "12345678-1234-5678-9012-123456789abc",
        "workspace": {
            "id": "workspace-uuid-here",
            "owner_id": "owner-uuid-here",
            "settings": {"theme_id": "theme-uuid-here"},
        },
        "collaborators": [{"user_id": "collab1-uuid", "role": "editor"}, {"user_id": "collab2-uuid", "role": "viewer"}],
    }

    processed = datason.auto_deserialize(nested_data, config=api_config)
    print("   ✅ All UUIDs remain strings at any nesting level")
    print(f"   ✅ Root: {type(processed['user_id'])}")
    print(f"   ✅ Nested: {type(processed['workspace']['owner_id'])}")
    print(f"   ✅ Deep nested: {type(processed['workspace']['settings']['theme_id'])}")
    print(f"   ✅ In arrays: {type(processed['collaborators'][0]['user_id'])}")

    # Pattern 2: Consistent configuration
    print("\n2️⃣ Consistent configuration pattern:")
    print("""
    # ✅ Do this - consistent config throughout your app
    API_CONFIG = get_api_config()

    # Use everywhere
    result1 = datason.auto_deserialize(data1, config=API_CONFIG)
    result2 = datason.auto_deserialize(data2, config=API_CONFIG)

    # ❌ Don't do this - inconsistent behavior
    result1 = datason.auto_deserialize(data1)  # Default config
    result2 = datason.auto_deserialize(data2, config=get_api_config())  # Different!
    """)

    # Pattern 3: Custom configuration
    print("\n3️⃣ Custom configuration for specific needs:")
    print("""
    from datason.config import SerializationConfig

    # For strict APIs with size limits
    strict_config = SerializationConfig(
        uuid_format="string",     # Keep UUIDs as strings
        parse_uuids=False,        # Don't auto-convert
        max_size=1_000_000,      # 1MB limit
        max_depth=10,            # Prevent deep nesting attacks
        sort_keys=True           # Consistent JSON output
    )
    """)


# =============================================================================
# TROUBLESHOOTING GUIDE
# =============================================================================


def demonstrate_troubleshooting():
    """Show common issues and how to fix them."""

    print("\n🔧 TROUBLESHOOTING GUIDE")
    print("=" * 60)

    print("""
❌ ISSUE: "ValidationError: str type expected"
✅ SOLUTION: Use get_api_config()

❌ ISSUE: Some UUIDs are strings, others are objects
✅ SOLUTION: Use consistent configuration throughout your app

❌ ISSUE: Performance is slow with large datasets
✅ SOLUTION: Process entire dataset at once, not item by item

❌ ISSUE: Nested UUIDs aren't handled consistently
✅ SOLUTION: Datason handles ALL nesting levels automatically

❌ ISSUE: Need different behavior for different endpoints
✅ SOLUTION: Create custom SerializationConfig instances

❌ ISSUE: Integration with existing codebase
✅ SOLUTION: Add config parameter incrementally - backward compatible
""")


# =============================================================================
# MAIN DEMO RUNNER
# =============================================================================


def main():
    """Run the complete demonstration."""
    print("UUID + Pydantic Compatibility Quickstart")
    print("=" * 50)

    # Step 1: Show the problem
    api_data = demonstrate_problem()

    # Step 2: Show the solution
    demonstrate_solution(api_data)  # Don't assign since we don't use it later

    # Step 3: FastAPI integration
    demonstrate_fastapi_integration()


if __name__ == "__main__":
    main()
