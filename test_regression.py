#!/usr/bin/env python3
"""Test script to debug security regression."""

from datason import serialize
from datason.core import SecurityError


def test_depth_bomb_with_debug():
    """Test depth bomb attack with debug output."""
    print("Testing depth bomb attack with debug...")

    # Monkey patch the serialize function to add debug output
    import datason.core

    original_serialize_core = datason.core._serialize_core

    def debug_serialize_core(obj, config, _depth, _seen, _type_handler):
        # Debug the depth checking logic
        if config:
            max_depth = config.max_depth
        else:
            from datason.core import MAX_SERIALIZATION_DEPTH

            max_depth = MAX_SERIALIZATION_DEPTH

        if _depth > 45:  # Debug near the limit
            print(
                f"  DEBUG: Depth {_depth}, max_depth={max_depth}, condition=({_depth} > {max_depth}) = {_depth > max_depth}"
            )

        if _depth > 0 and _depth % 10 == 0:  # Print every 10 levels
            print(f"  DEBUG: Processing at depth {_depth}, obj type: {type(obj).__name__}")
        if _depth > 55:  # Print details near the limit
            print(f"  DEBUG: Depth {_depth}, obj: {obj}")
        return original_serialize_core(obj, config, _depth, _seen, _type_handler)

    datason.core._serialize_core = debug_serialize_core

    try:
        # Create deeply nested structure (more than MAX_SERIALIZATION_DEPTH=50)
        data = {}
        current = data
        depth_created = 0
        for i in range(60):
            current["nest"] = {}
            current = current["nest"]
            depth_created += 1
        current["value"] = "payload"

        print(f"Created nested structure with depth: {depth_created}")

        result = serialize(data)
        print("❌ ERROR: Should have raised SecurityError but got result")
        print(f"Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"Result has {len(result)} keys")
            # Let's trace the depth of the result
            current_result = result
            result_depth = 0
            while "nest" in current_result and isinstance(current_result["nest"], dict):
                current_result = current_result["nest"]
                result_depth += 1
                if result_depth > 70:  # Safety break
                    break
            print(f"Result depth: {result_depth}")
        return False
    except SecurityError as e:
        print(f"✅ SUCCESS: SecurityError raised as expected: {e}")
        return True
    except Exception as e:
        print(f"❌ ERROR: Wrong exception type: {type(e).__name__}: {e}")
        return False
    finally:
        # Restore original function
        datason.core._serialize_core = original_serialize_core


def test_simple_depth():
    """Test with a simpler depth to see where the limit is."""
    print("\nTesting simple depth limits...")

    for test_depth in [10, 20, 30, 40, 50, 55, 60]:
        print(f"Testing depth {test_depth}...")
        data = {}
        current = data
        for i in range(test_depth):
            current["nest"] = {}
            current = current["nest"]
        current["value"] = "payload"

        try:
            serialize(data)
            print(f"  Depth {test_depth}: ✅ Serialized successfully")
        except SecurityError as e:
            print(f"  Depth {test_depth}: ❌ SecurityError: {e}")
            break
        except Exception as e:
            print(f"  Depth {test_depth}: ❌ Other error: {type(e).__name__}: {e}")
            break


def test_size_bomb():
    """Test size bomb attack."""
    print("\nTesting size bomb attack...")

    # Create large dictionary (more than MAX_OBJECT_SIZE=10_000_000)
    try:
        data = {f"key_{i}": f"value_{i}" for i in range(10_000_001)}
        serialize(data)
        print("❌ ERROR: Should have raised SecurityError but got result")
        return False
    except SecurityError as e:
        print(f"✅ SUCCESS: SecurityError raised as expected: {e}")
        return True
    except Exception as e:
        print(f"❌ ERROR: Wrong exception type: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("Security Regression Debug Test")
    print("=" * 40)

    test_depth_bomb_with_debug()
    test_simple_depth()

    print("\n" + "=" * 30)
    if test_size_bomb():
        print("✅ All security tests passed")
    else:
        print("❌ Security regression detected!")
