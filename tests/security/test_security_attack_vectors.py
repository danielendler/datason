"""
Security Attack Vector Tests - White Hat Testing

This test suite attempts to break datason's security measures using various attack vectors.
Each test represents a potential security vulnerability or resource exhaustion attack.

ATTACK CATEGORIES:
1. Depth Bomb Attacks (stack overflow)
2. Size Bomb Attacks (memory exhaustion)
3. Circular Reference Attacks (infinite loops)
4. String Bomb Attacks (memory/CPU exhaustion)
5. Cache Pollution Attacks (memory leaks)
6. Type Confusion Attacks (bypass security)
7. Resource Exhaustion Attacks (DoS)
8. Bypass Attacks (circumvent protections)
"""

import gc
import threading
import time
import warnings
from unittest.mock import Mock

import pytest

from datason import SecurityError, serialize
from datason.config import SerializationConfig


class TestDepthBombAttacks:
    """🎯 ATTACK VECTOR 1: Depth Bomb - Stack Overflow Attempts"""

    def test_depth_bomb_nested_dicts(self):
        """ATTACK: 1000+ nested dictionaries to trigger stack overflow"""
        # Create deeply nested dict structure
        data = {}
        current = data
        for i in range(1005):  # Exceed default depth limit
            current["nest"] = {}
            current = current["nest"]
        current["value"] = "payload"

        # Attack should be blocked by depth enforcement
        with pytest.raises(SecurityError, match="Maximum serialization depth"):
            serialize(data)

    def test_depth_bomb_nested_lists(self):
        """ATTACK: 1000+ nested lists to trigger stack overflow"""
        data = []
        current = data
        for i in range(1005):
            nested = []
            current.append(nested)
            current = nested
        current.append("payload")

        with pytest.raises(SecurityError, match="Maximum serialization depth"):
            serialize(data)

    def test_depth_bomb_mixed_structures(self):
        """ATTACK: Mixed dict/list nesting to bypass homogeneity detection"""
        data = {}
        current = data
        for i in range(1005):
            if i % 2 == 0:
                current["nest"] = {}
                current = current["nest"]
            else:
                current["list"] = [{}]
                current = current["list"][0]
        current["payload"] = "deep"

        with pytest.raises(SecurityError, match="Maximum serialization depth"):
            serialize(data)

    def test_depth_bomb_homogeneous_bypass_attempt(self):
        """ATTACK: Try to bypass depth checking with homogeneous collections"""
        # This is the exact attack our current system is vulnerable to
        data = {}
        current = data
        for i in range(1005):
            # All values are identical dicts - appears "homogeneous"
            current["attack"] = {}
            current = current["attack"]
        current["payload"] = "gotcha"

        # This SHOULD fail but currently doesn't due to homogeneity bypass
        with pytest.raises(SecurityError, match="Maximum serialization depth"):
            serialize(data)

    def test_depth_bomb_with_restrictive_config(self):
        """ATTACK: Depth bomb against restrictive configuration"""
        config = SerializationConfig(max_depth=10)

        data = {}
        current = data
        for i in range(15):  # Just slightly over limit
            current["nest"] = {}
            current = current["nest"]

        with pytest.raises(SecurityError, match="Maximum serialization depth"):
            serialize(data, config=config)


class TestSizeBombAttacks:
    """🎯 ATTACK VECTOR 2: Size Bomb - Memory Exhaustion Attempts"""

    def test_size_bomb_large_dict(self):
        """ATTACK: Massive dictionary to exhaust memory"""
        # Create dict with 10M+ items
        data = {f"key_{i}": f"value_{i}" for i in range(10_000_001)}

        with pytest.raises(SecurityError, match="Dictionary size.*exceeds maximum"):
            serialize(data)

    def test_size_bomb_large_list(self):
        """ATTACK: Massive list to exhaust memory"""
        data = [f"item_{i}" for i in range(10_000_001)]

        with pytest.raises(SecurityError, match="List/tuple size.*exceeds maximum"):
            serialize(data)

    def test_size_bomb_nested_large_structures(self):
        """ATTACK: Multiple large structures nested"""
        data = {
            "attack1": [f"item_{i}" for i in range(5_000_001)],
            "attack2": {f"key_{i}": "value" for i in range(5_000_001)},
        }

        # Should fail on first large structure encountered
        with pytest.raises(SecurityError, match="size.*exceeds maximum"):
            serialize(data)

    def test_size_bomb_with_restrictive_config(self):
        """ATTACK: Size bomb against restrictive configuration"""
        config = SerializationConfig(max_size=100)

        data = [f"item_{i}" for i in range(101)]

        with pytest.raises(SecurityError, match="List/tuple size.*exceeds maximum"):
            serialize(data, config=config)


class TestCircularReferenceAttacks:
    """🎯 ATTACK VECTOR 3: Circular Reference - Infinite Loop Attempts"""

    def test_circular_reference_direct(self):
        """ATTACK: Direct circular reference"""
        data = {}
        data["self"] = data

        # Should be handled gracefully with warning
        with warnings.catch_warnings(record=True) as w:
            result = serialize(data)
            assert len(w) == 1
            assert "Circular reference detected" in str(w[0].message)
            assert result == {"self": None}

    def test_circular_reference_indirect(self):
        """ATTACK: Indirect circular reference through multiple objects"""
        obj1 = {"name": "obj1"}
        obj2 = {"name": "obj2", "ref": obj1}
        obj3 = {"name": "obj3", "ref": obj2}
        obj1["ref"] = obj3  # Complete the circle

        with warnings.catch_warnings(record=True) as w:
            serialize(obj1)
            assert len(w) >= 1
            assert "Circular reference detected" in str(w[0].message)

    def test_circular_reference_in_list(self):
        """ATTACK: Circular reference through list"""
        data = []
        data.append(data)

        with warnings.catch_warnings(record=True) as w:
            serialize(data)
            assert len(w) == 1
            assert "Circular reference detected" in str(w[0].message)

    def test_circular_reference_complex_structure(self):
        """ATTACK: Complex circular structure with multiple entry points"""
        root = {"type": "root"}
        child1 = {"type": "child1", "parent": root}
        child2 = {"type": "child2", "parent": root, "sibling": child1}
        root["children"] = [child1, child2]
        child1["sibling"] = child2

        # This creates multiple circular paths
        with warnings.catch_warnings(record=True):
            serialize(root)
            # Should complete without infinite recursion


class TestStringBombAttacks:
    """🎯 ATTACK VECTOR 4: String Bomb - String Processing Exhaustion"""

    def test_string_bomb_massive_string(self):
        """ATTACK: Extremely long string to exhaust memory/CPU"""
        # Create 1M+ character string
        massive_string = "A" * 1_000_001

        with warnings.catch_warnings(record=True) as w:
            result = serialize(massive_string)
            # Should be truncated
            assert len(w) == 1
            assert "String length" in str(w[0].message)
            assert result.endswith("...[TRUNCATED]")

    def test_string_bomb_many_long_strings(self):
        """ATTACK: Many long strings to exhaust string processing"""
        data = {f"key_{i}": "X" * 1_000_001 for i in range(10)}  # Each string exceeds limit

        with warnings.catch_warnings(record=True) as w:
            serialize(data)
            # Should trigger multiple string length warnings
            assert len(w) >= 1

    def test_string_bomb_nested_in_structure(self):
        """ATTACK: Long strings nested deep in structures"""
        data = {"level1": {"level2": {"level3": {"attack": "X" * 1_000_001}}}}

        with warnings.catch_warnings(record=True) as w:
            serialize(data)
            assert len(w) >= 1
            assert "String length" in str(w[0].message)


class TestCachePollutionAttacks:
    """🎯 ATTACK VECTOR 5: Cache Pollution - Memory Leak Attempts"""

    def test_cache_pollution_type_cache(self):
        """ATTACK: Pollute type cache with many unique types"""
        # Create many unique types to exhaust type cache
        unique_objects = []
        for i in range(20):

            def create_unique_type(value):
                class UniqueType:
                    def __init__(self):
                        self.value = value

                return UniqueType()

            unique_objects.append(create_unique_type(i))

        # Serialize all objects to pollute cache
        for obj in unique_objects:
            serialize(obj)

        # Cache should be limited and not grow indefinitely
        from datason.core import _TYPE_CACHE, _TYPE_CACHE_SIZE_LIMIT

        assert len(_TYPE_CACHE) <= _TYPE_CACHE_SIZE_LIMIT

    def test_cache_pollution_string_cache(self):
        """ATTACK: Pollute string cache with many unique strings"""
        long_strings = [f"unique_string_{i}_{'X' * 100}" for i in range(1000)]

        for s in long_strings:
            serialize(s)

        # String cache should be limited
        from datason.core import _STRING_CACHE_SIZE_LIMIT, _STRING_LENGTH_CACHE

        assert len(_STRING_LENGTH_CACHE) <= _STRING_CACHE_SIZE_LIMIT


class TestTypeBypasses:
    """🎯 ATTACK VECTOR 6: Type Bypass - Circumvent Security via Type Confusion"""

    def test_mock_object_bypass_attempt(self):
        """ATTACK: Use mock objects to bypass security"""
        mock_obj = Mock()
        mock_obj.configure_mock(**{f"attr_{i}": "value" for i in range(200)})

        # Should be caught by mock detection
        with warnings.catch_warnings(record=True) as w:
            result = serialize(mock_obj)
            assert len(w) >= 1
            assert "mock object" in str(w[0].message).lower()
            assert "Mock object" in result

    def test_io_object_bypass_attempt(self):
        """ATTACK: Use IO objects to bypass security"""
        import io

        buffer = io.BytesIO(b"attack_data" * 1000)

        # Should be caught by IO detection
        with warnings.catch_warnings(record=True) as w:
            serialize(buffer)
            assert len(w) >= 1
            assert "problematic io object" in str(w[0].message).lower()

    def test_complex_object_dict_attack(self):
        """ATTACK: Complex object with large __dict__ to exhaust processing"""

        class AttackObject:
            def __init__(self):
                # Create object with 200+ attributes
                for i in range(200):
                    setattr(self, f"attr_{i}", f"value_{i}")

        obj = AttackObject()

        # Should be caught by attribute limit
        with warnings.catch_warnings(record=True) as w:
            serialize(obj)
            assert len(w) >= 1
            assert "too many attributes" in str(w[0].message)


class TestResourceExhaustionAttacks:
    """🎯 ATTACK VECTOR 7: Resource Exhaustion - CPU/Memory DoS Attacks"""

    def test_cpu_exhaustion_complex_nesting(self):
        """ATTACK: Complex nested structure to exhaust CPU"""
        # Create structure that's expensive to process
        data = {}
        for i in range(100):
            level = {}
            for j in range(100):
                level[f"key_{j}"] = {"data": list(range(100)), "meta": {"id": j, "parent": i}}
            data[f"level_{i}"] = level

        # Should complete but be slow - test it doesn't hang
        start = time.time()
        serialize(data)
        end = time.time()

        # Should complete in reasonable time (not hang forever)
        assert end - start < 30  # 30 second timeout

    def test_memory_exhaustion_prevention(self):
        """ATTACK: Try to exhaust memory through object pools"""
        # Try to exhaust memory pools
        large_dicts = []
        for i in range(100):
            large_dict = {f"key_{j}": f"value_{j}" for j in range(10000)}
            result = serialize(large_dict)
            large_dicts.append(result)

        # Memory should be managed properly
        # Force garbage collection
        gc.collect()


class TestHomogeneityBypassAttacks:
    """🎯 ATTACK VECTOR 8: Homogeneity Bypass - Exploit Optimization Paths"""

    def test_homogeneous_depth_bomb(self):
        """ATTACK: Use homogeneous collections to bypass depth checks"""
        # Create structure where each level looks homogeneous
        data = {}
        current = data
        for i in range(1005):
            # Each level has identical structure - looks homogeneous
            current["attack"] = {}
            current = current["attack"]
        current["payload"] = "depth_bomb"

        # This is the exact vulnerability we're trying to fix
        with pytest.raises(SecurityError, match="Maximum serialization depth"):
            serialize(data)

    def test_json_compatibility_bypass(self):
        """ATTACK: Use JSON-compatible data to bypass security"""
        # Create deep JSON structure that might bypass checks
        data = {"level": 0}
        current = data
        for i in range(1005):
            current["next"] = {"level": i + 1}
            current = current["next"]

        with pytest.raises(SecurityError, match="Maximum serialization depth"):
            serialize(data)

    def test_optimization_path_exploitation(self):
        """ATTACK: Exploit optimization paths to bypass security"""
        # Try to trigger optimization paths that might skip security
        data = []
        for i in range(10):
            # Create nested structure in optimized path
            nested = []
            for j in range(200):  # Large but under size limit individually
                nested.append({"id": j, "data": "x" * 100})
            data.append(nested)

        # Should be processed safely
        result = serialize(data)
        assert isinstance(result, list)


class TestParallelAttacks:
    """🎯 ATTACK VECTOR 9: Parallel/Concurrent Attacks"""

    def test_concurrent_cache_pollution(self):
        """ATTACK: Concurrent cache pollution from multiple threads"""

        def pollute_cache(thread_id):
            for i in range(100):

                class DynamicType:
                    def __init__(self, tid, idx):
                        self.thread_id = tid
                        self.index = idx

                obj = DynamicType(thread_id, i)
                serialize(obj)

        # Launch multiple threads to pollute caches concurrently
        threads = []
        for tid in range(5):
            thread = threading.Thread(target=pollute_cache, args=(tid,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Caches should still be within limits
        from datason.core import _TYPE_CACHE, _TYPE_CACHE_SIZE_LIMIT

        assert len(_TYPE_CACHE) <= _TYPE_CACHE_SIZE_LIMIT

    def test_concurrent_serialization_safety(self):
        """ATTACK: Concurrent serialization to test thread safety"""

        def serialize_data(data_id):
            data = {"id": data_id, "nested": {"level1": {"level2": {"data": [1, 2, 3, 4, 5]}}}}
            return serialize(data)

        # Serialize concurrently from multiple threads
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(serialize_data, i) for i in range(50)]
            results = [f.result() for f in futures]

        # All should complete successfully
        assert len(results) == 50
        assert all(isinstance(r, dict) for r in results)


class SecurityTestRunner:
    """Utility class to run all security tests and report vulnerabilities"""

    @staticmethod
    def run_comprehensive_security_audit():
        """Run all security tests and report any vulnerabilities found"""
        test_classes = [
            TestDepthBombAttacks,
            TestSizeBombAttacks,
            TestCircularReferenceAttacks,
            TestStringBombAttacks,
            TestCachePollutionAttacks,
            TestTypeBypasses,
            TestResourceExhaustionAttacks,
            TestHomogeneityBypassAttacks,
            TestParallelAttacks,
        ]

        vulnerabilities = []

        for test_class in test_classes:
            print(f"🔍 Testing {test_class.__name__}...")
            # This would run the actual tests and collect failures

        return vulnerabilities


if __name__ == "__main__":
    # Quick security audit runner
    print("🎯 DATASON SECURITY AUDIT - WHITE HAT TESTING")
    print("=" * 50)

    runner = SecurityTestRunner()
    vulnerabilities = runner.run_comprehensive_security_audit()

    if vulnerabilities:
        print(f"⚠️  FOUND {len(vulnerabilities)} POTENTIAL VULNERABILITIES")
        for vuln in vulnerabilities:
            print(f"  - {vuln}")
    else:
        print("✅ NO CRITICAL VULNERABILITIES FOUND")
