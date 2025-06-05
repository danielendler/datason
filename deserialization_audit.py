#!/usr/bin/env python3
"""
Comprehensive Deserialization Audit for datason

This script tests the round-trip capabilities of all supported types
to identify gaps between serialization and deserialization.

Usage:
    python deserialization_audit.py
"""

import json
import sys
import uuid
import warnings
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

# Test imports - graceful fallbacks for optional dependencies
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import sklearn.linear_model

    HAS_SKLEARN = True
except ImportError:
    sklearn = None
    HAS_SKLEARN = False

# Import datason
try:
    import datason
    from datason.config import SerializationConfig
    from datason.deserializers import deserialize_fast
except ImportError:
    print("❌ ERROR: datason not found. Please install datason first.")
    sys.exit(1)


class DeserializationAudit:
    """Comprehensive audit of datason's deserialization capabilities."""

    def __init__(self):
        self.results = {
            "basic_types": {},
            "complex_types": {},
            "ml_types": {},
            "pandas_types": {},
            "numpy_types": {},
            "metadata_types": {},
            "summary": {},
        }
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def log_test(self, category: str, test_name: str, success: bool, error: str = None):
        """Log a test result."""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            print(f"✅ {category}/{test_name}")
        else:
            self.failed_tests += 1
            print(f"❌ {category}/{test_name}: {error}")

        # Ensure category exists
        if category not in self.results:
            self.results[category] = {}

        self.results[category][test_name] = {"success": success, "error": error}

    def test_round_trip(
        self, category: str, test_name: str, original_data: Any, config: SerializationConfig = None
    ) -> bool:
        """Test complete round-trip: serialize → JSON → deserialize."""
        try:
            # Clear caches to ensure clean state for each test (helps with test order dependencies)
            from datason.deserializers import _clear_deserialization_caches

            _clear_deserialization_caches()

            # Step 1: Serialize with datason
            serialized = datason.serialize(original_data, config=config) if config else datason.serialize(original_data)

            # Step 2: Convert to JSON and back (real-world scenario)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)

            # Step 3: Deserialize with datason
            reconstructed = deserialize_fast(parsed, config)

            # Step 4: Verify reconstruction
            try:
                success = self._verify_reconstruction(original_data, reconstructed)

                if success:
                    self.log_test(category, test_name, True)
                else:
                    self.log_test(
                        category,
                        test_name,
                        False,
                        f"Reconstruction mismatch: {type(original_data).__name__} → {type(reconstructed).__name__}",
                    )

                return success
            except Exception as verify_error:
                self.log_test(category, test_name, False, f"Verification error: {verify_error}")
                return False

        except Exception as e:
            self.log_test(category, test_name, False, f"{type(e).__name__}: {str(e)}")
            return False

    def test_metadata_round_trip(self, category: str, test_name: str, original_data: Any) -> bool:
        """Test round-trip with type metadata for perfect reconstruction."""
        try:
            # Clear caches to ensure clean state for each test (helps with test order dependencies)
            from datason.deserializers import _clear_deserialization_caches

            _clear_deserialization_caches()

            # Use type metadata configuration
            config = SerializationConfig(include_type_hints=True)

            # Step 1: Serialize with type metadata
            serialized = datason.serialize(original_data, config=config)

            # Step 2: Convert to JSON and back
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)

            # Step 3: Deserialize (should auto-detect metadata)
            reconstructed = deserialize_fast(parsed, config)

            # Step 4: Verify perfect reconstruction
            try:
                success = self._verify_exact_reconstruction(original_data, reconstructed)

                if success:
                    self.log_test(category, f"{test_name}_with_metadata", True)
                else:
                    self.log_test(
                        category,
                        f"{test_name}_with_metadata",
                        False,
                        f"Metadata reconstruction failed: {type(original_data).__name__} → {type(reconstructed).__name__}",
                    )

                return success
            except Exception as verify_error:
                self.log_test(
                    category, f"{test_name}_with_metadata", False, f"Metadata verification error: {verify_error}"
                )
                return False

        except Exception as e:
            self.log_test(category, f"{test_name}_with_metadata", False, f"{type(e).__name__}: {str(e)}")
            return False

    def _verify_reconstruction(self, original: Any, reconstructed: Any) -> bool:
        """Basic verification - types may differ but values should be equivalent."""
        if original is None:
            return reconstructed is None

        if isinstance(original, (str, int, float, bool)):
            return original == reconstructed

        if isinstance(original, datetime):
            return isinstance(reconstructed, datetime) and original.replace(microsecond=0) == reconstructed.replace(
                microsecond=0
            )

        if isinstance(original, uuid.UUID):
            return isinstance(reconstructed, uuid.UUID) and original == reconstructed

        if isinstance(original, (list, tuple)):
            if not isinstance(reconstructed, (list, tuple)) or len(original) != len(reconstructed):
                return False
            return all(self._verify_reconstruction(o, r) for o, r in zip(original, reconstructed))

        if isinstance(original, dict):
            if not isinstance(reconstructed, dict) or set(original.keys()) != set(reconstructed.keys()):
                return False
            return all(self._verify_reconstruction(original[k], reconstructed[k]) for k in original)

        if isinstance(original, set):
            return isinstance(reconstructed, set) and original == reconstructed

        # Handle pandas objects specially
        if HAS_PANDAS:
            if isinstance(original, pd.DataFrame):
                if not isinstance(reconstructed, pd.DataFrame):
                    return False
                try:
                    pd.testing.assert_frame_equal(original, reconstructed)
                    return True
                except AssertionError:
                    return False
            elif isinstance(original, pd.Series):
                if not isinstance(reconstructed, pd.Series):
                    return False
                try:
                    pd.testing.assert_series_equal(original, reconstructed)
                    return True
                except AssertionError:
                    return False

        # Handle numpy arrays specially
        if HAS_NUMPY and isinstance(original, np.ndarray):
            if not isinstance(reconstructed, np.ndarray):
                return False
            try:
                np.testing.assert_array_equal(original, reconstructed)
                return True
            except AssertionError:
                return False

        # Handle PyTorch tensors specially
        if HAS_TORCH and hasattr(original, "__module__") and "torch" in str(type(original)):
            if not (hasattr(reconstructed, "__module__") and "torch" in str(type(reconstructed))):
                return False
            try:
                import torch

                return torch.equal(original, reconstructed)
            except Exception:
                return False

        # For complex types, basic equality check
        try:
            return original == reconstructed
        except Exception:
            # If equality fails, check string representation as fallback
            return str(original) == str(reconstructed)

    def _verify_exact_reconstruction(self, original: Any, reconstructed: Any) -> bool:
        """Exact verification - types AND values must match perfectly."""
        if type(original) is not type(reconstructed):
            return False

        return self._verify_reconstruction(original, reconstructed)

    def test_basic_types(self):
        """Test basic Python types."""
        print("\n🔍 Testing Basic Types...")

        test_cases = [
            ("none", None),
            ("string", "hello world"),
            ("integer", 42),
            ("float", 3.14),
            ("boolean_true", True),
            ("boolean_false", False),
            ("list", [1, 2, 3, "hello"]),
            ("dict", {"a": 1, "b": "hello", "c": None}),
            ("tuple", (1, "hello", 3.14)),
            ("set", {1, 2, 3, "hello"}),
        ]

        for test_name, data in test_cases:
            self.test_round_trip("basic_types", test_name, data)
            self.test_metadata_round_trip("basic_types", test_name, data)

    def test_complex_types(self):
        """Test complex Python types."""
        print("\n🔍 Testing Complex Types...")

        test_cases = [
            ("datetime", datetime(2023, 12, 1, 15, 30, 45, 123456)),
            ("uuid", uuid.UUID("12345678-1234-5678-9012-123456789abc")),
            ("decimal", Decimal("123.456")),
            ("path", Path("/home/test.txt")),  # Use safe path for testing
            ("complex_number", complex(1, 2)),
            (
                "nested_structure",
                {
                    "timestamp": datetime.now(),
                    "id": uuid.uuid4(),
                    "data": [1, 2, {"nested": "value"}],
                    "tags": {"python", "datason"},
                },
            ),
        ]

        for test_name, data in test_cases:
            self.test_round_trip("complex_types", test_name, data)
            self.test_metadata_round_trip("complex_types", test_name, data)

    def test_pandas_types(self):
        """Test pandas types if available."""
        if not HAS_PANDAS:
            print("\n⚠️  Skipping pandas tests - pandas not available")
            return

        print("\n🔍 Testing Pandas Types...")

        # Simple DataFrame
        df = pd.DataFrame({"int_col": [1, 2, 3], "float_col": [1.1, 2.2, 3.3], "str_col": ["a", "b", "c"]})
        self.test_round_trip("pandas_types", "dataframe_simple", df)
        self.test_metadata_round_trip("pandas_types", "dataframe_simple", df)

        # DataFrame with specific dtypes
        df_typed = pd.DataFrame(
            {
                "int32_col": pd.array([1, 2, 3], dtype="int32"),
                "category_col": pd.Categorical(["A", "B", "A"]),
                "datetime_col": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            }
        )
        self.test_round_trip("pandas_types", "dataframe_typed", df_typed)
        self.test_metadata_round_trip("pandas_types", "dataframe_typed", df_typed)

        # Series
        series = pd.Series([1, 2, 3, 4], name="test_series")
        self.test_round_trip("pandas_types", "series", series)
        self.test_metadata_round_trip("pandas_types", "series", series)

        # Series with categorical data
        cat_series = pd.Series(pd.Categorical(["A", "B", "A", "C"]))
        self.test_round_trip("pandas_types", "series_categorical", cat_series)
        self.test_metadata_round_trip("pandas_types", "series_categorical", cat_series)

    def test_numpy_types(self):
        """Test numpy types if available."""
        if not HAS_NUMPY:
            print("\n⚠️  Skipping numpy tests - numpy not available")
            return

        print("\n🔍 Testing NumPy Types...")

        test_cases = [
            ("array_1d", np.array([1, 2, 3, 4, 5])),
            ("array_2d", np.array([[1, 2], [3, 4], [5, 6]])),
            ("array_float32", np.array([1.1, 2.2, 3.3], dtype=np.float32)),
            ("array_int64", np.array([1, 2, 3], dtype=np.int64)),
            ("scalar_int", np.int32(42)),
            ("scalar_float", np.float64(3.14)),
            ("scalar_bool", np.bool_(True)),
        ]

        for test_name, data in test_cases:
            self.test_round_trip("numpy_types", test_name, data)
            self.test_metadata_round_trip("numpy_types", test_name, data)

    def test_ml_types(self):
        """Test ML library types if available."""
        print("\n🔍 Testing ML Types...")

        # PyTorch tensors
        if HAS_TORCH:
            tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            self.test_round_trip("ml_types", "pytorch_tensor", tensor)
            self.test_metadata_round_trip("ml_types", "pytorch_tensor", tensor)
        else:
            print("⚠️  Skipping PyTorch tests - torch not available")

        # Scikit-learn models
        if HAS_SKLEARN:
            model = sklearn.linear_model.LogisticRegression(random_state=42)
            self.test_round_trip("ml_types", "sklearn_model_unfitted", model)
            self.test_metadata_round_trip("ml_types", "sklearn_model_unfitted", model)

            # Fitted model
            if HAS_NUMPY:
                X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
                y = np.array([0, 0, 1, 1])
                model.fit(X, y)
                self.test_round_trip("ml_types", "sklearn_model_fitted", model)
                self.test_metadata_round_trip("ml_types", "sklearn_model_fitted", model)
        else:
            print("⚠️  Skipping scikit-learn tests - sklearn not available")

    def test_configuration_edge_cases(self):
        """Test edge cases with different configurations."""
        print("\n🔍 Testing Configuration Edge Cases...")

        # Test DataFrame orientation configurations
        if HAS_PANDAS:
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

            orientations = ["records", "split", "index", "values", "table"]
            for orient in orientations:
                try:
                    config = SerializationConfig(dataframe_orient=orient)
                    self.test_round_trip("pandas_types", f"dataframe_orient_{orient}", df, config)
                except Exception as e:
                    self.log_test("pandas_types", f"dataframe_orient_{orient}", False, str(e))

    def test_problematic_cases(self):
        """Test known problematic cases."""
        print("\n🔍 Testing Known Problematic Cases...")

        # Large nested structures
        large_nested = {
            "level1": {f"item_{i}": {"timestamp": datetime.now(), "data": list(range(10))} for i in range(10)}
        }
        self.test_round_trip("complex_types", "large_nested", large_nested)

        # Mixed type lists
        mixed_list = [1, "hello", datetime.now(), uuid.uuid4(), {"nested": "dict"}]
        self.test_round_trip("complex_types", "mixed_list", mixed_list)

        # Circular reference prevention
        circular = {"name": "test"}
        circular["self"] = circular
        try:
            _ = datason.serialize(circular)  # We don't need to use the result
            self.log_test("complex_types", "circular_reference", True)
        except Exception as e:
            self.log_test("complex_types", "circular_reference", False, str(e))

    def generate_report(self):
        """Generate a comprehensive audit report."""
        print(f"\n{'=' * 60}")
        print("📊 DESERIALIZATION AUDIT REPORT")
        print(f"{'=' * 60}")

        print("\n📈 Overall Statistics:")
        print(f"   Total Tests: {self.total_tests}")
        print(f"   Passed: {self.passed_tests} ({self.passed_tests / self.total_tests * 100:.1f}%)")
        print(f"   Failed: {self.failed_tests} ({self.failed_tests / self.total_tests * 100:.1f}%)")

        # Category breakdown
        for category, tests in self.results.items():
            if category == "summary" or not tests:
                continue

            category_total = len(tests)
            category_passed = sum(1 for test in tests.values() if test["success"])
            category_failed = category_total - category_passed

            print(f"\n📂 {category.replace('_', ' ').title()}:")
            print(f"   Passed: {category_passed}/{category_total} ({category_passed / category_total * 100:.1f}%)")

            if category_failed > 0:
                print("   Failed tests:")
                for test_name, test_result in tests.items():
                    if not test_result["success"]:
                        print(f"     ❌ {test_name}: {test_result['error']}")

        # Identify critical gaps
        print("\n🚨 Critical Gaps Identified:")

        critical_failures = []
        for category, tests in self.results.items():
            if category == "summary":
                continue
            for test_name, test_result in tests.items():
                if not test_result["success"] and "metadata" not in test_name:
                    critical_failures.append(f"{category}/{test_name}")

        if critical_failures:
            print(f"   Found {len(critical_failures)} critical round-trip failures:")
            for failure in critical_failures[:10]:  # Show first 10
                print(f"     • {failure}")
            if len(critical_failures) > 10:
                print(f"     ... and {len(critical_failures) - 10} more")
        else:
            print("   No critical round-trip failures found! 🎉")

        # Metadata gaps
        metadata_failures = []
        for category, tests in self.results.items():
            if category == "summary":
                continue
            for test_name, test_result in tests.items():
                if not test_result["success"] and "metadata" in test_name:
                    metadata_failures.append(f"{category}/{test_name}")

        if metadata_failures:
            print("\n⚠️  Type Metadata Gaps:")
            print(f"   Found {len(metadata_failures)} metadata round-trip failures:")
            for failure in metadata_failures[:10]:
                print(f"     • {failure}")
            if len(metadata_failures) > 10:
                print(f"     ... and {len(metadata_failures) - 10} more")

        # Recommendations
        print("\n💡 Recommendations:")

        if self.failed_tests == 0:
            print("   🎉 All tests passed! datason has excellent round-trip support.")
        elif critical_failures:
            print("   🔥 HIGH PRIORITY: Fix basic round-trip failures first")
            print("      - These block production ML workflows")
            print("      - Focus on type reconstruction in deserializers.py")

        if metadata_failures:
            print("   📋 MEDIUM PRIORITY: Enhance type metadata support")
            print("      - Improve _deserialize_with_type_metadata() function")
            print("      - Add metadata serialization for missing types")

        print("\n🎯 Next Steps:")
        print("   1. Fix critical round-trip failures (basic functionality)")
        print("   2. Enhance type metadata deserialization")
        print("   3. Add comprehensive round-trip tests to CI/CD")
        print("   4. Update roadmap to prioritize deserialization completeness")


def main():
    """Run the comprehensive deserialization audit."""
    print("🔍 Starting Comprehensive Deserialization Audit...")
    print(f"datason version: {getattr(datason, '__version__', 'unknown')}")
    print(f"Optional dependencies: pandas={HAS_PANDAS}, numpy={HAS_NUMPY}, torch={HAS_TORCH}, sklearn={HAS_SKLEARN}")

    # Suppress warnings during testing
    warnings.filterwarnings("ignore")

    audit = DeserializationAudit()

    # Run all test categories
    audit.test_basic_types()
    audit.test_complex_types()
    audit.test_pandas_types()
    audit.test_numpy_types()
    audit.test_ml_types()
    audit.test_configuration_edge_cases()
    audit.test_problematic_cases()

    # Generate comprehensive report
    audit.generate_report()

    # Exit with error code if tests failed
    if audit.failed_tests > 0:
        print(f"\n❌ Audit completed with {audit.failed_tests} failures")
        sys.exit(1)
    else:
        print("\n✅ Audit completed successfully - all tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
