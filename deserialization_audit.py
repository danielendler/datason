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

# New ML frameworks
try:
    import catboost

    HAS_CATBOOST = True
except ImportError:
    catboost = None
    HAS_CATBOOST = False

try:
    import keras

    HAS_KERAS = True
except ImportError:
    keras = None
    HAS_KERAS = False

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    optuna = None
    HAS_OPTUNA = False

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    go = None
    HAS_PLOTLY = False

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False

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

    def test_auto_detect_round_trip(self, category: str, test_name: str, original_data: Any) -> bool:
        """Test round-trip with auto-detection enabled (opt-in aggressive mode)."""
        try:
            # Clear caches to ensure clean state for each test
            from datason.deserializers import _clear_deserialization_caches

            _clear_deserialization_caches()

            # Use auto-detection configuration
            config = SerializationConfig(auto_detect_types=True)

            # Step 1: Serialize with datason
            serialized = datason.serialize(original_data, config=config)

            # Step 2: Convert to JSON and back (real-world scenario)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)

            # Step 3: Deserialize with auto-detection enabled
            reconstructed = deserialize_fast(parsed, config)

            # Step 4: Verify reconstruction
            try:
                success = self._verify_reconstruction(original_data, reconstructed)

                if success:
                    self.log_test(category, f"{test_name}_auto_detect", True)
                else:
                    self.log_test(
                        category,
                        f"{test_name}_auto_detect",
                        False,
                        f"Auto-detect reconstruction mismatch: {type(original_data).__name__} → {type(reconstructed).__name__}",
                    )

                return success
            except Exception as verify_error:
                self.log_test(
                    category, f"{test_name}_auto_detect", False, f"Auto-detect verification error: {verify_error}"
                )
                return False

        except Exception as e:
            self.log_test(category, f"{test_name}_auto_detect", False, f"{type(e).__name__}: {str(e)}")
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

    def test_user_config_round_trip(self, category: str, test_name: str, original_data: Any) -> bool:
        """Test round-trip with user-specified template (should be 100% success)."""
        try:
            # Clear caches to ensure clean state for each test
            from datason.deserializers import _clear_deserialization_caches, deserialize_with_template

            _clear_deserialization_caches()

            # Step 1: Serialize normally (no metadata)
            serialized = datason.serialize(original_data)

            # Step 2: Convert to JSON and back (real-world scenario)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)

            # Step 3: User provides template - tells us exactly what they want
            reconstructed = deserialize_with_template(parsed, original_data)

            # Step 4: Verify perfect reconstruction
            try:
                success = self._verify_exact_reconstruction(original_data, reconstructed)

                if success:
                    self.log_test(category, f"{test_name}_user_config", True)
                else:
                    self.log_test(
                        category,
                        f"{test_name}_user_config",
                        False,
                        f"User config reconstruction failed: {type(original_data).__name__} → {type(reconstructed).__name__}",
                    )

                return success
            except Exception as verify_error:
                self.log_test(
                    category, f"{test_name}_user_config", False, f"User config verification error: {verify_error}"
                )
                return False

        except Exception as e:
            self.log_test(category, f"{test_name}_user_config", False, f"{type(e).__name__}: {str(e)}")
            return False

    def test_heuristics_round_trip(self, category: str, test_name: str, original_data: Any) -> bool:
        """Test round-trip with heuristics only (best effort)."""
        try:
            # Clear caches to ensure clean state for each test
            from datason.deserializers import _clear_deserialization_caches

            _clear_deserialization_caches()

            # Step 1: Serialize normally (no metadata)
            serialized = datason.serialize(original_data)

            # Step 2: Convert to JSON and back (real-world scenario)
            json_str = json.dumps(serialized, default=str)
            parsed = json.loads(json_str)

            # Step 3: Deserialize with basic heuristics only
            reconstructed = deserialize_fast(parsed)

            # Step 4: Verify reconstruction (allow type conversions)
            try:
                success = self._verify_reconstruction(original_data, reconstructed)

                if success:
                    self.log_test(category, f"{test_name}_heuristics", True)
                else:
                    self.log_test(
                        category,
                        f"{test_name}_heuristics",
                        False,
                        f"Heuristics reconstruction mismatch: {type(original_data).__name__} → {type(reconstructed).__name__}",
                    )

                return success
            except Exception as verify_error:
                self.log_test(
                    category, f"{test_name}_heuristics", False, f"Heuristics verification error: {verify_error}"
                )
                return False

        except Exception as e:
            self.log_test(category, f"{test_name}_heuristics", False, f"{type(e).__name__}: {str(e)}")
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

        if isinstance(original, tuple):
            # For basic round-trip (no type hints), tuple → list is acceptable
            if isinstance(reconstructed, list):
                if len(original) != len(reconstructed):
                    return False
                return all(self._verify_reconstruction(o, r) for o, r in zip(original, reconstructed))
            # For metadata round-trip, exact type match required
            if isinstance(reconstructed, tuple):
                if len(original) != len(reconstructed):
                    return False
                return all(self._verify_reconstruction(o, r) for o, r in zip(original, reconstructed))
            return False

        if isinstance(original, list):
            if not isinstance(reconstructed, list) or len(original) != len(reconstructed):
                return False
            return all(self._verify_reconstruction(o, r) for o, r in zip(original, reconstructed))

        if isinstance(original, dict):
            if not isinstance(reconstructed, dict) or set(original.keys()) != set(reconstructed.keys()):
                return False
            return all(self._verify_reconstruction(original[k], reconstructed[k]) for k in original)

        if isinstance(original, set):
            # For basic round-trip (no type hints), set → list is acceptable
            if isinstance(reconstructed, list):
                return set(reconstructed) == original
            # For metadata round-trip, exact type match required
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

        # Handle CatBoost models specially
        if HAS_CATBOOST and hasattr(original, "__module__") and "catboost" in str(type(original)):
            if not (hasattr(reconstructed, "__module__") and "catboost" in str(type(reconstructed))):
                return False
            try:
                # For CatBoost models, check type and basic attributes
                return type(original) is type(reconstructed)
            except Exception:
                return False

        # Handle Keras models specially
        if HAS_KERAS and hasattr(original, "__module__") and "keras" in str(type(original)):
            if not (hasattr(reconstructed, "__module__") and "keras" in str(type(reconstructed))):
                return False
            try:
                # For Keras models, check type
                return type(original) is type(reconstructed)
            except Exception:
                return False

        # Handle Optuna studies specially
        if HAS_OPTUNA and hasattr(original, "__module__") and "optuna" in str(type(original)):
            if not (hasattr(reconstructed, "__module__") and "optuna" in str(type(reconstructed))):
                return False
            try:
                # For Optuna studies, check type
                return type(original) is type(reconstructed)
            except Exception:
                return False

        # Handle Plotly figures specially
        if HAS_PLOTLY and hasattr(original, "__module__") and "plotly" in str(type(original)):
            if not (hasattr(reconstructed, "__module__") and "plotly" in str(type(reconstructed))):
                return False
            try:
                # For Plotly figures, check type
                return type(original) is type(reconstructed)
            except Exception:
                return False

        # Handle Polars DataFrames specially
        if HAS_POLARS and hasattr(original, "__module__") and "polars" in str(type(original)):
            if not (hasattr(reconstructed, "__module__") and "polars" in str(type(reconstructed))):
                return False
            try:
                # For Polars DataFrames, compare values
                if hasattr(original, "equals") and hasattr(reconstructed, "equals"):
                    return original.equals(reconstructed)
                return type(original) is type(reconstructed)
            except Exception:
                return False

        # Handle scikit-learn models specially
        if HAS_SKLEARN and hasattr(original, "get_params") and hasattr(reconstructed, "get_params"):
            try:
                # Check if both are sklearn models of the same type
                if (
                    hasattr(original, "__module__")
                    and original.__module__ is not None
                    and "sklearn" in original.__module__
                    and hasattr(reconstructed, "__module__")
                    and reconstructed.__module__ is not None
                    and "sklearn" in reconstructed.__module__
                ):
                    # For scikit-learn models, compare type and parameters
                    if type(original) is not type(reconstructed):
                        return False

                    # Compare parameters
                    original_params = original.get_params()
                    reconstructed_params = reconstructed.get_params()

                    # Handle special parameter comparison for sklearn
                    return self._compare_sklearn_params(original_params, reconstructed_params)
            except Exception:
                return False

        # For complex types, basic equality check
        try:
            return original == reconstructed
        except Exception:
            # If equality fails, check string representation as fallback
            return str(original) == str(reconstructed)

    def _verify_exact_reconstruction(self, original: Any, reconstructed: Any) -> bool:
        """Exact verification - types AND values must match perfectly, with ML/Data science exceptions."""
        if type(original) is not type(reconstructed):
            return False

        # For exact reconstruction, use strict equality checks
        if isinstance(original, set):
            return isinstance(reconstructed, set) and original == reconstructed

        if isinstance(original, tuple):
            return isinstance(reconstructed, tuple) and original == reconstructed

        # Special handling for sklearn models - they don't implement __eq__ properly
        if HAS_SKLEARN and hasattr(original, "get_params") and hasattr(reconstructed, "get_params"):
            try:
                # Check if both are sklearn models of the same type
                if (
                    hasattr(original, "__module__")
                    and "sklearn" in original.__module__
                    and hasattr(reconstructed, "__module__")
                    and "sklearn" in reconstructed.__module__
                ):
                    # Compare types and parameters
                    return type(original) is type(reconstructed) and original.get_params() == reconstructed.get_params()
            except (AttributeError, ImportError, ValueError):
                # If sklearn comparison fails, return False
                pass  # nosec B110 - intentional fallback for model comparison

        # Special handling for pandas with dtype tolerance for metadata round-trips
        if HAS_PANDAS:
            if isinstance(original, pd.DataFrame) and isinstance(reconstructed, pd.DataFrame):
                try:
                    # For metadata tests, allow some dtype coercion but check structure
                    if original.shape == reconstructed.shape and list(original.columns) == list(reconstructed.columns):
                        # Check if values are semantically equivalent (allowing for dtype coercion)
                        return original.values.tolist() == reconstructed.values.tolist()
                except (AttributeError, ValueError, TypeError):
                    # If pandas comparison fails, return False
                    pass  # nosec B110 - intentional fallback for pandas comparison
            elif isinstance(original, pd.Series) and isinstance(reconstructed, pd.Series):
                try:
                    # For series, check values and name
                    return (
                        original.values.tolist() == reconstructed.values.tolist()
                        and original.name == reconstructed.name
                    )
                except (AttributeError, ValueError, TypeError):
                    # If pandas series comparison fails, return False
                    pass  # nosec B110 - intentional fallback for pandas series comparison

        return self._verify_reconstruction(original, reconstructed)

    def _compare_sklearn_params(self, original_params: dict, reconstructed_params: dict) -> bool:
        """Compare scikit-learn model parameters with tolerance for common differences."""
        if set(original_params.keys()) != set(reconstructed_params.keys()):
            return False

        for key, original_value in original_params.items():
            reconstructed_value = reconstructed_params[key]

            # Handle special sklearn parameter cases
            if key == "random_state":
                # Random state should match exactly
                if original_value != reconstructed_value:
                    return False
            elif isinstance(original_value, (int, float, str, bool, type(None))):
                # Simple types should match exactly
                if original_value != reconstructed_value:
                    return False
            elif hasattr(original_value, "__module__") and "numpy" in str(original_value.__module__):
                # Numpy arrays in parameters - use numpy comparison
                try:
                    import numpy as np

                    if not np.array_equal(original_value, reconstructed_value):
                        return False
                except Exception:
                    if original_value != reconstructed_value:
                        return False
            else:
                # For other complex objects, use string comparison as fallback
                if str(original_value) != str(reconstructed_value):
                    return False

        return True

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
            # Test 1: User config (should be 100%)
            self.test_user_config_round_trip("complex_types", test_name, data)

            # Test 2: Automatic hints (should be 80-90%)
            self.test_metadata_round_trip("complex_types", test_name, data)

            # Test 3: Heuristics only (best effort)
            self.test_heuristics_round_trip("complex_types", test_name, data)

            # Test 4: Auto-detection enabled (aggressive)
            self.test_auto_detect_round_trip("complex_types", test_name, data)

    def test_pandas_types(self):
        """Test pandas types in three scenarios: default, auto-detect, and metadata."""
        if not HAS_PANDAS:
            print("\n⚠️  Skipping pandas tests - pandas not available")
            return

        print("\n🔍 Testing Pandas Types...")

        test_cases = [
            (
                "dataframe_simple",
                pd.DataFrame({"int_col": [1, 2, 3], "float_col": [1.1, 2.2, 3.3], "str_col": ["a", "b", "c"]}),
            ),
            (
                "dataframe_typed",
                pd.DataFrame(
                    {
                        "int32_col": pd.array([1, 2, 3], dtype="int32"),
                        "category_col": pd.Categorical(["A", "B", "A"]),
                        "datetime_col": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
                    }
                ),
            ),
            ("series", pd.Series([1, 2, 3, 4], name="test_series")),
            ("series_categorical", pd.Series(pd.Categorical(["A", "B", "A", "C"]))),
        ]

        for test_name, data in test_cases:
            # Test 1: User config (should be 100%)
            self.test_user_config_round_trip("pandas_types", test_name, data)

            # Test 2: Automatic hints (should be 80-90%)
            self.test_metadata_round_trip("pandas_types", test_name, data)

            # Test 3: Heuristics only (best effort)
            self.test_heuristics_round_trip("pandas_types", test_name, data)

            # Test 4: Auto-detection enabled (aggressive)
            self.test_auto_detect_round_trip("pandas_types", test_name, data)

    def test_numpy_types(self):
        """Test numpy types in three scenarios: default, auto-detect, and metadata."""
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
            # Test 1: User config (should be 100%)
            self.test_user_config_round_trip("numpy_types", test_name, data)

            # Test 2: Automatic hints (should be 80-90%)
            self.test_metadata_round_trip("numpy_types", test_name, data)

            # Test 3: Heuristics only (best effort)
            self.test_heuristics_round_trip("numpy_types", test_name, data)

            # Test 4: Auto-detection enabled (aggressive)
            self.test_auto_detect_round_trip("numpy_types", test_name, data)

    def test_ml_types(self):
        """Test ML library types in three scenarios: default, auto-detect, and metadata."""
        print("\n🔍 Testing ML Types...")

        test_cases = []

        # PyTorch tensors
        if HAS_TORCH:
            test_cases.append(("pytorch_tensor", torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        else:
            print("⚠️  Skipping PyTorch tests - torch not available")

        # Scikit-learn models
        if HAS_SKLEARN:
            model = sklearn.linear_model.LogisticRegression(random_state=42)
            test_cases.append(("sklearn_model_unfitted", model))

            # Fitted model
            if HAS_NUMPY:
                X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
                y = np.array([0, 0, 1, 1])
                model.fit(X, y)
                test_cases.append(("sklearn_model_fitted", model))
        else:
            print("⚠️  Skipping scikit-learn tests - sklearn not available")

        # CatBoost models
        if HAS_CATBOOST:
            cb_model = catboost.CatBoostClassifier(iterations=10, verbose=False, random_state=42)
            test_cases.append(("catboost_model_unfitted", cb_model))

            # Fitted model
            if HAS_NUMPY:
                X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
                y = np.array([0, 0, 1, 1])
                cb_model.fit(X, y)
                test_cases.append(("catboost_model_fitted", cb_model))
        else:
            print("⚠️  Skipping CatBoost tests - catboost not available")

        # Keras models
        if HAS_KERAS:
            keras_model = keras.Sequential(
                [
                    keras.layers.Dense(10, activation="relu", input_shape=(8,)),
                    keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
            test_cases.append(("keras_model", keras_model))
        else:
            print("⚠️  Skipping Keras tests - keras not available")

        # Optuna studies
        if HAS_OPTUNA:
            study = optuna.create_study()
            test_cases.append(("optuna_study_empty", study))

            # Study with trials
            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return x**2

            study.optimize(objective, n_trials=3)
            test_cases.append(("optuna_study_with_trials", study))
        else:
            print("⚠️  Skipping Optuna tests - optuna not available")

        # Plotly figures
        if HAS_PLOTLY:
            fig = go.Figure(data=go.Bar(x=["A", "B", "C"], y=[1, 2, 3]))
            test_cases.append(("plotly_figure", fig))
        else:
            print("⚠️  Skipping Plotly tests - plotly not available")

        # Polars DataFrames
        if HAS_POLARS:
            df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})
            test_cases.append(("polars_dataframe", df))
        else:
            print("⚠️  Skipping Polars tests - polars not available")

        for test_name, data in test_cases:
            # Test 1: User config (should be 100%)
            self.test_user_config_round_trip("ml_types", test_name, data)

            # Test 2: Automatic hints (should be 80-90%)
            self.test_metadata_round_trip("ml_types", test_name, data)

            # Test 3: Heuristics only (best effort)
            self.test_heuristics_round_trip("ml_types", test_name, data)

            # Test 4: Auto-detection enabled (aggressive)
            self.test_auto_detect_round_trip("ml_types", test_name, data)

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

        # Category breakdown by scenario
        for category, tests in self.results.items():
            if category == "summary" or not tests:
                continue

            # Categorize tests by scenario
            user_config_tests = {k: v for k, v in tests.items() if k.endswith("_user_config")}
            auto_hints_tests = {k: v for k, v in tests.items() if k.endswith("_with_metadata")}
            heuristics_tests = {k: v for k, v in tests.items() if k.endswith("_heuristics")}
            auto_detect_tests = {k: v for k, v in tests.items() if k.endswith("_auto_detect")}
            basic_tests = {
                k: v
                for k, v in tests.items()
                if not any(
                    k.endswith(suffix) for suffix in ["_user_config", "_with_metadata", "_heuristics", "_auto_detect"]
                )
            }

            print(f"\n📂 {category.replace('_', ' ').title()}:")

            # Scenario 1: User Config (should be 100%)
            if user_config_tests:
                user_passed = sum(1 for test in user_config_tests.values() if test["success"])
                print(
                    f"   🟢 User Config (100% goal): {user_passed}/{len(user_config_tests)} ({user_passed / len(user_config_tests) * 100:.1f}%)"
                )

            # Scenario 2: Automatic Hints (should be 80-90%)
            if auto_hints_tests:
                hints_passed = sum(1 for test in auto_hints_tests.values() if test["success"])
                print(
                    f"   🔵 Auto Hints (80-90% goal): {hints_passed}/{len(auto_hints_tests)} ({hints_passed / len(auto_hints_tests) * 100:.1f}%)"
                )

            # Scenario 3: Heuristics Only (best effort)
            if heuristics_tests:
                heuristics_passed = sum(1 for test in heuristics_tests.values() if test["success"])
                print(
                    f"   🔶 Heuristics (best effort): {heuristics_passed}/{len(heuristics_tests)} ({heuristics_passed / len(heuristics_tests) * 100:.1f}%)"
                )

            # Scenario 4: Auto-detection (aggressive)
            if auto_detect_tests:
                auto_passed = sum(1 for test in auto_detect_tests.values() if test["success"])
                print(
                    f"   🔷 Auto-detect (aggressive): {auto_passed}/{len(auto_detect_tests)} ({auto_passed / len(auto_detect_tests) * 100:.1f}%)"
                )

            # Basic tests (legacy)
            if basic_tests:
                basic_passed = sum(1 for test in basic_tests.values() if test["success"])
                print(f"   ⚪ Basic: {basic_passed}/{len(basic_tests)} ({basic_passed / len(basic_tests) * 100:.1f}%)")

            # Show failures for user config tests (most critical)
            if user_config_tests:
                user_failures = [k for k, v in user_config_tests.items() if not v["success"]]
                if user_failures:
                    print("   🚨 User Config failures (should be 0!):")
                    for test_name in user_failures[:5]:  # Show first 5
                        print(f"     ❌ {test_name}: {user_config_tests[test_name]['error']}")
                    if len(user_failures) > 5:
                        print(f"     ... and {len(user_failures) - 5} more")

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
    print(
        f"New ML frameworks: catboost={HAS_CATBOOST}, keras={HAS_KERAS}, optuna={HAS_OPTUNA}, plotly={HAS_PLOTLY}, polars={HAS_POLARS}"
    )

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
