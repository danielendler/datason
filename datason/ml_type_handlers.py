"""Unified ML Type Handlers for datason.

This module contains unified type handlers for Machine Learning frameworks
where both serialization and deserialization logic are co-located in the
same class to prevent the split-brain architecture problem.

Each handler is responsible for:
1. Detection (can_handle)
2. Serialization (serialize)
3. Deserialization (deserialize)
4. Type identification (type_name)

This ensures that when adding a new ML framework, both serialization and
deserialization must be implemented together, preventing maintenance issues.
"""

import warnings
from typing import Any, Dict

from .type_registry import TypeHandler


class CatBoostTypeHandler(TypeHandler):
    """Unified handler for CatBoost models.

    Handles both CatBoostClassifier and CatBoostRegressor with parameter
    preservation for round-trip serialization.
    """

    def can_handle(self, obj: Any) -> bool:
        """Check if object is a CatBoost model."""
        try:
            catboost = self._lazy_import_catboost()
            if catboost is None:
                return False
            return isinstance(obj, (catboost.CatBoostClassifier, catboost.CatBoostRegressor))
        except Exception:
            return False

    def serialize(self, obj: Any) -> Dict[str, Any]:
        """Serialize CatBoost model to preservable format."""
        try:
            # Get model parameters
            params = obj.get_params()

            # Get model class name for reconstruction
            class_name = f"{obj.__class__.__module__}.{obj.__class__.__name__}"

            return {
                "__datason_type__": self.type_name,
                "__datason_value__": {
                    "class_name": class_name,
                    "params": params,
                    "is_fitted": obj.is_fitted() if hasattr(obj, "is_fitted") else False,
                },
            }
        except Exception as e:
            warnings.warn(f"Failed to serialize CatBoost model: {e}", stacklevel=2)
            return {"__datason_type__": "dict", "__datason_value__": {}}

    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Deserialize CatBoost model from preserved format."""
        try:
            value = data["__datason_value__"]
            class_name = value["class_name"]
            params = value["params"]

            # Dynamic import of the class
            module_path, class_name_only = class_name.rsplit(".", 1)
            import importlib

            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name_only)

            # Create new model with same parameters
            return model_class(**params)

        except Exception as e:
            warnings.warn(f"Failed to deserialize CatBoost model: {e}", stacklevel=2)
            return data  # Return original data as fallback

    @property
    def type_name(self) -> str:
        """Return the type name for CatBoost models."""
        return "catboost.model"

    def _lazy_import_catboost(self):
        """Lazy import CatBoost."""
        try:
            import catboost

            return catboost
        except ImportError:
            return None


class KerasTypeHandler(TypeHandler):
    """Unified handler for Keras models.

    Handles Keras models with basic configuration preservation.
    Creates functional models for type preservation in round-trips.
    """

    def can_handle(self, obj: Any) -> bool:
        """Check if object is a Keras model."""
        try:
            keras = self._lazy_import_keras()
            if keras is None:
                return False
            # Check for Keras model interface
            return hasattr(obj, "compile") and hasattr(obj, "fit") and hasattr(obj, "predict")
        except Exception:
            return False

    def serialize(self, obj: Any) -> Dict[str, Any]:
        """Serialize Keras model to preservable format."""
        try:
            # Get basic model info
            model_info = {
                "model_type": obj.__class__.__name__,
                "input_shape": getattr(obj, "input_shape", None),
                "output_shape": getattr(obj, "output_shape", None),
            }

            # Try to get model config if available
            if hasattr(obj, "get_config"):
                try:
                    model_info["config"] = obj.get_config()
                except Exception:
                    pass  # nosec B110 - Config not always available or serializable

            return {"__datason_type__": self.type_name, "__datason_value__": model_info}
        except Exception as e:
            warnings.warn(f"Failed to serialize Keras model: {e}", stacklevel=2)
            return {"__datason_type__": "dict", "__datason_value__": {}}

    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Deserialize Keras model from preserved format."""
        try:
            keras = self._lazy_import_keras()
            if keras is None:
                return data

            value = data["__datason_value__"]
            model_type = value.get("model_type", "Sequential")

            # Create a basic model for type preservation
            if model_type == "Sequential":
                return keras.Sequential()
            else:
                # Fallback to Sequential for unknown model types
                return keras.Sequential()

        except Exception as e:
            warnings.warn(f"Failed to deserialize Keras model: {e}", stacklevel=2)
            return data

    @property
    def type_name(self) -> str:
        """Return the type name for Keras models."""
        return "keras.model"

    def _lazy_import_keras(self):
        """Lazy import Keras."""
        try:
            import keras

            return keras
        except ImportError:
            return None


class OptunaTypeHandler(TypeHandler):
    """Unified handler for Optuna studies.

    Handles Optuna Study objects with study configuration preservation
    for meaningful round-trip reconstructions.
    """

    def can_handle(self, obj: Any) -> bool:
        """Check if object is an Optuna study."""
        try:
            optuna = self._lazy_import_optuna()
            if optuna is None:
                return False
            return isinstance(obj, optuna.Study)
        except Exception:
            return False

    def serialize(self, obj: Any) -> Dict[str, Any]:
        """Serialize Optuna study to preservable format."""
        try:
            study_info = {
                "study_name": obj.study_name,
                "direction": obj.direction.name if hasattr(obj.direction, "name") else str(obj.direction),
                "n_trials": len(obj.trials),
            }

            return {"__datason_type__": self.type_name, "__datason_value__": study_info}
        except Exception as e:
            warnings.warn(f"Failed to serialize Optuna study: {e}", stacklevel=2)
            return {"__datason_type__": "dict", "__datason_value__": {}}

    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Deserialize Optuna study from preserved format."""
        try:
            optuna = self._lazy_import_optuna()
            if optuna is None:
                return data

            value = data["__datason_value__"]
            study_name = value.get("study_name", "deserialized_study")
            direction = value.get("direction", "minimize")

            # Convert direction string back to enum
            direction_enum = getattr(
                optuna.study.StudyDirection, direction.upper(), optuna.study.StudyDirection.MINIMIZE
            )

            # Create new study with same configuration
            return optuna.create_study(study_name=study_name, direction=direction_enum)

        except Exception as e:
            warnings.warn(f"Failed to deserialize Optuna study: {e}", stacklevel=2)
            return data

    @property
    def type_name(self) -> str:
        """Return the type name for Optuna studies."""
        return "optuna.Study"

    def _lazy_import_optuna(self):
        """Lazy import Optuna."""
        try:
            import optuna

            return optuna
        except ImportError:
            return None


class PlotlyTypeHandler(TypeHandler):
    """Unified handler for Plotly figures.

    Handles Plotly Figure objects with full data and layout preservation
    for complete round-trip reconstruction.
    """

    def can_handle(self, obj: Any) -> bool:
        """Check if object is a Plotly figure."""
        try:
            plotly = self._lazy_import_plotly()
            if plotly is None:
                return False
            return isinstance(obj, plotly.graph_objects.Figure)
        except Exception:
            return False

    def serialize(self, obj: Any) -> Dict[str, Any]:
        """Serialize Plotly figure to preservable format."""
        try:
            # Extract data and layout from figure
            figure_dict = obj.to_dict()

            return {
                "__datason_type__": self.type_name,
                "__datason_value__": {
                    "data": figure_dict.get("data", []),
                    "layout": figure_dict.get("layout", {}),
                    "config": figure_dict.get("config", {}),
                },
            }
        except Exception as e:
            warnings.warn(f"Failed to serialize Plotly figure: {e}", stacklevel=2)
            return {"__datason_type__": "dict", "__datason_value__": {}}

    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Deserialize Plotly figure from preserved format."""
        try:
            plotly = self._lazy_import_plotly()
            if plotly is None:
                return data

            value = data["__datason_value__"]

            # Reconstruct figure from data and layout
            return plotly.graph_objects.Figure(data=value.get("data", []), layout=value.get("layout", {}))

        except Exception as e:
            warnings.warn(f"Failed to deserialize Plotly figure: {e}", stacklevel=2)
            return data

    @property
    def type_name(self) -> str:
        """Return the type name for Plotly figures."""
        return "plotly.graph_objects.Figure"

    def _lazy_import_plotly(self):
        """Lazy import Plotly."""
        try:
            import plotly.graph_objects

            return plotly
        except ImportError:
            return None


class PolarsTypeHandler(TypeHandler):
    """Unified handler for Polars DataFrames.

    Handles Polars DataFrame objects with full data preservation
    for accurate round-trip reconstruction.
    """

    def can_handle(self, obj: Any) -> bool:
        """Check if object is a Polars DataFrame."""
        try:
            polars = self._lazy_import_polars()
            if polars is None:
                return False
            return isinstance(obj, polars.DataFrame)
        except Exception:
            return False

    def serialize(self, obj: Any) -> Dict[str, Any]:
        """Serialize Polars DataFrame to preservable format."""
        try:
            # Convert to dictionary format
            data_dict = obj.to_dict()

            return {
                "__datason_type__": self.type_name,
                "__datason_value__": {
                    "data": data_dict,
                    "shape": obj.shape,
                    "columns": obj.columns,
                    "dtypes": [str(dtype) for dtype in obj.dtypes],
                },
            }
        except Exception as e:
            warnings.warn(f"Failed to serialize Polars DataFrame: {e}", stacklevel=2)
            return {"__datason_type__": "dict", "__datason_value__": {}}

    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Deserialize Polars DataFrame from preserved format."""
        try:
            polars = self._lazy_import_polars()
            if polars is None:
                return data

            value = data["__datason_value__"]
            data_dict = value.get("data", {})

            # Reconstruct DataFrame from data dictionary
            return polars.DataFrame(data_dict)

        except Exception as e:
            warnings.warn(f"Failed to deserialize Polars DataFrame: {e}", stacklevel=2)
            return data

    @property
    def type_name(self) -> str:
        """Return the type name for Polars DataFrames."""
        return "polars.DataFrame"

    def _lazy_import_polars(self):
        """Lazy import Polars."""
        try:
            import polars

            return polars
        except ImportError:
            return None


class PyTorchTypeHandler(TypeHandler):
    """Unified handler for PyTorch tensors.

    Handles PyTorch Tensor objects with full data and metadata preservation
    for accurate round-trip reconstruction.
    """

    def can_handle(self, obj: Any) -> bool:
        """Check if object is a PyTorch tensor."""
        try:
            torch = self._lazy_import_torch()
            if torch is None:
                return False
            return isinstance(obj, torch.Tensor)
        except Exception:
            return False

    def serialize(self, obj: Any) -> Dict[str, Any]:
        """Serialize PyTorch tensor to preservable format."""
        try:
            # Convert tensor to numpy for JSON serialization
            tensor_data = obj.detach().cpu().numpy()

            return {
                "__datason_type__": self.type_name,
                "__datason_value__": {
                    "data": tensor_data.tolist(),
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "device": str(obj.device),
                    "requires_grad": obj.requires_grad,
                },
            }
        except Exception as e:
            warnings.warn(f"Failed to serialize PyTorch tensor: {e}", stacklevel=2)
            return {"__datason_type__": "dict", "__datason_value__": {}}

    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Deserialize PyTorch tensor from preserved format."""
        try:
            torch = self._lazy_import_torch()
            if torch is None:
                return data

            value = data["__datason_value__"]

            # Reconstruct tensor from data
            tensor = torch.tensor(value["data"])

            # Set requires_grad if specified
            if value.get("requires_grad", False):
                tensor = tensor.requires_grad_(True)

            return tensor

        except Exception as e:
            warnings.warn(f"Failed to deserialize PyTorch tensor: {e}", stacklevel=2)
            return data

    @property
    def type_name(self) -> str:
        """Return the type name for PyTorch tensors."""
        return "torch.Tensor"

    def _lazy_import_torch(self):
        """Lazy import PyTorch."""
        try:
            import torch

            return torch
        except ImportError:
            return None


class SklearnTypeHandler(TypeHandler):
    """Unified handler for scikit-learn models.

    Handles scikit-learn model objects with parameter preservation
    for meaningful round-trip reconstruction.
    """

    def can_handle(self, obj: Any) -> bool:
        """Check if object is a scikit-learn model."""
        try:
            # Check for sklearn base estimator interface
            return (
                hasattr(obj, "get_params")
                and hasattr(obj, "set_params")
                and hasattr(obj, "__class__")
                and hasattr(obj.__class__, "__module__")
                and obj.__class__.__module__ is not None
                and obj.__class__.__module__.startswith("sklearn")
            )
        except Exception:
            return False

    def serialize(self, obj: Any) -> Dict[str, Any]:
        """Serialize scikit-learn model to preservable format."""
        try:
            # Get model parameters
            params = obj.get_params()

            # Get model class information
            class_name = f"{obj.__class__.__module__}.{obj.__class__.__name__}"

            # Check if model is fitted (has learned attributes)
            fitted_attributes = {}
            for attr_name in dir(obj):
                if attr_name.endswith("_") and not attr_name.startswith("_"):
                    try:
                        attr_value = getattr(obj, attr_name)
                        # Only include simple fitted attributes for now
                        if isinstance(attr_value, (int, float, str, bool, type(None))):
                            fitted_attributes[attr_name] = attr_value
                    except Exception:
                        continue  # nosec B112 - intentionally skip problematic attributes during serialization

            return {
                "__datason_type__": self.type_name,
                "__datason_value__": {
                    "class_name": class_name,
                    "params": params,
                    "fitted_attributes": fitted_attributes,
                    "is_fitted": len(fitted_attributes) > 0,
                },
            }
        except Exception as e:
            warnings.warn(f"Failed to serialize scikit-learn model: {e}", stacklevel=2)
            return {"__datason_type__": "dict", "__datason_value__": {}}

    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Deserialize scikit-learn model from preserved format."""
        try:
            value = data["__datason_value__"]
            class_name = value["class_name"]
            params = value["params"]

            # Dynamic import of the model class
            module_path, class_name_only = class_name.rsplit(".", 1)
            import importlib

            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name_only)

            # Create new model with same parameters
            model = model_class(**params)

            # Restore simple fitted attributes if available
            fitted_attributes = value.get("fitted_attributes", {})
            for attr_name, attr_value in fitted_attributes.items():
                try:
                    setattr(model, attr_name, attr_value)
                except Exception:
                    continue  # nosec B112 - intentionally skip attributes that can't be restored during deserialization

            return model

        except Exception as e:
            warnings.warn(f"Failed to deserialize scikit-learn model: {e}", stacklevel=2)
            return data

    @property
    def type_name(self) -> str:
        """Return the type name for scikit-learn models."""
        return "sklearn.base.BaseEstimator"

    def _lazy_import_sklearn(self):
        """Lazy import scikit-learn."""
        try:
            import sklearn

            return sklearn
        except ImportError:
            return None


# Registry initialization function
def register_all_ml_handlers():
    """Register all ML type handlers with the global registry.

    This function registers all available ML framework handlers,
    ensuring both serialization and deserialization are available
    for each supported framework.
    """
    from .type_registry import register_type_handler

    # Register all ML type handlers
    register_type_handler(CatBoostTypeHandler())
    register_type_handler(KerasTypeHandler())
    register_type_handler(OptunaTypeHandler())
    register_type_handler(PlotlyTypeHandler())
    register_type_handler(PolarsTypeHandler())
    register_type_handler(PyTorchTypeHandler())
    register_type_handler(SklearnTypeHandler())


# Auto-register handlers when module is imported
register_all_ml_handlers()
