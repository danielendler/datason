"""Machine Learning and AI library serializers for datason.

This module provides specialized serialization support for popular ML/AI libraries
including PyTorch, TensorFlow, scikit-learn, JAX, scipy, and others.

ML libraries are imported lazily to improve startup performance.
"""

import base64
import io
import os
import warnings
from typing import Any, Dict, Optional

# Aggressively suppress TensorFlow/Keras logging at module import time
# This must be done before any TensorFlow imports happen anywhere
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # ERROR only
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # Suppress oneDNN messages
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")  # Suppress deterministic warnings
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # Suppress GPU memory messages
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # Suppress CUDA messages entirely
os.environ.setdefault("XLA_FLAGS", "--xla_hlo_profile=false")  # Suppress XLA messages
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")  # Disable XLA devices

# Set up logging suppression before any imports
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)  # TensorFlow uses absl logging

# Suppress warnings at module level
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
warnings.filterwarnings("ignore", message=".*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*oneDNN.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*GPU.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*CUDA.*", category=UserWarning)

# Lazy import cache - libraries are imported only when first used
_LAZY_IMPORTS = {
    "torch": None,
    "tensorflow": None,
    "jax": None,
    "jnp": None,  # JAX numpy alias
    "sklearn": None,
    "BaseEstimator": None,  # sklearn base estimator class
    "scipy": None,
    "PIL_Image": None,  # PIL Image class
    "PIL": None,  # PIL package
    "transformers": None,
    "catboost": None,
    "keras": None,
    "optuna": None,
    "plotly": None,
    "polars": None,
    "pandas": None,  # pandas package
    "numpy": None,  # numpy package
}


def _lazy_import_torch():
    """Lazily import torch."""
    # Check if torch has been patched to None for testing
    import sys

    current_module = sys.modules.get(__name__)
    if current_module and hasattr(current_module, "__dict__") and "torch" in current_module.__dict__:
        patched_value = current_module.__dict__["torch"]
        if patched_value is None:
            return None
        _LAZY_IMPORTS["torch"] = patched_value
        return patched_value

    # Defensive check for missing key
    if "torch" not in _LAZY_IMPORTS:
        _LAZY_IMPORTS["torch"] = None

    if _LAZY_IMPORTS["torch"] is None:
        try:
            import torch

            _LAZY_IMPORTS["torch"] = torch
        except ImportError:
            _LAZY_IMPORTS["torch"] = False
    return _LAZY_IMPORTS["torch"] if _LAZY_IMPORTS["torch"] is not False else None


def _lazy_import_tensorflow():
    """Lazily import tensorflow."""
    # Check if tf has been patched to None for testing
    import sys

    current_module = sys.modules.get(__name__)
    if current_module and hasattr(current_module, "__dict__") and "tf" in current_module.__dict__:
        patched_value = current_module.__dict__["tf"]
        if patched_value is None:
            return None
        _LAZY_IMPORTS["tensorflow"] = patched_value
        return patched_value

    if _LAZY_IMPORTS["tensorflow"] is None:
        try:
            # Suppress TensorFlow logging to reduce test verbosity
            import logging
            import os
            import warnings

            # Suppress TensorFlow C++ logging
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 3 = ERROR only
            os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # Suppress oneDNN messages
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # Suppress CUDA initialization messages

            # Suppress warnings before import
            warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
            warnings.filterwarnings("ignore", message=".*deprecated.*", category=DeprecationWarning)

            # Set logging level before importing
            logging.getLogger("tensorflow").setLevel(logging.ERROR)

            import tensorflow as tf

            # Also suppress Python-level TF logging
            tf.get_logger().setLevel("ERROR")

            # Disable autograph verbosity
            tf.autograph.set_verbosity(0)

            _LAZY_IMPORTS["tensorflow"] = tf
        except ImportError:
            _LAZY_IMPORTS["tensorflow"] = False
    return _LAZY_IMPORTS["tensorflow"] if _LAZY_IMPORTS["tensorflow"] is not False else None


def _lazy_import_jax():
    """Lazily import jax."""
    # Check if jax has been patched to None for testing
    import sys

    current_module = sys.modules.get(__name__)
    if current_module and hasattr(current_module, "__dict__") and "jax" in current_module.__dict__:
        patched_value = current_module.__dict__["jax"]
        if patched_value is None:
            return None, None
        _LAZY_IMPORTS["jax"] = patched_value
        _LAZY_IMPORTS["jnp"] = getattr(patched_value, "numpy", None)
        return (
            _LAZY_IMPORTS["jax"],
            _LAZY_IMPORTS["jnp"],
        )

    if _LAZY_IMPORTS["jax"] is None or _LAZY_IMPORTS["jnp"] is None:
        try:
            import jax
            import jax.numpy as jnp

            _LAZY_IMPORTS["jax"] = jax
            _LAZY_IMPORTS["jnp"] = jnp
        except ImportError:
            _LAZY_IMPORTS["jax"] = False
            _LAZY_IMPORTS["jnp"] = False
    return (
        _LAZY_IMPORTS["jax"] if _LAZY_IMPORTS["jax"] is not False else None,
        _LAZY_IMPORTS["jnp"] if _LAZY_IMPORTS["jnp"] is not False else None,
    )


def _lazy_import_sklearn():
    """Lazily import sklearn."""
    # Check if sklearn or BaseEstimator has been patched to None for testing
    import sys

    current_module = sys.modules.get(__name__)
    if current_module and hasattr(current_module, "__dict__"):
        if "sklearn" in current_module.__dict__:
            patched = current_module.__dict__["sklearn"]
            if patched is None:
                return None, None
            # Don't cache Mock objects - they should be temporary test patches
            if not (hasattr(patched, "_mock_name") or str(type(patched)).startswith("<class 'unittest.mock.")):
                _LAZY_IMPORTS["sklearn"] = patched
        if "BaseEstimator" in current_module.__dict__:
            patched_base = current_module.__dict__["BaseEstimator"]
            if patched_base is None:
                return None, None
            # Don't cache Mock objects - they should be temporary test patches
            if not (
                hasattr(patched_base, "_mock_name") or str(type(patched_base)).startswith("<class 'unittest.mock.")
            ):
                _LAZY_IMPORTS["BaseEstimator"] = patched_base

    if _LAZY_IMPORTS["sklearn"] is None or _LAZY_IMPORTS["BaseEstimator"] is None:
        try:
            import sklearn
            from sklearn.base import BaseEstimator

            _LAZY_IMPORTS["sklearn"] = sklearn
            _LAZY_IMPORTS["BaseEstimator"] = BaseEstimator
        except ImportError:
            _LAZY_IMPORTS["sklearn"] = False
            _LAZY_IMPORTS["BaseEstimator"] = False
    return (
        _LAZY_IMPORTS["sklearn"] if _LAZY_IMPORTS["sklearn"] is not False else None,
        (_LAZY_IMPORTS["BaseEstimator"] if _LAZY_IMPORTS["BaseEstimator"] is not False else None),
    )


def _lazy_import_scipy():
    """Lazily import scipy."""
    # Check if scipy has been patched to None for testing
    import sys

    current_module = sys.modules.get(__name__)
    if current_module and hasattr(current_module, "__dict__") and "scipy" in current_module.__dict__:
        patched_value = current_module.__dict__["scipy"]
        if patched_value is None:
            return None
        _LAZY_IMPORTS["scipy"] = patched_value
        return patched_value

    if _LAZY_IMPORTS["scipy"] is None:
        try:
            import scipy.sparse

            _LAZY_IMPORTS["scipy"] = scipy
        except ImportError:
            _LAZY_IMPORTS["scipy"] = False
    return _LAZY_IMPORTS["scipy"] if _LAZY_IMPORTS["scipy"] is not False else None


def _lazy_import_pil():
    """Lazily import PIL."""
    # Check if Image has been patched to None for testing
    import sys

    current_module = sys.modules.get(__name__)
    if current_module and hasattr(current_module, "__dict__") and "Image" in current_module.__dict__:
        patched_value = current_module.__dict__["Image"]
        if patched_value is None:
            return None
        _LAZY_IMPORTS["PIL_Image"] = patched_value
        return patched_value

    # Defensive check for missing key
    if "PIL_Image" not in _LAZY_IMPORTS:
        _LAZY_IMPORTS["PIL_Image"] = None

    if _LAZY_IMPORTS["PIL_Image"] is None:
        try:
            from PIL import Image

            _LAZY_IMPORTS["PIL_Image"] = Image
        except ImportError:
            _LAZY_IMPORTS["PIL_Image"] = False
    return _LAZY_IMPORTS["PIL_Image"] if _LAZY_IMPORTS["PIL_Image"] is not False else None


def _lazy_import_transformers():
    """Lazily import transformers."""
    # Check if transformers has been patched to None for testing
    import sys

    current_module = sys.modules.get(__name__)
    if current_module and hasattr(current_module, "__dict__") and "transformers" in current_module.__dict__:
        patched_value = current_module.__dict__["transformers"]
        if patched_value is None:
            return None
        _LAZY_IMPORTS["transformers"] = patched_value
        return patched_value

    if _LAZY_IMPORTS["transformers"] is None:
        try:
            import transformers

            _LAZY_IMPORTS["transformers"] = transformers
        except ImportError:
            _LAZY_IMPORTS["transformers"] = False
    return _LAZY_IMPORTS["transformers"] if _LAZY_IMPORTS["transformers"] is not False else None


def _lazy_import_catboost():
    """Lazily import catboost."""
    import sys

    current_module = sys.modules.get(__name__)
    if current_module and hasattr(current_module, "__dict__") and "catboost" in current_module.__dict__:
        patched_value = current_module.__dict__["catboost"]
        if patched_value is None:
            return None
        _LAZY_IMPORTS["catboost"] = patched_value
        return patched_value

    # Defensive check for missing key
    if "catboost" not in _LAZY_IMPORTS:
        _LAZY_IMPORTS["catboost"] = None

    if _LAZY_IMPORTS["catboost"] is None:
        try:
            import catboost

            _LAZY_IMPORTS["catboost"] = catboost
        except ImportError:
            _LAZY_IMPORTS["catboost"] = False
    return _LAZY_IMPORTS["catboost"] if _LAZY_IMPORTS["catboost"] is not False else None


def _lazy_import_keras():
    """Lazily import keras."""
    import sys

    current_module = sys.modules.get(__name__)
    if current_module and hasattr(current_module, "__dict__") and "keras" in current_module.__dict__:
        patched_value = current_module.__dict__["keras"]
        if patched_value is None:
            return None
        _LAZY_IMPORTS["keras"] = patched_value
        return patched_value

    if _LAZY_IMPORTS["keras"] is None:
        try:
            # Temporarily redirect stdout/stderr to suppress console output during import
            import sys
            from contextlib import redirect_stderr, redirect_stdout
            from io import StringIO

            # Create null streams to absorb output
            null_stream = StringIO()

            with redirect_stdout(null_stream), redirect_stderr(null_stream):
                import keras

                # Suppress Keras logging
                keras.utils.disable_interactive_logging()

                # Additional TensorFlow logging suppression after import
                try:
                    import tensorflow as tf

                    tf.get_logger().setLevel("ERROR")
                    tf.autograph.set_verbosity(0)
                    # Disable various TensorFlow verbose options
                    if hasattr(tf.config, "experimental"):
                        try:
                            tf.config.experimental.enable_op_determinism()
                        except (AttributeError, RuntimeError, ValueError):
                            pass
                except (ImportError, AttributeError):
                    pass

            _LAZY_IMPORTS["keras"] = keras
        except ImportError:
            _LAZY_IMPORTS["keras"] = False
    return _LAZY_IMPORTS["keras"] if _LAZY_IMPORTS["keras"] is not False else None


def _lazy_import_optuna():
    """Lazily import optuna."""
    import sys

    current_module = sys.modules.get(__name__)
    if current_module and hasattr(current_module, "__dict__") and "optuna" in current_module.__dict__:
        patched_value = current_module.__dict__["optuna"]
        if patched_value is None:
            return None
        _LAZY_IMPORTS["optuna"] = patched_value
        return patched_value

    if _LAZY_IMPORTS["optuna"] is None:
        try:
            import optuna

            _LAZY_IMPORTS["optuna"] = optuna
        except ImportError:
            _LAZY_IMPORTS["optuna"] = False
    return _LAZY_IMPORTS["optuna"] if _LAZY_IMPORTS["optuna"] is not False else None


def _lazy_import_plotly():
    """Lazily import plotly."""
    import sys

    current_module = sys.modules.get(__name__)
    if current_module and hasattr(current_module, "__dict__") and "plotly" in current_module.__dict__:
        patched_value = current_module.__dict__["plotly"]
        if patched_value is None:
            return None
        _LAZY_IMPORTS["plotly"] = patched_value
        return patched_value

    if _LAZY_IMPORTS["plotly"] is None:
        try:
            import plotly.graph_objects as go

            _LAZY_IMPORTS["plotly"] = go
        except ImportError:
            _LAZY_IMPORTS["plotly"] = False
    return _LAZY_IMPORTS["plotly"] if _LAZY_IMPORTS["plotly"] is not False else None


def _lazy_import_polars():
    """Lazily import polars."""
    import sys

    current_module = sys.modules.get(__name__)
    if current_module and hasattr(current_module, "__dict__") and "polars" in current_module.__dict__:
        patched_value = current_module.__dict__["polars"]
        if patched_value is None:
            return None
        _LAZY_IMPORTS["polars"] = patched_value
        return patched_value

    if _LAZY_IMPORTS["polars"] is None:
        try:
            import polars as pl

            _LAZY_IMPORTS["polars"] = pl
        except ImportError:
            _LAZY_IMPORTS["polars"] = False
    return _LAZY_IMPORTS["polars"] if _LAZY_IMPORTS["polars"] is not False else None


def serialize_pytorch_tensor(tensor: Any) -> Dict[str, Any]:
    """Serialize a PyTorch tensor to a JSON-compatible format.

    Args:
        tensor: PyTorch tensor to serialize

    Returns:
        Dictionary containing tensor data and metadata
    """
    torch = _lazy_import_torch()
    if torch is None:
        return {"__datason_type__": "torch.Tensor", "__datason_value__": str(tensor)}

    # Convert to CPU and detach from computation graph
    cpu_tensor = tensor.detach().cpu()

    return {
        "__datason_type__": "torch.Tensor",
        "__datason_value__": {
            "shape": list(cpu_tensor.shape),
            "dtype": str(cpu_tensor.dtype),
            "data": cpu_tensor.numpy().tolist(),
            "device": str(tensor.device),
            "requires_grad": (tensor.requires_grad if hasattr(tensor, "requires_grad") else False),
        },
    }


def serialize_tensorflow_tensor(tensor: Any) -> Dict[str, Any]:
    """Serialize a TensorFlow tensor to a JSON-compatible format.

    Args:
        tensor: TensorFlow tensor to serialize

    Returns:
        Dictionary containing tensor data and metadata
    """
    tf = _lazy_import_tensorflow()
    if tf is None:
        return {"__datason_type__": "tf.Tensor", "__datason_value__": str(tensor)}

    return {
        "__datason_type__": "tf.Tensor",
        "__datason_value__": {
            "shape": tensor.shape.as_list(),
            "dtype": str(tensor.dtype.name),
            "data": tensor.numpy().tolist(),
        },
    }


def serialize_jax_array(array: Any) -> Dict[str, Any]:
    """Serialize a JAX array to a JSON-compatible format.

    Args:
        array: JAX array to serialize

    Returns:
        Dictionary containing array data and metadata
    """
    jax, jnp = _lazy_import_jax()
    if jax is None:
        return {"__datason_type__": "jax.Array", "__datason_value__": str(array)}

    return {
        "__datason_type__": "jax.Array",
        "__datason_value__": {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "data": array.tolist(),
        },
    }


def serialize_sklearn_model(model: Any) -> Dict[str, Any]:
    """Serialize a scikit-learn model to a JSON-compatible format.

    Args:
        model: Scikit-learn model to serialize

    Returns:
        Dictionary containing model metadata and parameters
    """
    sklearn, BaseEstimator = _lazy_import_sklearn()
    if sklearn is None or BaseEstimator is None:
        return {"__datason_type__": "sklearn.model", "__datason_value__": str(model)}

    try:
        # Get model parameters
        params = model.get_params() if hasattr(model, "get_params") else {}

        # Try to serialize parameters safely
        safe_params: Dict[str, Any] = {}
        for key, value in params.items():
            try:
                # Only include JSON-serializable parameters
                if isinstance(value, (str, int, float, bool, type(None))):
                    safe_params[key] = value
                elif isinstance(value, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in value):
                    safe_params[key] = list(value)
                else:
                    safe_params[key] = str(value)
            except Exception:
                safe_params[key] = str(value)

        return {
            "__datason_type__": "sklearn.model",
            "__datason_value__": {
                "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
                "params": safe_params,
                "fitted": hasattr(model, "n_features_in_") or hasattr(model, "feature_names_in_"),
            },
        }
    except Exception:
        warnings.warn("Could not serialize sklearn model due to an internal error.", stacklevel=2)
        return {
            "__datason_type__": "sklearn.model",
            "__datason_value__": {"error": "An internal error occurred during serialization."},
        }


def serialize_scipy_sparse(matrix: Any) -> Dict[str, Any]:
    """Serialize a scipy sparse matrix to a JSON-compatible format.

    Args:
        matrix: Scipy sparse matrix to serialize

    Returns:
        Dictionary containing sparse matrix data and metadata
    """
    scipy = _lazy_import_scipy()
    if scipy is None:
        return {"__datason_type__": "scipy.sparse", "__datason_value__": str(matrix)}

    try:
        # Convert to COO format for easier serialization
        coo_matrix = matrix.tocoo()

        return {
            "__datason_type__": "scipy.sparse",
            "__datason_value__": {
                "format": type(matrix).__name__,
                "shape": list(coo_matrix.shape),
                "dtype": str(coo_matrix.dtype),
                "data": coo_matrix.data.tolist(),
                "row": coo_matrix.row.tolist(),
                "col": coo_matrix.col.tolist(),
                "nnz": coo_matrix.nnz,
            },
        }
    except Exception:
        warnings.warn("Could not serialize scipy sparse matrix due to an internal error.", stacklevel=2)
        return {
            "__datason_type__": "scipy.sparse",
            "__datason_value__": {"error": "An internal error occurred during serialization."},
        }


def serialize_pil_image(image: Any) -> Dict[str, Any]:
    """Serialize a PIL Image to a JSON-compatible format.

    Args:
        image: PIL Image to serialize

    Returns:
        Dictionary containing image data and metadata
    """
    Image = _lazy_import_pil()
    if Image is None:
        return {"__datason_type__": "PIL.Image", "__datason_value__": str(image)}

    try:
        # Convert image to base64 string
        format_name = image.format or "PNG"
        buffer = io.BytesIO()
        image.save(buffer, format=format_name)
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return {
            "__datason_type__": "PIL.Image",
            "__datason_value__": {
                "format": format_name,
                "size": image.size,
                "mode": image.mode,
                "data": img_str,
            },
        }
    except Exception:
        warnings.warn("Could not serialize PIL Image due to an internal error.", stacklevel=2)
        return {
            "__datason_type__": "PIL.Image",
            "__datason_value__": {"error": "An internal error occurred during serialization."},
        }


def serialize_huggingface_tokenizer(tokenizer: Any) -> Dict[str, Any]:
    """Serialize a HuggingFace tokenizer to a JSON-compatible format.

    Args:
        tokenizer: HuggingFace tokenizer to serialize

    Returns:
        Dictionary containing tokenizer metadata
    """
    transformers = _lazy_import_transformers()
    if transformers is None:
        return {"__datason_type__": "transformers.tokenizer", "__datason_value__": str(tokenizer)}

    try:
        return {
            "__datason_type__": "transformers.tokenizer",
            "__datason_value__": {
                "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}",
                "vocab_size": len(tokenizer) if hasattr(tokenizer, "__len__") else None,
                "model_max_length": getattr(tokenizer, "model_max_length", None),
                "name_or_path": getattr(tokenizer, "name_or_path", None),
            },
        }
    except Exception:
        warnings.warn("Could not serialize HuggingFace tokenizer due to an internal error.", stacklevel=2)
        return {
            "__datason_type__": "transformers.tokenizer",
            "__datason_value__": {"error": "An internal error occurred during serialization."},
        }


def serialize_catboost_model(model: Any) -> Dict[str, Any]:
    """Serialize a CatBoost model to a JSON-compatible format.

    Args:
        model: CatBoost model to serialize

    Returns:
        Dictionary containing model metadata and parameters
    """
    catboost = _lazy_import_catboost()
    if catboost is None:
        return {"__datason_type__": "catboost.model", "__datason_value__": str(model)}

    try:
        # Get model parameters
        params = model.get_params() if hasattr(model, "get_params") else {}

        # Try to serialize parameters safely
        safe_params: Dict[str, Any] = {}
        for key, value in params.items():
            try:
                # Only include JSON-serializable parameters
                if isinstance(value, (str, int, float, bool, type(None))):
                    safe_params[key] = value
                elif isinstance(value, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in value):
                    safe_params[key] = list(value)
                else:
                    safe_params[key] = str(value)
            except Exception:
                safe_params[key] = str(value)

        # Check if model is fitted using tree_count_ which is more reliable
        tree_count = getattr(model, "tree_count_", None)
        fitted = tree_count is not None and tree_count > 0

        return {
            "__datason_type__": "catboost.model",
            "__datason_value__": {
                "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
                "params": safe_params,
                "fitted": fitted,
                "tree_count": getattr(model, "tree_count_", None),
            },
        }
    except Exception:
        warnings.warn("Could not serialize CatBoost model due to an internal error.", stacklevel=2)
        return {
            "__datason_type__": "catboost.model",
            "__datason_value__": {"error": "An internal error occurred during serialization."},
        }


def serialize_keras_model(model: Any) -> Dict[str, Any]:
    """Serialize a Keras model to a JSON-compatible format.

    Args:
        model: Keras model to serialize

    Returns:
        Dictionary containing model metadata and architecture
    """
    keras = _lazy_import_keras()
    if keras is None:
        return {"__datason_type__": "keras.model", "__datason_value__": str(model)}

    try:
        # Get model configuration
        config = model.get_config() if hasattr(model, "get_config") else {}

        # Extract basic metadata
        return {
            "__datason_type__": "keras.model",
            "__datason_value__": {
                "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
                "name": getattr(model, "name", None),
                "built": getattr(model, "built", False),
                "trainable": getattr(model, "trainable", True),
                "layers_count": len(model.layers) if hasattr(model, "layers") else 0,
                "input_shape": getattr(model, "input_shape", None),
                "output_shape": getattr(model, "output_shape", None),
                "config_summary": str(config)[:500] if config else None,  # Truncate for safety
            },
        }
    except Exception:
        warnings.warn("Could not serialize Keras model due to an internal error.", stacklevel=2)
        return {
            "__datason_type__": "keras.model",
            "__datason_value__": {"error": "An internal error occurred during serialization."},
        }


def serialize_optuna_study(study: Any) -> Dict[str, Any]:
    """Serialize an Optuna study to a JSON-compatible format.

    Args:
        study: Optuna study to serialize

    Returns:
        Dictionary containing study metadata and configuration
    """
    optuna = _lazy_import_optuna()
    if optuna is None:
        return {"__datason_type__": "optuna.study", "__datason_value__": str(study)}

    try:
        # Get trial count safely
        trials_count = len(study.trials) if hasattr(study, "trials") else 0

        # Get best value/params only if trials exist
        best_value = None
        best_params = {}
        if trials_count > 0:
            try:
                best_value = study.best_value if hasattr(study, "best_value") else None
                best_params = study.best_params if hasattr(study, "best_params") else {}
            except Exception:  # nosec B110
                # Ignore errors when getting best values if no trials are completed
                pass

        # Avoid deprecated attributes
        return {
            "__datason_type__": "optuna.study",
            "__datason_value__": {
                "study_name": study.study_name,
                "direction": str(study.direction) if hasattr(study, "direction") else None,
                "user_attrs": study.user_attrs if hasattr(study, "user_attrs") else {},
                "trials_count": trials_count,
                "best_value": best_value,
                "best_params": best_params,
            },
        }
    except Exception:
        warnings.warn("Could not serialize Optuna study due to an internal error.", stacklevel=2)
        return {
            "__datason_type__": "optuna.study",
            "__datason_value__": {"error": "An internal error occurred during serialization."},
        }


def serialize_plotly_figure(figure: Any) -> Dict[str, Any]:
    """Serialize a Plotly figure to a JSON-compatible format.

    Args:
        figure: Plotly figure to serialize

    Returns:
        Dictionary containing figure data and layout
    """
    go = _lazy_import_plotly()
    if go is None:
        return {"__datason_type__": "plotly.figure", "__datason_value__": str(figure)}

    try:
        # Get figure dictionary representation
        fig_dict = figure.to_dict() if hasattr(figure, "to_dict") else {}

        return {
            "__datason_type__": "plotly.figure",
            "__datason_value__": {
                "data": fig_dict.get("data", []),
                "layout": fig_dict.get("layout", {}),
                "config": fig_dict.get("config", {}),
                "frames": fig_dict.get("frames", []),
            },
        }
    except Exception:
        warnings.warn("Could not serialize Plotly figure due to an internal error.", stacklevel=2)
        return {
            "__datason_type__": "plotly.figure",
            "__datason_value__": {"error": "An internal error occurred during serialization."},
        }


def serialize_polars_dataframe(dataframe: Any) -> Dict[str, Any]:
    """Serialize a Polars DataFrame to a JSON-compatible format.

    Args:
        dataframe: Polars DataFrame to serialize

    Returns:
        Dictionary containing DataFrame data and metadata
    """
    pl = _lazy_import_polars()
    if pl is None:
        return {"__datason_type__": "polars.dataframe", "__datason_value__": str(dataframe)}

    try:
        # Convert to dict for serialization
        data_dict = dataframe.to_dict(as_series=False) if hasattr(dataframe, "to_dict") else {}

        # Convert shape tuple to list for JSON serialization
        shape = dataframe.shape if hasattr(dataframe, "shape") else (0, 0)
        shape_list = list(shape) if isinstance(shape, tuple) else [0, 0]

        return {
            "__datason_type__": "polars.dataframe",
            "__datason_value__": {
                "data": data_dict,
                "columns": dataframe.columns if hasattr(dataframe, "columns") else [],
                "shape": shape_list,
                "dtypes": {col: str(dtype) for col, dtype in zip(dataframe.columns, dataframe.dtypes)}
                if hasattr(dataframe, "dtypes")
                else {},
            },
        }
    except Exception:
        warnings.warn("Could not serialize Polars DataFrame due to an internal error.", stacklevel=2)
        return {
            "__datason_type__": "polars.dataframe",
            "__datason_value__": {"error": "An internal error occurred during serialization."},
        }


def detect_and_serialize_ml_object(obj: Any) -> Optional[Dict[str, Any]]:
    """Detect and serialize ML/AI objects automatically.

    Args:
        obj: Object that might be from an ML/AI library

    Returns:
        Serialized object or None if not an ML/AI object
    """

    # Helper function to safely check attributes
    def safe_hasattr(obj: Any, attr: str) -> bool:
        try:
            return hasattr(obj, attr)
        except Exception:
            return False

    # PyTorch tensors
    torch = _lazy_import_torch()
    if torch is not None and isinstance(obj, torch.Tensor):
        return serialize_pytorch_tensor(obj)

    # TensorFlow tensors
    tf = _lazy_import_tensorflow()
    if (
        tf is not None
        and safe_hasattr(obj, "numpy")
        and safe_hasattr(obj, "shape")
        and safe_hasattr(obj, "dtype")
        and "tensorflow" in str(type(obj))
    ):
        return serialize_tensorflow_tensor(obj)

    # JAX arrays
    jax, jnp = _lazy_import_jax()
    if jax is not None and safe_hasattr(obj, "shape") and safe_hasattr(obj, "dtype") and "jax" in str(type(obj)):
        return serialize_jax_array(obj)

    # Scikit-learn models
    sklearn, BaseEstimator = _lazy_import_sklearn()
    if sklearn is not None and isinstance(BaseEstimator, type):
        try:
            if isinstance(obj, BaseEstimator):
                return serialize_sklearn_model(obj)
        except (TypeError, AttributeError):
            # Handle case where BaseEstimator is a Mock or invalid type
            pass

    # Scipy sparse matrices
    scipy = _lazy_import_scipy()
    if scipy is not None and safe_hasattr(obj, "tocoo") and "scipy.sparse" in str(type(obj)):
        return serialize_scipy_sparse(obj)

    # PIL Images
    Image = _lazy_import_pil()
    if Image is not None and isinstance(obj, Image.Image):
        return serialize_pil_image(obj)

    # HuggingFace tokenizers
    transformers = _lazy_import_transformers()
    if transformers is not None and safe_hasattr(obj, "encode") and "transformers" in str(type(obj)):
        return serialize_huggingface_tokenizer(obj)

    # CatBoost models - use proper isinstance check like other frameworks
    catboost = _lazy_import_catboost()
    if catboost is not None:
        try:
            if isinstance(obj, (catboost.CatBoostClassifier, catboost.CatBoostRegressor)):
                return serialize_catboost_model(obj)
        except (TypeError, AttributeError):
            pass

    # Keras models - use proper isinstance check like other frameworks
    keras = _lazy_import_keras()
    if keras is not None:
        try:
            # Check for common Keras model types
            keras_model_types = []
            if hasattr(keras, "Model"):
                keras_model_types.append(keras.Model)
            if hasattr(keras, "Sequential"):
                keras_model_types.append(keras.Sequential)
            if hasattr(keras, "models"):
                if hasattr(keras.models, "Model"):
                    keras_model_types.append(keras.models.Model)
                if hasattr(keras.models, "Sequential"):
                    keras_model_types.append(keras.models.Sequential)

            if keras_model_types and isinstance(obj, tuple(keras_model_types)):
                return serialize_keras_model(obj)
        except (TypeError, AttributeError):
            pass

    # Optuna studies - use proper isinstance check like other frameworks
    optuna = _lazy_import_optuna()
    if optuna is not None:
        try:
            if hasattr(optuna, "Study") and isinstance(obj, optuna.Study):
                return serialize_optuna_study(obj)
        except (TypeError, AttributeError):
            pass

    # Plotly figures - use proper isinstance check like other frameworks
    plotly = _lazy_import_plotly()
    if plotly is not None:
        try:
            import plotly.graph_objects as go

            if isinstance(obj, go.Figure):
                return serialize_plotly_figure(obj)
        except (TypeError, AttributeError, ImportError):
            pass

    # Polars DataFrames - use proper isinstance check like other frameworks
    polars = _lazy_import_polars()
    if polars is not None:
        try:
            if hasattr(polars, "DataFrame") and isinstance(obj, polars.DataFrame):
                return serialize_polars_dataframe(obj)
        except (TypeError, AttributeError):
            pass

    return None


def get_ml_library_info() -> Dict[str, bool]:
    """Get information about which ML libraries are available.

    Returns:
        Dictionary mapping library names to availability status
    """
    return {
        "torch": _lazy_import_torch() is not None,
        "tensorflow": _lazy_import_tensorflow() is not None,
        "jax": _lazy_import_jax()[0] is not None,
        "sklearn": _lazy_import_sklearn()[0] is not None,
        "scipy": _lazy_import_scipy() is not None,
        "PIL": _lazy_import_pil() is not None,
        "transformers": _lazy_import_transformers() is not None,
        "catboost": _lazy_import_catboost() is not None,
        "keras": _lazy_import_keras() is not None,
        "optuna": _lazy_import_optuna() is not None,
        "plotly": _lazy_import_plotly() is not None,
        "polars": _lazy_import_polars() is not None,
    }


# Module-level attribute access for testing patches
def __getattr__(name: str):
    """Support dynamic attribute access for test patches."""
    if name == "torch":
        return _lazy_import_torch()
    elif name == "tf":
        return _lazy_import_tensorflow()
    elif name == "jax":
        jax, _ = _lazy_import_jax()
        return jax
    elif name == "sklearn":
        sklearn, _ = _lazy_import_sklearn()
        return sklearn
    elif name == "BaseEstimator":
        _, base_estimator = _lazy_import_sklearn()
        return base_estimator
    elif name == "scipy":
        return _lazy_import_scipy()
    elif name == "Image":
        return _lazy_import_pil()
    elif name == "transformers":
        return _lazy_import_transformers()
    elif name == "catboost":
        return _lazy_import_catboost()
    elif name == "keras":
        return _lazy_import_keras()
    elif name == "optuna":
        return _lazy_import_optuna()
    elif name == "plotly":
        return _lazy_import_plotly()
    elif name == "polars":
        return _lazy_import_polars()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
