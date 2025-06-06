"""Machine Learning and AI library serializers for datason.

This module provides specialized serialization support for popular ML/AI libraries
including PyTorch, TensorFlow, scikit-learn, JAX, scipy, and others.

ML libraries are imported lazily to improve startup performance.
"""

import base64
import io
import warnings
from typing import Any, Dict, Optional

# Lazy import cache - libraries are imported only when first used
_LAZY_IMPORTS = {
    "torch": None,
    "tensorflow": None,
    "jax": None,
    "jnp": None,
    "sklearn": None,
    "BaseEstimator": None,
    "scipy": None,
    "PIL_Image": None,
    "transformers": None,
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
            import tensorflow as tf

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
            _LAZY_IMPORTS["sklearn"] = patched
        if "BaseEstimator" in current_module.__dict__:
            patched_base = current_module.__dict__["BaseEstimator"]
            if patched_base is None:
                return None, None
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
    except Exception as e:
        warnings.warn(f"Could not serialize sklearn model: {e}", stacklevel=2)
        return {"__datason_type__": "sklearn.model", "__datason_value__": {"error": str(e)}}


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
    except Exception as e:
        warnings.warn(f"Could not serialize scipy sparse matrix: {e}", stacklevel=2)
        return {"__datason_type__": "scipy.sparse", "__datason_value__": {"error": str(e)}}


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
    except Exception as e:
        warnings.warn(f"Could not serialize PIL Image: {e}", stacklevel=2)
        return {"__datason_type__": "PIL.Image", "__datason_value__": {"error": str(e)}}


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
    except Exception as e:
        warnings.warn(f"Could not serialize HuggingFace tokenizer: {e}", stacklevel=2)
        return {"__datason_type__": "transformers.tokenizer", "__datason_value__": {"error": str(e)}}


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
    if sklearn is not None and isinstance(BaseEstimator, type) and isinstance(obj, BaseEstimator):
        return serialize_sklearn_model(obj)

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
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
