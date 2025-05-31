"""Machine Learning and AI library serializers for datason.

This module provides specialized serialization support for popular ML/AI libraries
including PyTorch, TensorFlow, scikit-learn, JAX, scipy, and others.
"""

import base64
import io
from typing import TYPE_CHECKING, Any, Dict, Optional
import warnings

if TYPE_CHECKING:
    try:
        import torch  # type: ignore
    except ImportError:
        pass

    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        pass

    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
    except ImportError:
        pass

    try:
        import sklearn  # type: ignore
        from sklearn.base import BaseEstimator  # type: ignore
    except ImportError:
        pass

    try:
        import scipy.sparse  # type: ignore
    except ImportError:
        pass

    try:
        from PIL import Image  # type: ignore
    except ImportError:
        pass

    try:
        import transformers  # type: ignore
    except ImportError:
        pass

# Runtime imports with fallbacks
try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

try:
    import sklearn
    from sklearn.base import BaseEstimator
except ImportError:
    sklearn = None
    BaseEstimator = None

try:
    import scipy.sparse
except ImportError:
    scipy = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import transformers
except ImportError:
    transformers = None


def serialize_pytorch_tensor(tensor: Any) -> Dict[str, Any]:
    """Serialize a PyTorch tensor to a JSON-compatible format.

    Args:
        tensor: PyTorch tensor to serialize

    Returns:
        Dictionary containing tensor data and metadata
    """
    if torch is None:
        return {"_type": "torch.Tensor", "_data": str(tensor)}

    # Convert to CPU and detach from computation graph
    cpu_tensor = tensor.detach().cpu()

    return {
        "_type": "torch.Tensor",
        "_shape": list(cpu_tensor.shape),
        "_dtype": str(cpu_tensor.dtype),
        "_data": cpu_tensor.numpy().tolist(),
        "_device": str(tensor.device),
        "_requires_grad": tensor.requires_grad
        if hasattr(tensor, "requires_grad")
        else False,
    }


def serialize_tensorflow_tensor(tensor: Any) -> Dict[str, Any]:
    """Serialize a TensorFlow tensor to a JSON-compatible format.

    Args:
        tensor: TensorFlow tensor to serialize

    Returns:
        Dictionary containing tensor data and metadata
    """
    if tf is None:
        return {"_type": "tf.Tensor", "_data": str(tensor)}

    return {
        "_type": "tf.Tensor",
        "_shape": tensor.shape.as_list(),
        "_dtype": str(tensor.dtype.name),
        "_data": tensor.numpy().tolist(),
    }


def serialize_jax_array(array: Any) -> Dict[str, Any]:
    """Serialize a JAX array to a JSON-compatible format.

    Args:
        array: JAX array to serialize

    Returns:
        Dictionary containing array data and metadata
    """
    if jax is None:
        return {"_type": "jax.Array", "_data": str(array)}

    return {
        "_type": "jax.Array",
        "_shape": list(array.shape),
        "_dtype": str(array.dtype),
        "_data": array.tolist(),
    }


def serialize_sklearn_model(model: Any) -> Dict[str, Any]:
    """Serialize a scikit-learn model to a JSON-compatible format.

    Args:
        model: Scikit-learn model to serialize

    Returns:
        Dictionary containing model metadata and parameters
    """
    if sklearn is None or BaseEstimator is None:
        return {"_type": "sklearn.model", "_data": str(model)}

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
                elif isinstance(value, (list, tuple)) and all(
                    isinstance(x, (str, int, float, bool)) for x in value
                ):
                    safe_params[key] = list(value)
                else:
                    safe_params[key] = str(value)
            except Exception:
                safe_params[key] = str(value)

        return {
            "_type": "sklearn.model",
            "_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "_params": safe_params,
            "_fitted": hasattr(model, "n_features_in_")
            or hasattr(model, "feature_names_in_"),
        }
    except Exception as e:
        warnings.warn(f"Could not serialize sklearn model: {e}", stacklevel=2)
        return {"_type": "sklearn.model", "_error": str(e)}


def serialize_scipy_sparse(matrix: Any) -> Dict[str, Any]:
    """Serialize a scipy sparse matrix to a JSON-compatible format.

    Args:
        matrix: Scipy sparse matrix to serialize

    Returns:
        Dictionary containing sparse matrix data and metadata
    """
    if scipy is None:
        return {"_type": "scipy.sparse", "_data": str(matrix)}

    try:
        # Convert to COO format for easier serialization
        coo_matrix = matrix.tocoo()

        return {
            "_type": "scipy.sparse",
            "_format": type(matrix).__name__,
            "_shape": list(coo_matrix.shape),
            "_dtype": str(coo_matrix.dtype),
            "_data": coo_matrix.data.tolist(),
            "_row": coo_matrix.row.tolist(),
            "_col": coo_matrix.col.tolist(),
            "_nnz": coo_matrix.nnz,
        }
    except Exception as e:
        warnings.warn(f"Could not serialize scipy sparse matrix: {e}", stacklevel=2)
        return {"_type": "scipy.sparse", "_error": str(e)}


def serialize_pil_image(image: Any) -> Dict[str, Any]:
    """Serialize a PIL Image to a JSON-compatible format.

    Args:
        image: PIL Image to serialize

    Returns:
        Dictionary containing image data and metadata
    """
    if Image is None:
        return {"_type": "PIL.Image", "_data": str(image)}

    try:
        # Convert image to base64 string
        format_name = image.format or "PNG"
        buffer = io.BytesIO()
        image.save(buffer, format=format_name)
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return {
            "_type": "PIL.Image",
            "_format": format_name,
            "_size": image.size,
            "_mode": image.mode,
            "_data": img_str,
        }
    except Exception as e:
        warnings.warn(f"Could not serialize PIL Image: {e}", stacklevel=2)
        return {"_type": "PIL.Image", "_error": str(e)}


def serialize_huggingface_tokenizer(tokenizer: Any) -> Dict[str, Any]:
    """Serialize a HuggingFace tokenizer to a JSON-compatible format.

    Args:
        tokenizer: HuggingFace tokenizer to serialize

    Returns:
        Dictionary containing tokenizer metadata
    """
    if transformers is None:
        return {"_type": "transformers.tokenizer", "_data": str(tokenizer)}

    try:
        return {
            "_type": "transformers.tokenizer",
            "_class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}",
            "_vocab_size": len(tokenizer) if hasattr(tokenizer, "__len__") else None,
            "_model_max_length": getattr(tokenizer, "model_max_length", None),
            "_name_or_path": getattr(tokenizer, "name_or_path", None),
        }
    except Exception as e:
        warnings.warn(f"Could not serialize HuggingFace tokenizer: {e}", stacklevel=2)
        return {"_type": "transformers.tokenizer", "_error": str(e)}


def detect_and_serialize_ml_object(obj: Any) -> Optional[Dict[str, Any]]:
    """Detect and serialize ML/AI objects automatically.

    Args:
        obj: Object that might be from an ML/AI library

    Returns:
        Serialized object or None if not an ML/AI object
    """
    # PyTorch tensors
    if torch is not None and isinstance(obj, torch.Tensor):
        return serialize_pytorch_tensor(obj)

    # TensorFlow tensors
    if (
        tf is not None
        and hasattr(obj, "numpy")
        and hasattr(obj, "shape")
        and hasattr(obj, "dtype")
        and "tensorflow" in str(type(obj))
    ):
        return serialize_tensorflow_tensor(obj)

    # JAX arrays
    if (
        jax is not None
        and hasattr(obj, "shape")
        and hasattr(obj, "dtype")
        and "jax" in str(type(obj))
    ):
        return serialize_jax_array(obj)

    # Scikit-learn models
    if (
        sklearn is not None
        and BaseEstimator is not None
        and isinstance(obj, BaseEstimator)
    ):
        return serialize_sklearn_model(obj)

    # Scipy sparse matrices
    if scipy is not None and hasattr(obj, "tocoo") and "scipy.sparse" in str(type(obj)):
        return serialize_scipy_sparse(obj)

    # PIL Images
    if Image is not None and isinstance(obj, Image.Image):
        return serialize_pil_image(obj)

    # HuggingFace tokenizers
    if (
        transformers is not None
        and hasattr(obj, "encode")
        and "transformers" in str(type(obj))
    ):
        return serialize_huggingface_tokenizer(obj)

    return None


def get_ml_library_info() -> Dict[str, bool]:
    """Get information about which ML libraries are available.

    Returns:
        Dictionary mapping library names to availability status
    """
    return {
        "torch": torch is not None,
        "tensorflow": tf is not None,
        "jax": jax is not None,
        "sklearn": sklearn is not None,
        "scipy": scipy is not None,
        "PIL": Image is not None,
        "transformers": transformers is not None,
    }
