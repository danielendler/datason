"""Register stdlib handlers and defer optional libraries until first use.

Optional plugin modules are imported only for matching objects or known tag
namespaces. Availability discovery probes top-level specs without importing the
libraries. Import targets and priorities are fixed by the table below.
"""

from importlib.util import find_spec

from .._registry import default_registry
from ._lazy import LazyPlugin
from .datetime import DatetimePlugin
from .decimal import DecimalPlugin
from .path import PathPlugin
from .structured import StructuredPlugin
from .uuid import UUIDPlugin

# name, priority, class, object module roots, wire tag prefixes
_OPTIONAL = (
    ("numpy", 200, "NumpyPlugin", ("numpy",), ("numpy.",)),
    ("pandas", 201, "PandasPlugin", ("pandas",), ("pandas.",)),
    ("scipy_sparse", 250, "ScipySparsePlugin", ("scipy",), ("scipy.sparse.",)),
    ("torch", 300, "TorchPlugin", ("torch",), ("torch.",)),
    ("tensorflow", 301, "TensorFlowPlugin", ("tensorflow",), ("tf.",)),
    ("sklearn", 302, "SklearnPlugin", ("sklearn",), ("sklearn.",)),
    (
        "ml_misc",
        350,
        "MlMiscPlugin",
        ("polars", "jax", "jaxlib", "catboost", "optuna", "plotly"),
        ("polars.", "jax.", "catboost.", "optuna.", "plotly."),
    ),
    ("pydantic", 10_001, "PydanticPlugin", ("pydantic",), ()),
)


def _register_builtins() -> None:
    for plugin_cls in (DatetimePlugin, UUIDPlugin, DecimalPlugin, PathPlugin, StructuredPlugin):
        default_registry.register(plugin_cls())
    for name, priority, class_name, roots, tags in _OPTIONAL:
        # ml_misc also reads metadata-only CatBoost/Optuna exports without those libraries.
        dependency = "scipy" if name == "scipy_sparse" else name
        if name == "ml_misc" or find_spec(dependency) is not None:
            default_registry.register(LazyPlugin(name, priority, class_name, roots, tags))


_register_builtins()
