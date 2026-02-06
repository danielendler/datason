"""Built-in type handler plugins.

Plugins are auto-registered when this package is imported.
Each plugin handles serialization/deserialization for a specific
family of types (datetime, uuid, numpy, etc.).

Third-party plugins (numpy, pandas) are registered only if the
library is installed — import errors are silently skipped.
"""

from .._registry import default_registry
from .datetime import DatetimePlugin
from .decimal import DecimalPlugin
from .path import PathPlugin
from .uuid import UUIDPlugin


def _register_builtins() -> None:
    """Register all built-in plugins with the default registry."""
    # Stdlib plugins (always available)
    for plugin_cls in (DatetimePlugin, UUIDPlugin, DecimalPlugin, PathPlugin):
        default_registry.register(plugin_cls())

    # Data science plugins (optional dependencies)
    _register_optional_plugins()


def _register_optional_plugins() -> None:
    """Register plugins for optional third-party libraries."""
    try:
        from .numpy import NumpyPlugin

        default_registry.register(NumpyPlugin())
    except ImportError:
        pass

    try:
        from .pandas import PandasPlugin

        default_registry.register(PandasPlugin())
    except ImportError:
        pass


_register_builtins()
