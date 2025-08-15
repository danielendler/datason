"""Standard library JSON compatibility module.

This package re-exports Python's built-in :mod:`json` module so that
``import datason.json as json`` provides a strict drop-in replacement.
All functions, classes, and exceptions behave exactly like the
standard library version.
"""

import json as _json
from json import *  # noqa: F401,F403

__all__ = getattr(_json, "__all__", [])  # type: ignore[attr-defined]
__doc__ = _json.__doc__
