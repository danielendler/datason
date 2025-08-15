# Lightweight profiling utilities for optional stage timing.
"""Internal profiling utilities for optional stage timing.

These helpers provide minimal-overhead timing that can be enabled via the
``DATASON_PROFILE`` environment variable.  When the variable is set to ``"1"``,
code wrapped in :func:`profile_run` and :func:`stage` will record wall clock
execution times for named stages.  When the variable is unset or ``"0"`` these
functions become near no-ops and the overhead is intended to be effectively
zero.

The collected timings can be forwarded to an optional *profile sink* – a
callable that receives the dictionary of stage timings when the top level run
completes.  The sink can be configured via :func:`set_profile_sink`.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Callable, Dict, Optional
import contextvars

# Evaluate the flag once for minimal overhead in hot paths
ENABLED = os.getenv("DATASON_PROFILE") == "1"

# Context variable storing the timing dictionary for the current run
_current_timings: contextvars.ContextVar[Optional[Dict[str, float]]] = contextvars.ContextVar(
    "datason_profile_timings", default=None
)

# Optional sink callable invoked with timings at the end of a profiling run
_profile_sink: Optional[Callable[[Dict[str, float]], None]] = None


def set_profile_sink(sink: Optional[Callable[[Dict[str, float]], None]]) -> None:
    """Register a callable that receives timing dictionaries.

    The *sink* is invoked once for each top-level :func:`profile_run` when
    profiling is enabled.  Passing ``None`` removes the sink.
    """

    global _profile_sink
    _profile_sink = sink


@contextmanager
def profile_run() -> Dict[str, float]:
    """Create a profiling run and yield the timings dictionary.

    When profiling is disabled this yields an empty dictionary.  When enabled it
    yields a dictionary that will be populated by :func:`stage` contexts.
    """

    if not ENABLED:
        yield {}
        return

    timings: Dict[str, float] = {}
    token = _current_timings.set(timings)
    try:
        yield timings
    finally:
        _current_timings.reset(token)
        if _profile_sink is not None:
            try:
                _profile_sink(timings)
            except Exception:
                # Profiling must never interfere with normal execution
                pass


@contextmanager
def stage(name: str):
    """Measure a named stage within a :func:`profile_run`.

    The collected timings are stored in the current run's dictionary.  When
    profiling is disabled this context manager adds virtually no overhead.
    """

    if not ENABLED:
        yield
        return

    timings = _current_timings.get()
    if timings is None:
        # No active profiling run
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        timings[name] = timings.get(name, 0.0) + (end - start)
