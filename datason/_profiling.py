"""Internal profiling utilities for optional stage timing.

These helpers provide minimal-overhead timing that can be enabled via the
``DATASON_PROFILE`` environment variable. When enabled, blocks wrapped in
``profile_run`` and ``stage`` accumulate wall-clock timings for named stages.
When disabled they devolve to no-ops with negligible overhead. Collected
timings can be forwarded to an optional *profile sink* – a callable that
receives the stage dictionary when the top level run completes.
"""

from __future__ import annotations

import contextvars
import os
import time
from typing import Callable

# Evaluate the flag once for minimal overhead in hot paths
ENABLED = os.getenv("DATASON_PROFILE") == "1"

# Context variable storing the timing dictionary for the current run
_current_timings: contextvars.ContextVar[dict[str, float] | None] = contextvars.ContextVar(
    "datason_profile_timings", default=None
)

# Optional sink callable invoked with timings at the end of a profiling run
_profile_sink: Callable[[dict[str, float]], None] | None = None


def set_profile_sink(sink: Callable[[dict[str, float]], None] | None) -> None:
    """Register a callable that receives timing dictionaries.

    The *sink* is invoked once for each top-level ``profile_run`` when profiling
    is enabled. Passing ``None`` removes the sink.
    """

    global _profile_sink
    _profile_sink = sink


class _NullProfileRun:
    def __enter__(self) -> dict[str, float]:  # noqa: D401 - trivial context
        return {}

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: D401 - trivial context
        return False


class _ProfileRun:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self.token: contextvars.Token | None = None

    def __enter__(self) -> dict[str, float]:  # noqa: D401 - context manager
        self.token = _current_timings.set(self.timings)
        return self.timings

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: D401 - context manager
        if self.token is not None:
            _current_timings.reset(self.token)

        # Call the registered profile sink
        sink = _profile_sink
        if sink is not None:
            try:
                sink(self.timings)
            except Exception:  # nosec B110
                # Profiling must never interfere with normal execution
                pass

        # Note: Individual events are now emitted immediately by _Stage.__exit__
        # This avoids accumulation issues and ensures proper event timing

        return False


class _NullStage:
    def __enter__(self) -> None:  # noqa: D401 - trivial context
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: D401 - trivial context
        return False


class _Stage:
    def __init__(self, name: str) -> None:
        self.name = name
        self.timings: dict[str, float] | None = None
        self.start: float = 0.0

    def __enter__(self) -> None:  # noqa: D401 - context manager
        self.timings = _current_timings.get()
        if self.timings is not None:
            self.start = time.perf_counter()
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: D401 - context manager
        timings = self.timings
        if timings is not None:
            end = time.perf_counter()
            duration_seconds = end - self.start
            timings[self.name] = timings.get(self.name, 0.0) + duration_seconds

            # Also emit individual events to external profile_sink for benchmarks
            try:
                import datason

                external_sink = getattr(datason, "profile_sink", None)
                if external_sink is not None and hasattr(external_sink, "append"):
                    external_sink.append(
                        {
                            "stage": self.name,
                            "duration": int(duration_seconds * 1_000_000_000),  # Convert to nanoseconds
                        }
                    )
            except Exception:  # nosec B110
                # Profiling must never interfere with normal execution
                pass
        return False


_NULL_PROFILE_RUN = _NullProfileRun()
_NULL_STAGE = _NullStage()


def profile_run() -> _ProfileRun | _NullProfileRun:
    """Create a profiling run and yield the timings dictionary."""

    if not ENABLED:
        return _NULL_PROFILE_RUN
    return _ProfileRun()


def stage(name: str) -> _Stage | _NullStage:
    """Measure a named stage within a ``profile_run``."""

    if not ENABLED:
        return _NULL_STAGE
    return _Stage(name)
