"""
Optional numba support.

Import ``jit``, ``prange`` and ``NUMBA_AVAILABLE`` from here. Without numba,
``jit`` is a no-op decorator and ``prange`` is ``range``, so decorated code
still runs (slowly) as plain Python.
"""

try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        """No-op stand-in for numba.jit; returns the function unchanged."""

        def decorator(func):
            return func

        return decorator

    prange = range

__all__ = ["NUMBA_AVAILABLE", "jit", "prange"]
