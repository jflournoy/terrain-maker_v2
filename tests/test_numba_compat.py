"""The numba fallback must leave decorated code runnable without numba."""

import importlib
import sys
from unittest import mock


def test_fallback_without_numba():
    with mock.patch.dict(sys.modules, {"numba": None}):
        import src.terrain._numba_compat as compat

        compat = importlib.reload(compat)
        try:
            assert compat.NUMBA_AVAILABLE is False
            assert compat.prange is range

            @compat.jit(nopython=True, cache=True)
            def add(a, b):
                return a + b

            assert add(2, 3) == 5
        finally:
            sys.modules.pop("numba", None)
    importlib.reload(compat)  # restore the real numba-backed module
