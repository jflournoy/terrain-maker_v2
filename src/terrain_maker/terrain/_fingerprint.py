"""Content fingerprints for cache keys.

A cache key must change whenever the cached result could: a different input array, or a
transform with different code or parameters. Transforms here are closures that carry their
parameters in closure cells, so a transform is identified by its name, its bytecode and the
values it closes over. Anything that cannot be identified by content (an arbitrary object)
falls back to its id, which makes the key unique to this process: a cache miss, never a
wrong hit.
"""

import hashlib
import logging
import types

import numpy as np

_PLAIN = (bool, int, float, complex, str, bytes, type(None))


def array_fingerprint(array) -> str:
    """Hex digest of an array's dtype, shape and bytes."""
    array = np.ascontiguousarray(array)
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{array.dtype.str}{array.shape}".encode())
    h.update(array.tobytes())
    return h.hexdigest()


def _update(h, value, depth):
    if depth > 6:
        h.update(f"<deep:{id(value)}>".encode())
    elif isinstance(value, _PLAIN):
        h.update(f"{type(value).__name__}:{value!r}".encode())
    elif isinstance(value, np.ndarray):
        h.update(array_fingerprint(value).encode())
    elif isinstance(value, np.generic):
        h.update(f"{value.dtype.str}:{value.item()!r}".encode())
    elif isinstance(value, (list, tuple)):
        h.update(f"{type(value).__name__}[{len(value)}]".encode())
        for item in value:
            _update(h, item, depth + 1)
    elif isinstance(value, dict):
        h.update(f"dict[{len(value)}]".encode())
        for key in sorted(value, key=repr):
            _update(h, key, depth + 1)
            _update(h, value[key], depth + 1)
    elif callable(value) and hasattr(value, "__code__"):
        _update_function(h, value, depth + 1)
    elif isinstance(value, logging.Logger):  # closures capture loggers; output-neutral
        h.update(f"<logger:{value.name}>".encode())
    elif isinstance(value, types.ModuleType):
        h.update(f"<module:{value.__name__}>".encode())
    elif hasattr(value, "to_gdal"):  # affine.Affine
        _update(h, tuple(value), depth + 1)
    else:
        h.update(f"<{type(value).__qualname__}:{id(value)}>".encode())


def _update_function(h, func, depth):
    code = func.__code__
    h.update(f"fn:{getattr(func, '__name__', '')}:{code.co_qualname if hasattr(code, 'co_qualname') else code.co_name}".encode())
    h.update(code.co_code)
    _update(h, code.co_consts, depth + 1)
    _update(h, func.__defaults__, depth + 1)
    for cell in func.__closure__ or ():
        try:
            contents = cell.cell_contents
        except ValueError:  # empty cell
            h.update(b"<empty>")
            continue
        _update(h, contents, depth + 1)
    for attr in sorted(k for k in vars(func) if not k.startswith("__")):
        _update(h, attr, depth + 1)
        _update(h, getattr(func, attr), depth + 1)


def callable_fingerprint(func) -> str:
    """Hex digest of a function's name, code, defaults and closed-over values."""
    h = hashlib.blake2b(digest_size=16)
    _update(h, func, 0)
    return h.hexdigest()
