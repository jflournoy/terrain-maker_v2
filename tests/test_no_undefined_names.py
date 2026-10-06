"""Every name a script uses must be defined somewhere: imported, assigned, or built in.

An example calling a function it never imported runs fine until that line, which in a
data pipeline can be after minutes of loading (san_diego_flow_demo.py died this way on
downsample_then_reproject, failing 71 tests at once). Python's own symbol tables find
these statically, like pyflakes' undefined-name check, with no extra dependency.
"""

import builtins
import symtable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKED = sorted(
    p
    for folder in ("src", "examples", "tools", "scripts")
    for p in (ROOT / folder).rglob("*.py")
    if "__pycache__" not in p.parts
)
MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__spec__", "__loader__", "__package__",
                  "__path__", "__builtins__", "__annotations__"}


def _tables(table):
    yield table
    for child in table.get_children():
        yield from _tables(child)


def undefined_names(source, filename="<string>"):
    """(name, scope) pairs used as globals but bound nowhere in the module."""
    module = symtable.symtable(source, filename, "exec")
    if any(s.get_name() == "*" for s in module.get_symbols()):
        return []  # a star import binds names we cannot see
    bound = {s.get_name() for s in module.get_symbols() if s.is_assigned() or s.is_imported()}
    for table in _tables(module):
        bound |= {s.get_name() for s in table.get_symbols()
                  if s.is_declared_global() and s.is_assigned()}
    known = bound | set(dir(builtins)) | MODULE_DUNDERS
    missing = []
    for table in _tables(module):
        for s in table.get_symbols():
            global_use = s.is_global() if table is not module else not (s.is_assigned() or s.is_imported())
            if s.is_referenced() and global_use and s.get_name() not in known:
                missing.append((s.get_name(), table.get_name()))
    return sorted(set(missing))


def test_checker_finds_a_missing_import():
    source = "import numpy as np\n\ndef f(x):\n    return np.sum(downsample(x))\n"
    assert undefined_names(source) == [("downsample", "f")]


def test_checker_accepts_locals_globals_builtins_and_comprehensions():
    source = (
        "import os\nLIMIT = 3\n\ndef g():\n    global CACHE\n    CACHE = {}\n\n"
        "class C:\n    size = LIMIT\n    def m(self, xs):\n        return [len(x) for x in xs if os.sep]\n\n"
        "def h():\n    return CACHE, C, __file__\n"
    )
    assert undefined_names(source) == []


@pytest.mark.parametrize("path", CHECKED, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_undefined_names(path):
    assert undefined_names(path.read_text(), str(path)) == []
