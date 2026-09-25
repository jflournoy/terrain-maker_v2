"""Deprecated import path: ``src.X`` is now ``terrain_maker.X``.

Only reachable when running from the repository root (the old workflow).
Each ``src.*`` module is an alias of the real ``terrain_maker.*`` module,
so old imports, pickles and ``isinstance`` checks keep working.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

_OLD, _NEW = "src.", "terrain_maker."


class _AliasLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD):
            return None
        new_name = _NEW + fullname[len(_OLD) :]
        if importlib.util.find_spec(new_name) is None:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        new_name = _NEW + spec.name[len(_OLD) :]
        warnings.warn(
            f"'{spec.name}' is deprecated; import '{new_name}' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return importlib.import_module(new_name)

    def exec_module(self, module):
        pass  # already executed under its real name


if not any(isinstance(f, _AliasLoader) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasLoader())
