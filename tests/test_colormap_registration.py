"""The library's colormap names must resolve after importing terrain_maker alone.

Examples and library code ask matplotlib for "rocket" and "mako" (seaborn
colormaps) and "boreal_mako" (ours). Those names exist only after something
registers them. seaborn registers its maps as an import side effect, so they
vanished silently when an unrelated `import seaborn` was removed from core.py.
Each check runs in a fresh interpreter: inside pytest another test module may
already have imported seaborn and masked the gap.
"""

import subprocess
import sys

import pytest

NAMES_USED_BY_LIBRARY_AND_EXAMPLES = [
    "rocket",
    "mako",
    "mako_r",
    "boreal_mako",
    "boreal_mako_print",
    "michigan",
]


def _resolve_in_fresh_interpreter(module: str, name: str) -> subprocess.CompletedProcess:
    code = (
        f"import {module}\n"
        "import matplotlib\n"
        f"matplotlib.colormaps[{name!r}]\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )


@pytest.mark.parametrize("name", NAMES_USED_BY_LIBRARY_AND_EXAMPLES)
def test_colormap_resolves_after_importing_color_mapping(name):
    result = _resolve_in_fresh_interpreter("terrain_maker.terrain.color_mapping", name)
    assert result.returncode == 0, result.stderr[-2000:]
