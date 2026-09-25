"""
Library log output: visible by default, never duplicated.

Scripts that don't configure logging should still see library progress
messages; scripts that do (logging.basicConfig, file handlers) should get
each message exactly once, through their own handlers.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# One message from core, one from a module that used print() before
EMIT = """
import numpy as np
from rasterio import Affine
from src.terrain.core import Terrain
from src.terrain.visualization.line_layers import expand_lines_variable_width_sparse
Terrain(np.ones((5, 5), dtype=np.float32), Affine.identity())
mask = np.zeros((20, 20), dtype=bool)
mask[10, 5:15] = True
expand_lines_variable_width_sparse(mask, np.arange(400.0).reshape(20, 20), max_width=3)
"""


def _run(prelude: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", prelude + EMIT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout + result.stderr


def test_messages_visible_without_logging_config():
    output = _run("")
    assert output.count("Initializing Terrain...") == 1
    assert output.count("Sparse expansion:") == 1


def test_messages_not_duplicated_with_basic_config():
    output = _run(
        "import logging\n"
        "logging.basicConfig(level=logging.INFO, format='APP %(name)s: %(message)s')\n"
    )
    assert output.count("Initializing Terrain...") == 1
    assert "APP src.terrain.core: Initializing Terrain..." in output
    assert output.count("Sparse expansion:") == 1
    assert "APP src.terrain.visualization.line_layers: " in output
