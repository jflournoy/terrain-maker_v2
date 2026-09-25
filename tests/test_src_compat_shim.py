"""The old `src.*` import path keeps working (with a warning) after the rename."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _run(code):
    return subprocess.run(
        [sys.executable, "-W", "always::FutureWarning", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_old_import_path_aliases_new_modules():
    result = _run(
        "import src.terrain.core as old\n"
        "from src.terrain.mesh_operations import create_boundary_extension as f\n"
        "import terrain_maker.terrain.core as new\n"
        "import terrain_maker.terrain.mesh_operations as m\n"
        "assert old is new, (old, new)\n"
        "assert f is m.create_boundary_extension\n"
    )
    assert result.returncode == 0, result.stderr
    assert "terrain_maker" in result.stderr  # FutureWarning points at the new name


def test_old_from_src_import_config():
    result = _run("from src import config\nimport terrain_maker.config as c\nassert config is c\n")
    assert result.returncode == 0, result.stderr
