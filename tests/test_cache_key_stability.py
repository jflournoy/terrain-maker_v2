"""
Cache keys must be stable across Python processes and sensitive to array content.

Python's built-in hash() of bytes is salted per process (PYTHONHASHSEED), so any
key built from it changes on every run and the on-disk cache never hits.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

TRANSFORM_KEY_SCRIPT = """
import numpy as np
from src.terrain.cache import TransformCache
cache = TransformCache(cache_dir={cache_dir!r})
print(cache.compute_transform_hash(
    upstream_hash="abc",
    transform_name="mask",
    params={{"mask": np.arange(12, dtype=np.float32).reshape(3, 4)}},
))
"""

PIPELINE_KEY_SCRIPT = """
import numpy as np
from src.terrain.cache import PipelineCache
cache = PipelineCache(cache_dir={cache_dir!r})
cache.define_target("t", params={{"mask": np.arange(12, dtype=np.float32).reshape(3, 4)}})
print(cache.compute_target_key("t"))
"""


def _key_in_fresh_process(script: str, cache_dir: Path, seed: str) -> str:
    """Run a key-computing script in a new interpreter with a given hash seed."""
    code = textwrap.dedent(script.format(cache_dir=str(cache_dir)))
    env = {"PYTHONHASHSEED": seed, "PATH": ""}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


class TestKeysStableAcrossProcesses:
    def test_transform_cache_key_with_array_param(self, tmp_path):
        key_a = _key_in_fresh_process(TRANSFORM_KEY_SCRIPT, tmp_path, seed="1")
        key_b = _key_in_fresh_process(TRANSFORM_KEY_SCRIPT, tmp_path, seed="2")
        assert key_a == key_b

    def test_pipeline_cache_key_with_array_param(self, tmp_path):
        key_a = _key_in_fresh_process(PIPELINE_KEY_SCRIPT, tmp_path, seed="1")
        key_b = _key_in_fresh_process(PIPELINE_KEY_SCRIPT, tmp_path, seed="2")
        assert key_a == key_b


class TestMeshKeySensitiveToArrayContent:
    def test_same_shape_different_values_give_different_keys(self, tmp_path):
        from src.terrain.mesh_cache import MeshCache

        cache = MeshCache(cache_dir=tmp_path)
        zeros = np.zeros((4, 4), dtype=np.uint8)
        ones = np.ones((4, 4), dtype=np.uint8)

        key_zeros = cache.compute_mesh_hash("dem", {"water_mask": zeros})
        key_ones = cache.compute_mesh_hash("dem", {"water_mask": ones})

        assert key_zeros != key_ones

    def test_same_values_give_same_key(self, tmp_path):
        from src.terrain.mesh_cache import MeshCache

        cache = MeshCache(cache_dir=tmp_path)
        mask = np.eye(4, dtype=np.uint8)

        assert cache.compute_mesh_hash("dem", {"water_mask": mask}) == cache.compute_mesh_hash(
            "dem", {"water_mask": mask.copy()}
        )
