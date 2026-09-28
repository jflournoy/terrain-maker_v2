"""Memory guards fail fast with advice instead of letting the OS kill the process."""

import numpy as np
import pytest
from rasterio import Affine

from terrain_maker.terrain._memory import ArrayTooLargeError, check_memory


def test_small_request_passes():
    check_memory(1_000, bytes_per_cell=8, operation="tiny op")


def test_request_over_limit_raises_with_advice(monkeypatch):
    monkeypatch.setenv("TERRAIN_MAKER_MEMORY_LIMIT_GB", "0.001")  # ~1 MB
    with pytest.raises(ArrayTooLargeError) as err:
        check_memory(10_000_000, bytes_per_cell=40, operation="distance transform")
    message = str(err.value)
    assert "distance transform" in message
    assert "10,000,000" in message
    assert "Downsample" in message and "TERRAIN_MAKER_MEMORY_LIMIT_GB" in message


def test_is_a_memory_error(monkeypatch):
    monkeypatch.setenv("TERRAIN_MAKER_MEMORY_LIMIT_GB", "0.001")
    with pytest.raises(MemoryError):
        check_memory(10_000_000, bytes_per_cell=40, operation="x")


def test_limit_defaults_to_available_memory(monkeypatch):
    monkeypatch.delenv("TERRAIN_MAKER_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.setattr(
        "terrain_maker.terrain._memory.available_memory_bytes", lambda: 2 * 1024**3
    )
    check_memory(10_000_000, bytes_per_cell=40, operation="x")  # 0.4 GB of 1 GB budget
    with pytest.raises(ArrayTooLargeError):
        check_memory(100_000_000, bytes_per_cell=40, operation="x")  # 4 GB


def test_proximity_grid_is_guarded(monkeypatch):
    from terrain_maker.terrain.core import Terrain

    dem = np.ones((200, 200), dtype=np.float32)
    terrain = Terrain(dem, Affine(30, 0, 320000, 0, -30, 4700000), dem_crs="EPSG:32617")
    terrain.transforms.append(lambda data, trans: (data, trans, None))
    terrain.apply_transforms()
    monkeypatch.setenv("TERRAIN_MAKER_MEMORY_LIMIT_GB", "0.0001")
    with pytest.raises(ArrayTooLargeError, match="proximity"):
        terrain.compute_proximity_mask_grid(
            np.array([320000.0 + 3000]), np.array([4700000.0 - 3000]), 300, input_crs="EPSG:32617"
        )


def test_shoreline_colors_are_guarded(monkeypatch):
    from terrain_maker.terrain.water import shoreline_water_colors

    water = np.zeros((300, 300), dtype=bool)
    water[100:200, 100:200] = True
    monkeypatch.setenv("TERRAIN_MAKER_MEMORY_LIMIT_GB", "0.0001")
    ys, xs = np.nonzero(water)
    with pytest.raises(ArrayTooLargeError):
        shoreline_water_colors(water, ys, xs)
