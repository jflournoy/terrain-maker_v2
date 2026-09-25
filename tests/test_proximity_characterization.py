"""
Characterization tests for Terrain proximity / ring masks and ring coloring.

Pins mask digests across clustering, reprojection, NaN and out-of-bounds
points so the shared logic can be factored out without changing results.
Update EXPECTED only for an intended behavior change.
"""

import hashlib
import logging

import numpy as np
import pytest
from rasterio import Affine

pytest.importorskip("bpy")

from terrain_maker.terrain.core import Terrain  # noqa: E402

# 40 x 50 grid of 30 m UTM 17N pixels near Detroit
ORIGIN_X, ORIGIN_Y, PIXEL = 320000.0, 4700000.0, 30.0
UTM = "EPSG:32617"


def _terrain():
    yy, xx = np.mgrid[0:40, 0:50]
    dem = (200 + 3 * np.sin(xx / 6) + 2 * np.cos(yy / 5)).astype(np.float32)
    terrain = Terrain(dem, Affine(PIXEL, 0, ORIGIN_X, 0, -PIXEL, ORIGIN_Y), dem_crs=UTM)
    terrain.transforms.append(lambda data, trans: (data, trans, None))
    terrain.apply_transforms()
    return terrain


def _points():
    """UTM points: two close together, one apart, one NaN, one outside the grid."""
    xs = ORIGIN_X + PIXEL * np.array([10.2, 12.7, 38.4, np.nan, 90.0])
    ys = ORIGIN_Y - PIXEL * np.array([8.6, 9.1, 30.3, 5.0, 5.0])
    return xs, ys


def _wgs84_points():
    from pyproj import Transformer

    xs, ys = _points()
    lons, lats = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True).transform(xs, ys)
    return np.asarray(lons), np.asarray(lats)


def _digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _summary(mask):
    return {"shape": tuple(mask.shape), "n": int(mask.sum()), "digest": _digest(mask)}


def _grid(terrain, cluster, wgs84):
    xs, ys = _wgs84_points() if wgs84 else _points()
    crs = "EPSG:4326" if wgs84 else UTM
    return terrain.compute_proximity_mask_grid(
        xs, ys, radius_meters=150, input_crs=crs, cluster_threshold_meters=cluster
    )


def _ring(terrain, inner, cluster):
    xs, ys = _points()
    return terrain.compute_ring_mask_grid(
        xs, ys, inner, 240, input_crs=UTM, cluster_threshold_meters=cluster
    )


def _mesh(terrain, cluster, wgs84):
    terrain.create_mesh(boundary_extension=False, verbose=False)
    xs, ys = _wgs84_points() if wgs84 else _points()
    crs = "EPSG:4326" if wgs84 else UTM
    return terrain.compute_proximity_mask(
        xs, ys, radius_meters=150, input_crs=crs, cluster_threshold_meters=cluster
    )


def _mock_ring():
    """Ring mask on a terrain without a transform: points in normalized grid space."""

    class Mock:
        data_layers = {}
        _transformed_dem = np.zeros((20, 30))
        _transformed_transform = None
        logger = logging.getLogger("test")

    Mock.compute_ring_mask_grid = Terrain.compute_ring_mask_grid
    return Mock().compute_ring_mask_grid(
        np.array([0.2, 0.7, np.nan]), np.array([0.3, 0.6, 0.1]), 30, 120
    )


def _ring_color():
    terrain = _terrain()
    terrain.y_valid, terrain.x_valid = np.nonzero(np.ones((40, 50), dtype=bool))
    terrain.colors = np.full((2000, 4), 200, dtype=np.float32)
    terrain.apply_ring_color(_ring(_terrain(), 60, None), ring_color=(0.1, 0.5, 0.9))
    return {"digest": _digest(terrain.colors)}


CASES = {
    "grid_utm": lambda: _summary(_grid(_terrain(), None, False)),
    "grid_wgs84": lambda: _summary(_grid(_terrain(), None, True)),
    "grid_clustered": lambda: _summary(_grid(_terrain(), 100, False)),
    "ring_filled": lambda: _summary(_ring(_terrain(), 0, None)),
    "ring_annulus": lambda: _summary(_ring(_terrain(), 90, None)),
    "ring_clustered": lambda: _summary(_ring(_terrain(), 90, 100)),
    "ring_mock_no_transform": lambda: _summary(_mock_ring()),
    "mesh_utm": lambda: _summary(_mesh(_terrain(), None, False)),
    "mesh_wgs84_clustered": lambda: _summary(_mesh(_terrain(), 100, True)),
    "ring_color": _ring_color,
}

EXPECTED = {
    "grid_clustered": {"shape": (40, 50), "n": 159, "digest": "4a52ea765c152fb9"},
    "grid_utm": {"shape": (40, 50), "n": 185, "digest": "62276698a6f2fed4"},
    "grid_wgs84": {"shape": (40, 50), "n": 185, "digest": "62276698a6f2fed4"},
    "mesh_utm": {"shape": (2000,), "n": 186, "digest": "cf48d13631ced0d7"},
    "mesh_wgs84_clustered": {"shape": (2000,), "n": 157, "digest": "171961ab6e3aabc0"},
    "ring_annulus": {"shape": (40, 50), "n": 366, "digest": "8492292604f5ef29"},
    "ring_clustered": {"shape": (40, 50), "n": 342, "digest": "01b00bfdf76229f6"},
    "ring_color": {"digest": "bc4585546257f7b3"},
    "ring_filled": {"shape": (40, 50), "n": 429, "digest": "d1859fde5dfd746a"},
    "ring_mock_no_transform": {"shape": (20, 30), "n": 96, "digest": "f81c77db057ad056"},
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_proximity_output_is_pinned(name):
    assert CASES[name]() == EXPECTED[name]
