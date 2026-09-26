"""
Characterization tests for Terrain.add_data_layer and blended color mapping.

Pins the stored layer (data, grid, CRS, bookkeeping keys) for every grid
path of add_data_layer, and the vertex colors of set_blended_color_mapping
(grid/vertex masks, RGB/RGBA colormaps, boundary padding, water), so both
can be restructured without changing results. Update EXPECTED only for an
intended behavior change.
"""

import hashlib

import numpy as np
import pytest
from rasterio import Affine

pytest.importorskip("bpy")

from terrain_maker.terrain.color_mapping import elevation_colormap  # noqa: E402
from terrain_maker.terrain.core import Terrain  # noqa: E402
from terrain_maker.terrain.transforms import reproject_raster  # noqa: E402

DEM_TRANSFORM = Affine(0.001, 0, -83.05, 0, -0.001, 42.35)


def _digest(arr):
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.round(arr.astype(np.float64), 4)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _dem():
    yy, xx = np.mgrid[0:40, 0:48]
    return (180 + 6 * np.sin(xx / 7) + 4 * np.cos(yy / 5)).astype(np.float32)


def _terrain(transformed=False):
    terrain = Terrain(_dem(), DEM_TRANSFORM, dem_crs="EPSG:4326")
    if transformed:
        terrain.transforms.append(reproject_raster(src_crs="EPSG:4326", dst_crs="EPSG:32617"))
        terrain.apply_transforms()
    return terrain


def _score(shape=(20, 24), dtype=np.float32):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return ((xx + 2 * yy) % 17).astype(dtype)


# ----- add_data_layer -------------------------------------------------------


def _layer_summary(terrain, name):
    info = terrain.data_layers[name]
    return {
        "keys": sorted(info),
        "shape": tuple(info["data"].shape),
        "dtype": str(info["data"].dtype),
        "data": _digest(info["data"]),
        "transform": tuple(round(v, 9) for v in tuple(info["transform"])[:6]),
        "crs": str(info["crs"]),
        "target_layer": info["target_layer"],
    }


def _add(name, **kwargs):
    terrain = _terrain(transformed=kwargs.pop("transformed", False))
    data = kwargs.pop("data", _score())
    terrain.add_data_layer(name, data, **kwargs)
    return _layer_summary(terrain, name)


HALF = Affine(0.002, 0, -83.05, 0, -0.002, 42.35)
LAYER_CASES = {
    "no_target": lambda: _add("s", transform=HALF, crs="EPSG:4326"),
    "same_grid": lambda: _add(
        "s", data=_score((40, 48)), transform=DEM_TRANSFORM, crs="EPSG:4326", target_layer="dem"
    ),
    "resample_to_dem": lambda: _add("s", transform=HALF, crs="EPSG:4326", target_layer="dem"),
    "reproject_to_transformed_dem": lambda: _add(
        "s", transform=HALF, crs="EPSG:4326", target_layer="dem", transformed=True
    ),
    "same_extent_as": lambda: _add("s", same_extent_as="dem"),
    "same_extent_as_transformed": lambda: _add("s", same_extent_as="dem", transformed=True),
    "target_crs_first_layer_grid": lambda: _add(
        "s", transform=HALF, crs="EPSG:4326", target_crs="EPSG:4326"
    ),
    "int_dtype_nodata_0": lambda: _add(
        "s", data=_score(dtype=np.int16), transform=HALF, crs="EPSG:4326", target_layer="dem"
    ),
    "explicit_nodata": lambda: _add(
        "s", transform=HALF, crs="EPSG:4326", target_layer="dem", nodata=3.0
    ),
}

# ----- blended colors -------------------------------------------------------


def _rgb_colormap(values):
    return elevation_colormap(values, cmap_name="plasma")[..., :3]


def _blend(vertex_mask=False, rgb=False, water=False, after_mesh=False):
    terrain = Terrain(_dem(), DEM_TRANSFORM, dem_crs="EPSG:4326")
    terrain.transforms.append(lambda data, trans: (data, trans, None))
    terrain.apply_transforms()
    terrain.add_data_layer(
        "score", _score((40, 48)), DEM_TRANSFORM, "EPSG:4326", target_layer="dem"
    )
    terrain.apply_transforms()
    yy, xx = np.mgrid[0:40, 0:48]
    mask = (yy - 20) ** 2 + (xx - 24) ** 2 < 12**2
    if vertex_mask:
        mask = mask[np.nonzero(np.ones((40, 48), dtype=bool))]
    terrain.set_blended_color_mapping(
        elevation_colormap,
        ["dem"],
        _rgb_colormap if rgb else (lambda v: elevation_colormap(v, cmap_name="viridis")),
        ["score"],
        mask,
    )
    terrain.create_mesh(verbose=False)
    colors = terrain.colors
    if after_mesh or water:
        water_mask = np.zeros((40, 48), dtype=bool)
        water_mask[3:12, 5:30] = True
        colors = terrain.compute_colors(water_mask=water_mask if water else None)
    return {"shape": tuple(colors.shape), "dtype": str(colors.dtype), "colors": _digest(colors)}


BLEND_CASES = {
    "grid_mask": lambda: _blend(),
    "vertex_mask": lambda: _blend(vertex_mask=True),
    "rgb_overlay": lambda: _blend(rgb=True),
    "after_mesh_padding": lambda: _blend(after_mesh=True),
    "water": lambda: _blend(water=True),
}

CASES = {
    **{f"layer_{k}": v for k, v in LAYER_CASES.items()},
    **{f"blend_{k}": v for k, v in BLEND_CASES.items()},
}
EXPECTED = {
    "blend_after_mesh_padding": {
        "shape": (3513, 4),
        "dtype": "uint8",
        "colors": "e7b435f89a580c3b",
    },
    "blend_grid_mask": {"shape": (1920, 4), "dtype": "uint8", "colors": "6b03f6df941b40c7"},
    "blend_rgb_overlay": {"shape": (1920, 4), "dtype": "uint8", "colors": "5f2ac1a2532a32d0"},
    "blend_vertex_mask": {"shape": (1920, 4), "dtype": "uint8", "colors": "6b03f6df941b40c7"},
    "blend_water": {"shape": (3513, 4), "dtype": "uint8", "colors": "abf501ec525a6a7c"},
    "layer_explicit_nodata": {
        "keys": [
            "crs",
            "data",
            "original_crs",
            "original_data",
            "original_transform",
            "target_layer",
            "transform",
            "transformed",
        ],
        "shape": (40, 48),
        "dtype": "float32",
        "data": "0fd2f6955f1fc5f6",
        "transform": (0.001, 0.0, -83.05, 0.0, -0.001, 42.35),
        "crs": "EPSG:4326",
        "target_layer": "dem",
    },
    "layer_int_dtype_nodata_0": {
        "keys": [
            "crs",
            "data",
            "original_crs",
            "original_data",
            "original_transform",
            "target_layer",
            "transform",
            "transformed",
        ],
        "shape": (40, 48),
        "dtype": "int16",
        "data": "eddb9edcc9c06803",
        "transform": (0.001, 0.0, -83.05, 0.0, -0.001, 42.35),
        "crs": "EPSG:4326",
        "target_layer": "dem",
    },
    "layer_no_target": {
        "keys": ["crs", "data", "target_layer", "transform", "transformed"],
        "shape": (20, 24),
        "dtype": "float32",
        "data": "d506371554df6281",
        "transform": (0.002, 0.0, -83.05, 0.0, -0.002, 42.35),
        "crs": "EPSG:4326",
        "target_layer": None,
    },
    "layer_reproject_to_transformed_dem": {
        "keys": [
            "crs",
            "data",
            "original_crs",
            "original_data",
            "original_transform",
            "target_layer",
            "transform",
            "transformed",
        ],
        "shape": (48, 43),
        "dtype": "float32",
        "data": "39b908baa4f15912",
        "transform": (95.210039706, 0.0, 331046.948210094, 0.0, -95.210039706, 4690672.815281517),
        "crs": "EPSG:32617",
        "target_layer": "dem",
    },
    "layer_resample_to_dem": {
        "keys": [
            "crs",
            "data",
            "original_crs",
            "original_data",
            "original_transform",
            "target_layer",
            "transform",
            "transformed",
        ],
        "shape": (40, 48),
        "dtype": "float32",
        "data": "a3d212fed720ed87",
        "transform": (0.001, 0.0, -83.05, 0.0, -0.001, 42.35),
        "crs": "EPSG:4326",
        "target_layer": "dem",
    },
    "layer_same_extent_as": {
        "keys": [
            "crs",
            "data",
            "original_crs",
            "original_data",
            "original_transform",
            "target_layer",
            "transform",
            "transformed",
        ],
        "shape": (40, 48),
        "dtype": "float32",
        "data": "a3d212fed720ed87",
        "transform": (0.001, 0.0, -83.05, 0.0, -0.001, 42.35),
        "crs": "EPSG:4326",
        "target_layer": "dem",
    },
    "layer_same_extent_as_transformed": {
        "keys": [
            "crs",
            "data",
            "original_crs",
            "original_data",
            "original_transform",
            "target_layer",
            "transform",
            "transformed",
        ],
        "shape": (48, 43),
        "dtype": "float32",
        "data": "39b908baa4f15912",
        "transform": (95.210039706, 0.0, 331046.948210094, 0.0, -95.210039706, 4690672.815281517),
        "crs": "EPSG:32617",
        "target_layer": "dem",
    },
    "layer_same_grid": {
        "keys": ["crs", "data", "target_layer", "transform", "transformed"],
        "shape": (40, 48),
        "dtype": "float32",
        "data": "e4a2fbf6c4d23e38",
        "transform": (0.001, 0.0, -83.05, 0.0, -0.001, 42.35),
        "crs": "EPSG:4326",
        "target_layer": "dem",
    },
    "layer_target_crs_first_layer_grid": {
        "keys": [
            "crs",
            "data",
            "original_crs",
            "original_data",
            "original_transform",
            "target_layer",
            "transform",
            "transformed",
        ],
        "shape": (40, 48),
        "dtype": "float32",
        "data": "a3d212fed720ed87",
        "transform": (0.001, 0.0, -83.05, 0.0, -0.001, 42.35),
        "crs": "EPSG:4326",
        "target_layer": None,
    },
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_output_is_pinned(name):
    assert CASES[name]() == EXPECTED[name]
