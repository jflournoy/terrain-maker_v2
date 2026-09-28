"""Every color mode produces the same layout: an (H, W, 4) grid aligned with the DEM."""

import numpy as np
import pytest
from rasterio import Affine

from terrain_maker.terrain.color_mapping import elevation_colormap
from terrain_maker.terrain.core import Terrain

SHAPE = (30, 40)


@pytest.fixture
def terrain():
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    dem = (100 + 5 * np.sin(xx / 5) + 2 * np.cos(yy / 3)).astype(np.float32)
    t = Terrain(dem, Affine(30, 0, 320000, 0, -30, 4700000), dem_crs="EPSG:32617")
    t.transforms.append(lambda data, trans: (data, trans, None))
    t.apply_transforms()
    return t


def _mask():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[5:15, 10:30] = True
    return mask


def _viridis(values):
    return elevation_colormap(values, cmap_name="viridis")


MODES = {
    "standard": lambda t: t.set_color_mapping(elevation_colormap, ["dem"]),
    "blended": lambda t: t.set_blended_color_mapping(
        elevation_colormap, ["dem"], _viridis, ["dem"], _mask()
    ),
    "multi": lambda t: t.set_multi_color_mapping(
        elevation_colormap,
        ["dem"],
        [{"colormap": _viridis, "source_layers": ["dem"], "priority": 1, "mask": _mask()}],
    ),
}


@pytest.mark.parametrize("mode", sorted(MODES))
def test_every_mode_returns_rgba_grid_before_mesh(terrain, mode):
    MODES[mode](terrain)
    colors = terrain.compute_colors()
    assert colors.shape == SHAPE + (4,)
    assert terrain.colors is colors


@pytest.mark.parametrize("mode", sorted(MODES))
def test_water_mask_is_applied_in_every_mode(terrain, mode):
    MODES[mode](terrain)
    dry = terrain.compute_colors().copy()
    water = np.zeros(SHAPE, dtype=bool)
    water[20:28, 5:20] = True

    wet = terrain.compute_colors(water_mask=water)

    assert not np.array_equal(wet[water, :3], dry[water, :3])
    np.testing.assert_array_equal(wet[~water], dry[~water])
    assert np.all(wet[water, 2] > wet[water, 0])  # blue dominates red


def test_blend_uses_overlay_inside_mask_only(terrain):
    MODES["blended"](terrain)
    colors = terrain.compute_colors()
    base = elevation_colormap(terrain._layer_array("dem"))
    overlay = _viridis(terrain._layer_array("dem"))
    np.testing.assert_array_equal(colors[_mask(), :3], overlay[_mask()][:, :3])
    np.testing.assert_array_equal(colors[~_mask(), :3], base[~_mask()][:, :3])


def test_vertex_mask_without_mesh_raises(terrain):
    terrain.set_blended_color_mapping(
        elevation_colormap, ["dem"], _viridis, ["dem"], np.ones(17, dtype=bool)
    )
    with pytest.raises(ValueError, match="create_mesh"):
        terrain.compute_colors()


def test_multi_overlay_mask_of_wrong_shape_is_an_error_not_a_threshold(terrain):
    terrain.set_multi_color_mapping(
        elevation_colormap,
        ["dem"],
        [{"colormap": _viridis, "source_layers": ["dem"], "priority": 1, "mask": np.ones((3, 3))}],
    )
    with pytest.raises(ValueError, match="mask"):
        terrain.compute_colors()
