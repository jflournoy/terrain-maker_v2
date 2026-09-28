"""build_mesh produces the full mesh without Blender; colors map to vertices by dtype."""

import numpy as np
from rasterio import Affine

from terrain_maker.terrain.color_mapping import elevation_colormap
from terrain_maker.terrain.core import Terrain
from terrain_maker.terrain.mesh_operations import MeshData, vertex_colors_rgba


def _terrain():
    yy, xx = np.mgrid[0:20, 0:25]
    dem = (100 + 5 * np.sin(xx / 4) + 2 * np.cos(yy / 3)).astype(np.float32)
    t = Terrain(dem, Affine(30, 0, 320000, 0, -30, 4700000), dem_crs="EPSG:32617")
    t.transforms.append(lambda data, trans: (data, trans, None))
    t.apply_transforms()
    t.set_color_mapping(elevation_colormap, ["dem"])
    return t


def test_build_mesh_returns_mesh_data_without_blender_object():
    terrain = _terrain()
    mesh = terrain.build_mesh(verbose=False)
    assert isinstance(mesh, MeshData)
    assert mesh.vertices is terrain.vertices and mesh.faces is terrain.faces
    assert mesh.colors.shape == (20, 25, 4)
    assert len(mesh.boundary_colors) == len(mesh.vertices) - len(mesh.y_valid)
    assert not hasattr(terrain, "terrain_obj")


def test_vertex_colors_cover_surface_and_skirt():
    mesh = _terrain().build_mesh(verbose=False)
    colors = mesh.vertex_colors()
    assert colors.shape == (len(mesh.vertices), 4) and colors.dtype == np.float32
    n = len(mesh.y_valid)
    expected_surface = mesh.colors[mesh.y_valid, mesh.x_valid] / 255.0
    np.testing.assert_allclose(colors[:n], expected_surface, rtol=1e-6)
    np.testing.assert_allclose(colors[n:, :3], mesh.boundary_colors / 255.0, rtol=1e-6)


def test_dark_uint8_colors_are_scaled_by_dtype_not_value():
    # All values <= 1 in a uint8 image still mean 0-255 (previously left unscaled)
    grid = np.ones((2, 2, 3), dtype=np.uint8)
    colors = vertex_colors_rgba(4, grid, np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    np.testing.assert_allclose(colors[:, :3], 1 / 255.0)
    np.testing.assert_allclose(colors[:, 3], 1.0)


def test_float_colors_in_unit_range_are_kept():
    grid = np.full((1, 2, 4), 0.5, dtype=np.float32)
    colors = vertex_colors_rgba(3, grid, np.array([0, 0]), np.array([0, 1]))
    np.testing.assert_allclose(colors[:2], 0.5)
    np.testing.assert_allclose(colors[2], 1.0)  # uncolored vertex is white
