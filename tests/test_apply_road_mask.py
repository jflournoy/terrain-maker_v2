"""apply_road_mask marks road vertices or raises; it never returns without the layer.

It used to return (with a warning only if a logger was passed) when the layer could not be
created or the mesh had no loops, read 0 for vertices outside the mask, and clamped loops of
vertices past the surface grid (the skirt) onto the last surface vertex's road value.
"""

import bpy
import numpy as np
import pytest

from terrain_maker.terrain.blender_integration import apply_road_mask


def _mesh():
    """3x3 surface grid (vertices 0-8) plus one skirt quad below it (vertices 9-12)."""
    verts = [(x, y, 0.0) for y in range(3) for x in range(3)]
    verts += [(0, 0, -1.0), (2, 0, -1.0), (2, 2, -1.0), (0, 2, -1.0)]
    faces = [(0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7), (9, 10, 11, 12)]
    mesh = bpy.data.meshes.new("road_mask_test")
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new("road_mask_test", mesh)
    y_valid = np.repeat(np.arange(3), 3)
    x_valid = np.tile(np.arange(3), 3)
    return obj, y_valid, x_valid


def _vertex_red(obj):
    layer = obj.data.vertex_colors["RoadMask"]
    colors = np.zeros(len(layer.data) * 4, dtype=np.float32)
    layer.data.foreach_get("color", colors)
    loop_vertex = np.zeros(len(layer.data), dtype=np.int32)
    obj.data.loops.foreach_get("vertex_index", loop_vertex)
    return {int(v): float(r) for v, r in zip(loop_vertex, colors.reshape(-1, 4)[:, 0])}


def test_marks_road_vertices():
    obj, y, x = _mesh()
    mask = np.zeros((3, 3))
    mask[1, :] = 1.0
    apply_road_mask(obj, mask, y, x)
    red = _vertex_red(obj)
    assert [red[v] for v in (3, 4, 5)] == [1.0, 1.0, 1.0]
    assert red[0] == 0.0 and red[8] == 0.0


def test_skirt_vertices_are_not_road_even_if_last_surface_vertex_is():
    obj, y, x = _mesh()
    mask = np.zeros((3, 3))
    mask[2, 2] = 1.0  # vertex 8, the last surface vertex
    apply_road_mask(obj, mask, y, x)
    red = _vertex_red(obj)
    assert red[8] == 1.0
    assert all(red[v] == 0.0 for v in (9, 10, 11, 12))


def test_mask_not_covering_the_grid_raises():
    obj, y, x = _mesh()
    with pytest.raises(ValueError, match="outside the road mask"):
        apply_road_mask(obj, np.zeros((2, 3)), y, x)


def test_mesh_without_loops_raises():
    mesh = bpy.data.meshes.new("empty_mask_test")
    obj = bpy.data.objects.new("empty_mask_test", mesh)
    with pytest.raises(ValueError, match="no faces"):
        apply_road_mask(obj, np.zeros((3, 3)), np.array([], int), np.array([], int))


def test_vertex_colors_leave_skirt_vertices_alone():
    """apply_vertex_colors used to clamp skirt vertices onto the last surface vertex's color."""
    from terrain_maker.terrain.blender_integration import apply_vertex_colors

    obj, y, x = _mesh()
    layer = obj.data.vertex_colors.new(name="TerrainColors")
    preset = np.tile([0.1, 0.2, 0.3, 1.0], len(layer.data)).astype(np.float32)
    layer.data.foreach_set("color", preset)  # what boundary coloring left on the skirt

    colors = np.zeros((3, 3, 4), dtype=np.float32)
    colors[..., 3] = 1.0
    colors[2, 2] = [0.9, 0.0, 0.0, 1.0]  # last surface vertex is red
    apply_vertex_colors(obj, colors, y, x)

    got = np.zeros(len(layer.data) * 4, dtype=np.float32)
    layer.data.foreach_get("color", got)
    loop_vertex = np.zeros(len(layer.data), dtype=np.int32)
    obj.data.loops.foreach_get("vertex_index", loop_vertex)
    by_vertex = {int(v): c for v, c in zip(loop_vertex, got.reshape(-1, 4))}
    byte = 1.5 / 255  # Blender stores vertex colors as bytes
    assert by_vertex[8][0] == pytest.approx(0.9, abs=byte)
    for v in (9, 10, 11, 12):
        np.testing.assert_allclose(by_vertex[v], [0.1, 0.2, 0.3, 1.0], atol=byte)


def test_vertex_colors_for_more_vertices_than_the_mesh_has_raise():
    from terrain_maker.terrain.blender_integration import apply_vertex_colors

    obj, _, _ = _mesh()
    with pytest.raises(ValueError, match="vertices"):
        apply_vertex_colors(obj, np.ones((20, 4), dtype=np.float32))
