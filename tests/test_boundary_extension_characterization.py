"""
Characterization tests for create_boundary_extension.

Pins the skirt geometry (vertices, faces, colors) that Terrain.create_mesh
produces for each edge mode, so the 1,100-line function can be split into
smaller steps without changing its output. Update EXPECTED only for an
intended behavior change.
"""

import hashlib

import numpy as np
import pytest
from rasterio import Affine

pytest.importorskip("bpy")

from terrain_maker.terrain import mesh_operations  # noqa: E402
from terrain_maker.terrain.color_mapping import elevation_colormap  # noqa: E402
from terrain_maker.terrain.core import Terrain  # noqa: E402


def _dem(irregular=False):
    yy, xx = np.mgrid[0:30, 0:30]
    dem = (100 + 5 * np.sin(xx / 5) + 3 * np.cos(yy / 4)).astype(np.float32)
    if irregular:
        dem[(yy - 15) ** 2 + (xx - 15) ** 2 > 13**2] = np.nan
    return dem


CONFIGS = {
    "rect_fractional_two_tier": dict(),
    "rect_integer_two_tier": dict(use_fractional_edges=False),
    "morph_two_tier": dict(use_rectangle_edges=False, use_fractional_edges=False),
    "morph_single_tier": dict(
        use_rectangle_edges=False, use_fractional_edges=False, two_tier_edge=False
    ),
    "morph_smoothed": dict(
        use_rectangle_edges=False, use_fractional_edges=False, smooth_boundary=True
    ),
    "morph_catmull_rom": dict(
        use_rectangle_edges=False,
        use_fractional_edges=False,
        use_catmull_rom=True,
        catmull_rom_subdivisions=3,
    ),
    "morph_blended_colors": dict(
        use_rectangle_edges=False, use_fractional_edges=False, edge_blend_colors=True
    ),
    "morph_irregular": dict(use_rectangle_edges=False, use_fractional_edges=False),
    "rect_blended_colors": dict(edge_blend_colors=True),
}


def _digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _run(name, monkeypatch):
    captured = {}
    original = mesh_operations.create_boundary_extension

    def recording(*args, **kwargs):
        result = original(*args, **kwargs)
        captured["result"] = result
        return result

    monkeypatch.setattr(mesh_operations, "create_boundary_extension", recording)

    terrain = Terrain(
        _dem(irregular=name == "morph_irregular"), Affine(0.001, 0, -83.0, 0, -0.001, 42.5)
    )
    terrain.transforms.append(lambda data, trans: (data, trans, None))
    terrain.apply_transforms()
    terrain.set_color_mapping(elevation_colormap, source_layers=["dem"])
    terrain.compute_colors()
    terrain.create_mesh(boundary_extension=True, verbose=False, **CONFIGS[name])

    result = captured["result"]
    vertices, faces = np.asarray(result[0]), np.asarray(result[1], dtype=np.int64)
    summary = {
        "n_vertices": len(vertices),
        "n_faces": len(faces),
        "vertices": _digest(np.round(vertices, 6)),
        "faces": _digest(faces),
    }
    if len(result) == 3:
        summary["colors"] = _digest(np.asarray(result[2]))
    return summary


EXPECTED = {
    "morph_blended_colors": {
        "n_vertices": 232,
        "n_faces": 232,
        "vertices": "4b3efdc78ded8c49",
        "faces": "a56b15a7f7bfc855",
        "colors": "ca84fa5f774dd1d5",
    },
    "morph_catmull_rom": {
        "n_vertices": 1044,
        "n_faces": 696,
        "vertices": "2a28f338342dbbff",
        "faces": "1142b1582d7a1e67",
        "colors": "53f06d1567901d3e",
    },
    "morph_irregular": {
        "n_vertices": 144,
        "n_faces": 144,
        "vertices": "50a15815893445c5",
        "faces": "4baa641178294a1a",
        "colors": "73082060cda6dbaf",
    },
    "morph_single_tier": {
        "n_vertices": 116,
        "n_faces": 116,
        "vertices": "8fe3631fa446f432",
        "faces": "0f2f0bb5a660cabc",
    },
    "morph_smoothed": {
        "n_vertices": 348,
        "n_faces": 348,
        "vertices": "9c26e83a0a7d9b42",
        "faces": "83adb4ed1f4887ad",
        "colors": "f94484cad170ac8b",
    },
    "morph_two_tier": {
        "n_vertices": 232,
        "n_faces": 232,
        "vertices": "4b3efdc78ded8c49",
        "faces": "a56b15a7f7bfc855",
        "colors": "5b8ad2356ad9c637",
    },
    "rect_blended_colors": {
        "n_vertices": 1077,
        "n_faces": 718,
        "vertices": "f542060c3d4ab95e",
        "faces": "0346789aef758a09",
        "colors": "84023a442ec7e2d1",
    },
    "rect_fractional_two_tier": {
        "n_vertices": 1077,
        "n_faces": 718,
        "vertices": "f542060c3d4ab95e",
        "faces": "0346789aef758a09",
        "colors": "f93c9730962aa114",
    },
    "rect_integer_two_tier": {
        "n_vertices": 232,
        "n_faces": 232,
        "vertices": "d23452107ca7c009",
        "faces": "833e2e926943aac1",
        "colors": "5b8ad2356ad9c637",
    },
}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_skirt_output_is_pinned(name, monkeypatch):
    summary = _run(name, monkeypatch)
    assert summary == EXPECTED[name]
