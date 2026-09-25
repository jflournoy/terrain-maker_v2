"""
Characterization tests for Terrain.create_mesh.

Pins the stored mesh (vertices, faces), vertex colors after water coloring,
boundary colors and model offset, so create_mesh can be split into steps
without changing results. Update EXPECTED only for an intended change.
"""

import hashlib

import numpy as np
import pytest
from rasterio import Affine

pytest.importorskip("bpy")

from terrain_maker.terrain.color_mapping import elevation_colormap  # noqa: E402
from terrain_maker.terrain.core import Terrain  # noqa: E402


def _dem():
    yy, xx = np.mgrid[0:30, 0:30]
    dem = (100 + 5 * np.sin(xx / 5) + 3 * np.cos(yy / 4)).astype(np.float32)
    dem[5:14, 6:16] = 95.0  # flat lake for slope-based water detection
    dem[(yy - 15) ** 2 + (xx - 15) ** 2 > 14**2] = np.nan
    return dem


def _lake_mask(shape):
    mask = np.zeros(shape, dtype=bool)
    mask[18:26, 10:20] = True
    return mask


CONFIGS = {
    "default": dict(),
    "detect_water": dict(detect_water=True),
    "given_water_mask": dict(water_mask="same"),
    "resampled_water_mask": dict(water_mask="half"),
    "not_centered_morph": dict(center_model=False, use_rectangle_edges=False),
    "no_extension": dict(boundary_extension=False),
    "single_tier_morph": dict(two_tier_edge=False, use_rectangle_edges=False),
}


def _digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _run(name):
    dem = _dem()
    terrain = Terrain(dem, Affine(0.001, 0, -83.0, 0, -0.001, 42.5))
    terrain.transforms.append(lambda data, trans: (data, trans, None))
    terrain.apply_transforms()
    terrain.set_color_mapping(elevation_colormap, source_layers=["dem"])

    kwargs = dict(CONFIGS[name])
    if kwargs.get("water_mask") == "same":
        kwargs["water_mask"] = _lake_mask(dem.shape)
    elif kwargs.get("water_mask") == "half":
        kwargs["water_mask"] = _lake_mask(dem.shape)[::2, ::2]
    terrain.create_mesh(verbose=False, **kwargs)

    boundary_colors = getattr(terrain, "boundary_colors", None)
    return {
        "n_vertices": len(terrain.vertices),
        "n_faces": len(terrain.faces),
        "vertices": _digest(np.round(terrain.vertices, 6)),
        "faces": _digest(
            np.array([len(f) for f in terrain.faces] + [i for f in terrain.faces for i in f])
        ),
        "colors": _digest(terrain.colors),
        "boundary_colors": None if boundary_colors is None else _digest(boundary_colors),
        "model_offset": [round(float(v), 6) for v in terrain.model_offset],
    }


EXPECTED = {
    "default": {
        "n_vertices": 1690,
        "n_faces": 1310,
        "vertices": "cc1539a6c2b028e4",
        "faces": "15190cb153f26896",
        "colors": "eb0867c8094edcd7",
        "boundary_colors": "f93c9730962aa114",
        "model_offset": [0.15, 0.15, 98.967953],
    },
    "detect_water": {
        "n_vertices": 1690,
        "n_faces": 1310,
        "vertices": "cc1539a6c2b028e4",
        "faces": "15190cb153f26896",
        "colors": "09f14b19db0ff16c",
        "boundary_colors": "f93c9730962aa114",
        "model_offset": [0.15, 0.15, 98.967953],
    },
    "given_water_mask": {
        "n_vertices": 1690,
        "n_faces": 1310,
        "vertices": "cc1539a6c2b028e4",
        "faces": "15190cb153f26896",
        "colors": "f99dfa2286adf119",
        "boundary_colors": "f93c9730962aa114",
        "model_offset": [0.15, 0.15, 98.967953],
    },
    "no_extension": {
        "n_vertices": 613,
        "n_faces": 592,
        "vertices": "570819b97f098ec6",
        "faces": "c78c92f9605600f2",
        "colors": "eb0867c8094edcd7",
        "boundary_colors": None,
        "model_offset": [0.15, 0.15, 98.967953],
    },
    "not_centered_morph": {
        "n_vertices": 765,
        "n_faces": 744,
        "vertices": "143c3fbd200e19bc",
        "faces": "43e916a08f768636",
        "colors": "eb0867c8094edcd7",
        "boundary_colors": "0fcb600a21103371",
        "model_offset": [0.0, 0.0, 0.0],
    },
    "resampled_water_mask": {
        "n_vertices": 1690,
        "n_faces": 1310,
        "vertices": "cc1539a6c2b028e4",
        "faces": "15190cb153f26896",
        "colors": "f99dfa2286adf119",
        "boundary_colors": "f93c9730962aa114",
        "model_offset": [0.15, 0.15, 98.967953],
    },
    "single_tier_morph": {
        "n_vertices": 689,
        "n_faces": 668,
        "vertices": "f00190cf58204f33",
        "faces": "fbf99ed9f5f11632",
        "colors": "eb0867c8094edcd7",
        "boundary_colors": None,
        "model_offset": [0.15, 0.15, 98.967953],
    },
}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_create_mesh_output_is_pinned(name):
    assert _run(name) == EXPECTED[name]
