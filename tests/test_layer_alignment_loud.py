"""Aligning a layer to its target either succeeds or raises; it never leaves a misaligned layer.

_align_layers_to_targets used to warn and skip on a missing or untransformed target, skip
whenever shapes matched (even with different georeferencing), log and continue when the
reprojection failed, and fill cells outside the source with 0 instead of nodata.
"""

import numpy as np
import pytest
from rasterio.transform import from_origin

from terrain_maker.terrain.core import Terrain

CRS = "EPSG:4326"


def _terrain(tmp_path, layer_transform, layer_data, target="dem"):
    dem = np.zeros((10, 10), dtype=np.float32)
    terrain = Terrain(dem, from_origin(0.0, 10.0, 1.0, 1.0), cache_dir=str(tmp_path))
    terrain.data_layers["dem"].update(
        transformed=True, transformed_data=dem, transformed_transform=from_origin(0.0, 10.0, 1.0, 1.0),
        transformed_crs=CRS)
    terrain.data_layers["scores"] = {
        "data": layer_data, "transform": layer_transform, "crs": CRS, "target_layer": target,
        "transformed": True, "transformed_data": layer_data,
        "transformed_transform": layer_transform, "transformed_crs": CRS,
    }
    return terrain


def test_missing_target_raises(tmp_path):
    t = _terrain(tmp_path, from_origin(0.0, 10.0, 1.0, 1.0), np.ones((10, 10), np.float32), target="nope")
    with pytest.raises(ValueError, match="nope"):
        t._align_layers_to_targets()


def test_untransformed_target_raises(tmp_path):
    t = _terrain(tmp_path, from_origin(0.0, 10.0, 1.0, 1.0), np.ones((5, 5), np.float32))
    t.data_layers["dem"]["transformed"] = False
    with pytest.raises(RuntimeError, match="not been transformed"):
        t._align_layers_to_targets()


def test_reprojection_failure_raises(tmp_path, monkeypatch):
    import rasterio.warp

    t = _terrain(tmp_path, from_origin(0.0, 10.0, 2.0, 2.0), np.ones((5, 5), np.float32))
    monkeypatch.setattr(rasterio.warp, "reproject", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(RuntimeError, match="scores.*boom"):
        t._align_layers_to_targets()


def test_same_shape_different_georeferencing_is_realigned(tmp_path):
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    shifted = from_origin(2.0, 10.0, 1.0, 1.0)  # same shape, two columns east of the DEM
    t = _terrain(tmp_path, shifted, data)
    t._align_layers_to_targets()
    aligned = t.data_layers["scores"]["transformed_data"]
    assert np.isnan(aligned[:, :2]).all()  # DEM columns 0-1 lie west of the layer
    np.testing.assert_allclose(aligned[:, 2:], data[:, :8])


def test_cells_outside_source_are_nan_not_zero(tmp_path):
    small = np.full((5, 5), 7.0, dtype=np.float32)  # covers only the DEM's north-west quarter
    t = _terrain(tmp_path, from_origin(0.0, 10.0, 1.0, 1.0), small)
    t._align_layers_to_targets()
    aligned = t.data_layers["scores"]["transformed_data"]
    assert np.isnan(aligned[6:, 6:]).all()
