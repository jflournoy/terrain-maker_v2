"""Transform-aware edges map every edge pixel or raise; failed transforms are not dropped.

pyproj reports a failed transform as inf; the integer variant then threw inside a broad
except and the pixel was silently counted and dropped, and the fractional variant passed inf
into the mesh. A DEM layer without a CRS was assumed to be EPSG:4326.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.transform import from_origin

from terrain_maker.terrain.mesh import rectangle_edges as re_


def _terrain(crs="EPSG:4326", transformed_crs="EPSG:32617"):
    tf = from_origin(-83.5, 42.5, 0.01, 0.01)
    layer = {"data": np.zeros((10, 10)), "transform": tf, "crs": crs, "transformed": True,
             "transformed_data": np.zeros((10, 10)), "transformed_transform": tf,
             "transformed_crs": transformed_crs}
    return SimpleNamespace(dem_shape=(10, 10), dem_transform=tf, data_layers={"dem": layer})


class _InfTransformer:
    @staticmethod
    def from_crs(*a, **k):
        return _InfTransformer()

    def transform(self, x, y):
        return float("inf"), float("inf")


@pytest.mark.parametrize("fn", [
    lambda t: re_.generate_transform_aware_rectangle_edges(t, coord_to_index={}),
    re_.generate_transform_aware_rectangle_edges_fractional,
], ids=["integer", "fractional"])
def test_failed_transform_raises(fn, monkeypatch):
    import pyproj

    monkeypatch.setattr(pyproj, "Transformer", _InfTransformer)
    with pytest.raises(ValueError, match="Edge pixel"):
        fn(_terrain())


@pytest.mark.parametrize("fn", [
    lambda t: re_.generate_transform_aware_rectangle_edges(t, coord_to_index={}),
    re_.generate_transform_aware_rectangle_edges_fractional,
], ids=["integer", "fractional"])
def test_missing_crs_raises(fn):
    with pytest.raises(ValueError, match="crs"):
        fn(_terrain(crs=None))
