"""Lake loading returns real lakes or raises; an empty set means the area has no lakes.

download_hydrolakes returned an empty FeatureCollection when the shapefile or geopandas was
missing, and download_water_bodies cached it, so every later run also had no lakes.
rasterize_lakes_to_mask skipped features whose geometry failed to parse, with no log.
"""

import sys

import pytest

from terrain_maker.terrain import water_bodies as wb

BBOX = (42.0, -83.5, 42.5, -83.0)


def test_missing_hydrolakes_shapefile_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # so data/hydrolakes/... is not found either
    with pytest.raises(FileNotFoundError, match="hydrosheds"):
        wb.download_hydrolakes(BBOX, str(tmp_path))


def test_missing_shapefile_is_not_cached_as_no_lakes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        wb.download_water_bodies(BBOX, str(tmp_path), data_source="hydrolakes")
    assert not list(tmp_path.glob("*.geojson"))


def test_missing_geopandas_raises(tmp_path, monkeypatch):
    shp = tmp_path / "HydroLAKES_polys_v10.shp"
    shp.write_bytes(b"")
    monkeypatch.setitem(sys.modules, "geopandas", None)
    with pytest.raises(ImportError, match="geopandas"):
        wb.download_hydrolakes(BBOX, str(tmp_path))


def test_invalid_lake_geometry_raises_naming_the_feature():
    lakes = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[-83.4, 42.1], [-83.3, 42.1],
                                                                               [-83.3, 42.2], [-83.4, 42.1]]]},
         "properties": {}},
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": "garbage"}, "properties": {}},
    ]}
    with pytest.raises(ValueError, match="feature 2"):
        wb.rasterize_lakes_to_mask(lakes, BBOX, 0.01)
