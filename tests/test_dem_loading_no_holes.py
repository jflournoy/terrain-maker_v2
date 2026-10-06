"""DEM loaders read every tile they find, or raise naming the ones they could not.

load_dem_files skipped unopenable files, files with an unexpected dtype and bad ZIPs, then
merged the rest: a mosaic with holes and only a log line. load_filtered_hgt_files skipped
unparseable names silently. load_geotiff_cropped_to_dem fell back to a full read when its
windowed read failed.
"""

import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from terrain_maker.terrain import data_loading as dl


def _tile(path, lat, lon, dtype="int16"):
    data = np.full((11, 11), 100, dtype=dtype)
    with rasterio.open(path, "w", driver="GTiff", height=11, width=11, count=1, dtype=dtype,
                       crs="EPSG:4326", transform=from_origin(lon, lat + 1, 0.1, 0.1)) as dst:
        dst.write(data, 1)


def test_good_tiles_load(tmp_path):
    _tile(tmp_path / "N42W084.tif", 42, -84)
    _tile(tmp_path / "N42W083.tif", 42, -83)
    dem, _ = dl.load_dem_files(str(tmp_path), pattern="*.tif")
    assert dem.shape[1] > 11


def test_unopenable_tile_raises_naming_it(tmp_path):
    _tile(tmp_path / "N42W084.tif", 42, -84)
    (tmp_path / "N42W083.tif").write_bytes(b"not a raster")
    with pytest.raises(ValueError, match="N42W083"):
        dl.load_dem_files(str(tmp_path), pattern="*.tif")


def test_unexpected_dtype_raises(tmp_path):
    _tile(tmp_path / "N42W084.tif", 42, -84)
    _tile(tmp_path / "N42W083.tif", 42, -83, dtype="uint8")
    with pytest.raises(ValueError, match="uint8"):
        dl.load_dem_files(str(tmp_path), pattern="*.tif")


def test_bad_zip_raises(tmp_path):
    (tmp_path / "NASADEM_HGT_N42W084.zip").write_bytes(b"not a zip")
    with zipfile.ZipFile(tmp_path / "NASADEM_HGT_N42W083.zip", "w") as zf:
        zf.writestr("n42w083.hgt", b"\x00" * 10)
    with pytest.raises(ValueError, match="N42W084"):
        dl._extract_dem_from_zips(tmp_path, "*.hgt")


def test_filtered_loader_rejects_unparseable_names(tmp_path):
    _tile(tmp_path / "N42W084.hgt", 42, -84)
    _tile(tmp_path / "mystery.hgt", 42, -83)
    with pytest.raises(ValueError, match="mystery"):
        dl.load_filtered_hgt_files(tmp_path, min_latitude=40)


def test_filtered_loader_raises_on_unopenable(tmp_path):
    _tile(tmp_path / "N42W084.hgt", 42, -84)
    (tmp_path / "N43W084.hgt").write_bytes(b"junk")
    with pytest.raises(ValueError, match="N43W084"):
        dl.load_filtered_hgt_files(tmp_path, min_latitude=40)


def test_windowed_read_failure_raises(tmp_path, monkeypatch):
    path = tmp_path / "precip.tif"
    _tile(path, 42, -84, dtype="float32")
    import rasterio.windows

    monkeypatch.setattr(rasterio.windows, "from_bounds",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad window")))
    with pytest.raises(RuntimeError, match="bad window"):
        dl.load_geotiff_cropped_to_dem(path, (5, 5), from_origin(-84, 43, 0.2, 0.2), "EPSG:4326")
