"""Reading the PRISM normals archive: BIL or GeoTIFF inside, nodata masked.

The PRISM server's zip holds an ESRI BIL raster (.bil + .hdr). The reader only looked for
.tif and never masked nodata, so PRISM's -9999 ocean cells would have become precipitation.
No network: the archive is built locally and served through a patched requests.get.
"""

import io
import zipfile
from unittest.mock import Mock

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from terrain_maker.terrain.precipitation_downloader import download_real_prism_annual

BBOX = (32.5, -117.6, 33.5, -116.0)  # min_lat, min_lon, max_lat, max_lon
TRANSFORM = from_origin(-118.0, 34.0, 0.1, 0.1)  # covers 34.0..32.0 N, -118.0..-116.0 W
SHAPE = (20, 20)


def _raster(tmp_path, driver, suffix):
    data = np.full(SHAPE, 400.0, dtype=np.float32)
    data[:, :5] = -9999.0  # ocean, west edge
    path = tmp_path / f"PRISM_ppt_30yr_normal_4kmM4_annual{suffix}"
    with rasterio.open(path, "w", driver=driver, height=SHAPE[0], width=SHAPE[1], count=1,
                       dtype="float32", transform=TRANSFORM, nodata=-9999.0) as dst:
        dst.write(data, 1)
    return path


def _serve_zip(monkeypatch, files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for path in files:
            zf.write(path, path.name)
    response = Mock(status_code=200, content=buffer.getvalue())
    monkeypatch.setattr("requests.get", lambda url, *a, **k: response)


def _members(path):
    return sorted(p for p in path.parent.iterdir() if p.stem == path.stem and p.suffix != ".xml")


@pytest.mark.parametrize("driver,suffix", [("EHdr", ".bil"), ("GTiff", ".tif")])
def test_reads_archive_and_masks_nodata(tmp_path, monkeypatch, driver, suffix):
    src_dir, out_dir = tmp_path / "src", tmp_path / "out"
    src_dir.mkdir()
    raster = _raster(src_dir, driver, suffix)
    _serve_zip(monkeypatch, _members(raster))

    data, transform = download_real_prism_annual(BBOX, output_dir=str(out_dir))

    assert data.shape == (10, 16)  # 33.5..32.5 N by -117.6..-116.0 W at 0.1 degrees
    assert transform.c == pytest.approx(-117.6) and transform.f == pytest.approx(33.5)
    assert np.isnan(data[:, :1]).all()  # -117.6..-117.5 lies in the -9999 ocean strip
    assert np.nanmin(data) == 400.0


def test_archive_without_a_raster_raises(tmp_path, monkeypatch):
    readme = tmp_path / "readme.txt"
    readme.write_text("no raster here")
    _serve_zip(monkeypatch, [readme])
    with pytest.raises(Exception, match="no .bil or .tif raster"):
        download_real_prism_annual(BBOX, output_dir=str(tmp_path / "out"))
