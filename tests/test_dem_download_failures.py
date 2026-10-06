"""DEM downloads deliver every tile or say which ones failed and why.

download_dem_by_bbox used to drop any tile that failed (no library, no credentials, network
error) and return the rest, so a mosaic silently had holes; _geocode_place_name returned
Detroit's bbox for every place name.
"""

from unittest.mock import patch

import pytest

from terrain_maker.terrain import dem_downloader as dd

BBOX = (42.2, -83.8, 43.6, -82.6)  # spans N42/N43 x W084/W083: 4 tiles


def _fail_on(bad):
    def fake(tile, output_dir, username=None, password=None):
        if tile in bad:
            raise RuntimeError(f"HTTP 503 for {tile}")
        (output_dir / f"NASADEM_HGT_{tile.upper()}.zip").write_bytes(b"zip")
        return True
    return fake


def test_all_tiles_ok_returns_all(tmp_path):
    with patch.object(dd, "_download_srtm_tile", side_effect=_fail_on(set())):
        files = dd.download_dem_by_bbox(BBOX, str(tmp_path))
    assert len(files) == len(dd.calculate_required_srtm_tiles(BBOX)) == 4


def test_failed_tile_raises_naming_each_tile_and_reason(tmp_path):
    with patch.object(dd, "_download_srtm_tile", side_effect=_fail_on({"N42W084", "N43W083"})):
        with pytest.raises(RuntimeError) as err:
            dd.download_dem_by_bbox(BBOX, str(tmp_path))
    msg = str(err.value)
    assert "N42W084" in msg and "N43W083" in msg and "HTTP 503" in msg and "2 of 4" in msg


def test_expected_missing_tiles_are_allowed_and_only_those(tmp_path):
    with patch.object(dd, "_download_srtm_tile", side_effect=_fail_on({"N42W084"})):
        files = dd.download_dem_by_bbox(BBOX, str(tmp_path), expected_missing=["N42W084"])
    assert len(files) == 3
    with patch.object(dd, "_download_srtm_tile", side_effect=_fail_on({"N42W084", "N43W083"})):
        with pytest.raises(RuntimeError, match="N43W083"):
            dd.download_dem_by_bbox(BBOX, str(tmp_path / "b"), expected_missing=["N42W084"])


def test_tile_without_credentials_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "NASADEMConnection", object)
    monkeypatch.setattr(dd, "_load_earthdata_credentials", lambda u, p: (None, None))
    with pytest.raises(RuntimeError, match="credentials"):
        dd._download_srtm_tile("N42W084", tmp_path)


def test_tile_without_nasadem_library_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "NASADEMConnection", None)
    with pytest.raises(ImportError, match="NASADEM"):
        dd._download_srtm_tile("N42W084", tmp_path)


def test_geocoding_is_not_faked():
    with pytest.raises(NotImplementedError, match="Seville"):
        dd._geocode_place_name("Seville, Spain")
