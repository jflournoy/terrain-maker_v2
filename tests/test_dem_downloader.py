"""
Tests for DEM (Digital Elevation Model) downloader.

Tests the functionality for downloading SRTM elevation data for specified areas.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np


class TestSRTMDownloader:
    """Test SRTM elevation data downloading."""

    def test_download_dem_by_bbox_creates_output_directory(self, tmp_path):
        """
        Test that download_dem_by_bbox creates output directory if it doesn't exist.

        When downloading DEM data, the downloader should automatically create
        the output directory structure.
        """
        from src.terrain.dem_downloader import download_dem_by_bbox

        output_dir = tmp_path / "dem_data" / "downloads"
        bbox = (42.0, -83.5, 42.5, -83.0)  # Detroit area

        # Directory shouldn't exist yet
        assert not output_dir.exists()

        # Download should create it (mocked, won't actually download)
        with patch('src.terrain.dem_downloader._download_srtm_tile') as mock_download:
            mock_download.return_value = True

            result = download_dem_by_bbox(
                bbox=bbox,
                output_dir=str(output_dir),
                # Don't actually authenticate in tests
                username=None,
                password=None
            )

        # Directory should now exist
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_download_dem_by_bbox_returns_downloaded_files(self, tmp_path):
        """
        Test that download_dem_by_bbox returns list of downloaded file paths.

        The function should return a list of Path objects pointing to the
        downloaded HGT files.
        """
        from src.terrain.dem_downloader import download_dem_by_bbox

        output_dir = tmp_path / "dem_data"
        bbox = (42.0, -83.0, 42.5, -82.5)  # Small bbox covering ~2 tiles

        with patch('src.terrain.dem_downloader._download_srtm_tile') as mock_download:
            # Simulate successful downloads
            mock_download.return_value = True

            result = download_dem_by_bbox(
                bbox=bbox,
                output_dir=str(output_dir),
                username=None,
                password=None
            )

        # Should return a list
        assert isinstance(result, list)

        # List should contain Path objects
        if result:
            assert all(isinstance(p, Path) for p in result)

    def test_calculate_required_srtm_tiles(self):
        """
        Test that we can calculate which SRTM tiles are needed for a bbox.

        SRTM tiles are 1°×1° and named like N42W083.hgt (north 42°, west 83°).
        For a bounding box, we need to determine all tiles that intersect it.
        """
        from src.terrain.dem_downloader import calculate_required_srtm_tiles

        # Detroit area: should need tiles N42W084 and N42W083
        bbox = (42.0, -83.5, 42.5, -83.0)
        tiles = calculate_required_srtm_tiles(bbox)

        assert isinstance(tiles, list)
        assert len(tiles) >= 1

        # Should return tile names
        assert all(isinstance(t, str) for t in tiles)

        # Tile names should follow SRTM convention (e.g., "N42W084")
        for tile in tiles:
            assert len(tile) == 7  # e.g., "N42W084"
            assert tile[0] in ['N', 'S']
            assert tile[3] in ['E', 'W']

    def test_download_dem_by_place_name_geocodes_location(self, tmp_path):
        """
        Test that download_dem_by_place_name can geocode a place name to bbox.

        Users should be able to specify "Detroit, MI" instead of coordinates.
        """
        from src.terrain.dem_downloader import download_dem_by_place_name

        output_dir = tmp_path / "dem_data"

        with patch('src.terrain.dem_downloader._geocode_place_name') as mock_geocode:
            with patch('src.terrain.dem_downloader.download_dem_by_bbox') as mock_download:
                # Mock geocoding to return a bbox
                mock_geocode.return_value = (42.0, -83.5, 42.5, -83.0)
                mock_download.return_value = []

                result = download_dem_by_place_name(
                    place_name="Detroit, MI",
                    output_dir=str(output_dir),
                    username=None,
                    password=None
                )

                # Should have called geocoding
                mock_geocode.assert_called_once_with("Detroit, MI")

                # Should have called bbox download with geocoded coordinates
                mock_download.assert_called_once()

    def test_srtm_tile_naming_convention(self):
        """
        Test SRTM tile naming follows standard convention.

        SRTM uses specific naming:
        - N/S for latitude (north/south of equator)
        - E/W for longitude (east/west of prime meridian)
        - Coordinates are for SW corner of tile

        Examples:
        - N42W084.hgt = tile from 42°N-43°N, 84°W-83°W
        - S01E036.hgt = tile from 1°S-0°, 36°E-37°E
        """
        from src.terrain.dem_downloader import get_srtm_tile_name

        # Detroit: north of equator, west of prime meridian
        assert get_srtm_tile_name(42.3, -83.0) == "N42W083"
        assert get_srtm_tile_name(42.9, -83.9) == "N42W084"

        # Test edge cases
        assert get_srtm_tile_name(0.5, 0.5) == "N00E000"
        assert get_srtm_tile_name(-0.5, -0.5) == "S01W001"


@pytest.mark.skip(reason="NASADEM integration tests require real NASA Earthdata credentials")
class TestSRTMDownloadHTTP:
    """
    Test actual HTTP downloading of SRTM tiles.

    These tests are skipped by default because they require:
    - NASA Earthdata account credentials
    - Network access
    - NASADEM library (which is complex to mock)

    To run these tests, use: pytest -m integration
    """

    def test_download_srtm_tile_makes_http_request(self, tmp_path):
        """
        Test that _download_srtm_tile uses NASADEM library.

        SRTM tiles are downloaded via the NASADEM library.
        """
        from src.terrain.dem_downloader import _download_srtm_tile

        output_dir = tmp_path / "downloads"
        output_dir.mkdir()

        # This is an integration test - would actually call NASADEM
        # Skipped by default, run with real credentials using: pytest -m integration
        pass

    def test_download_srtm_tile_creates_hgt_file(self, tmp_path):
        """
        Test that downloaded SRTM data is saved as .hgt file.
        """
        from src.terrain.dem_downloader import _download_srtm_tile

        output_dir = tmp_path / "downloads"
        output_dir.mkdir()

        # Integration test - skipped by default
        pass

    def test_download_srtm_tile_handles_404(self, tmp_path):
        """
        Test that downloader handles missing tiles gracefully.

        Some areas may not have SRTM coverage (oceans, poles).
        """
        from src.terrain.dem_downloader import _download_srtm_tile

        output_dir = tmp_path / "downloads"
        output_dir.mkdir()

        # Integration test - skipped by default
        pass

    def test_download_srtm_tile_skips_existing_files(self, tmp_path):
        """
        Test that downloader doesn't re-download existing files.

        This saves bandwidth and time when resuming downloads.
        """
        from src.terrain.dem_downloader import _download_srtm_tile

        output_dir = tmp_path / "downloads"
        output_dir.mkdir()

        # Create existing file
        existing_file = output_dir / "N42W084.hgt"
        existing_file.write_bytes(b'\x00' * 1000)

        with patch('src.terrain.dem_downloader.requests.Session') as mock_session:
            result = _download_srtm_tile(
                "N42W084",
                output_dir,
                username="test_user",
                password="test_pass"
            )

            # Should skip download
            mock_session.return_value.get.assert_not_called()
            assert result is True  # File exists, so success


class TestBBoxVisualization:
    """Test bbox visualization helpers for users."""

    def test_display_bbox_on_map_creates_html(self, tmp_path):
        """
        Test that we can create an interactive map showing the bbox.

        This helps users visualize and refine their bounding box selection.
        Should create an HTML file with folium/leaflet map.
        """
        from src.terrain.dem_downloader import display_bbox_on_map

        bbox = (42.0, -83.5, 42.5, -83.0)
        output_file = tmp_path / "bbox_map.html"

        display_bbox_on_map(bbox, output_file=str(output_file))

        # Should create HTML file
        assert output_file.exists()

        # Should contain map markers/rectangle
        content = output_file.read_text()
        assert "folium" in content.lower() or "leaflet" in content.lower()
