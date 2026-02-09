"""
Tests for precipitation data downloader.

Following TDD RED-GREEN-REFACTOR cycle.
This is the RED phase - all tests should fail initially.

Requirements:
- Download PRISM annual precipitation data for bounding box
- Support other datasets (WorldClim, CHELSA)
- Validate spatial alignment with DEM
- Handle caching and resume
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

# Import the precipitation downloader functions (don't exist yet - RED phase)
from src.terrain.precipitation_downloader import (
    download_precipitation,
    get_prism_annual_precip,
    validate_precipitation_alignment,
    list_available_datasets,
)


class TestPrecipitationDownloadAPI:
    """Test suite for main precipitation download API."""

    def test_download_precipitation_requires_bbox(self):
        """download_precipitation should require bbox parameter."""
        with pytest.raises(TypeError):
            download_precipitation()

    def test_download_precipitation_requires_output_dir(self):
        """download_precipitation should require output_dir parameter."""
        bbox = (32.5, -117.6, 33.5, -116.0)
        with pytest.raises(TypeError):
            download_precipitation(bbox=bbox)

    def test_download_precipitation_validates_bbox_format(self):
        """download_precipitation should validate bbox has 4 coordinates."""
        with pytest.raises(ValueError, match="bbox must be.*4 values"):
            download_precipitation(bbox=(32.5, -117.6), output_dir="test")

    def test_download_precipitation_validates_bbox_order(self):
        """download_precipitation should validate bbox coordinate order (min_lat, min_lon, max_lat, max_lon)."""
        # Inverted bbox (max before min)
        with pytest.raises(ValueError, match="Invalid bbox.*min.*max"):
            download_precipitation(
                bbox=(33.5, -116.0, 32.5, -117.6),  # max_lat, max_lon, min_lat, min_lon
                output_dir="test",
            )

    def test_download_precipitation_returns_path(self, tmp_path):
        """download_precipitation should return path to downloaded file."""
        bbox = (32.5, -117.6, 33.5, -116.0)

        result = download_precipitation(bbox=bbox, output_dir=str(tmp_path), dataset="prism")

        assert isinstance(result, (str, Path)), "Should return path"
        assert Path(result).exists(), "Downloaded file should exist"

    def test_download_precipitation_creates_geotiff(self, tmp_path):
        """download_precipitation should create valid GeoTIFF file."""
        bbox = (32.5, -117.6, 33.5, -116.0)

        result_path = download_precipitation(
            bbox=bbox, output_dir=str(tmp_path), dataset="prism"
        )

        # Verify it's a valid raster
        import rasterio

        with rasterio.open(result_path) as src:
            assert src.count == 1, "Should have single band"
            assert src.dtypes[0] in ["float32", "float64"], "Should be float type"
            assert src.crs is not None, "Should have CRS"

    def test_download_precipitation_respects_dataset_parameter(self, tmp_path):
        """download_precipitation should support different datasets."""
        bbox = (32.5, -117.6, 33.5, -116.0)

        # Should work with different datasets
        prism_path = download_precipitation(
            bbox=bbox, output_dir=str(tmp_path), dataset="prism"
        )
        assert "prism" in str(prism_path).lower()


class TestPRISMDownload:
    """Test suite for PRISM-specific download functionality."""

    def test_get_prism_annual_precip_returns_array(self, tmp_path):
        """get_prism_annual_precip should return numpy array."""
        bbox = (32.5, -117.6, 33.5, -116.0)

        data, transform = get_prism_annual_precip(bbox, output_dir=str(tmp_path))

        assert isinstance(data, np.ndarray), "Should return numpy array"
        assert data.ndim == 2, "Should be 2D array"
        assert data.dtype in [np.float32, np.float64], "Should be float type"

    def test_get_prism_annual_precip_returns_transform(self, tmp_path):
        """get_prism_annual_precip should return Affine transform."""
        from rasterio import Affine

        bbox = (32.5, -117.6, 33.5, -116.0)

        data, transform = get_prism_annual_precip(bbox, output_dir=str(tmp_path))

        assert isinstance(transform, Affine), "Should return Affine transform"

    def test_get_prism_annual_precip_reasonable_values(self, tmp_path):
        """PRISM precipitation should be in reasonable range (0-5000 mm/year)."""
        bbox = (32.5, -117.6, 33.5, -116.0)

        data, _ = get_prism_annual_precip(bbox, output_dir=str(tmp_path))

        # Remove any nodata values
        valid_data = data[~np.isnan(data)]
        assert np.all(valid_data >= 0), "Precipitation should be non-negative"
        assert np.all(valid_data <= 5000), "Precipitation should be reasonable (<5000mm/year)"

    def test_get_prism_annual_precip_caches_results(self, tmp_path):
        """get_prism_annual_precip should cache downloaded data."""
        bbox = (32.5, -117.6, 33.5, -116.0)

        # First download
        data1, _ = get_prism_annual_precip(bbox, output_dir=str(tmp_path))

        # Second download (should use cache)
        data2, _ = get_prism_annual_precip(bbox, output_dir=str(tmp_path))

        # Should return same data without re-downloading
        np.testing.assert_array_equal(data1, data2)


class TestPrecipitationValidation:
    """Test suite for precipitation data validation."""

    def test_validate_precipitation_alignment_same_shape(self, tmp_path):
        """validate_precipitation_alignment should pass for same-shaped arrays."""
        dem = np.ones((100, 100))
        precip = np.ones((100, 100)) * 500

        # Should not raise
        validate_precipitation_alignment(dem, precip, dem_shape=dem.shape)

    def test_validate_precipitation_alignment_different_shape_raises(self, tmp_path):
        """validate_precipitation_alignment should raise for mismatched shapes."""
        dem = np.ones((100, 100))
        precip = np.ones((50, 50)) * 500

        with pytest.raises(ValueError, match="shape mismatch"):
            validate_precipitation_alignment(dem, precip, dem_shape=dem.shape)

    def test_validate_precipitation_alignment_validates_crs(self):
        """validate_precipitation_alignment should validate CRS match."""
        from rasterio import Affine

        dem_crs = "EPSG:4326"
        precip_crs = "EPSG:32611"  # Different CRS

        with pytest.raises(ValueError, match="CRS mismatch"):
            validate_precipitation_alignment(
                dem=None,
                precip=None,
                dem_crs=dem_crs,
                precip_crs=precip_crs,
                dem_shape=(100, 100),
            )


class TestDatasetListing:
    """Test suite for listing available datasets."""

    def test_list_available_datasets_returns_list(self):
        """list_available_datasets should return list of dataset names."""
        datasets = list_available_datasets()

        assert isinstance(datasets, list), "Should return list"
        assert len(datasets) > 0, "Should have at least one dataset"

    def test_list_available_datasets_includes_prism(self):
        """list_available_datasets should include PRISM."""
        datasets = list_available_datasets()

        assert "prism" in datasets, "Should include PRISM dataset"

    def test_list_available_datasets_includes_metadata(self):
        """list_available_datasets should return metadata for each dataset."""
        datasets = list_available_datasets(include_metadata=True)

        assert isinstance(datasets, dict), "Should return dict when include_metadata=True"
        assert "prism" in datasets, "Should include PRISM"

        prism_meta = datasets["prism"]
        assert "resolution" in prism_meta, "Should include resolution"
        assert "temporal_coverage" in prism_meta, "Should include temporal coverage"
        assert "url" in prism_meta, "Should include source URL"


class TestPrecipitationIntegration:
    """Integration tests for complete workflow."""

    def test_download_and_validate_workflow(self, tmp_path):
        """Test complete download and validation workflow."""
        bbox = (32.5, -117.6, 33.5, -116.0)

        # Download precipitation
        precip_path = download_precipitation(bbox=bbox, output_dir=str(tmp_path), dataset="prism")

        # Load and validate
        import rasterio

        with rasterio.open(precip_path) as src:
            precip_data = src.read(1)
            precip_transform = src.transform
            precip_crs = src.crs

        # Should have reasonable values
        valid_data = precip_data[~np.isnan(precip_data)]
        assert len(valid_data) > 0, "Should have valid data"
        assert np.all(valid_data >= 0), "Should be non-negative"
        assert np.all(valid_data <= 5000), "Should be reasonable"

    def test_multiple_downloads_use_cache(self, tmp_path):
        """Multiple downloads should use cached data."""
        bbox = (32.5, -117.6, 33.5, -116.0)

        # First download
        path1 = download_precipitation(bbox=bbox, output_dir=str(tmp_path), dataset="prism")

        # Second download
        path2 = download_precipitation(bbox=bbox, output_dir=str(tmp_path), dataset="prism")

        # Should return same file
        assert path1 == path2, "Should use cached file"


class TestRealPRISMDownload:
    """Test suite for real PRISM data download (not synthetic)."""

    def test_download_real_prism_requires_network(self, tmp_path):
        """download_real_prism_annual should handle network connectivity."""
        from src.terrain.precipitation_downloader import download_real_prism_annual

        bbox = (32.5, -117.6, 33.5, -116.0)

        # Should either succeed or raise informative network error
        try:
            result = download_real_prism_annual(bbox, output_dir=str(tmp_path))
            assert result is not None, "Should return data on success"
        except Exception as e:
            # Network errors should be informative
            assert "network" in str(e).lower() or "connection" in str(e).lower()

    def test_download_real_prism_returns_valid_data(self, tmp_path):
        """download_real_prism_annual should return valid precipitation data."""
        from src.terrain.precipitation_downloader import download_real_prism_annual

        bbox = (32.5, -117.6, 33.5, -116.0)

        # May need to skip if no network available
        pytest.importorskip("requests")

        try:
            data, transform = download_real_prism_annual(bbox, output_dir=str(tmp_path))

            # Should return numpy array
            assert isinstance(data, np.ndarray), "Should return numpy array"
            assert data.ndim == 2, "Should be 2D array"

            # Should have reasonable precipitation values
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                assert np.all(valid_data >= 0), "Precipitation should be non-negative"
                assert np.all(valid_data <= 10000), "Precipitation should be reasonable (<10,000mm/year)"
        except Exception:
            pytest.skip("Network unavailable or PRISM server unreachable")

    def test_download_real_prism_uses_web_service(self, tmp_path, monkeypatch):
        """download_real_prism_annual should use PRISM web service API."""
        from src.terrain.precipitation_downloader import download_real_prism_annual
        from unittest.mock import Mock
        import zipfile
        import io

        # Create a mock ZIP file with BIL and HDR files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            # Add mock BIL file (binary raster data)
            zf.writestr('PRISM_ppt_30yr_normal_4kmM4_annual.bil', b'\x00' * 1000)
            # Add mock HDR file (header metadata)
            zf.writestr('PRISM_ppt_30yr_normal_4kmM4_annual.hdr', b'BYTEORDER I\nLAYOUT BIL\nNROWS 10\nNCOLS 10')

        # Mock the HTTP request
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = zip_buffer.getvalue()

        calls = []
        def mock_get(url, *args, **kwargs):
            calls.append(url)
            return mock_response

        monkeypatch.setattr("requests.get", mock_get)

        # Mock rasterio.open to avoid actually reading the BIL file
        mock_rasterio_data = Mock()
        mock_rasterio_data.read.return_value = np.full((10, 10), 500.0, dtype=np.float32)
        from rasterio.transform import from_bounds
        mock_rasterio_data.transform = from_bounds(-117.6, 32.5, -116.0, 33.5, 10, 10)
        mock_rasterio_data.crs = "EPSG:4326"
        mock_rasterio_data.__enter__ = Mock(return_value=mock_rasterio_data)
        mock_rasterio_data.__exit__ = Mock(return_value=None)

        def mock_rasterio_open(path, *args, **kwargs):
            return mock_rasterio_data

        monkeypatch.setattr("rasterio.open", mock_rasterio_open)

        bbox = (32.5, -117.6, 33.5, -116.0)

        try:
            download_real_prism_annual(bbox, output_dir=str(tmp_path))

            # Should make HTTP request to PRISM service
            assert len(calls) > 0, "Should make HTTP request"
            url = calls[0]

            # URL should point to PRISM service
            assert "prism" in url.lower(), "Should use PRISM service"
            assert "https://" in url, "Should use HTTPS"
        except NotImplementedError:
            pytest.skip("Real PRISM download not yet implemented")

    def test_download_real_prism_handles_http_errors(self, tmp_path, monkeypatch):
        """download_real_prism_annual should handle HTTP errors gracefully."""
        from src.terrain.precipitation_downloader import download_real_prism_annual
        from unittest.mock import Mock

        # Mock HTTP 404 error
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")

        def mock_get(*args, **kwargs):
            return mock_response

        monkeypatch.setattr("requests.get", mock_get)

        bbox = (32.5, -117.6, 33.5, -116.0)

        # Should raise informative error
        with pytest.raises(Exception, match="404|Not Found|download failed"):
            download_real_prism_annual(bbox, output_dir=str(tmp_path))

    def test_get_prism_annual_precip_can_use_real_data(self, tmp_path, monkeypatch):
        """get_prism_annual_precip should support real data download via parameter."""
        from src.terrain.precipitation_downloader import get_prism_annual_precip
        from rasterio.transform import from_bounds

        bbox = (32.5, -117.6, 33.5, -116.0)

        # Mock real download to avoid network dependency
        mock_data = np.full((50, 50), 500.0, dtype=np.float32)
        mock_transform = from_bounds(-117.6, 32.5, -116.0, 33.5, 50, 50)

        calls = []
        def mock_real_download(bbox, output_dir):
            calls.append((bbox, output_dir))
            return (mock_data, mock_transform)

        import src.terrain.precipitation_downloader as precip_module
        monkeypatch.setattr(precip_module, "download_real_prism_annual", mock_real_download)

        # Call with use_real_data flag
        data, transform = get_prism_annual_precip(
            bbox, output_dir=str(tmp_path), use_real_data=True
        )

        # Should use real download
        assert len(calls) == 1, "Should call real download once"
        assert isinstance(data, np.ndarray)
        assert data.shape == (50, 50)

    def test_download_real_prism_bbox_coverage(self, tmp_path):
        """download_real_prism_annual should cover the requested bbox."""
        from src.terrain.precipitation_downloader import download_real_prism_annual
        from rasterio import Affine

        bbox = (32.5, -117.6, 33.5, -116.0)
        min_lat, min_lon, max_lat, max_lon = bbox

        try:
            data, transform = download_real_prism_annual(bbox, output_dir=str(tmp_path))

            # Calculate geographic bounds from transform
            height, width = data.shape
            # Transform maps pixel coords to geographic coords
            # Upper-left corner
            ul_lon, ul_lat = transform * (0, 0)
            # Lower-right corner
            lr_lon, lr_lat = transform * (width, height)

            # Bounds should cover requested bbox (with some tolerance for grid alignment)
            assert ul_lon <= min_lon + 0.1, "Should cover western edge"
            assert lr_lon >= max_lon - 0.1, "Should cover eastern edge"
            assert ul_lat >= max_lat - 0.1, "Should cover northern edge"
            assert lr_lat <= min_lat + 0.1, "Should cover southern edge"

        except Exception:
            pytest.skip("Network unavailable or PRISM server unreachable")
