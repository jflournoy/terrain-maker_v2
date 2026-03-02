"""
Tests for snow analysis module.

Tests for existing SNODAS functions from src.snow.snodas, plus
TDD RED phase tests for the planned SnowAnalysis class.
"""

import pytest
import numpy as np
import numpy.ma as ma
from pathlib import Path
import tempfile
import gzip
from datetime import datetime

from src.snow.snodas import (
    _read_snodas_header,
    _gunzip_snodas_file,
    _read_snodas_binary,
    _load_snodas_data,
    _load_processed_snodas,
)


# =============================================================================
# Tests for existing snodas.py functions
# =============================================================================


class TestGunzipSnodasFile:
    """Test SNODAS .gz file decompression."""

    def test_gunzip_creates_uncompressed_file(self, tmp_path):
        """_gunzip_snodas_file creates uncompressed output."""
        # Create a test .gz file
        test_data = b"Test SNODAS data content"
        gz_file = tmp_path / "test_file.dat.gz"
        with gzip.open(gz_file, "wb") as f:
            f.write(test_data)

        # Decompress
        output_file = _gunzip_snodas_file(gz_file, keep_original=True)

        # Verify output exists and has correct content
        assert output_file.exists()
        assert output_file == tmp_path / "test_file.dat"
        with open(output_file, "rb") as f:
            assert f.read() == test_data
        # Original should still exist
        assert gz_file.exists()

    def test_gunzip_removes_original_when_requested(self, tmp_path):
        """_gunzip_snodas_file removes original if keep_original=False."""
        test_data = b"Test data"
        gz_file = tmp_path / "test_file.dat.gz"
        with gzip.open(gz_file, "wb") as f:
            f.write(test_data)

        output_file = _gunzip_snodas_file(gz_file, keep_original=False)

        assert output_file.exists()
        assert not gz_file.exists()  # Original removed

    def test_gunzip_handles_corruption_gracefully(self, tmp_path):
        """_gunzip_snodas_file raises exception for corrupted .gz file."""
        # Create invalid .gz file
        gz_file = tmp_path / "corrupted.dat.gz"
        with open(gz_file, "wb") as f:
            f.write(b"Not actually gzipped data")

        # Should raise an exception
        with pytest.raises(Exception):
            _gunzip_snodas_file(gz_file)


class TestReadSnodasHeader:
    """Test SNODAS header file parsing."""

    def test_read_header_basic_fields(self, tmp_path):
        """_read_snodas_header extracts basic metadata fields."""
        header_content = """Data layer name: Snow Depth
Data units: meters
Data slope: 0.001
Data intercept: 0.0
No data value: -9999.0
Number of columns: 6935
Number of rows: 3351
Data bytes per pixel: 2
Minimum data value: 0
Maximum data value: 8000
Horizontal datum: WGS84
Horizontal precision: 0.00833333
Projected: No
Geographically corrected: Yes
Benchmark x-axis coordinate: -124.733749997
Benchmark y-axis coordinate: 52.871249997
x-axis resolution: 0.00833333
y-axis resolution: 0.00833333
x-axis offset: 0.0
y-axis offset: 0.0
Minimum x-axis coordinate: -124.73375
Maximum x-axis coordinate: -66.94208
Minimum y-axis coordinate: 24.94958
Maximum y-axis coordinate: 52.87125
Benchmark column: 1
Benchmark row: 1
Start year: 2024
Start month: 1
Start day: 15
Start hour: 6
Start minute: 0
Start second: 0
"""
        header_file = tmp_path / "test_header.txt"
        header_file.write_text(header_content)

        meta = _read_snodas_header(header_file)

        # Check basic fields
        assert meta["data_units"] == "meters"
        assert meta["data_slope"] == 0.001
        assert meta["data_intercept"] == 0.0
        assert meta["no_data_value"] == -9999.0
        assert meta["width"] == 6935
        assert meta["height"] == 3351
        assert meta["crs"] == "EPSG:4326"
        # Check transform exists
        assert "transform" in meta

    def test_read_header_creates_transform(self, tmp_path):
        """_read_snodas_header creates valid Affine transform."""
        header_content = """Number of columns: 100
Number of rows: 50
Minimum x-axis coordinate: -120.0
Maximum x-axis coordinate: -119.0
Minimum y-axis coordinate: 45.0
Maximum y-axis coordinate: 45.5
"""
        header_file = tmp_path / "test_header.txt"
        header_file.write_text(header_content)

        meta = _read_snodas_header(header_file)

        # Transform should be an Affine object
        from affine import Affine

        assert isinstance(meta["transform"], Affine)

    def test_read_header_handles_not_applicable_values(self, tmp_path):
        """_read_snodas_header skips 'Not applicable' values."""
        header_content = """Data units: meters
Some field: Not applicable
Number of columns: 100
Number of rows: 50
"""
        header_file = tmp_path / "test_header.txt"
        header_file.write_text(header_content)

        meta = _read_snodas_header(header_file)

        assert "some_field" not in meta
        assert meta["data_units"] == "meters"


class TestReadSnodasBinary:
    """Test SNODAS binary data file reading."""

    def test_read_binary_basic(self, tmp_path):
        """_read_snodas_binary reads big-endian 16-bit data."""
        # Create simple binary data (5x5 grid)
        data = np.arange(25, dtype=">i2").reshape(5, 5)
        binary_file = tmp_path / "test.dat"
        data.tofile(binary_file)

        meta = {
            "height": 5,
            "width": 5,
            "no_data_value": -9999,
        }

        result = _read_snodas_binary(binary_file, meta)

        assert result.shape == (5, 5)
        np.testing.assert_array_equal(result, data)

    def test_read_binary_applies_slope_and_intercept(self, tmp_path):
        """_read_snodas_binary applies slope and intercept transformations."""
        # Raw data: [0, 1, 2, 3, 4]
        raw_data = np.arange(5, dtype=">i2")
        binary_file = tmp_path / "test.dat"
        raw_data.tofile(binary_file)

        meta = {
            "height": 1,
            "width": 5,
            "data_slope": 0.1,
            "data_intercept": 10.0,
            "no_data_value": -9999,
        }

        result = _read_snodas_binary(binary_file, meta)

        # Expected: raw * 0.1 + 10.0 = [[10.0, 10.1, 10.2, 10.3, 10.4]] (2D array)
        expected = np.array([[10.0, 10.1, 10.2, 10.3, 10.4]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_read_binary_masks_no_data_values(self, tmp_path):
        """_read_snodas_binary masks no_data values."""
        # Data with no_data value (-9999)
        data = np.array([0, 100, -9999, 200, -9999], dtype=">i2")
        binary_file = tmp_path / "test.dat"
        data.tofile(binary_file)

        meta = {
            "height": 1,
            "width": 5,
            "no_data_value": -9999,
        }

        result = _read_snodas_binary(binary_file, meta)

        # Should be a masked array
        assert isinstance(result, ma.MaskedArray)
        # Positions 2 and 4 should be masked
        assert result.mask[0, 2]
        assert result.mask[0, 4]
        # Other positions should not be masked
        assert not result.mask[0, 0]
        assert not result.mask[0, 1]
        assert not result.mask[0, 3]


class TestLoadSnodasData:
    """Test _load_snodas_data function."""

    def test_load_snodas_data_returns_data_and_metadata(self, tmp_path):
        """_load_snodas_data returns both data array and metadata dict."""
        # Create test header
        header_content = """Data units: meters
Data slope: 0.001
Data intercept: 0.0
No data value: -9999.0
Number of columns: 5
Number of rows: 3
Minimum x-axis coordinate: -120.0
Maximum x-axis coordinate: -119.0
Minimum y-axis coordinate: 45.0
Maximum y-axis coordinate: 45.5
"""
        header_file = tmp_path / "test_header.txt.gz"
        with gzip.open(header_file, "wt") as f:
            f.write(header_content)

        # Create test binary data (3x5 = 15 values)
        data = np.arange(15, dtype=">i2").reshape(3, 5)
        binary_file = tmp_path / "test_data.dat.gz"
        with gzip.open(binary_file, "wb") as f:
            f.write(data.tobytes())

        result_data, meta = _load_snodas_data(binary_file, header_file)

        # Check data
        assert result_data.shape == (3, 5)
        # Check metadata
        assert "width" in meta
        assert "height" in meta
        assert meta["width"] == 5
        assert meta["height"] == 3

    def test_load_snodas_data_uses_existing_uncompressed(self, tmp_path):
        """_load_snodas_data uses existing uncompressed files."""
        # Create uncompressed header
        header_content = """Number of columns: 2
Number of rows: 2
No data value: -9999
Minimum x-axis coordinate: -120.0
Maximum x-axis coordinate: -119.0
Minimum y-axis coordinate: 45.0
Maximum y-axis coordinate: 45.5
"""
        header_file = tmp_path / "test_header.txt"
        header_file.write_text(header_content)

        # Create compressed version
        header_gz = tmp_path / "test_header.txt.gz"
        with gzip.open(header_gz, "wt") as f:
            f.write("WRONG CONTENT")  # This shouldn't be used

        # Create uncompressed binary
        data = np.array([[1, 2], [3, 4]], dtype=">i2")
        binary_file = tmp_path / "test_data.dat"
        data.tofile(binary_file)

        # Create compressed version
        binary_gz = tmp_path / "test_data.dat.gz"
        with gzip.open(binary_gz, "wb") as f:
            f.write(b"WRONG")  # This shouldn't be used

        result_data, meta = _load_snodas_data(binary_gz, header_gz)

        # Should use the uncompressed files
        assert result_data.shape == (2, 2)
        assert meta["width"] == 2


class TestLoadProcessedSnodas:
    """Test _load_processed_snodas function."""

    def test_load_processed_snodas_reads_npz(self, tmp_path):
        """_load_processed_snodas reads .npz files correctly."""
        from affine import Affine

        # Create test .npz file
        test_data = np.random.rand(10, 10).astype(np.float32)
        test_transform = Affine(0.1, 0, -120, 0, -0.1, 45)
        npz_file = tmp_path / "test.npz"

        np.savez_compressed(
            npz_file,
            data=test_data,
            transform=np.array(list(test_transform), dtype=np.float64),
            crs=np.string_("EPSG:4326"),
            height=np.int32(10),
            width=np.int32(10),
            no_data_value=np.float32(-9999),
        )

        data, meta = _load_processed_snodas(npz_file)

        assert data.shape == (10, 10)
        assert "transform" in meta
        assert "crs" in meta
        assert meta["height"] == 10
        assert meta["width"] == 10

    def test_load_processed_snodas_masks_no_data(self, tmp_path):
        """_load_processed_snodas masks no_data values."""
        # Create data with no_data values
        test_data = np.array([[1.0, 2.0, -9999.0], [4.0, -9999.0, 6.0]], dtype=np.float32)
        npz_file = tmp_path / "test.npz"

        np.savez_compressed(npz_file, data=test_data, no_data_value=np.float32(-9999.0))

        data, meta = _load_processed_snodas(npz_file)

        # Should be masked array
        assert isinstance(data, ma.MaskedArray)
        # Check that -9999 values are masked
        assert data.mask[0, 2]
        assert data.mask[1, 1]
        assert not data.mask[0, 0]

    def test_load_processed_snodas_handles_missing_fields(self, tmp_path):
        """_load_processed_snodas handles .npz with minimal fields."""
        # Create minimal .npz file
        test_data = np.random.rand(5, 5).astype(np.float32)
        npz_file = tmp_path / "test.npz"
        np.savez_compressed(npz_file, data=test_data)

        data, meta = _load_processed_snodas(npz_file)

        # Should still load data
        assert data.shape == (5, 5)
        # Metadata might be minimal
        assert isinstance(meta, dict)


# =============================================================================
# TDD RED Phase: Tests for planned SnowAnalysis class (not yet implemented)
# =============================================================================


@pytest.mark.xfail(reason="SnowAnalysis class not yet implemented (TDD RED phase)")
class TestSnowAnalysisInit:
    """Test SnowAnalysis initialization."""

    def test_init_without_terrain(self):
        """SnowAnalysis can be initialized without terrain."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()
        assert analyzer.terrain is None
        assert analyzer.snodas_root_dir is None
        assert analyzer.cache_dir == Path("snow_analysis_cache")
        assert analyzer.processed_files == {}
        assert analyzer.stats is None
        assert analyzer.metadata is None
        assert analyzer.sledding_score is None

    def test_init_with_custom_cache_dir(self):
        """SnowAnalysis respects custom cache directory."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis(cache_dir="custom_cache")
        assert analyzer.cache_dir == Path("custom_cache")
        assert analyzer.cache_dir.exists()

    def test_init_with_snodas_root_dir(self, tmp_path):
        """SnowAnalysis accepts snodas_root_dir."""
        from src.snow.analysis import SnowAnalysis

        root_dir = tmp_path / "snodas_data"
        root_dir.mkdir()
        analyzer = SnowAnalysis(snodas_root_dir=str(root_dir))
        assert analyzer.snodas_root_dir == root_dir

    def test_set_terrain(self):
        """set_terrain() updates the terrain reference."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()
        mock_terrain = type("MockTerrain", (), {"dem_bounds": (0, 0, 1, 1)})()
        analyzer.set_terrain(mock_terrain)
        assert analyzer.terrain == mock_terrain


@pytest.mark.xfail(reason="SnowAnalysis class not yet implemented (TDD RED phase)")
class TestCalculateTPI:
    """Test Topographic Position Index calculation."""

    def test_tpi_flat_terrain_returns_zeros(self):
        """TPI of perfectly flat terrain should be all zeros."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()
        flat_dem = np.ones((10, 10)) * 100.0

        tpi = analyzer.calculate_tpi(flat_dem, window_size=3)

        assert tpi.shape == flat_dem.shape
        np.testing.assert_allclose(tpi, 0.0, atol=1e-6)

    def test_tpi_single_peak(self):
        """TPI should be positive at a peak."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()
        dem = np.zeros((5, 5))
        dem[2, 2] = 10.0

        tpi = analyzer.calculate_tpi(dem, window_size=3)

        assert tpi[2, 2] > 0
        assert tpi[1, 2] < 0

    def test_tpi_single_valley(self):
        """TPI should be negative in a valley."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()
        dem = np.ones((5, 5)) * 10.0
        dem[2, 2] = 0.0

        tpi = analyzer.calculate_tpi(dem, window_size=3)

        assert tpi[2, 2] < 0

    def test_tpi_custom_window_size(self):
        """TPI calculation respects custom window_size."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()
        dem = np.random.rand(20, 20) * 100

        tpi_small = analyzer.calculate_tpi(dem, window_size=3)
        tpi_large = analyzer.calculate_tpi(dem, window_size=7)

        assert not np.allclose(tpi_small, tpi_large)


@pytest.mark.xfail(reason="SnowAnalysis class not yet implemented (TDD RED phase)")
class TestCalculateRoughness:
    """Test terrain roughness calculation."""

    def test_roughness_flat_terrain_is_zero(self):
        """Roughness of flat terrain should be zero."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()
        flat_dem = np.ones((10, 10)) * 100.0

        roughness = analyzer.calculate_roughness(flat_dem, window_size=3)

        assert roughness.shape == flat_dem.shape
        np.testing.assert_allclose(roughness, 0.0, atol=1e-6)

    def test_roughness_rough_terrain_is_positive(self):
        """Roughness of variable terrain should be > 0."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()
        rough_dem = np.random.rand(10, 10) * 100

        roughness = analyzer.calculate_roughness(rough_dem, window_size=3)

        assert np.mean(roughness) > 0
        assert roughness.shape == rough_dem.shape

    def test_roughness_increases_with_variability(self):
        """More variable terrain should have higher roughness."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()
        gentle = np.tile(np.arange(10), (10, 1))
        steep = gentle * 10

        roughness_gentle = analyzer.calculate_roughness(gentle, window_size=3)
        roughness_steep = analyzer.calculate_roughness(steep, window_size=3)

        assert np.mean(roughness_steep) > np.mean(roughness_gentle)


@pytest.mark.xfail(reason="SnowAnalysis class not yet implemented (TDD RED phase)")
class TestSleddingScore:
    """Test sledding suitability score calculation."""

    def test_sledding_score_requires_stats(self):
        """calculate_sledding_score raises error without stats."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()

        with pytest.raises(ValueError, match="No snow stats"):
            analyzer.calculate_sledding_score()

    def test_sledding_score_basic_calculation(self):
        """calculate_sledding_score computes score from stats."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()

        stats = {
            "median_max_depth": np.ones((10, 10)) * 300,
            "mean_snow_day_ratio": np.ones((10, 10)) * 0.5,
            "interseason_cv": np.ones((10, 10)) * 0.1,
            "mean_intraseason_cv": np.ones((10, 10)) * 0.1,
        }
        analyzer.stats = stats

        score = analyzer.calculate_sledding_score(stats, min_depth_mm=100, min_coverage=0.3)

        assert score.shape == (10, 10)
        assert np.all((score >= 0) & (score <= 1))
        assert np.mean(score) > 0.5

    def test_sledding_score_zero_for_no_snow(self):
        """calculate_sledding_score gives low score with no snow."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()

        stats = {
            "median_max_depth": np.zeros((10, 10)),
            "mean_snow_day_ratio": np.zeros((10, 10)),
            "interseason_cv": np.ones((10, 10)) * 0.5,
            "mean_intraseason_cv": np.ones((10, 10)) * 0.5,
        }
        analyzer.stats = stats

        score = analyzer.calculate_sledding_score(stats)

        assert np.mean(score) < 0.3


@pytest.mark.xfail(reason="SnowAnalysis class not yet implemented (TDD RED phase)")
class TestProcessSnowDataWorkflow:
    """Test process_snow_data high-level workflow."""

    def test_process_snow_data_requires_terrain(self):
        """process_snow_data raises error without terrain."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()

        with pytest.raises(ValueError, match="No terrain provided"):
            analyzer.process_snow_data()

    def test_process_snow_data_requires_valid_snodas_dir(self, tmp_path):
        """process_snow_data raises error with invalid SNODAS directory."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis(snodas_root_dir=str(tmp_path / "nonexistent"))

        mock_terrain = type("MockTerrain", (), {"dem_bounds": (-120.0, 45.0, -119.0, 45.5)})()
        analyzer.set_terrain(mock_terrain)

        with pytest.raises(ValueError, match="SNODAS directory not found"):
            analyzer.process_snow_data()


@pytest.mark.xfail(reason="SnowAnalysis class not yet implemented (TDD RED phase)")
class TestBatchProcessSnodas:
    """Test batch_process_snodas_data on SnowAnalysis class."""

    def test_batch_process_returns_empty_without_files(self, tmp_path):
        """batch_process_snodas_data returns empty dict without files."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis(snodas_root_dir=str(tmp_path))
        extent = (-120.0, 45.0, -119.0, 45.5)

        result = analyzer.batch_process_snodas_data(extent)

        assert result == {}

    def test_batch_process_returns_empty_without_root_dir(self):
        """batch_process_snodas_data returns empty dict without root dir."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()
        extent = (-120.0, 45.0, -119.0, 45.5)

        result = analyzer.batch_process_snodas_data(extent)

        assert result == {}


@pytest.mark.xfail(reason="SnowAnalysis class not yet implemented (TDD RED phase)")
class TestVisualization:
    """Test visualization functions."""

    def test_visualize_requires_data(self):
        """visualize_snow_data raises error without data."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()

        with pytest.raises((ValueError, AttributeError)):
            analyzer.visualize_snow_data(data_type="sledding_score")

    def test_visualize_with_sledding_score(self):
        """visualize_snow_data works with sledding_score data."""
        import matplotlib

        matplotlib.use("Agg")
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()
        analyzer.sledding_score = np.random.rand(10, 10)

        try:
            analyzer.visualize_snow_data(data_type="sledding_score")
        except Exception as e:
            if "display" not in str(e).lower():
                raise


@pytest.mark.xfail(reason="SnowAnalysis class not yet implemented (TDD RED phase)")
class TestStoreSnowStatsInTerrain:
    """Test _store_snow_stats_in_terrain function."""

    def test_store_stats_skips_without_terrain(self):
        """_store_snow_stats_in_terrain does nothing without terrain."""
        from src.snow.analysis import SnowAnalysis

        analyzer = SnowAnalysis()
        stats = {"median_max_depth": np.ones((10, 10))}
        metadata = {"transform": None, "crs": "EPSG:4326"}

        analyzer._store_snow_stats_in_terrain(stats, metadata)
        assert analyzer.terrain is None

    def test_store_stats_adds_arrays_to_terrain(self):
        """_store_snow_stats_in_terrain adds array stats to terrain."""
        from affine import Affine
        from src.snow.analysis import SnowAnalysis

        added_layers = []

        class MockTerrain:
            dem_transform = Affine.identity()

            def add_data_layer(self, name, data, transform, crs, target_layer=None):
                added_layers.append(name)

        mock_terrain = MockTerrain()
        analyzer = SnowAnalysis()
        analyzer.terrain = mock_terrain

        stats = {
            "median_max_depth": np.ones((10, 10)),
            "mean_snow_day_ratio": np.ones((10, 10)),
            "some_scalar": 42,
        }
        metadata = {"transform": Affine.identity(), "crs": "EPSG:4326"}

        analyzer._store_snow_stats_in_terrain(stats, metadata)

        assert len(added_layers) == 2
        assert "snodas_median_max_depth" in added_layers
        assert "snodas_mean_snow_day_ratio" in added_layers


# 🔴 TDD RED PHASE COMPLETE - Round 2
# SnowAnalysis class tests marked as xfail until implementation.
# SNODAS I/O function tests now point to existing src.snow.snodas module.
