"""
Tests for data loading operations.

Tests DEM file loading and merging functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import patch
import rasterio
from rasterio.transform import Affine


class TestLoadDemFiles:
    """Tests for load_dem_files function."""

    def test_load_dem_files_imports(self):
        """Test that load_dem_files can be imported."""
        from src.terrain.data_loading import load_dem_files

        assert callable(load_dem_files)

    def test_load_dem_files_nonexistent_directory_raises(self):
        """Test that nonexistent directory raises ValueError."""
        from src.terrain.data_loading import load_dem_files

        with pytest.raises(ValueError, match="does not exist"):
            load_dem_files("/nonexistent/directory/path")

    def test_load_dem_files_file_not_directory_raises(self):
        """Test that file path instead of directory raises ValueError."""
        from src.terrain.data_loading import load_dem_files

        with tempfile.NamedTemporaryFile() as tmp:
            with pytest.raises(ValueError, match="not a directory"):
                load_dem_files(tmp.name)

    def test_load_dem_files_no_matching_files_raises(self):
        """Test that directory with no matching files raises ValueError."""
        from src.terrain.data_loading import load_dem_files

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No files matching"):
                load_dem_files(tmpdir, pattern="*.hgt")

    def test_load_dem_files_returns_tuple(self):
        """Test that function returns (array, transform) tuple."""
        from src.terrain.data_loading import load_dem_files

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple test DEM file
            dem_data = np.random.rand(10, 10).astype(np.float32)
            transform = Affine.translation(0, 10) * Affine.scale(1, -1)

            test_file = Path(tmpdir) / "test.tif"
            with rasterio.open(
                test_file,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype=dem_data.dtype,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(dem_data, 1)

            result = load_dem_files(tmpdir, pattern="*.tif")

            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], np.ndarray)
            assert isinstance(result[1], Affine)

    def test_load_dem_files_loads_single_file(self):
        """Test loading a single DEM file."""
        from src.terrain.data_loading import load_dem_files

        with tempfile.TemporaryDirectory() as tmpdir:
            dem_data = np.arange(100).reshape(10, 10).astype(np.float32)
            transform = Affine.translation(0, 10) * Affine.scale(1, -1)

            test_file = Path(tmpdir) / "test.tif"
            with rasterio.open(
                test_file,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype=dem_data.dtype,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(dem_data, 1)

            merged_dem, merged_transform = load_dem_files(tmpdir, pattern="*.tif")

            # Should match input data
            assert merged_dem.shape == (10, 10)
            assert np.allclose(merged_dem, dem_data)

    def test_load_dem_files_loads_multiple_files(self):
        """Test loading and merging multiple DEM files."""
        from src.terrain.data_loading import load_dem_files

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two adjacent DEM tiles
            dem1 = np.ones((10, 10), dtype=np.float32)
            dem2 = np.ones((10, 10), dtype=np.float32) * 2

            # First tile at (0, 0)
            transform1 = Affine.translation(0, 10) * Affine.scale(1, -1)
            file1 = Path(tmpdir) / "tile1.tif"
            with rasterio.open(
                file1,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype=dem1.dtype,
                crs="EPSG:4326",
                transform=transform1,
            ) as dst:
                dst.write(dem1, 1)

            # Second tile at (10, 0) - to the right
            transform2 = Affine.translation(10, 10) * Affine.scale(1, -1)
            file2 = Path(tmpdir) / "tile2.tif"
            with rasterio.open(
                file2,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype=dem2.dtype,
                crs="EPSG:4326",
                transform=transform2,
            ) as dst:
                dst.write(dem2, 1)

            merged_dem, merged_transform = load_dem_files(tmpdir, pattern="*.tif")

            # Merged should be larger than individual tiles
            assert merged_dem.shape[0] >= 10
            assert merged_dem.shape[1] >= 10

    def test_load_dem_files_custom_pattern(self):
        """Test loading with custom file pattern."""
        from src.terrain.data_loading import load_dem_files

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different extensions
            dem_data = np.ones((10, 10), dtype=np.float32)
            transform = Affine.translation(0, 10) * Affine.scale(1, -1)

            # Create .tif file
            tif_file = Path(tmpdir) / "test.tif"
            with rasterio.open(
                tif_file,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype=dem_data.dtype,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(dem_data, 1)

            # Create .img file (HFA format)
            img_file = Path(tmpdir) / "test.img"
            with rasterio.open(
                img_file,
                "w",
                driver="HFA",
                height=10,
                width=10,
                count=1,
                dtype=dem_data.dtype,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(dem_data, 1)

            # Load only .tif files
            merged_dem, _ = load_dem_files(tmpdir, pattern="*.tif")
            assert merged_dem is not None

            # Load only .img files
            merged_dem, _ = load_dem_files(tmpdir, pattern="*.img")
            assert merged_dem is not None

    def test_load_dem_files_recursive_search(self):
        """Test recursive subdirectory search."""
        from src.terrain.data_loading import load_dem_files

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectory
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()

            # Create DEM in subdirectory
            dem_data = np.ones((10, 10), dtype=np.float32)
            transform = Affine.translation(0, 10) * Affine.scale(1, -1)

            test_file = subdir / "test.tif"
            with rasterio.open(
                test_file,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype=dem_data.dtype,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(dem_data, 1)

            # Non-recursive should find nothing
            with pytest.raises(ValueError, match="No files matching"):
                load_dem_files(tmpdir, pattern="*.tif", recursive=False)

            # Recursive should find the file
            merged_dem, _ = load_dem_files(tmpdir, pattern="*.tif", recursive=True)
            assert merged_dem is not None
            assert merged_dem.shape == (10, 10)


class TestLoadScoreGrid:
    """Tests for load_score_grid function - loads georeferenced raster data from NPZ files."""

    def test_load_score_grid_imports(self):
        """Test that load_score_grid can be imported."""
        from src.terrain.data_loading import load_score_grid

        assert callable(load_score_grid)

    def test_load_score_grid_with_transform(self):
        """Load NPZ with transform metadata returns both array and Affine."""
        from src.terrain.data_loading import load_score_grid

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data with transform
            score_data = np.random.rand(100, 100).astype(np.float32)
            transform_tuple = (0.001, 0, -83.0, 0, -0.001, 42.5)  # lon/lat transform

            test_file = Path(tmpdir) / "scores.npz"
            np.savez_compressed(
                test_file,
                data=score_data,
                transform=np.array(transform_tuple),
            )

            loaded_data, loaded_transform = load_score_grid(test_file)

            assert isinstance(loaded_data, np.ndarray)
            assert loaded_data.shape == (100, 100)
            assert np.allclose(loaded_data, score_data)
            assert isinstance(loaded_transform, Affine)
            assert loaded_transform.a == transform_tuple[0]
            assert loaded_transform.c == transform_tuple[2]

    def test_load_score_grid_without_transform(self):
        """Load NPZ without transform returns array and None."""
        from src.terrain.data_loading import load_score_grid

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data WITHOUT transform
            score_data = np.random.rand(50, 50).astype(np.float32)

            test_file = Path(tmpdir) / "scores_no_transform.npz"
            np.savez_compressed(test_file, data=score_data)

            loaded_data, loaded_transform = load_score_grid(test_file)

            assert isinstance(loaded_data, np.ndarray)
            assert loaded_data.shape == (50, 50)
            assert np.allclose(loaded_data, score_data)
            assert loaded_transform is None

    def test_load_score_grid_flexible_keys(self):
        """Finds data under various key names."""
        from src.terrain.data_loading import load_score_grid

        with tempfile.TemporaryDirectory() as tmpdir:
            score_data = np.random.rand(20, 20).astype(np.float32)

            # Test with "score" key
            file1 = Path(tmpdir) / "with_score_key.npz"
            np.savez_compressed(file1, score=score_data)
            data1, _ = load_score_grid(file1)
            assert np.allclose(data1, score_data)

            # Test with "values" key
            file2 = Path(tmpdir) / "with_values_key.npz"
            np.savez_compressed(file2, values=score_data)
            data2, _ = load_score_grid(file2)
            assert np.allclose(data2, score_data)

            # Test with arbitrary key (fallback to first array)
            file3 = Path(tmpdir) / "with_arbitrary_key.npz"
            np.savez_compressed(file3, my_custom_array=score_data)
            data3, _ = load_score_grid(file3)
            assert np.allclose(data3, score_data)

    def test_load_score_grid_custom_keys(self):
        """User can specify custom keys to search for."""
        from src.terrain.data_loading import load_score_grid

        with tempfile.TemporaryDirectory() as tmpdir:
            score_data = np.random.rand(30, 30).astype(np.float32)

            test_file = Path(tmpdir) / "custom_key.npz"
            np.savez_compressed(test_file, sledding_score=score_data)

            # Should find with custom key list
            data, _ = load_score_grid(test_file, data_keys=["sledding_score"])
            assert np.allclose(data, score_data)

    def test_load_score_grid_file_not_found(self):
        """Raises FileNotFoundError for missing file."""
        from src.terrain.data_loading import load_score_grid

        with pytest.raises(FileNotFoundError):
            load_score_grid(Path("/nonexistent/path/scores.npz"))

    def test_load_score_grid_preserves_dtype(self):
        """Preserves original data type."""
        from src.terrain.data_loading import load_score_grid

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test float32
            float32_data = np.random.rand(10, 10).astype(np.float32)
            f32_file = Path(tmpdir) / "float32.npz"
            np.savez_compressed(f32_file, data=float32_data)
            loaded, _ = load_score_grid(f32_file)
            assert loaded.dtype == np.float32

            # Test float64
            float64_data = np.random.rand(10, 10).astype(np.float64)
            f64_file = Path(tmpdir) / "float64.npz"
            np.savez_compressed(f64_file, data=float64_data)
            loaded, _ = load_score_grid(f64_file)
            assert loaded.dtype == np.float64


class TestParseHgtFilename:
    """Tests for parse_hgt_filename function - parses SRTM HGT filenames."""

    def test_parse_hgt_filename_imports(self):
        """Test that parse_hgt_filename can be imported."""
        from src.terrain.data_loading import parse_hgt_filename

        assert callable(parse_hgt_filename)

    def test_parse_hgt_filename_northern_western(self):
        """N42W083.hgt -> lat=42, lon=-83."""
        from src.terrain.data_loading import parse_hgt_filename

        lat, lon = parse_hgt_filename("N42W083.hgt")
        assert lat == 42
        assert lon == -83

    def test_parse_hgt_filename_northern_eastern(self):
        """N45E010.hgt -> lat=45, lon=10."""
        from src.terrain.data_loading import parse_hgt_filename

        lat, lon = parse_hgt_filename("N45E010.hgt")
        assert lat == 45
        assert lon == 10

    def test_parse_hgt_filename_southern_western(self):
        """S15W047.hgt -> lat=-15, lon=-47."""
        from src.terrain.data_loading import parse_hgt_filename

        lat, lon = parse_hgt_filename("S15W047.hgt")
        assert lat == -15
        assert lon == -47

    def test_parse_hgt_filename_southern_eastern(self):
        """S33E018.hgt -> lat=-33, lon=18."""
        from src.terrain.data_loading import parse_hgt_filename

        lat, lon = parse_hgt_filename("S33E018.hgt")
        assert lat == -33
        assert lon == 18

    def test_parse_hgt_filename_with_path(self):
        """Works with Path object containing directory."""
        from src.terrain.data_loading import parse_hgt_filename

        lat, lon = parse_hgt_filename(Path("/some/dir/N42W083.hgt"))
        assert lat == 42
        assert lon == -83

    def test_parse_hgt_filename_invalid_returns_none(self):
        """Invalid filenames return (None, None)."""
        from src.terrain.data_loading import parse_hgt_filename

        lat, lon = parse_hgt_filename("not_a_valid_hgt.hgt")
        assert lat is None
        assert lon is None


def _create_test_dem_file(path: Path, value: float, lat: int, lon: int) -> None:
    """Helper to create a test DEM file with proper transform."""
    dem_data = np.ones((10, 10), dtype=np.float32) * value
    # HGT files have SW corner at the lat/lon in filename
    transform = Affine.translation(lon, lat + 1) * Affine.scale(0.1, -0.1)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=dem_data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(dem_data, 1)


class TestLoadFilteredHgtFiles:
    """Tests for load_filtered_hgt_files function - loads HGT files filtered by lat/lon."""

    def test_load_filtered_hgt_files_imports(self):
        """Test that load_filtered_hgt_files can be imported."""
        from src.terrain.data_loading import load_filtered_hgt_files

        assert callable(load_filtered_hgt_files)

    def test_load_filtered_hgt_files_filters_by_latitude(self):
        """Only loads HGT files within latitude range."""
        from src.terrain.data_loading import load_filtered_hgt_files

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test files with different latitudes (use .tif extension)
            # Each file has a different elevation value so we can verify which were loaded
            _create_test_dem_file(tmpdir / "N40W083.tif", 100, 40, -83)  # lat 40 - should be filtered out
            _create_test_dem_file(tmpdir / "N42W083.tif", 200, 42, -83)  # lat 42 - should load
            _create_test_dem_file(tmpdir / "N44W083.tif", 300, 44, -83)  # lat 44 - should load

            # Filter to lat >= 42
            dem, transform = load_filtered_hgt_files(tmpdir, min_latitude=42, pattern="*.tif")

            # Should only include tiles with lat >= 42 (values 200 and 300)
            assert dem is not None
            unique_vals = set(np.unique(dem))
            # Should have 200 and 300 (and possibly 0 for nodata gaps between tiles)
            assert 200 in unique_vals, "Should include tile at lat 42 (value 200)"
            assert 300 in unique_vals, "Should include tile at lat 44 (value 300)"
            # Should NOT have 100 (the filtered-out tile)
            assert 100 not in unique_vals, "Should NOT include tile at lat 40 (value 100)"

    def test_load_filtered_hgt_files_filters_by_longitude(self):
        """Only loads HGT files within longitude range."""
        from src.terrain.data_loading import load_filtered_hgt_files

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test files with different longitudes
            _create_test_dem_file(tmpdir / "N42W090.tif", 100, 42, -90)  # lon -90 - filter out
            _create_test_dem_file(tmpdir / "N42W085.tif", 200, 42, -85)  # lon -85 - load
            _create_test_dem_file(tmpdir / "N42W080.tif", 300, 42, -80)  # lon -80 - load

            # Filter to lon >= -85
            dem, transform = load_filtered_hgt_files(tmpdir, min_longitude=-85, pattern="*.tif")

            assert dem is not None
            unique_vals = set(np.unique(dem))
            # Should have 200 and 300 (and possibly 0 for nodata gaps)
            assert 200 in unique_vals, "Should include tile at lon -85 (value 200)"
            assert 300 in unique_vals, "Should include tile at lon -80 (value 300)"
            # Should NOT have 100 (the filtered-out tile)
            assert 100 not in unique_vals, "Should NOT include tile at lon -90 (value 100)"

    def test_load_filtered_hgt_files_no_filter(self):
        """Without filters, loads all HGT files."""
        from src.terrain.data_loading import load_filtered_hgt_files

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test files
            _create_test_dem_file(tmpdir / "N42W083.tif", 100, 42, -83)
            _create_test_dem_file(tmpdir / "N43W083.tif", 200, 43, -83)

            # No filters - should load all
            dem, transform = load_filtered_hgt_files(tmpdir, pattern="*.tif")

            assert dem is not None
            # Should include both tiles (100 and 200)
            assert np.min(dem) == 100
            assert np.max(dem) == 200

    def test_load_filtered_hgt_files_returns_tuple(self):
        """Returns (array, transform) tuple."""
        from src.terrain.data_loading import load_filtered_hgt_files

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            _create_test_dem_file(tmpdir / "N42W083.tif", 100, 42, -83)

            result = load_filtered_hgt_files(tmpdir, pattern="*.tif")

            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], np.ndarray)
            assert isinstance(result[1], Affine)

    def test_load_filtered_hgt_files_no_matching_raises(self):
        """Raises ValueError when no files match filters."""
        from src.terrain.data_loading import load_filtered_hgt_files

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create file at lat 42
            _create_test_dem_file(tmpdir / "N42W083.tif", 100, 42, -83)

            # Filter to lat >= 50 (no files match)
            with pytest.raises(ValueError, match="No HGT files found matching"):
                load_filtered_hgt_files(tmpdir, min_latitude=50, pattern="*.tif")
