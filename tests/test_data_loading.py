"""
Tests for data loading operations.

Tests DEM file loading and merging functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
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
