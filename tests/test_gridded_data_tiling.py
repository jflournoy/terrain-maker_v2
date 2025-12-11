"""
Tests for automatic tiling in GriddedDataLoader.

Tests cover:
- Tile layout creation and specifications
- Memory monitoring and failsafe
- Aggregation strategies
- End-to-end tiled pipeline execution
"""

import numpy as np
import pytest
from unittest import mock
from pathlib import Path
import tempfile

from src.terrain.gridded_data import (
    GriddedDataLoader,
    TiledDataConfig,
    TileSpecGridded,
    MemoryMonitor,
    MemoryLimitExceeded,
)


class MockTerrain:
    """Mock Terrain object for testing."""

    def __init__(self):
        self.dem_bounds = (0.0, 0.0, 10.0, 10.0)  # minx, miny, maxx, maxy


class TestTiledDataConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TiledDataConfig()
        assert config.max_output_pixels == 4096 * 4096
        assert config.target_tile_outputs == 2000
        assert config.max_memory_percent == 85.0
        assert config.max_swap_percent == 50.0
        assert config.enable_memory_monitoring is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = TiledDataConfig(
            max_output_pixels=1000000,
            target_tile_outputs=1000,
            max_memory_percent=75.0,
        )
        assert config.max_output_pixels == 1000000
        assert config.target_tile_outputs == 1000
        assert config.max_memory_percent == 75.0


class TestMemoryMonitor:
    """Test memory monitoring."""

    def test_memory_monitor_disabled_without_psutil(self):
        """Test that monitor gracefully handles missing psutil."""
        config = TiledDataConfig(enable_memory_monitoring=True)
        with mock.patch.dict("sys.modules", {"psutil": None}):
            monitor = MemoryMonitor(config)
            # Should not raise even without psutil
            monitor.check_memory(force=True)

    def test_memory_monitor_check_interval(self):
        """Test that check_interval throttles checks."""
        config = TiledDataConfig(memory_check_interval=10.0)

        mock_psutil = mock.MagicMock()
        mock_psutil.virtual_memory.return_value.percent = 50.0
        mock_psutil.swap_memory.return_value.percent = 30.0

        monitor = MemoryMonitor(config)
        monitor.psutil = mock_psutil

        # First check should execute
        monitor.check_memory(force=True)
        assert mock_psutil.virtual_memory.call_count == 1

        # Second check within interval should be skipped
        monitor.check_memory(force=False)
        assert mock_psutil.virtual_memory.call_count == 1  # No additional call

    def test_memory_limit_exceeded_on_high_memory(self):
        """Test that MemoryLimitExceeded is raised when memory is high."""
        config = TiledDataConfig(max_memory_percent=50.0, enable_memory_monitoring=True)

        mock_psutil = mock.MagicMock()
        mock_psutil.virtual_memory.return_value.percent = 85.0
        mock_psutil.swap_memory.return_value.percent = 30.0

        monitor = MemoryMonitor(config)
        monitor.psutil = mock_psutil

        with pytest.raises(MemoryLimitExceeded):
            monitor.check_memory(force=True)

    def test_memory_limit_exceeded_on_high_swap(self):
        """Test that MemoryLimitExceeded is raised when swap is high."""
        config = TiledDataConfig(max_swap_percent=40.0, enable_memory_monitoring=True)

        mock_psutil = mock.MagicMock()
        mock_psutil.virtual_memory.return_value.percent = 50.0
        mock_psutil.swap_memory.return_value.percent = 60.0

        monitor = MemoryMonitor(config)
        monitor.psutil = mock_psutil

        with pytest.raises(MemoryLimitExceeded):
            monitor.check_memory(force=True)


class TestTileSpecification:
    """Test tile specification creation."""

    def test_single_tile(self):
        """Test creation of single tile when output fits."""
        terrain = MockTerrain()
        config = TiledDataConfig(target_tile_outputs=5000)
        loader = GriddedDataLoader(terrain, tile_config=config)

        target_shape = (1000, 1000)
        extent = (0.0, 0.0, 10.0, 10.0)

        tiles = loader._create_tile_specs(target_shape, extent)

        assert len(tiles) == 1
        tile = tiles[0]
        assert tile.target_shape == (1000, 1000)
        assert tile.extent == extent
        assert tile.out_slice == (slice(0, 1000), slice(0, 1000))

    def test_multiple_tiles(self):
        """Test creation of multiple tiles."""
        terrain = MockTerrain()
        config = TiledDataConfig(target_tile_outputs=100)
        loader = GriddedDataLoader(terrain, tile_config=config)

        target_shape = (300, 300)
        extent = (0.0, 0.0, 30.0, 30.0)

        tiles = loader._create_tile_specs(target_shape, extent)

        # 300x300 with 100x100 tiles = 3x3 = 9 tiles
        assert len(tiles) == 9

        # Check first tile
        assert tiles[0].target_shape == (100, 100)
        assert tiles[0].out_slice == (slice(0, 100), slice(0, 100))

        # Check corner tile
        assert tiles[8].target_shape == (100, 100)
        assert tiles[8].out_slice == (slice(200, 300), slice(200, 300))


class TestAggregation:
    """Test aggregation strategies."""

    def test_determine_aggregation_2d_array(self):
        """Test aggregation detection for 2D arrays."""
        terrain = MockTerrain()
        loader = GriddedDataLoader(terrain)

        tile_outputs = [np.ones((100, 100)), np.ones((100, 100))]
        strategy = loader._determine_aggregation(tile_outputs)
        assert strategy == "concatenate"

    def test_determine_aggregation_scalar(self):
        """Test aggregation detection for scalars."""
        terrain = MockTerrain()
        loader = GriddedDataLoader(terrain)

        tile_outputs = [0.5, 0.6]
        strategy = loader._determine_aggregation(tile_outputs)
        assert strategy == "mean"

    def test_determine_aggregation_dict_with_arrays(self):
        """Test aggregation detection for dict of arrays."""
        terrain = MockTerrain()
        loader = GriddedDataLoader(terrain)

        tile_outputs = [
            {"data": np.ones((100, 100))},
            {"data": np.ones((100, 100))},
        ]
        strategy = loader._determine_aggregation(tile_outputs)
        assert strategy == "concatenate"

    def test_assemble_grid(self):
        """Test grid assembly from tiles."""
        terrain = MockTerrain()
        loader = GriddedDataLoader(terrain)

        # Create 2x2 tiles
        tile1 = np.ones((100, 100))
        tile2 = np.ones((100, 100)) * 2
        tile3 = np.ones((100, 100)) * 3
        tile4 = np.ones((100, 100)) * 4

        tile_specs = [
            TileSpecGridded(
                src_slice=(slice(0, 100), slice(0, 100)),
                out_slice=(slice(0, 100), slice(0, 100)),
                extent=(0, 0, 5, 5),
                target_shape=(100, 100),
            ),
            TileSpecGridded(
                src_slice=(slice(0, 100), slice(100, 200)),
                out_slice=(slice(0, 100), slice(100, 200)),
                extent=(5, 0, 10, 5),
                target_shape=(100, 100),
            ),
            TileSpecGridded(
                src_slice=(slice(100, 200), slice(0, 100)),
                out_slice=(slice(100, 200), slice(0, 100)),
                extent=(0, 5, 5, 10),
                target_shape=(100, 100),
            ),
            TileSpecGridded(
                src_slice=(slice(100, 200), slice(100, 200)),
                out_slice=(slice(100, 200), slice(100, 200)),
                extent=(5, 5, 10, 10),
                target_shape=(100, 100),
            ),
        ]

        arrays = [tile1, tile2, tile3, tile4]
        target_shape = (200, 200)

        result = loader._assemble_grid(arrays, tile_specs, target_shape)

        assert result.shape == (200, 200)
        assert np.all(result[0:100, 0:100] == 1)
        assert np.all(result[0:100, 100:200] == 2)
        assert np.all(result[100:200, 0:100] == 3)
        assert np.all(result[100:200, 100:200] == 4)

    def test_average_statistics_scalar(self):
        """Test averaging scalar statistics."""
        terrain = MockTerrain()
        loader = GriddedDataLoader(terrain)

        tile_outputs = [0.5, 0.6, 0.7]
        result = loader._average_statistics(tile_outputs)
        assert np.isclose(result, 0.6)

    def test_average_statistics_dict(self):
        """Test averaging dict of statistics."""
        terrain = MockTerrain()
        loader = GriddedDataLoader(terrain)

        tile_outputs = [
            {"mean": 0.5, "std": 0.1},
            {"mean": 0.6, "std": 0.2},
            {"mean": 0.7, "std": 0.3},
        ]
        result = loader._average_statistics(tile_outputs)
        assert np.isclose(result["mean"], 0.6)
        assert np.isclose(result["std"], 0.2)


class TestPixelCounting:
    """Test array pixel counting."""

    def test_count_numpy_array(self):
        """Test pixel counting for numpy arrays."""
        terrain = MockTerrain()
        loader = GriddedDataLoader(terrain)

        arr = np.ones((100, 200))
        count = loader._get_array_pixel_count(arr)
        assert count == 20000

    def test_count_dict_of_arrays(self):
        """Test pixel counting for dict of arrays."""
        terrain = MockTerrain()
        loader = GriddedDataLoader(terrain)

        data = {"arr1": np.ones((100, 100)), "arr2": np.ones((50, 50))}
        count = loader._get_array_pixel_count(data)
        assert count == 12500

    def test_count_empty_dict(self):
        """Test pixel counting for empty dict."""
        terrain = MockTerrain()
        loader = GriddedDataLoader(terrain)

        data = {}
        count = loader._get_array_pixel_count(data)
        assert count == 0

    def test_count_non_array(self):
        """Test pixel counting for non-array data."""
        terrain = MockTerrain()
        loader = GriddedDataLoader(terrain)

        count = loader._get_array_pixel_count("string")
        assert count == 0


class TestOutputShapeDetection:
    """Test output shape detection."""

    def test_get_output_shape_array(self):
        """Test shape detection for arrays."""
        terrain = MockTerrain()
        loader = GriddedDataLoader(terrain)

        arr = np.ones((100, 200, 3))
        shape = loader._get_output_shape(arr)
        assert shape == (100, 200, 3)

    def test_get_output_shape_dict(self):
        """Test shape detection for dict of arrays."""
        terrain = MockTerrain()
        loader = GriddedDataLoader(terrain)

        data = {"arr1": np.ones((100, 200)), "arr2": np.ones((50, 50))}
        shape = loader._get_output_shape(data)
        assert shape == (100, 200)  # First array found

    def test_get_output_shape_scalar(self):
        """Test shape detection for scalars."""
        terrain = MockTerrain()
        loader = GriddedDataLoader(terrain)

        shape = loader._get_output_shape(0.5)
        assert shape == ()


class TestIntegration:
    """End-to-end integration tests."""

    def test_simple_pipeline_no_tiling(self):
        """Test simple pipeline that doesn't need tiling."""
        terrain = MockTerrain()
        config = TiledDataConfig(max_output_pixels=1000000)
        loader = GriddedDataLoader(terrain, tile_config=config)

        def load_data(source, extent=None, target_shape=None):
            # Return small array that doesn't need tiling
            return {"data": np.ones((100, 100))}

        def process_data(input_data):
            return {"data": input_data["data"] * 2}

        pipeline = [
            ("load", load_data, {}),
            ("process", process_data, {}),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            loader.cache_dir = Path(tmpdir)
            result = loader.run_pipeline(
                data_source="test_source",
                pipeline=pipeline,
                cache_name="test",
            )

            assert "data" in result
            assert result["data"].shape == (100, 100)
            assert np.all(result["data"] == 2)

    def test_should_tile_detection(self):
        """Test that tiling is triggered for large outputs."""
        terrain = MockTerrain()
        config = TiledDataConfig(max_output_pixels=100000)  # Low threshold
        loader = GriddedDataLoader(terrain, tile_config=config)

        def load_large_data(source, extent=None, target_shape=None):
            # Return large array that needs tiling
            return {"data": np.ones((500, 500))}

        pipeline = [("load", load_large_data, {})]

        with tempfile.TemporaryDirectory() as tmpdir:
            loader.cache_dir = Path(tmpdir)
            should_tile = loader._should_tile(
                data_source="test_source",
                pipeline=pipeline,
                cache_name="test",
            )

            assert should_tile is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
