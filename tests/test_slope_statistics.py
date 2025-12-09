"""
Unit tests for tiled slope statistics computation.

Tests focus on:
1. Geographic pixel mapping accuracy
2. Tile layout boundaries
3. Gradient computation with halos
4. Statistics aggregation correctness
5. Aspect vector averaging
6. Cliff detection capabilities
"""

import pytest
import numpy as np
from rasterio.transform import Affine
from src.snow.slope_statistics import (
    TiledSlopeConfig,
    SlopeStatistics,
    TileSpec,
    compute_pixel_mapping,
    compute_tile_layout,
    compute_tile_slopes,
    aggregate_by_geographic_mapping,
    compute_tiled_slope_statistics,
)


class TestPixelMapping:
    """Tests for compute_pixel_mapping()"""

    def test_pixel_mapping_returns_expected_fields(self):
        """Pixel mapping should return stride and shape information."""
        dem_shape = (1000, 1000)
        dem_transform = Affine.identity()
        target_shape = (100, 100)
        target_transform = Affine.identity()

        mapping = compute_pixel_mapping(
            dem_shape, dem_transform, "EPSG:4326",
            target_shape, target_transform, "EPSG:4326"
        )

        assert "row_stride" in mapping
        assert "col_stride" in mapping
        assert mapping["dem_shape"] == dem_shape
        assert mapping["target_shape"] == target_shape

    def test_pixel_mapping_stride_calculation(self):
        """Stride should be approximately dem_shape / target_shape."""
        dem_shape = (2000, 3000)
        target_shape = (100, 100)

        mapping = compute_pixel_mapping(
            dem_shape, Affine.identity(), "EPSG:4326",
            target_shape, Affine.identity(), "EPSG:4326"
        )

        expected_row_stride = dem_shape[0] / target_shape[0]
        expected_col_stride = dem_shape[1] / target_shape[1]

        assert abs(mapping["row_stride"] - expected_row_stride) < 1
        assert abs(mapping["col_stride"] - expected_col_stride) < 1

    def test_pixel_mapping_with_non_identity_transform(self):
        """Mapping should handle non-identity transforms."""
        dem_transform = Affine(10, 0, 100000, 0, -10, 500000)
        target_transform = Affine(100, 0, 100000, 0, -100, 500000)

        mapping = compute_pixel_mapping(
            (1000, 1000), dem_transform, "EPSG:2154",
            (100, 100), target_transform, "EPSG:2154"
        )

        # Should still compute valid stride
        assert mapping["row_stride"] > 0
        assert mapping["col_stride"] > 0


class TestTileLayout:
    """Tests for compute_tile_layout()"""

    def test_tile_layout_creates_non_empty_list(self):
        """Tile layout should return at least one tile."""
        dem_shape = (1000, 1000)
        target_shape = (100, 100)
        pixel_mapping = {
            "row_stride": 10.0,
            "col_stride": 10.0,
            "dem_shape": dem_shape,
            "target_shape": target_shape,
        }

        tiles = compute_tile_layout(dem_shape, target_shape, pixel_mapping)

        assert len(tiles) > 0
        assert all(isinstance(t, TileSpec) for t in tiles)

    def test_tile_layout_covers_entire_dem(self):
        """All tiles together should cover entire DEM."""
        dem_shape = (2000, 3000)
        target_shape = (200, 300)
        pixel_mapping = {
            "row_stride": 10.0,
            "col_stride": 10.0,
            "dem_shape": dem_shape,
            "target_shape": target_shape,
        }
        config = TiledSlopeConfig(target_tile_outputs=50)

        tiles = compute_tile_layout(dem_shape, target_shape, pixel_mapping, config)

        # Check that tiles cover the full output space
        max_out_row = max(t.out_slice[0].stop for t in tiles)
        max_out_col = max(t.out_slice[1].stop for t in tiles)

        assert max_out_row >= target_shape[0]
        assert max_out_col >= target_shape[1]

    def test_tile_spec_slices_are_valid(self):
        """All tile slices should be within array bounds."""
        dem_shape = (1000, 1000)
        target_shape = (100, 100)
        pixel_mapping = {
            "row_stride": 10.0,
            "col_stride": 10.0,
            "dem_shape": dem_shape,
            "target_shape": target_shape,
        }

        tiles = compute_tile_layout(dem_shape, target_shape, pixel_mapping)

        for tile in tiles:
            # src_slice (with halo) should be within bounds
            assert tile.src_slice[0].start >= 0
            assert tile.src_slice[0].stop <= dem_shape[0]
            assert tile.src_slice[1].start >= 0
            assert tile.src_slice[1].stop <= dem_shape[1]

            # core_slice offset should be valid
            assert tile.core_slice[0].start >= 0
            assert tile.core_slice[0].stop <= (
                tile.src_slice[0].stop - tile.src_slice[0].start
            )

    def test_tile_layout_with_large_stride(self):
        """Tile layout should handle large strides gracefully."""
        dem_shape = (10000, 10000)
        target_shape = (100, 100)
        pixel_mapping = {
            "row_stride": 100.0,
            "col_stride": 100.0,
            "dem_shape": dem_shape,
            "target_shape": target_shape,
        }
        config = TiledSlopeConfig(target_tile_outputs=50, max_tile_size=2000)

        tiles = compute_tile_layout(dem_shape, target_shape, pixel_mapping, config)

        # Tiles should still be created
        assert len(tiles) > 0
        # But should be clamped to max_tile_size
        for tile in tiles:
            tile_height = tile.src_slice[0].stop - tile.src_slice[0].start
            tile_width = tile.src_slice[1].stop - tile.src_slice[1].start
            # Including halo, might be slightly over, but core should respect limit
            core_height = tile.core_slice[0].stop - tile.core_slice[0].start
            assert core_height <= 2100  # Allow some margin for halo


class TestTileSlopes:
    """Tests for compute_tile_slopes()"""

    def test_compute_tile_slopes_returns_correct_shape(self):
        """Slope computation should return arrays matching core_slice."""
        # Create a simple 10x10 tile with halo
        tile_data = np.array([
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            [100, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 100],
            [100, 110, 120, 120, 120, 120, 120, 120, 120, 120, 110, 100],
            [100, 110, 120, 130, 130, 130, 130, 130, 130, 120, 110, 100],
            [100, 110, 120, 130, 140, 140, 140, 140, 130, 120, 110, 100],
            [100, 110, 120, 130, 140, 140, 140, 140, 130, 120, 110, 100],
            [100, 110, 120, 130, 130, 130, 130, 130, 130, 120, 110, 100],
            [100, 110, 120, 120, 120, 120, 120, 120, 120, 120, 110, 100],
            [100, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 100],
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        ], dtype=np.float32)

        core_slice = (slice(1, 11), slice(1, 11))

        slope_deg, aspect_deg = compute_tile_slopes(
            tile_data, core_slice, cell_size_m=10.0
        )

        assert slope_deg.shape == (10, 10)
        assert aspect_deg.shape == (10, 10)

    def test_compute_tile_slopes_produces_valid_ranges(self):
        """Slopes should be in valid range [0, 90) degrees."""
        # Create tilted terrain
        tile_data = np.array([
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5],
        ], dtype=np.float32)

        core_slice = (slice(1, 5), slice(1, 5))

        slope_deg, aspect_deg = compute_tile_slopes(
            tile_data, core_slice, cell_size_m=10.0
        )

        assert np.all(slope_deg >= 0)
        assert np.all(slope_deg < 90)
        assert np.all(aspect_deg >= 0)
        assert np.all(aspect_deg < 360)

    def test_compute_tile_slopes_flat_terrain(self):
        """Flat terrain should have zero slope."""
        tile_data = np.ones((6, 6), dtype=np.float32) * 100

        core_slice = (slice(1, 5), slice(1, 5))

        slope_deg, aspect_deg = compute_tile_slopes(
            tile_data, core_slice, cell_size_m=10.0
        )

        # Flat terrain should have very small slopes (numerical noise)
        assert np.max(slope_deg) < 0.1

    def test_compute_tile_slopes_steep_slope(self):
        """Steep terrain should have large slopes."""
        # Create sharp slope: 10m elevation change across 10m
        tile_data = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 5, 5, 5, 5, 0],
            [0, 10, 10, 10, 10, 0],
            [0, 10, 10, 10, 10, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.float32)

        core_slice = (slice(1, 5), slice(1, 5))

        slope_deg, aspect_deg = compute_tile_slopes(
            tile_data, core_slice, cell_size_m=10.0
        )

        # Should detect slopes around 45 degrees (arctan(10/10) ≈ 45°)
        # Peak slopes should be significant
        assert np.max(slope_deg) > 20


class TestAggregation:
    """Tests for aggregate_by_geographic_mapping()"""

    def test_aggregation_output_shape(self):
        """Aggregation should produce correct output shape."""
        output_shape = (10, 10)
        slope = np.random.rand(100, 100).astype(np.float32)
        aspect = np.random.rand(100, 100).astype(np.float32) * 360
        elevation = np.random.rand(100, 100).astype(np.float32) * 1000

        stats = aggregate_by_geographic_mapping(
            slope, aspect, elevation,
            row_stride=10, col_stride=10,
            output_shape=output_shape
        )

        for key in ["mean", "max", "min", "std", "p95", "roughness"]:
            assert stats[key].shape == output_shape

    def test_aggregation_mean_within_range(self):
        """Aggregated mean should be between min and max."""
        output_shape = (5, 5)
        # Create test data with known values
        slope = np.ones((50, 50), dtype=np.float32) * 15.0
        aspect = np.ones((50, 50), dtype=np.float32) * 45.0
        elevation = np.ones((50, 50), dtype=np.float32) * 1000.0

        stats = aggregate_by_geographic_mapping(
            slope, aspect, elevation,
            row_stride=10, col_stride=10,
            output_shape=output_shape
        )

        # All values should be 15.0
        assert np.allclose(stats["mean"], 15.0, atol=0.01)
        assert np.allclose(stats["max"], 15.0, atol=0.01)
        assert np.allclose(stats["min"], 15.0, atol=0.01)
        assert np.allclose(stats["std"], 0.0, atol=0.01)

    def test_aggregation_detects_variation(self):
        """Aggregation should detect slope variation within blocks."""
        output_shape = (2, 2)
        # Create data with variation within blocks
        slope = np.array([
            [10, 10, 20, 20],
            [10, 10, 20, 20],
            [30, 30, 40, 40],
            [30, 30, 40, 40],
        ], dtype=np.float32)
        aspect = np.zeros((4, 4), dtype=np.float32)
        elevation = np.zeros((4, 4), dtype=np.float32)

        stats = aggregate_by_geographic_mapping(
            slope, aspect, elevation,
            row_stride=2, col_stride=2,
            output_shape=output_shape
        )

        # Check block statistics
        assert stats["mean"][0, 0] == 10.0
        assert stats["max"][0, 0] == 10.0
        assert stats["mean"][0, 1] == 20.0
        assert stats["mean"][1, 0] == 30.0
        assert stats["mean"][1, 1] == 40.0

    def test_aggregation_aspect_vector_averaging(self):
        """Aspect should use vector averaging (not simple mean)."""
        output_shape = (1, 1)
        # Test case: aspects at 0° and 360° should average to 0°, not 180°
        aspect = np.array([
            [1, 1, 359, 359],
            [1, 1, 359, 359],
            [1, 1, 359, 359],
            [1, 1, 359, 359],
        ], dtype=np.float32)
        slope = np.ones((4, 4), dtype=np.float32)
        elevation = np.ones((4, 4), dtype=np.float32)

        stats = aggregate_by_geographic_mapping(
            slope, aspect, elevation,
            row_stride=2, col_stride=2,
            output_shape=output_shape
        )

        # Reconstruct aspect from sin/cos
        aspect_sin = stats["aspect_sin"][0, 0]
        aspect_cos = stats["aspect_cos"][0, 0]
        reconstructed_aspect = (
            np.degrees(np.arctan2(aspect_sin, aspect_cos)) % 360
        )

        # Should be close to 0° (or 360°), not 180°
        assert reconstructed_aspect < 90 or reconstructed_aspect > 270

    def test_aggregation_percentile_computation(self):
        """95th percentile should be appropriately high."""
        output_shape = (1, 1)
        # Create data where we know the percentiles
        slopes = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
        aspect = np.zeros((10, 10), dtype=np.float32)
        elevation = np.zeros((10, 10), dtype=np.float32)

        stats = aggregate_by_geographic_mapping(
            slopes, aspect, elevation,
            row_stride=10, col_stride=10,
            output_shape=output_shape
        )

        # 95th percentile of [1..100] should be around 95.5
        assert 94 < stats["p95"][0, 0] < 97


class TestSlopeStatisticsDataclass:
    """Tests for SlopeStatistics dataclass properties"""

    def test_dominant_aspect_computation(self):
        """dominant_aspect property should correctly compute aspect from sin/cos."""
        stats = SlopeStatistics(
            slope_mean=np.zeros((2, 2)),
            slope_max=np.zeros((2, 2)),
            slope_min=np.zeros((2, 2)),
            slope_std=np.zeros((2, 2)),
            slope_p95=np.zeros((2, 2)),
            roughness=np.zeros((2, 2)),
            aspect_sin=np.array([[0, 1], [-1, 0]], dtype=np.float32),
            aspect_cos=np.array([[1, 0], [0, -1]], dtype=np.float32),
        )

        aspect = stats.dominant_aspect

        assert aspect.shape == (2, 2)
        # [0] = ~0° (cos only), [1] = ~90° (equal sin/cos), [2] = ~270°, [3] = ~180°
        # For purely cos case, atan2(0, 1) = 0
        assert abs(aspect[0, 0]) < 1 or aspect[0, 0] > 359  # ~0°
        assert 89 < aspect[0, 1] < 91  # ~90°
        assert 269 < aspect[1, 0] < 271 or aspect[1, 0] < 1  # ~270° or ~0°
        assert 179 < aspect[1, 1] < 181  # ~180°

    def test_aspect_strength_computation(self):
        """aspect_strength should be 0 to 1."""
        stats = SlopeStatistics(
            slope_mean=np.zeros((2, 2)),
            slope_max=np.zeros((2, 2)),
            slope_min=np.zeros((2, 2)),
            slope_std=np.zeros((2, 2)),
            slope_p95=np.zeros((2, 2)),
            roughness=np.zeros((2, 2)),
            aspect_sin=np.array([[0.0, 0.707], [0.0, 0.0]], dtype=np.float32),
            aspect_cos=np.array([[1.0, 0.707], [0.0, 0.0]], dtype=np.float32),
        )

        strength = stats.aspect_strength

        assert strength.shape == (2, 2)
        assert np.all(strength >= 0)
        assert np.all(strength <= 1)
        # [0,0] = 1 (all cos), [0,1] ≈ 1 (both equal), [1,0] = 0, [1,1] = 0
        assert abs(strength[0, 0] - 1.0) < 0.01
        assert 0.99 < strength[0, 1] < 1.01
        assert strength[1, 0] < 0.01


class TestIntegration:
    """Integration tests for full pipeline"""

    def test_compute_tiled_slope_statistics_simple_dem(self):
        """Full pipeline should work on simple synthetic DEM."""
        # Create simple synthetic DEM (4x4 -> 2x2 gives stride 2, evenly divisible)
        dem = np.array([
            [100, 100, 100, 100],
            [100, 110, 110, 100],
            [100, 110, 120, 100],
            [100, 100, 100, 100],
        ], dtype=np.float32)

        dem_transform = Affine.identity()
        target_shape = (2, 2)
        target_transform = Affine.identity()

        result = compute_tiled_slope_statistics(
            dem, dem_transform, "EPSG:4326",
            target_shape, target_transform, "EPSG:4326",
        )

        assert isinstance(result, SlopeStatistics)
        assert result.slope_mean.shape == target_shape
        assert result.slope_max.shape == target_shape
        assert result.slope_min.shape == target_shape
        assert result.slope_std.shape == target_shape
        assert result.slope_p95.shape == target_shape
        assert result.roughness.shape == target_shape

    def test_compute_tiled_slope_statistics_cliff_detection(self):
        """Pipeline should detect cliffs (max > mean significantly)."""
        # Create DEM with a cliff
        dem = np.ones((100, 100), dtype=np.float32) * 100
        # Add a sharp cliff in a small region
        dem[40:60, 40:60] = 150  # Abrupt 50m rise

        dem_transform = Affine.identity()
        target_shape = (10, 10)
        target_transform = Affine.identity()

        result = compute_tiled_slope_statistics(
            dem, dem_transform, "EPSG:4326",
            target_shape, target_transform, "EPSG:4326",
        )

        # Find the pixel containing the cliff (around output pixel [4,4] or [5,5])
        cliff_region_max = np.max(result.slope_max[3:7, 3:7])
        flat_region_max = np.max(result.slope_max[0:2, 0:2])

        # Cliff region should have significantly higher max slope
        assert cliff_region_max > flat_region_max * 3

    def test_compute_tiled_slope_statistics_output_no_nans(self):
        """Output should not contain NaNs."""
        dem = np.random.rand(100, 100).astype(np.float32) * 100

        dem_transform = Affine.identity()
        target_shape = (10, 10)
        target_transform = Affine.identity()

        result = compute_tiled_slope_statistics(
            dem, dem_transform, "EPSG:4326",
            target_shape, target_transform, "EPSG:4326",
        )

        assert not np.any(np.isnan(result.slope_mean))
        assert not np.any(np.isnan(result.slope_max))
        assert not np.any(np.isnan(result.slope_min))
        assert not np.any(np.isnan(result.roughness))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
