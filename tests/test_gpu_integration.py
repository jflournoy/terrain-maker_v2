"""
Tests for GPU operation integration into existing codebase.

TDD RED phase: These tests verify that GPU-accelerated functions
integrate correctly with existing water detection and road processing.
"""

import numpy as np
import pytest


class TestWaterSlopeIntegration:
    """Tests for GPU slope integration in water detection."""

    def test_calculate_slope_uses_gpu_backend(self):
        """Water module's _calculate_slope should use GPU backend."""
        from src.terrain.water import _calculate_slope
        from src.terrain.gpu_ops import gpu_horn_slope

        # Create test DEM
        dem = np.random.rand(100, 100).astype(np.float32) * 1000

        # Both should produce identical results
        water_slope = _calculate_slope(dem)
        gpu_slope = gpu_horn_slope(dem)

        np.testing.assert_allclose(water_slope, gpu_slope, rtol=1e-4, atol=1e-6)

    def test_identify_water_by_slope_still_works(self):
        """Water detection should still work after GPU integration."""
        from src.terrain.water import identify_water_by_slope

        # Create DEM with flat water-like region
        dem = np.random.rand(100, 100).astype(np.float32) * 100
        dem[40:60, 40:60] = 50.0  # Flat region (water)

        water_mask = identify_water_by_slope(dem, slope_threshold=0.5)

        # Flat region should be detected as water
        assert water_mask[50, 50], "Center of flat region should be water"
        assert np.sum(water_mask[40:60, 40:60]) > 100, "Most of flat region should be water"

    def test_water_slope_handles_nan(self):
        """Water slope calculation should handle NaN after GPU integration."""
        from src.terrain.water import _calculate_slope

        dem = np.random.rand(50, 50).astype(np.float32) * 1000
        dem[20:25, 20:25] = np.nan

        slope = _calculate_slope(dem)

        # NaN regions should remain NaN
        assert np.all(np.isnan(slope[20:25, 20:25]))
        # Non-NaN regions should have valid values
        assert not np.any(np.isnan(slope[0:10, 0:10]))


class TestAdvancedVizSlopeIntegration:
    """Tests for GPU slope integration in advanced_viz.horn_slope."""

    def test_horn_slope_uses_gpu_backend(self):
        """advanced_viz.horn_slope should use GPU backend."""
        from src.terrain.advanced_viz import horn_slope
        from src.terrain.gpu_ops import gpu_horn_slope

        dem = np.random.rand(100, 100).astype(np.float32) * 1000

        # Both should produce identical results
        viz_slope = horn_slope(dem)
        gpu_slope = gpu_horn_slope(dem)

        np.testing.assert_allclose(viz_slope, gpu_slope, rtol=1e-4, atol=1e-6)

    def test_horn_slope_exported_from_terrain(self):
        """horn_slope should be importable from terrain package."""
        from src.terrain import horn_slope

        assert callable(horn_slope)

        # Should work correctly
        dem = np.random.rand(50, 50).astype(np.float32) * 500
        result = horn_slope(dem)
        assert result.shape == dem.shape


class TestRoadGaussianIntegration:
    """Tests for GPU Gaussian blur integration in roads module."""

    def test_smooth_road_mask_uses_gpu(self):
        """Road mask smoothing should use GPU Gaussian blur."""
        from src.terrain.roads import smooth_road_mask

        # Create binary road mask
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[45:55, 20:80] = 1.0  # Horizontal road

        smoothed = smooth_road_mask(mask, sigma=2.0)

        # Should still be float, with values in [0, 1]
        assert smoothed.dtype == np.float32 or smoothed.dtype == np.float64
        assert smoothed.min() >= 0.0
        assert smoothed.max() <= 1.0
        # Edges should be smoothed (not binary)
        assert 0.0 < smoothed[45, 20] < 1.0  # Edge pixel

    def test_smooth_road_mask_matches_gpu(self):
        """Road mask smoothing should match GPU Gaussian blur results."""
        from src.terrain.roads import smooth_road_mask
        from src.terrain.gpu_ops import gpu_gaussian_blur

        mask = np.zeros((100, 100), dtype=np.float32)
        mask[45:55, 20:80] = 1.0

        sigma = 2.0
        roads_result = smooth_road_mask(mask, sigma=sigma)
        gpu_result = gpu_gaussian_blur(mask, sigma=sigma)

        # Clip both to [0, 1] for comparison (road function clips)
        gpu_clipped = np.clip(gpu_result, 0, 1)

        # Interior should match closely
        margin = int(3 * sigma)
        np.testing.assert_allclose(
            roads_result[margin:-margin, margin:-margin],
            gpu_clipped[margin:-margin, margin:-margin],
            rtol=0.02,
            atol=0.02,
        )


class TestTransformsGpuIntegration:
    """Tests for GPU integration in transforms module."""

    def test_despeckle_scores_removes_noise(self):
        """despeckle_scores should remove isolated speckles via median filter."""
        from src.terrain.transforms import despeckle_scores

        # Create score data with isolated speckles
        np.random.seed(42)
        score = np.zeros((100, 100), dtype=np.float32)
        score[40:60, 40:60] = 0.8  # Main region
        score[45, 45] = 0.0  # Isolated speckle (should be removed)
        score[10, 10] = 1.0  # Isolated bright pixel

        # Run despeckle
        despeckled = despeckle_scores(score, kernel_size=3)

        # Should preserve shape
        assert despeckled.shape == score.shape
        # Isolated speckle in main region should be filled
        assert despeckled[45, 45] > 0.5, "Speckle should be filled by median"

    def test_smooth_score_data_preserves_edges(self):
        """smooth_score_data should preserve edges (bilateral filter)."""
        from src.terrain.transforms import smooth_score_data

        # Create score data with sharp edge
        score = np.zeros((100, 100), dtype=np.float32)
        score[:, 50:] = 1.0  # Sharp vertical edge

        smoothed = smooth_score_data(score, sigma_spatial=3.0)

        # Should preserve shape
        assert smoothed.shape == score.shape
        # Edge should still be relatively sharp (bilateral preserves edges)
        edge_gradient = np.abs(smoothed[:, 51] - smoothed[:, 49])
        assert np.mean(edge_gradient) > 0.5, "Edge should be mostly preserved"
