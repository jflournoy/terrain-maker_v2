"""
Tests for transform operations.

Tests the raster transformation functions that will be extracted from core.py.
"""

import pytest
import numpy as np


class TestDownsampleRaster:
    """Tests for downsample_raster function."""

    def test_downsample_raster_imports(self):
        """Test that downsample_raster can be imported."""
        from src.terrain.transforms import downsample_raster

        assert callable(downsample_raster)

    def test_downsample_raster_reduces_size(self):
        """Test that downsampling reduces array size."""
        from src.terrain.transforms import downsample_raster

        # 100x100 array
        data = np.random.rand(100, 100)

        # Downsample by 0.5 (50%)
        transform_func = downsample_raster(zoom_factor=0.5)
        downsampled, _, _ = transform_func(data)

        # Should be approximately 50x50
        assert downsampled.shape[0] == 50
        assert downsampled.shape[1] == 50

    def test_downsample_raster_preserves_nodata(self):
        """Test that nodata values are preserved."""
        from src.terrain.transforms import downsample_raster

        data = np.ones((20, 20))
        data[5:10, 5:10] = np.nan

        transform_func = downsample_raster(zoom_factor=0.5, nodata_value=np.nan)
        downsampled, _, _ = transform_func(data)

        # Should still have NaN values
        assert np.any(np.isnan(downsampled))

    def test_downsample_raster_returns_three_tuple(self):
        """Test that function returns (data, transform, crs) tuple."""
        from src.terrain.transforms import downsample_raster

        data = np.random.rand(10, 10)
        transform_func = downsample_raster(zoom_factor=0.5)
        result = transform_func(data)

        assert isinstance(result, tuple)
        assert len(result) == 3


class TestSmoothRaster:
    """Tests for smooth_raster function."""

    def test_smooth_raster_imports(self):
        """Test that smooth_raster can be imported."""
        from src.terrain.transforms import smooth_raster

        assert callable(smooth_raster)

    def test_smooth_raster_reduces_noise(self):
        """Test that smoothing reduces high-frequency noise."""
        from src.terrain.transforms import smooth_raster

        # Create noisy data
        np.random.seed(42)
        data = np.ones((50, 50)) + np.random.randn(50, 50) * 0.1

        transform_func = smooth_raster(window_size=5)
        smoothed, _, _ = transform_func(data)

        # Smoothed data should have lower variance
        assert np.std(smoothed) < np.std(data)

    def test_smooth_raster_preserves_shape(self):
        """Test that smoothing preserves array shape."""
        from src.terrain.transforms import smooth_raster

        data = np.random.rand(30, 40)
        transform_func = smooth_raster(window_size=3)
        smoothed, _, _ = transform_func(data)

        assert smoothed.shape == data.shape

    def test_smooth_raster_handles_nodata(self):
        """Test that nodata values are preserved in core region."""
        from src.terrain.transforms import smooth_raster

        data = np.ones((20, 20))
        data[5:15, 5:15] = np.nan

        transform_func = smooth_raster(window_size=3, nodata_value=np.nan)
        smoothed, _, _ = transform_func(data)

        # Core NaN region should still have NaN (allowing for filter edge effects)
        # Check center of NaN region, not edges
        assert np.all(np.isnan(smoothed[7:13, 7:13]))

    def test_smooth_raster_returns_three_tuple(self):
        """Test that function returns (data, transform, crs) tuple."""
        from src.terrain.transforms import smooth_raster

        data = np.random.rand(10, 10)
        transform_func = smooth_raster(window_size=3)
        result = transform_func(data)

        assert isinstance(result, tuple)
        assert len(result) == 3


class TestFlipRaster:
    """Tests for flip_raster function."""

    def test_flip_raster_imports(self):
        """Test that flip_raster can be imported."""
        from src.terrain.transforms import flip_raster

        assert callable(flip_raster)

    def test_flip_raster_horizontal(self):
        """Test horizontal (top-bottom) flip."""
        from src.terrain.transforms import flip_raster

        # Create array with distinct rows
        data = np.array([[1, 2], [3, 4], [5, 6]])

        transform_func = flip_raster(axis="horizontal")
        flipped, _, _ = transform_func(data)

        # After flip, first row should be last row
        assert np.array_equal(flipped[0], [5, 6])
        assert np.array_equal(flipped[2], [1, 2])

    def test_flip_raster_vertical(self):
        """Test vertical (left-right) flip."""
        from src.terrain.transforms import flip_raster

        # Create array with distinct columns
        data = np.array([[1, 2, 3], [4, 5, 6]])

        transform_func = flip_raster(axis="vertical")
        flipped, _, _ = transform_func(data)

        # After flip, first column should be last column
        assert np.array_equal(flipped[:, 0], [3, 6])
        assert np.array_equal(flipped[:, 2], [1, 4])

    def test_flip_raster_preserves_shape(self):
        """Test that flipping preserves array shape."""
        from src.terrain.transforms import flip_raster

        data = np.random.rand(25, 30)

        transform_func = flip_raster(axis="horizontal")
        flipped, _, _ = transform_func(data)

        assert flipped.shape == data.shape

    def test_flip_raster_invalid_axis_raises(self):
        """Test that invalid axis raises ValueError."""
        from src.terrain.transforms import flip_raster

        data = np.array([[1, 2], [3, 4]])
        transform_func = flip_raster(axis="invalid")

        with pytest.raises(ValueError):
            transform_func(data)

    def test_flip_raster_returns_three_tuple(self):
        """Test that function returns (data, transform, crs) tuple."""
        from src.terrain.transforms import flip_raster

        data = np.random.rand(10, 10)
        transform_func = flip_raster(axis="horizontal")
        result = transform_func(data)

        assert isinstance(result, tuple)
        assert len(result) == 3


class TestScaleElevation:
    """Tests for scale_elevation function."""

    def test_scale_elevation_imports(self):
        """Test that scale_elevation can be imported."""
        from src.terrain.transforms import scale_elevation

        assert callable(scale_elevation)

    def test_scale_elevation_multiplies_values(self):
        """Test that scaling multiplies elevation values."""
        from src.terrain.transforms import scale_elevation

        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        transform_func = scale_elevation(scale_factor=2.0)
        scaled, _, _ = transform_func(data)

        # All values should be doubled
        assert np.allclose(scaled, [[2.0, 4.0], [6.0, 8.0]])

    def test_scale_elevation_handles_nodata(self):
        """Test that nodata values are preserved."""
        from src.terrain.transforms import scale_elevation

        data = np.array([[1.0, np.nan], [3.0, 4.0]])

        transform_func = scale_elevation(scale_factor=2.0, nodata_value=np.nan)
        scaled, _, _ = transform_func(data)

        # NaN should remain NaN
        assert np.isnan(scaled[0, 1])
        # Other values should be scaled
        assert scaled[0, 0] == 2.0
        assert scaled[1, 0] == 6.0

    def test_scale_elevation_preserves_shape(self):
        """Test that scaling preserves array shape."""
        from src.terrain.transforms import scale_elevation

        data = np.random.rand(20, 30)
        transform_func = scale_elevation(scale_factor=1.5)
        scaled, _, _ = transform_func(data)

        assert scaled.shape == data.shape

    def test_scale_elevation_with_factor_one(self):
        """Test that scale factor of 1.0 leaves data unchanged."""
        from src.terrain.transforms import scale_elevation

        data = np.random.rand(10, 10)
        transform_func = scale_elevation(scale_factor=1.0)
        scaled, _, _ = transform_func(data)

        assert np.allclose(scaled, data)

    def test_scale_elevation_returns_three_tuple(self):
        """Test that function returns (data, transform, crs) tuple."""
        from src.terrain.transforms import scale_elevation

        data = np.random.rand(10, 10)
        transform_func = scale_elevation(scale_factor=2.0)
        result = transform_func(data)

        assert isinstance(result, tuple)
        assert len(result) == 3


class TestFeaturePreservingSmooth:
    """Tests for feature_preserving_smooth bilateral filter function."""

    def test_feature_preserving_smooth_imports(self):
        """Test that feature_preserving_smooth can be imported."""
        from src.terrain.transforms import feature_preserving_smooth

        assert callable(feature_preserving_smooth)

    def test_feature_preserving_smooth_returns_three_tuple(self):
        """Test that function returns (data, transform, crs) tuple."""
        from src.terrain.transforms import feature_preserving_smooth

        data = np.random.rand(20, 20)
        transform_func = feature_preserving_smooth(sigma_spatial=2.0)
        result = transform_func(data)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_feature_preserving_smooth_preserves_shape(self):
        """Test that smoothing preserves array shape."""
        from src.terrain.transforms import feature_preserving_smooth

        data = np.random.rand(30, 40)
        transform_func = feature_preserving_smooth(sigma_spatial=2.0)
        smoothed, _, _ = transform_func(data)

        assert smoothed.shape == data.shape

    def test_feature_preserving_smooth_reduces_noise(self):
        """Test that smoothing reduces high-frequency noise in flat areas."""
        from src.terrain.transforms import feature_preserving_smooth

        # Create flat terrain with noise
        np.random.seed(42)
        flat_terrain = np.ones((50, 50)) * 100  # Flat at 100m
        noisy_terrain = flat_terrain + np.random.randn(50, 50) * 5  # +/- 5m noise

        transform_func = feature_preserving_smooth(sigma_spatial=3.0)
        smoothed, _, _ = transform_func(noisy_terrain)

        # Smoothed data should have lower variance (noise reduced)
        assert np.std(smoothed) < np.std(noisy_terrain)

    def test_feature_preserving_smooth_preserves_edges(self):
        """Test that sharp elevation changes (edges) are preserved."""
        from src.terrain.transforms import feature_preserving_smooth

        # Create terrain with sharp ridge (cliff)
        terrain = np.zeros((100, 100))
        terrain[:, 50:] = 100  # Sharp 100m cliff at column 50

        # Add some noise
        np.random.seed(42)
        noisy_terrain = terrain + np.random.randn(100, 100) * 2

        transform_func = feature_preserving_smooth(sigma_spatial=3.0, sigma_intensity=10.0)
        smoothed, _, _ = transform_func(noisy_terrain)

        # The edge gradient at column 50 should still be steep
        # (bilateral filter preserves edges)
        edge_gradient = np.abs(smoothed[:, 52] - smoothed[:, 48]).mean()
        assert edge_gradient > 80, f"Edge should be preserved, got gradient {edge_gradient}"

    def test_feature_preserving_smooth_handles_nodata(self):
        """Test that nodata values are preserved."""
        from src.terrain.transforms import feature_preserving_smooth

        data = np.ones((30, 30)) * 50
        data[10:20, 10:20] = np.nan  # NaN region

        transform_func = feature_preserving_smooth(sigma_spatial=2.0, nodata_value=np.nan)
        smoothed, _, _ = transform_func(data)

        # Core NaN region should still have NaN
        assert np.all(np.isnan(smoothed[12:18, 12:18]))

    def test_feature_preserving_smooth_auto_intensity_sigma(self):
        """Test that intensity sigma is auto-calculated when None."""
        from src.terrain.transforms import feature_preserving_smooth

        # Create terrain with known elevation range
        data = np.linspace(0, 100, 400).reshape(20, 20)

        # Should not raise, should auto-calculate sigma_intensity
        transform_func = feature_preserving_smooth(sigma_spatial=2.0, sigma_intensity=None)
        smoothed, _, _ = transform_func(data)

        # Should return valid data
        assert not np.any(np.isnan(smoothed))
        assert smoothed.shape == data.shape

    def test_feature_preserving_smooth_with_transform(self):
        """Test that affine transform is passed through unchanged."""
        from src.terrain.transforms import feature_preserving_smooth
        from rasterio import Affine

        data = np.random.rand(20, 20)
        input_transform = Affine(30.0, 0, 500000, 0, -30.0, 5000000)

        transform_func = feature_preserving_smooth(sigma_spatial=2.0)
        _, output_transform, _ = transform_func(data, input_transform)

        # Transform should be unchanged by smoothing
        assert output_transform == input_transform

    def test_feature_preserving_smooth_large_sigma_capped(self):
        """Test that excessively large sigma values are capped to prevent memory issues."""
        from src.terrain.transforms import feature_preserving_smooth
        import logging

        data = np.random.rand(30, 30)

        # Create transform with very large sigma (should be capped)
        transform_func = feature_preserving_smooth(sigma_spatial=100.0)

        # Should not raise, should complete without memory error
        smoothed, _, _ = transform_func(data)

        # Should return valid data of same shape
        assert smoothed.shape == data.shape
        assert not np.any(np.isnan(smoothed))


class TestSmoothScoreData:
    """Tests for smooth_score_data function (TDD).

    Smooths score grids (0-1 range) using bilateral filtering to reduce
    blocky pixelation from low-resolution source data (SNODAS ~925m) when
    displayed on high-resolution terrain (~30m DEM).
    """

    def test_smooth_score_data_imports(self):
        """Test that smooth_score_data can be imported."""
        from src.terrain.transforms import smooth_score_data

        assert callable(smooth_score_data)

    def test_smooth_score_data_returns_array(self):
        """Test that smooth_score_data returns a numpy array."""
        from src.terrain.transforms import smooth_score_data

        # Simple score grid (0-1 range)
        scores = np.array([
            [0.2, 0.2, 0.8, 0.8],
            [0.2, 0.2, 0.8, 0.8],
            [0.3, 0.3, 0.7, 0.7],
            [0.3, 0.3, 0.7, 0.7],
        ], dtype=np.float32)

        smoothed = smooth_score_data(scores)

        assert isinstance(smoothed, np.ndarray)
        assert smoothed.shape == scores.shape

    def test_smooth_score_data_preserves_range(self):
        """Test that smoothed scores stay in 0-1 range."""
        from src.terrain.transforms import smooth_score_data

        scores = np.random.rand(50, 50).astype(np.float32)

        smoothed = smooth_score_data(scores)

        assert np.nanmin(smoothed) >= 0.0
        assert np.nanmax(smoothed) <= 1.0

    def test_smooth_score_data_reduces_blockiness(self):
        """Test that smoothing reduces sharp block boundaries.

        Simulates the blocky appearance of upsampled low-res SNODAS data.
        After smoothing, the variance in gradient magnitude should decrease.
        """
        from src.terrain.transforms import smooth_score_data

        # Create blocky score pattern (simulating upsampled low-res data)
        scores = np.zeros((40, 40), dtype=np.float32)
        scores[0:20, 0:20] = 0.3   # Block 1
        scores[0:20, 20:40] = 0.8  # Block 2
        scores[20:40, 0:20] = 0.6  # Block 3
        scores[20:40, 20:40] = 0.4 # Block 4

        smoothed = smooth_score_data(scores, sigma_spatial=3.0)

        # Calculate gradient magnitude to measure edge sharpness
        grad_y_orig, grad_x_orig = np.gradient(scores)
        grad_y_smooth, grad_x_smooth = np.gradient(smoothed)

        # Max gradient should be lower after smoothing (softer edges)
        max_grad_orig = np.max(np.abs(grad_x_orig) + np.abs(grad_y_orig))
        max_grad_smooth = np.max(np.abs(grad_x_smooth) + np.abs(grad_y_smooth))

        assert max_grad_smooth < max_grad_orig, (
            f"Smoothing should reduce max gradient: {max_grad_smooth:.4f} >= {max_grad_orig:.4f}"
        )

    def test_smooth_score_data_preserves_score_regions(self):
        """Test that smoothing preserves distinct score regions.

        High scores should stay high, low scores should stay low.
        Only the transitions should be softened.
        """
        from src.terrain.transforms import smooth_score_data

        # Create distinct regions
        scores = np.zeros((40, 40), dtype=np.float32)
        scores[5:15, 5:15] = 0.9  # High score region (center-ish)
        scores[25:35, 25:35] = 0.1  # Low score region

        smoothed = smooth_score_data(scores, sigma_spatial=2.0, sigma_intensity=0.2)

        # Center of high region should still be high
        assert smoothed[10, 10] > 0.7, f"High region center degraded: {smoothed[10, 10]}"
        # Center of low region should still be low
        assert smoothed[30, 30] < 0.3, f"Low region center degraded: {smoothed[30, 30]}"

    def test_smooth_score_data_handles_nan(self):
        """Test that NaN values are preserved."""
        from src.terrain.transforms import smooth_score_data

        scores = np.random.rand(20, 20).astype(np.float32)
        scores[8:12, 8:12] = np.nan  # NaN region in center

        smoothed = smooth_score_data(scores)

        # NaN region should still be NaN
        assert np.all(np.isnan(smoothed[8:12, 8:12]))
        # Non-NaN regions should have valid values
        assert not np.any(np.isnan(smoothed[0:5, 0:5]))

    def test_smooth_score_data_sigma_spatial_effect(self):
        """Test that larger sigma_spatial produces more smoothing."""
        from src.terrain.transforms import smooth_score_data

        # Blocky pattern
        scores = np.zeros((30, 30), dtype=np.float32)
        scores[:, 15:] = 1.0

        smoothed_small = smooth_score_data(scores, sigma_spatial=1.0)
        smoothed_large = smooth_score_data(scores, sigma_spatial=5.0)

        # Larger sigma should produce smoother transition (lower max gradient)
        grad_small = np.abs(np.gradient(smoothed_small, axis=1)).max()
        grad_large = np.abs(np.gradient(smoothed_large, axis=1)).max()

        assert grad_large < grad_small, (
            f"Larger sigma should produce smoother transition: {grad_large} >= {grad_small}"
        )

    def test_smooth_score_data_default_sigma_intensity(self):
        """Test that default sigma_intensity auto-calculates from score range.

        For score data (0-1 range), sigma_intensity should be automatically
        set to a reasonable fraction of the range. The default behavior is
        edge-preserving: sharp 0→1 transitions are maintained (by design).
        """
        from src.terrain.transforms import smooth_score_data

        # Create scores with small variations (like upsampled SNODAS blocks)
        # Adjacent blocks have similar but not identical values
        scores = np.zeros((40, 40), dtype=np.float32)
        scores[0:20, 0:20] = 0.5  # Block 1
        scores[0:20, 20:40] = 0.55  # Block 2 (similar to block 1)
        scores[20:40, 0:20] = 0.45  # Block 3 (similar)
        scores[20:40, 20:40] = 0.52  # Block 4

        # Should work without explicit sigma_intensity
        smoothed = smooth_score_data(scores, sigma_spatial=3.0)

        # Transitions between similar blocks should be smoothed
        # Check gradient at block boundaries is reduced
        grad_y_orig, grad_x_orig = np.gradient(scores)
        grad_y_smooth, grad_x_smooth = np.gradient(smoothed)

        orig_max_grad = np.max(np.abs(grad_x_orig) + np.abs(grad_y_orig))
        smooth_max_grad = np.max(np.abs(grad_x_smooth) + np.abs(grad_y_smooth))

        assert smooth_max_grad < orig_max_grad, (
            f"Similar-value block boundaries should be smoothed: {smooth_max_grad} >= {orig_max_grad}"
        )


class TestDespeckleScores:
    """Tests for despeckle_scores function (TDD).

    Despeckle removes isolated outlier pixels using median filtering.
    Unlike bilateral smoothing which preserves edges, median filtering
    replaces isolated speckles with their neighborhood median.
    """

    def test_despeckle_scores_imports(self):
        """Test that despeckle_scores can be imported."""
        from src.terrain.transforms import despeckle_scores

        assert callable(despeckle_scores)

    def test_despeckle_scores_returns_array(self):
        """Test that despeckle_scores returns numpy array."""
        from src.terrain.transforms import despeckle_scores

        scores = np.random.rand(50, 50).astype(np.float32)
        result = despeckle_scores(scores)

        assert isinstance(result, np.ndarray)
        assert result.shape == scores.shape

    def test_despeckle_scores_removes_isolated_low_speckles(self):
        """Test that isolated low scores in high regions are removed.

        This is the key use case: a single low-score pixel (speckle)
        surrounded by high scores should be replaced with the median
        (which will be high).
        """
        from src.terrain.transforms import despeckle_scores

        # Create high-score region with isolated low speckles
        scores = np.full((20, 20), 0.8, dtype=np.float32)
        # Add isolated low speckles (single pixels)
        scores[5, 5] = 0.1  # Speckle 1
        scores[10, 10] = 0.05  # Speckle 2
        scores[15, 8] = 0.15  # Speckle 3

        despeckled = despeckle_scores(scores, kernel_size=3)

        # Isolated speckles should be replaced with median (≈0.8)
        assert despeckled[5, 5] > 0.5, f"Speckle at (5,5) not removed: {despeckled[5, 5]}"
        assert despeckled[10, 10] > 0.5, f"Speckle at (10,10) not removed: {despeckled[10, 10]}"
        assert despeckled[15, 8] > 0.5, f"Speckle at (15,8) not removed: {despeckled[15, 8]}"

    def test_despeckle_scores_removes_isolated_high_speckles(self):
        """Test that isolated high scores in low regions are removed."""
        from src.terrain.transforms import despeckle_scores

        # Create low-score region with isolated high speckles
        scores = np.full((20, 20), 0.2, dtype=np.float32)
        scores[7, 7] = 0.9  # Isolated high speckle

        despeckled = despeckle_scores(scores, kernel_size=3)

        # Isolated high speckle should be replaced with median (≈0.2)
        assert despeckled[7, 7] < 0.5, f"High speckle not removed: {despeckled[7, 7]}"

    def test_despeckle_scores_preserves_large_regions(self):
        """Test that large coherent regions are preserved.

        A 5x5 block of low scores should NOT be removed, even with
        a 3x3 kernel, because most pixels have same-value neighbors.
        """
        from src.terrain.transforms import despeckle_scores

        scores = np.full((30, 30), 0.8, dtype=np.float32)
        # Add a large low-score region (5x5 block)
        scores[10:15, 10:15] = 0.2

        despeckled = despeckle_scores(scores, kernel_size=3)

        # Center of large region should be preserved
        center_val = despeckled[12, 12]
        assert center_val < 0.5, f"Large region center changed: {center_val}"

    def test_despeckle_scores_kernel_size_effect(self):
        """Test that larger kernel removes larger speckle clusters."""
        from src.terrain.transforms import despeckle_scores

        # Create high region with 3x3 speckle cluster
        # A 3x3 kernel will preserve 3x3 clusters (center has 8 neighbors same value)
        # A 7x7 kernel should remove it (49 pixels, mostly high)
        scores = np.full((30, 30), 0.8, dtype=np.float32)
        scores[12:15, 12:15] = 0.1  # 3x3 low cluster

        # 3x3 kernel won't remove 3x3 cluster (center pixel has 8 low neighbors)
        despeckled_3 = despeckle_scores(scores, kernel_size=3)
        # 7x7 kernel should remove more of it (49 pixels, mostly high)
        despeckled_7 = despeckle_scores(scores, kernel_size=7)

        # Larger kernel should do better at removing the cluster
        cluster_mean_3 = despeckled_3[12:15, 12:15].mean()
        cluster_mean_7 = despeckled_7[12:15, 12:15].mean()

        assert cluster_mean_7 > cluster_mean_3, (
            f"Larger kernel should remove cluster better: 7x7={cluster_mean_7:.2f}, 3x3={cluster_mean_3:.2f}"
        )

    def test_despeckle_scores_handles_nan(self):
        """Test that NaN values are preserved."""
        from src.terrain.transforms import despeckle_scores

        scores = np.random.rand(20, 20).astype(np.float32)
        scores[5:8, 5:8] = np.nan

        despeckled = despeckle_scores(scores)

        # NaN region should remain NaN
        assert np.all(np.isnan(despeckled[5:8, 5:8]))
        # Non-NaN regions should have valid values
        assert not np.any(np.isnan(despeckled[0:4, 0:4]))

    def test_despeckle_scores_preserves_range(self):
        """Test that output stays in valid score range [0, 1]."""
        from src.terrain.transforms import despeckle_scores

        scores = np.random.rand(50, 50).astype(np.float32)
        despeckled = despeckle_scores(scores)

        assert np.nanmin(despeckled) >= 0.0
        assert np.nanmax(despeckled) <= 1.0
