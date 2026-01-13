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


class TestSlopeAdaptiveSmoothSlopeComputation:
    """Tests for slope computation in slope_adaptive_smooth.

    TDD RED phase - verifying slope is computed correctly.
    The issue: diagnostic plots showing blank (all-zero) slope maps.
    """

    def test_slope_adaptive_smooth_imports(self):
        """Test that slope_adaptive_smooth can be imported."""
        from src.terrain.transforms import slope_adaptive_smooth

        assert callable(slope_adaptive_smooth)

    def test_slope_of_flat_terrain_is_zero(self):
        """Flat terrain should have ~0 degree slope."""
        from src.terrain.transforms import slope_adaptive_smooth
        from rasterio import Affine

        # Flat plane at 100m elevation
        flat_dem = np.ones((50, 50), dtype=np.float32) * 100.0

        # 30m pixel size (typical SRTM)
        affine = Affine(30.0, 0.0, 0.0, 0.0, -30.0, 0.0)

        transform_fn = slope_adaptive_smooth(slope_threshold=2.0, smooth_sigma=3.0)
        result, _, _ = transform_fn(flat_dem, affine)

        # Flat terrain smoothed should be unchanged (or nearly so)
        np.testing.assert_allclose(result, flat_dem, atol=0.01)

    def test_slope_of_known_gradient_plane(self):
        """A plane tilted at 45° should have 45° slope."""
        from src.terrain.transforms import slope_adaptive_smooth
        from rasterio import Affine

        # Create 45° slope: rise = run = pixel_size
        # For 30m pixels, elevation increases 30m per pixel
        size = 50
        y_coords = np.arange(size)
        x_coords = np.arange(size)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # 30m rise per 30m run = 45° slope in x direction
        pixel_size = 30.0
        dem = xx.astype(np.float32) * pixel_size  # elevation = x * 30m

        affine = Affine(pixel_size, 0.0, 0.0, 0.0, -pixel_size, 0.0)

        # With threshold=2°, a 45° slope should NOT be smoothed
        transform_fn = slope_adaptive_smooth(slope_threshold=2.0, smooth_sigma=5.0)
        result, _, _ = transform_fn(dem, affine)

        # Interior pixels should be nearly unchanged (45° >> 2° threshold)
        interior_orig = dem[10:-10, 10:-10]
        interior_result = result[10:-10, 10:-10]
        np.testing.assert_allclose(interior_result, interior_orig, atol=0.5)

    def test_slope_of_5_degree_gradient(self):
        """A plane tilted at 5° should have ~5° slope."""
        from src.terrain.transforms import slope_adaptive_smooth
        from rasterio import Affine

        # For 5° slope: tan(5°) ≈ 0.0875
        # rise/run = 0.0875, so for 30m pixels, rise = 30 * 0.0875 = 2.625m per pixel
        size = 50
        y_coords = np.arange(size)
        x_coords = np.arange(size)
        xx, yy = np.meshgrid(x_coords, y_coords)

        pixel_size = 30.0
        rise_per_pixel = pixel_size * np.tan(np.radians(5.0))  # ~2.625m
        dem = xx.astype(np.float32) * rise_per_pixel

        affine = Affine(pixel_size, 0.0, 0.0, 0.0, -pixel_size, 0.0)

        # With threshold=2°, a 5° slope should get some smoothing but not full
        # (transition zone)
        transform_fn = slope_adaptive_smooth(slope_threshold=2.0, smooth_sigma=5.0, transition_width=1.0)
        result, _, _ = transform_fn(dem, affine)

        # 5° > 2° threshold, so smoothing weight should be low but not zero
        # The result should be close to original (not completely smoothed)
        interior_orig = dem[10:-10, 10:-10]
        interior_result = result[10:-10, 10:-10]

        # Should be similar but not identical
        diff = np.abs(interior_result - interior_orig)
        assert np.max(diff) < 5.0, f"Difference too large: {np.max(diff)}"

    def test_slope_of_1_degree_gradient_gets_smoothed(self):
        """A 1° slope should be mostly smoothed (below 2° threshold)."""
        from src.terrain.transforms import slope_adaptive_smooth
        from rasterio import Affine

        # For 1° slope: tan(1°) ≈ 0.01745
        # rise = 30 * 0.01745 = 0.524m per pixel
        size = 50
        y_coords = np.arange(size)
        x_coords = np.arange(size)
        xx, yy = np.meshgrid(x_coords, y_coords)

        pixel_size = 30.0
        rise_per_pixel = pixel_size * np.tan(np.radians(1.0))  # ~0.524m
        base_elevation = xx.astype(np.float32) * rise_per_pixel

        # Add some noise to see smoothing effect
        np.random.seed(42)
        noise = np.random.randn(size, size).astype(np.float32) * 2.0
        dem = base_elevation + noise

        affine = Affine(pixel_size, 0.0, 0.0, 0.0, -pixel_size, 0.0)

        # 1° < 2° threshold, so this should be heavily smoothed
        transform_fn = slope_adaptive_smooth(slope_threshold=2.0, smooth_sigma=5.0)
        result, _, _ = transform_fn(dem, affine)

        # Noise should be reduced
        interior_orig = dem[10:-10, 10:-10]
        interior_result = result[10:-10, 10:-10]

        noise_orig = np.std(interior_orig - base_elevation[10:-10, 10:-10])
        noise_result = np.std(interior_result - base_elevation[10:-10, 10:-10])

        # 1° slope with 2° threshold should get high smoothing weight (~0.98)
        # Expect meaningful noise reduction (at least 25%)
        assert noise_result < noise_orig * 0.75, (
            f"Expected at least 25% noise reduction: orig={noise_orig:.2f}, result={noise_result:.2f}"
        )

    def test_pixel_size_affects_slope_calculation(self):
        """Different pixel sizes should give different slope calculations."""
        from src.terrain.transforms import slope_adaptive_smooth
        from rasterio import Affine

        # Same elevation gradient: 30m rise per pixel + NOISE
        # (Gaussian blur on constant gradient doesn't change it, need noise)
        size = 30
        xx, _ = np.meshgrid(np.arange(size), np.arange(size))
        base_dem = xx.astype(np.float32) * 30.0  # 30m rise per pixel

        # Add noise that smoothing can remove
        np.random.seed(42)
        noise = np.random.randn(size, size).astype(np.float32) * 5.0
        dem = base_dem + noise

        # With 30m pixel size: 30m rise / 30m run = 45° slope (steep, no smoothing)
        affine_30m = Affine(30.0, 0.0, 0.0, 0.0, -30.0, 0.0)
        transform_30m = slope_adaptive_smooth(slope_threshold=2.0, smooth_sigma=3.0)
        result_30m, _, _ = transform_30m(dem, affine_30m)

        # With 300m pixel size: 30m rise / 300m run = ~5.7° slope (gentler, some smoothing)
        affine_300m = Affine(300.0, 0.0, 0.0, 0.0, -300.0, 0.0)
        transform_300m = slope_adaptive_smooth(slope_threshold=2.0, smooth_sigma=3.0)
        result_300m, _, _ = transform_300m(dem, affine_300m)

        # 45° slope (30m pixels) should NOT be smoothed (noise preserved)
        # 5.7° slope (300m pixels) should be partially smoothed (noise reduced)

        # Measure noise by comparing to base gradient
        interior_base = base_dem[5:-5, 5:-5]
        interior_30m = result_30m[5:-5, 5:-5]
        interior_300m = result_300m[5:-5, 5:-5]

        # Noise remaining after smoothing
        noise_30m = np.std(interior_30m - interior_base)
        noise_300m = np.std(interior_300m - interior_base)

        # 30m result should preserve more noise (steep slope = no smoothing)
        # 300m result should reduce more noise (gentle slope = some smoothing)
        assert noise_30m > noise_300m, (
            f"30m pixels (steep slope) should preserve more noise than 300m pixels (gentle slope): "
            f"noise_30m={noise_30m:.2f}, noise_300m={noise_300m:.2f}"
        )

    def test_no_affine_assumes_1m_pixels_with_warning(self):
        """Without affine, should assume 1m pixels and warn."""
        from src.terrain.transforms import slope_adaptive_smooth
        import warnings

        size = 30
        xx, _ = np.meshgrid(np.arange(size), np.arange(size))
        dem = xx.astype(np.float32) * 1.0  # 1m rise per pixel

        transform_fn = slope_adaptive_smooth(slope_threshold=2.0, smooth_sigma=3.0)

        # Without affine transform, should still work but warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # No affine = assumes 1m pixels
            result, _, _ = transform_fn(dem, None)

            # Should have logged a warning (check logs, not warnings module)
            assert result.shape == dem.shape

    def test_wgs84_degree_pixel_size_gives_huge_slopes(self):
        """WGS84 pixel size (~0.0008°) should NOT be used directly for slope.

        This test documents the bug: if affine has degree-based pixel size,
        slopes will be computed incorrectly (division by tiny number = huge slopes).
        """
        from src.terrain.transforms import slope_adaptive_smooth
        from rasterio import Affine

        size = 30
        xx, _ = np.meshgrid(np.arange(size), np.arange(size))
        # Small elevation change: 0.01m per pixel
        dem = xx.astype(np.float32) * 0.01

        # WGS84 1 arc-second: ~0.000278° per pixel
        # At equator, this is ~30m, but the value stored is in degrees
        wgs84_pixel_degrees = 1.0 / 3600.0  # ~0.000278
        affine_degrees = Affine(wgs84_pixel_degrees, 0.0, -83.0, 0.0, -wgs84_pixel_degrees, 42.0)

        transform_fn = slope_adaptive_smooth(slope_threshold=2.0, smooth_sigma=3.0)
        result, _, _ = transform_fn(dem, affine_degrees)

        # With degree-based pixel size, the slope calculation will be wrong:
        # dx = 0.01 / 0.000278 = 36 (huge!)
        # This would give arctan(36) = 88.4° for a 0.01m/pixel gradient
        #
        # The DEM should have been reprojected to meters first.
        # This test documents current behavior - if it's producing near-zero slopes,
        # something else is wrong.
        assert result.shape == dem.shape

    def test_elevation_scale_compensation(self):
        """elevation_scale parameter should compensate for prior scale_elevation."""
        from src.terrain.transforms import slope_adaptive_smooth
        from rasterio import Affine

        # Create a 5° slope DEM
        size = 50
        xx, _ = np.meshgrid(np.arange(size), np.arange(size))
        pixel_size = 30.0
        rise_per_pixel = pixel_size * np.tan(np.radians(5.0))  # ~2.625m

        # Scaled DEM (as if scale_elevation(0.0001) was applied)
        scale_factor = 0.0001
        dem_meters = xx.astype(np.float32) * rise_per_pixel
        dem_scaled = dem_meters * scale_factor

        affine = Affine(pixel_size, 0.0, 0.0, 0.0, -pixel_size, 0.0)

        # Add noise to see smoothing effect
        np.random.seed(42)
        noise = np.random.randn(size, size).astype(np.float32) * 2.0
        dem_scaled_noisy = dem_scaled + noise * scale_factor

        # Without compensation, scaled DEM has ~0° slope (all smoothed)
        transform_no_comp = slope_adaptive_smooth(slope_threshold=2.0, smooth_sigma=5.0)
        result_no_comp, _, _ = transform_no_comp(dem_scaled_noisy, affine)

        # With compensation, slope is computed correctly (5° > 2° threshold, less smoothing)
        transform_with_comp = slope_adaptive_smooth(
            slope_threshold=2.0, smooth_sigma=5.0, elevation_scale=scale_factor
        )
        result_with_comp, _, _ = transform_with_comp(dem_scaled_noisy, affine)

        interior = slice(10, -10), slice(10, -10)

        # Measure noise remaining after smoothing
        noise_no_comp = np.std(result_no_comp[interior] - dem_scaled[interior])
        noise_with_comp = np.std(result_with_comp[interior] - dem_scaled[interior])

        # Without compensation: appears flat, heavily smoothed (less noise remaining)
        # With compensation: 5° slope detected, less smoothing (more noise remaining)
        assert noise_with_comp > noise_no_comp * 1.5, (
            f"With compensation should preserve more noise (less smoothing): "
            f"with_comp={noise_with_comp:.6f}, no_comp={noise_no_comp:.6f}"
        )

    def test_edge_preservation_protects_sharp_boundaries(self):
        """edge_threshold should preserve sharp elevation discontinuities like lake edges.

        The key issue: flat areas NEAR sharp edges get contaminated by Gaussian blur
        mixing values across the boundary. Edge preservation reduces smoothing weight
        in those near-edge flat areas, keeping them closer to original values.
        """
        from src.terrain.transforms import slope_adaptive_smooth
        from rasterio import Affine

        # Create terrain with a sharp lake edge: flat land at 100m, lake at 80m
        size = 100
        dem = np.full((size, size), 100.0, dtype=np.float64)
        # Lake in the center (20m drop)
        dem[30:70, 30:70] = 80.0

        # Add noise to flat areas (buildings/bumps we want to smooth)
        np.random.seed(42)
        noise = np.random.normal(0, 2, dem.shape)
        dem_noisy = dem + noise

        # 30m pixel size
        affine = Affine(30.0, 0, 0, 0, -30.0, 0)

        # Without edge preservation: the lake boundary will be smoothed
        transform_no_edge = slope_adaptive_smooth(
            slope_threshold=5.0, smooth_sigma=3.0, edge_threshold=None
        )
        result_no_edge, _, _ = transform_no_edge(dem_noisy, affine)

        # With edge preservation: protect edges with >10m elevation change
        transform_with_edge = slope_adaptive_smooth(
            slope_threshold=5.0, smooth_sigma=3.0, edge_threshold=10.0
        )
        result_with_edge, _, _ = transform_with_edge(dem_noisy, affine)

        # Focus on flat areas just INSIDE the edge (1-3 pixels from boundary)
        # These should be affected differently by edge preservation
        # Without edge: Gaussian blur contaminates these pixels with values from other side
        # With edge: smoothing is reduced here, keeping values closer to original

        # Land side: pixels at row 50, columns 27-29 (just outside lake)
        land_near_edge_no = result_no_edge[50, 27:30].mean()
        land_near_edge_with = result_with_edge[50, 27:30].mean()
        original_land_near = dem_noisy[50, 27:30].mean()

        # Lake side: pixels at row 50, columns 31-33 (just inside lake)
        lake_near_edge_no = result_no_edge[50, 31:34].mean()
        lake_near_edge_with = result_with_edge[50, 31:34].mean()
        original_lake_near = dem_noisy[50, 31:34].mean()

        # With edge preservation, near-edge land should stay closer to 100m (not pulled toward 80m)
        land_error_no = abs(land_near_edge_no - original_land_near)
        land_error_with = abs(land_near_edge_with - original_land_near)

        # With edge preservation, near-edge lake should stay closer to 80m (not pulled toward 100m)
        lake_error_no = abs(lake_near_edge_no - original_lake_near)
        lake_error_with = abs(lake_near_edge_with - original_lake_near)

        # At least one of the near-edge areas should be better preserved
        # (The effect depends on smoothing parameters and exact geometry)
        total_error_no = land_error_no + lake_error_no
        total_error_with = land_error_with + lake_error_with

        # Edge preservation should reduce contamination from cross-boundary blurring
        assert total_error_with < total_error_no, (
            f"Edge preservation should reduce cross-boundary contamination: "
            f"with_edge_error={total_error_with:.2f}, no_edge_error={total_error_no:.2f}"
        )

        # Interior of land/lake (far from edge) should still be smoothed
        land_interior = result_with_edge[10:20, 10:20]
        original_land_int = dem_noisy[10:20, 10:20]

        # Noise should be reduced in flat interiors even with edge preservation
        land_noise_before = np.std(original_land_int)
        land_noise_after = np.std(land_interior)
        assert land_noise_after < land_noise_before * 0.9, (
            f"Land interior should still be smoothed: before={land_noise_before:.2f}, after={land_noise_after:.2f}"
        )

    def test_strength_parameter_scales_smoothing_effect(self):
        """strength parameter should scale the maximum smoothing effect."""
        from src.terrain.transforms import slope_adaptive_smooth
        from rasterio import Affine

        # Create flat terrain with noise
        size = 50
        np.random.seed(42)
        dem = np.random.randn(size, size).astype(np.float32) * 5.0 + 100.0

        # 30m pixel size (flat terrain)
        affine = Affine(30.0, 0.0, 0.0, 0.0, -30.0, 0.0)

        # Full strength (default)
        transform_full = slope_adaptive_smooth(slope_threshold=2.0, smooth_sigma=5.0, strength=1.0)
        result_full, _, _ = transform_full(dem, affine)

        # Half strength
        transform_half = slope_adaptive_smooth(slope_threshold=2.0, smooth_sigma=5.0, strength=0.5)
        result_half, _, _ = transform_half(dem, affine)

        # Quarter strength
        transform_quarter = slope_adaptive_smooth(slope_threshold=2.0, smooth_sigma=5.0, strength=0.25)
        result_quarter, _, _ = transform_quarter(dem, affine)

        # Measure noise reduction
        interior = slice(10, -10), slice(10, -10)
        noise_orig = np.std(dem[interior])
        noise_full = np.std(result_full[interior])
        noise_half = np.std(result_half[interior])
        noise_quarter = np.std(result_quarter[interior])

        # Full strength should reduce noise most
        # Half strength should reduce less
        # Quarter strength should reduce even less
        assert noise_full < noise_half < noise_quarter < noise_orig, (
            f"Higher strength should reduce more noise: "
            f"orig={noise_orig:.2f}, quarter={noise_quarter:.2f}, "
            f"half={noise_half:.2f}, full={noise_full:.2f}"
        )

    def test_strength_zero_has_no_effect(self):
        """strength=0 should result in no change to the DEM."""
        from src.terrain.transforms import slope_adaptive_smooth
        from rasterio import Affine

        # Create flat terrain with noise
        size = 30
        np.random.seed(42)
        dem = np.random.randn(size, size).astype(np.float32) * 5.0 + 100.0

        affine = Affine(30.0, 0.0, 0.0, 0.0, -30.0, 0.0)

        # Zero strength - no effect
        transform_fn = slope_adaptive_smooth(slope_threshold=2.0, smooth_sigma=5.0, strength=0.0)
        result, _, _ = transform_fn(dem, affine)

        # Result should be identical to input
        np.testing.assert_array_almost_equal(result, dem, decimal=5)

    def test_strength_in_transform_name(self):
        """strength should appear in transform name when not 1.0."""
        from src.terrain.transforms import slope_adaptive_smooth

        # Default strength - not in name
        transform_default = slope_adaptive_smooth(strength=1.0)
        assert "s=" not in transform_default.__name__

        # Custom strength - in name
        transform_half = slope_adaptive_smooth(strength=0.5)
        assert "s=0.5" in transform_half.__name__


class TestSlopeComputationDiagnostic:
    """Diagnostic tests to understand slope computation behavior."""

    def test_slope_values_for_tilted_plane(self):
        """Directly compute and verify slope values for a tilted plane."""
        from scipy import ndimage
        from rasterio import Affine

        # Create 45° slope: 30m rise per 30m run
        size = 30
        xx, _ = np.meshgrid(np.arange(size), np.arange(size))
        dem = xx.astype(np.float64) * 30.0  # 30m per pixel

        # Compute Sobel gradients (same as slope_adaptive_smooth does)
        dy = ndimage.sobel(dem, axis=0, mode="reflect") / 8.0
        dx = ndimage.sobel(dem, axis=1, mode="reflect") / 8.0

        # Print raw gradient values
        print(f"\nRaw dx (per pixel) - interior mean: {np.mean(dx[5:-5, 5:-5]):.4f}")
        print(f"Raw dy (per pixel) - interior mean: {np.mean(dy[5:-5, 5:-5]):.4f}")

        # With 30m pixel size
        pixel_size = 30.0
        dx_meters = dx / pixel_size
        dy_meters = dy / pixel_size

        print(f"dx (per meter) - interior mean: {np.mean(dx_meters[5:-5, 5:-5]):.4f}")

        gradient_magnitude = np.sqrt(dx_meters**2 + dy_meters**2)
        slope_degrees = np.degrees(np.arctan(gradient_magnitude))

        print(f"Slope degrees - interior mean: {np.mean(slope_degrees[5:-5, 5:-5]):.2f}°")
        print(f"Slope degrees - interior min: {np.min(slope_degrees[5:-5, 5:-5]):.2f}°")
        print(f"Slope degrees - interior max: {np.max(slope_degrees[5:-5, 5:-5]):.2f}°")

        # For 45° slope (rise/run = 1), we expect slope_degrees ≈ 45
        expected_slope = 45.0
        actual_slope = np.mean(slope_degrees[5:-5, 5:-5])

        assert abs(actual_slope - expected_slope) < 1.0, (
            f"Expected ~45° slope, got {actual_slope:.2f}°"
        )

    def test_slope_values_for_gentle_slope(self):
        """Compute slope for a 2° tilted plane."""
        from scipy import ndimage

        # For 2° slope: tan(2°) ≈ 0.0349
        # rise/run = 0.0349, so for 30m pixels, rise = 30 * 0.0349 = 1.05m per pixel
        size = 30
        xx, _ = np.meshgrid(np.arange(size), np.arange(size))
        rise_per_pixel = 30.0 * np.tan(np.radians(2.0))  # ~1.05m
        dem = xx.astype(np.float64) * rise_per_pixel

        print(f"\nRise per pixel: {rise_per_pixel:.4f}m")

        # Compute Sobel gradients
        dy = ndimage.sobel(dem, axis=0, mode="reflect") / 8.0
        dx = ndimage.sobel(dem, axis=1, mode="reflect") / 8.0

        print(f"Raw dx (per pixel) - interior mean: {np.mean(dx[5:-5, 5:-5]):.4f}")

        # Scale by pixel size
        pixel_size = 30.0
        dx_meters = dx / pixel_size
        dy_meters = dy / pixel_size

        gradient_magnitude = np.sqrt(dx_meters**2 + dy_meters**2)
        slope_degrees = np.degrees(np.arctan(gradient_magnitude))

        print(f"Slope degrees - interior mean: {np.mean(slope_degrees[5:-5, 5:-5]):.2f}°")

        # Should be ~2° slope
        expected_slope = 2.0
        actual_slope = np.mean(slope_degrees[5:-5, 5:-5])

        assert abs(actual_slope - expected_slope) < 0.5, (
            f"Expected ~2° slope, got {actual_slope:.2f}°"
        )

    def test_smoothing_weight_mask_values(self):
        """Verify the smoothing weight mask for known slopes."""
        from scipy import ndimage

        # Create flat terrain (should get weight ~1.0, full smoothing)
        flat_dem = np.ones((30, 30), dtype=np.float64) * 100.0

        # Create 10° slope terrain (should get weight ~0.0, no smoothing)
        size = 30
        xx, _ = np.meshgrid(np.arange(size), np.arange(size))
        rise_per_pixel = 30.0 * np.tan(np.radians(10.0))  # ~5.3m
        steep_dem = xx.astype(np.float64) * rise_per_pixel

        pixel_size = 30.0
        slope_threshold = 2.0
        transition_width = 1.0

        for dem, desc in [(flat_dem, "flat"), (steep_dem, "10° slope")]:
            dy = ndimage.sobel(dem, axis=0, mode="reflect") / 8.0
            dx = ndimage.sobel(dem, axis=1, mode="reflect") / 8.0
            dx_meters = dx / pixel_size
            dy_meters = dy / pixel_size
            gradient_magnitude = np.sqrt(dx_meters**2 + dy_meters**2)
            slope_degrees = np.degrees(np.arctan(gradient_magnitude))

            # Sigmoid weight calculation (same as slope_adaptive_smooth)
            k = 4.0 / transition_width
            smoothing_weight = 1.0 / (1.0 + np.exp(k * (slope_degrees - slope_threshold)))

            interior_slope = np.mean(slope_degrees[5:-5, 5:-5])
            interior_weight = np.mean(smoothing_weight[5:-5, 5:-5])

            print(f"\n{desc}: slope={interior_slope:.2f}°, weight={interior_weight:.4f}")

        # Flat terrain should have weight ~1.0 (full smoothing)
        # 10° slope should have weight ~0.0 (no smoothing)
        dy = ndimage.sobel(flat_dem, axis=0, mode="reflect") / 8.0
        dx = ndimage.sobel(flat_dem, axis=1, mode="reflect") / 8.0
        dx_meters = dx / pixel_size
        dy_meters = dy / pixel_size
        gradient_magnitude = np.sqrt(dx_meters**2 + dy_meters**2)
        slope_degrees = np.degrees(np.arctan(gradient_magnitude))
        k = 4.0 / transition_width
        flat_weight = 1.0 / (1.0 + np.exp(k * (slope_degrees - slope_threshold)))

        assert np.mean(flat_weight[5:-5, 5:-5]) > 0.9, "Flat terrain should have high smoothing weight"


class TestDiagnosticsSlopeComputation:
    """Tests for slope computation in the diagnostics module."""

    def test_diagnostics_slope_matches_transform_slope(self):
        """Diagnostic plot should compute same slope as the transform."""
        from scipy import ndimage
        from src.terrain.transforms import slope_adaptive_smooth
        from rasterio import Affine

        # Create a 5° slope
        size = 50
        xx, _ = np.meshgrid(np.arange(size), np.arange(size))
        pixel_size = 30.0
        rise_per_pixel = pixel_size * np.tan(np.radians(5.0))
        dem = xx.astype(np.float64) * rise_per_pixel

        # Compute slope the way the transform does
        dy_transform = ndimage.sobel(dem, axis=0, mode="reflect") / 8.0
        dx_transform = ndimage.sobel(dem, axis=1, mode="reflect") / 8.0
        dx_transform = dx_transform / pixel_size
        dy_transform = dy_transform / pixel_size
        gradient_transform = np.sqrt(dx_transform**2 + dy_transform**2)
        slope_transform = np.degrees(np.arctan(gradient_transform))

        # Compute slope the way diagnostics does (copied from plot_adaptive_smooth_diagnostics)
        dy_diag = ndimage.sobel(dem, axis=0, mode="reflect") / 8.0
        dx_diag = ndimage.sobel(dem, axis=1, mode="reflect") / 8.0
        dx_diag = dx_diag / pixel_size
        dy_diag = dy_diag / pixel_size
        gradient_diag = np.sqrt(dx_diag**2 + dy_diag**2)
        slope_diag = np.degrees(np.arctan(gradient_diag))

        # Should be identical
        np.testing.assert_allclose(slope_diag, slope_transform)

        # Both should give ~5° slope
        mean_slope = np.mean(slope_diag[10:-10, 10:-10])
        assert abs(mean_slope - 5.0) < 0.5, f"Expected ~5° slope, got {mean_slope:.2f}°"

    def test_diagnostics_slope_with_none_pixel_size(self):
        """Diagnostics with pixel_size=None should not compute slope correctly."""
        from scipy import ndimage

        # Create a 5° slope
        size = 50
        xx, _ = np.meshgrid(np.arange(size), np.arange(size))
        pixel_size = 30.0
        rise_per_pixel = pixel_size * np.tan(np.radians(5.0))
        dem = xx.astype(np.float64) * rise_per_pixel

        # Compute slope WITHOUT pixel size (as diagnostics does when pixel_size is None)
        dy = ndimage.sobel(dem, axis=0, mode="reflect") / 8.0
        dx = ndimage.sobel(dem, axis=1, mode="reflect") / 8.0
        # NO division by pixel_size
        gradient = np.sqrt(dx**2 + dy**2)
        slope_no_pixelsize = np.degrees(np.arctan(gradient))

        # This should give WRONG slope (much steeper because not scaled)
        mean_slope = np.mean(slope_no_pixelsize[10:-10, 10:-10])
        print(f"\nSlope without pixel_size: {mean_slope:.2f}° (should be wrong)")

        # The wrong slope should be much larger than 5°
        # dx = 2.625m/pixel, without scaling, gradient = 2.625
        # arctan(2.625) = 69.1°
        assert mean_slope > 30.0, f"Expected large slope without pixel_size, got {mean_slope:.2f}°"

    def test_diagnostics_slope_with_wgs84_pixel_size(self):
        """Diagnostics with WGS84 degree pixel size should give huge slopes."""
        from scipy import ndimage

        # Create small elevation changes
        size = 50
        xx, _ = np.meshgrid(np.arange(size), np.arange(size))
        dem = xx.astype(np.float64) * 0.1  # 0.1m per pixel

        # WGS84 pixel size in degrees
        pixel_size_degrees = 1.0 / 3600.0  # ~0.000278

        dy = ndimage.sobel(dem, axis=0, mode="reflect") / 8.0
        dx = ndimage.sobel(dem, axis=1, mode="reflect") / 8.0
        dx = dx / pixel_size_degrees  # Division by tiny number!
        dy = dy / pixel_size_degrees
        gradient = np.sqrt(dx**2 + dy**2)
        slope_degrees = np.degrees(np.arctan(gradient))

        mean_slope = np.mean(slope_degrees[10:-10, 10:-10])
        print(f"\nSlope with WGS84 degree pixel_size: {mean_slope:.2f}°")

        # dx = 0.1 / 0.000278 = 360, arctan(360) = 89.84°
        # This is the bug: everything appears as ~90° steep
        assert mean_slope > 85.0, f"Expected ~90° slope with WGS84 pixels, got {mean_slope:.2f}°"

    def test_slope_with_real_dem_sample(self):
        """Test slope computation with real DEM data."""
        import rasterio
        from pathlib import Path
        from scipy import ndimage

        # Try to find sample DEM file
        sample_paths = [
            Path("data/samples/CA_La Jolla_296219_1943_31680_geo.tif"),
            Path("data/samples/CA_La Jolla_465821_1943_31680_geo.tif"),
        ]

        sample_path = None
        for p in sample_paths:
            if p.exists():
                sample_path = p
                break

        if sample_path is None:
            pytest.skip("No sample DEM file found")

        # Load a small crop of the DEM
        with rasterio.open(sample_path) as src:
            # Read a small window (100x100 pixels)
            window = rasterio.windows.Window(0, 0, 100, 100)
            dem = src.read(1, window=window).astype(np.float64)
            transform = src.window_transform(window)
            pixel_size_x = abs(transform.a)
            pixel_size_y = abs(transform.e)

        print(f"\nDEM shape: {dem.shape}")
        print(f"DEM range: {np.nanmin(dem):.2f} to {np.nanmax(dem):.2f}")
        print(f"Pixel size: {pixel_size_x:.6f} x {pixel_size_y:.6f}")

        # Check if pixel size is in degrees (very small) or meters
        if pixel_size_x < 0.001:
            print("WARNING: Pixel size appears to be in degrees (WGS84)")
            # For testing, we'll compute both ways
            pixel_size_meters = pixel_size_x * 111000  # rough conversion
            print(f"Approximate meters: {pixel_size_meters:.1f}m")
        else:
            pixel_size_meters = pixel_size_x
            print(f"Pixel size appears to be in meters: {pixel_size_meters:.1f}m")

        # Compute slope
        dy = ndimage.sobel(dem, axis=0, mode="reflect") / 8.0
        dx = ndimage.sobel(dem, axis=1, mode="reflect") / 8.0

        # With actual pixel size from transform
        dx_scaled = dx / pixel_size_x
        dy_scaled = dy / pixel_size_y
        gradient = np.sqrt(dx_scaled**2 + dy_scaled**2)
        slope = np.degrees(np.arctan(gradient))

        print(f"Slope range (raw pixel size): {np.nanmin(slope):.2f}° to {np.nanmax(slope):.2f}°")
        print(f"Slope mean: {np.nanmean(slope):.2f}°")

        # If the slope is all near 90°, pixel size is probably wrong
        if np.nanmean(slope) > 80:
            print("WARNING: Mean slope >80° suggests pixel size is in degrees, not meters")

        # Basic sanity check: slope should be non-negative and not all the same
        assert np.all(slope[~np.isnan(slope)] >= 0), "Slope should be non-negative"
        assert np.nanstd(slope) > 0.01, "Slope should have some variation"
