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
