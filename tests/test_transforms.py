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
