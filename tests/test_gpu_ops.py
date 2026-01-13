"""
Tests for GPU-accelerated operations using PyTorch.

TDD RED phase: These tests define the expected behavior of GPU-accelerated
functions before implementation.
"""

import numpy as np
import pytest


class TestGpuSlopeCalculation:
    """Tests for GPU-accelerated slope calculation using Horn's method."""

    def test_gpu_horn_slope_imports(self):
        """GPU Horn slope function should be importable."""
        from src.terrain.gpu_ops import gpu_horn_slope

        assert callable(gpu_horn_slope)

    def test_gpu_horn_slope_returns_same_shape(self):
        """GPU slope calculation should preserve input shape."""
        from src.terrain.gpu_ops import gpu_horn_slope

        dem = np.random.rand(100, 100).astype(np.float32) * 1000
        result = gpu_horn_slope(dem)

        assert result.shape == dem.shape

    def test_gpu_horn_slope_flat_terrain_is_zero(self):
        """Flat terrain should have zero slope."""
        from src.terrain.gpu_ops import gpu_horn_slope

        # Constant elevation = flat terrain
        dem = np.full((50, 50), 100.0, dtype=np.float32)
        result = gpu_horn_slope(dem)

        # All slopes should be essentially zero
        assert np.allclose(result[1:-1, 1:-1], 0.0, atol=1e-6)

    def test_gpu_horn_slope_tilted_plane(self):
        """Tilted plane should have uniform non-zero slope."""
        from src.terrain.gpu_ops import gpu_horn_slope

        # Create plane with slope of 1 unit per pixel in x direction
        x = np.arange(100, dtype=np.float32)
        dem = np.tile(x, (100, 1))  # Slope in x, flat in y

        result = gpu_horn_slope(dem)

        # Interior should have uniform slope (ignoring edges)
        interior = result[5:-5, 5:-5]
        assert np.std(interior) < 0.1  # Should be uniform
        assert np.mean(interior) > 0.1  # Should be non-zero

    def test_gpu_horn_slope_handles_nan(self):
        """GPU slope should handle NaN values gracefully."""
        from src.terrain.gpu_ops import gpu_horn_slope

        dem = np.random.rand(50, 50).astype(np.float32) * 1000
        dem[20:25, 20:25] = np.nan  # Add NaN region

        result = gpu_horn_slope(dem)

        # NaN regions should remain NaN in output
        assert np.all(np.isnan(result[20:25, 20:25]))
        # Non-NaN regions should have valid values
        assert not np.any(np.isnan(result[0:10, 0:10]))

    def test_gpu_horn_slope_matches_scipy(self):
        """GPU slope should produce same results as scipy implementation."""
        from scipy import ndimage
        from src.terrain.gpu_ops import gpu_horn_slope

        dem = np.random.rand(100, 100).astype(np.float32) * 1000

        # scipy reference implementation
        dx = ndimage.convolve(dem, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0)
        dy = ndimage.convolve(dem, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0)
        scipy_slope = np.hypot(dx, dy)

        # GPU implementation
        gpu_slope = gpu_horn_slope(dem)

        # Should match within floating point tolerance
        np.testing.assert_allclose(gpu_slope, scipy_slope, rtol=1e-4, atol=1e-6)


class TestGpuGaussianBlur:
    """Tests for GPU-accelerated Gaussian blur."""

    def test_gpu_gaussian_blur_imports(self):
        """GPU Gaussian blur function should be importable."""
        from src.terrain.gpu_ops import gpu_gaussian_blur

        assert callable(gpu_gaussian_blur)

    def test_gpu_gaussian_blur_returns_same_shape(self):
        """GPU Gaussian blur should preserve input shape."""
        from src.terrain.gpu_ops import gpu_gaussian_blur

        data = np.random.rand(100, 100).astype(np.float32)
        result = gpu_gaussian_blur(data, sigma=2.0)

        assert result.shape == data.shape

    def test_gpu_gaussian_blur_smooths_noise(self):
        """Gaussian blur should reduce noise (lower standard deviation)."""
        from src.terrain.gpu_ops import gpu_gaussian_blur

        # Create noisy data
        np.random.seed(42)
        data = np.random.rand(100, 100).astype(np.float32)
        original_std = np.std(data)

        result = gpu_gaussian_blur(data, sigma=3.0)

        # Blurred data should have lower variance
        assert np.std(result) < original_std

    def test_gpu_gaussian_blur_preserves_constant(self):
        """Blurring a constant should return the same constant."""
        from src.terrain.gpu_ops import gpu_gaussian_blur

        data = np.full((50, 50), 42.0, dtype=np.float32)
        result = gpu_gaussian_blur(data, sigma=5.0)

        # Interior should remain constant (edges may have boundary effects)
        np.testing.assert_allclose(result[10:-10, 10:-10], 42.0, rtol=1e-4)

    def test_gpu_gaussian_blur_larger_sigma_more_smooth(self):
        """Larger sigma should produce smoother result."""
        from src.terrain.gpu_ops import gpu_gaussian_blur

        np.random.seed(42)
        data = np.random.rand(100, 100).astype(np.float32)

        result_small = gpu_gaussian_blur(data, sigma=1.0)
        result_large = gpu_gaussian_blur(data, sigma=5.0)

        # Larger sigma should have lower variance
        assert np.std(result_large) < np.std(result_small)

    def test_gpu_gaussian_blur_matches_scipy(self):
        """GPU Gaussian blur should produce similar results to scipy."""
        from scipy.ndimage import gaussian_filter
        from src.terrain.gpu_ops import gpu_gaussian_blur

        data = np.random.rand(100, 100).astype(np.float32) * 100
        sigma = 2.0

        scipy_result = gaussian_filter(data, sigma=sigma)
        gpu_result = gpu_gaussian_blur(data, sigma=sigma)

        # Compare interior (edges differ due to boundary handling)
        margin = int(3 * sigma)
        interior_scipy = scipy_result[margin:-margin, margin:-margin]
        interior_gpu = gpu_result[margin:-margin, margin:-margin]

        # Interior should match closely
        np.testing.assert_allclose(interior_gpu, interior_scipy, rtol=0.02, atol=0.2)

    def test_gpu_gaussian_blur_handles_nan(self):
        """GPU Gaussian blur should handle NaN values."""
        from src.terrain.gpu_ops import gpu_gaussian_blur

        data = np.random.rand(50, 50).astype(np.float32)
        data[20:25, 20:25] = np.nan

        result = gpu_gaussian_blur(data, sigma=2.0)

        # NaN regions should remain NaN
        assert np.all(np.isnan(result[22:23, 22:23]))  # Center of NaN region


class TestGpuMedianFilter:
    """Tests for GPU-accelerated median filter."""

    def test_gpu_median_filter_imports(self):
        """GPU median filter should be importable."""
        from src.terrain.gpu_ops import gpu_median_filter

        assert callable(gpu_median_filter)

    def test_gpu_median_filter_returns_same_shape(self):
        """GPU median filter should preserve input shape."""
        from src.terrain.gpu_ops import gpu_median_filter

        data = np.random.rand(100, 100).astype(np.float32)
        result = gpu_median_filter(data, kernel_size=3)

        assert result.shape == data.shape

    def test_gpu_median_filter_removes_salt_pepper_noise(self):
        """Median filter should remove salt-and-pepper noise."""
        from src.terrain.gpu_ops import gpu_median_filter

        # Create data with salt-and-pepper noise
        data = np.full((50, 50), 0.5, dtype=np.float32)
        data[10, 10] = 1.0  # Salt (bright outlier)
        data[20, 20] = 0.0  # Pepper (dark outlier)

        result = gpu_median_filter(data, kernel_size=3)

        # Outliers should be removed (replaced with median of neighbors)
        assert abs(result[10, 10] - 0.5) < 0.1, "Salt noise should be removed"
        assert abs(result[20, 20] - 0.5) < 0.1, "Pepper noise should be removed"

    def test_gpu_median_filter_preserves_edges(self):
        """Median filter should preserve sharp edges better than Gaussian."""
        from src.terrain.gpu_ops import gpu_median_filter

        # Create data with sharp edge
        data = np.zeros((50, 50), dtype=np.float32)
        data[:, 25:] = 1.0  # Sharp vertical edge

        result = gpu_median_filter(data, kernel_size=3)

        # Edge should remain relatively sharp
        assert result[25, 24] < 0.3, "Left of edge should stay dark"
        assert result[25, 26] > 0.7, "Right of edge should stay bright"

    def test_gpu_median_filter_matches_scipy(self):
        """GPU median filter should match scipy results."""
        from scipy.ndimage import median_filter
        from src.terrain.gpu_ops import gpu_median_filter

        np.random.seed(42)
        data = np.random.rand(50, 50).astype(np.float32)

        scipy_result = median_filter(data, size=3)
        gpu_result = gpu_median_filter(data, kernel_size=3)

        # Should match closely (interior, avoiding edge differences)
        np.testing.assert_allclose(
            gpu_result[2:-2, 2:-2], scipy_result[2:-2, 2:-2], rtol=1e-4, atol=1e-6
        )

    def test_gpu_median_filter_different_kernel_sizes(self):
        """Median filter should work with different kernel sizes."""
        from src.terrain.gpu_ops import gpu_median_filter

        data = np.random.rand(50, 50).astype(np.float32)

        # Various kernel sizes should work
        result_3 = gpu_median_filter(data, kernel_size=3)
        result_5 = gpu_median_filter(data, kernel_size=5)

        assert result_3.shape == data.shape
        assert result_5.shape == data.shape
        # Larger kernel should smooth more
        assert np.std(result_5) <= np.std(result_3)

    def test_gpu_median_filter_handles_nan(self):
        """GPU median filter should handle NaN values."""
        from src.terrain.gpu_ops import gpu_median_filter

        data = np.random.rand(50, 50).astype(np.float32)
        data[20:25, 20:25] = np.nan

        result = gpu_median_filter(data, kernel_size=3)

        # NaN regions should remain NaN
        assert np.all(np.isnan(result[21:24, 21:24]))  # Interior of NaN region


class TestGpuMaxMinFilter:
    """Tests for GPU-accelerated max and min filters."""

    def test_gpu_max_filter_imports(self):
        """GPU max filter should be importable."""
        from src.terrain.gpu_ops import gpu_max_filter

        assert callable(gpu_max_filter)

    def test_gpu_min_filter_imports(self):
        """GPU min filter should be importable."""
        from src.terrain.gpu_ops import gpu_min_filter

        assert callable(gpu_min_filter)

    def test_gpu_max_filter_returns_same_shape(self):
        """GPU max filter should preserve input shape."""
        from src.terrain.gpu_ops import gpu_max_filter

        data = np.random.rand(100, 100).astype(np.float32)
        result = gpu_max_filter(data, kernel_size=3)

        assert result.shape == data.shape

    def test_gpu_min_filter_returns_same_shape(self):
        """GPU min filter should preserve input shape."""
        from src.terrain.gpu_ops import gpu_min_filter

        data = np.random.rand(100, 100).astype(np.float32)
        result = gpu_min_filter(data, kernel_size=3)

        assert result.shape == data.shape

    def test_gpu_max_filter_finds_local_maxima(self):
        """Max filter should dilate bright regions."""
        from src.terrain.gpu_ops import gpu_max_filter

        # Single bright pixel
        data = np.zeros((10, 10), dtype=np.float32)
        data[5, 5] = 1.0

        result = gpu_max_filter(data, kernel_size=3)

        # 3x3 region around bright pixel should all be 1.0
        assert np.all(result[4:7, 4:7] == 1.0), "Max filter should dilate"

    def test_gpu_min_filter_finds_local_minima(self):
        """Min filter should erode bright regions."""
        from src.terrain.gpu_ops import gpu_min_filter

        # Single dark pixel
        data = np.ones((10, 10), dtype=np.float32)
        data[5, 5] = 0.0

        result = gpu_min_filter(data, kernel_size=3)

        # 3x3 region around dark pixel should all be 0.0
        assert np.all(result[4:7, 4:7] == 0.0), "Min filter should erode"

    def test_gpu_max_filter_matches_scipy(self):
        """GPU max filter should match scipy results."""
        from scipy.ndimage import maximum_filter
        from src.terrain.gpu_ops import gpu_max_filter

        np.random.seed(42)
        data = np.random.rand(50, 50).astype(np.float32)

        scipy_result = maximum_filter(data, size=3)
        gpu_result = gpu_max_filter(data, kernel_size=3)

        # Should match closely
        np.testing.assert_allclose(
            gpu_result[1:-1, 1:-1], scipy_result[1:-1, 1:-1], rtol=1e-4, atol=1e-6
        )

    def test_gpu_min_filter_matches_scipy(self):
        """GPU min filter should match scipy results."""
        from scipy.ndimage import minimum_filter
        from src.terrain.gpu_ops import gpu_min_filter

        np.random.seed(42)
        data = np.random.rand(50, 50).astype(np.float32)

        scipy_result = minimum_filter(data, size=3)
        gpu_result = gpu_min_filter(data, kernel_size=3)

        # Should match closely
        np.testing.assert_allclose(
            gpu_result[1:-1, 1:-1], scipy_result[1:-1, 1:-1], rtol=1e-4, atol=1e-6
        )

    def test_gpu_max_min_filter_relationship(self):
        """Max of negated data should equal negated min of original."""
        from src.terrain.gpu_ops import gpu_max_filter, gpu_min_filter

        data = np.random.rand(50, 50).astype(np.float32)

        max_result = gpu_max_filter(data, kernel_size=3)
        min_result = gpu_min_filter(data, kernel_size=3)

        # min(x) == -max(-x)
        max_of_neg = gpu_max_filter(-data, kernel_size=3)
        np.testing.assert_allclose(-max_of_neg, min_result, rtol=1e-5)
