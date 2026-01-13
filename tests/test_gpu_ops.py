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
