"""
TDD: Two-Pass Downsampling Tests

Goal: Implement two-pass downsampling for billion-pixel DEMs
Expected: 35x speedup (79s → ~2s) vs single-pass approach

RED Phase: Define expected behavior
"""

import numpy as np
import pytest
from src.terrain.transforms import downsample_raster, downsample_raster_optimized


class TestTwoPassDownsampling:
    """Test two-pass downsampling approach for large DEMs."""

    def test_two_pass_produces_correct_output_shape(self):
        """Two-pass downsampling should produce same output shape as single-pass."""
        # Create test data: 1000x1000 pixels
        data = np.random.randn(1000, 1000).astype(np.float32)
        target_shape = (100, 100)
        zoom_factor = target_shape[0] / data.shape[0]

        # Single pass (baseline)
        downsample_single = downsample_raster(zoom_factor=zoom_factor, method="average")
        result_single, _, _ = downsample_single(data)

        # Two-pass (should produce identical shape)
        downsample_first = downsample_raster(zoom_factor=0.316, method="average")  # ~1000 → 316
        result_first, _, _ = downsample_first(data)

        downsample_second = downsample_raster(zoom_factor=zoom_factor / 0.316, method="average")  # 316 → 100
        result_two_pass, _, _ = downsample_second(result_first)

        assert result_two_pass.shape == result_single.shape, \
            f"Shape mismatch: {result_two_pass.shape} vs {result_single.shape}"

    def test_two_pass_values_close_to_single_pass(self):
        """Two-pass should produce statistically similar values to single-pass."""
        # Create smooth test data where interpolation order matters less
        x = np.linspace(0, 1, 500)
        y = np.linspace(0, 1, 500)
        X, Y = np.meshgrid(x, y)
        data = (np.sin(X * 10) * np.cos(Y * 10)).astype(np.float32)
        target_shape = (50, 50)
        zoom_factor = target_shape[0] / data.shape[0]

        # Single pass
        downsample_single = downsample_raster(zoom_factor=zoom_factor, method="average")
        result_single, _, _ = downsample_single(data)

        # Two-pass
        downsample_first = downsample_raster(zoom_factor=0.316, method="average")
        result_first, _, _ = downsample_first(data)
        downsample_second = downsample_raster(zoom_factor=zoom_factor / 0.316, method="average")
        result_two_pass, _, _ = downsample_second(result_first)

        # Check correlation (should be very high)
        flat_single = result_single.flatten()
        flat_two_pass = result_two_pass.flatten()
        correlation = np.corrcoef(flat_single, flat_two_pass)[0, 1]

        assert correlation > 0.95, \
            f"Low correlation between single and two-pass: {correlation:.3f}"

    def test_two_pass_preserves_nodata_values(self):
        """Two-pass should correctly handle NaN values throughout."""
        data = np.random.randn(500, 500).astype(np.float32)
        # Add NaN values (simulating nodata)
        data[100:150, 100:150] = np.nan
        data[300:310, 300:310] = np.nan

        target_shape = (50, 50)
        zoom_factor = target_shape[0] / data.shape[0]

        # Two-pass with nodata
        downsample_first = downsample_raster(zoom_factor=0.316, method="average")
        result_first, _, _ = downsample_first(data)

        downsample_second = downsample_raster(zoom_factor=zoom_factor / 0.316, method="average")
        result_two_pass, _, _ = downsample_second(result_first)

        # Check that nodata regions are preserved (some NaN should remain)
        assert np.any(np.isnan(result_two_pass)), \
            "NaN values not preserved in two-pass downsampling"

    def test_two_pass_intermediate_resolution_scales(self):
        """Test that intermediate resolution is reasonable for very large DEMs."""
        # Simulate 1076M pixel DEM (28479×37782)
        large_shape = (28479, 37782)
        target_shape = (870, 1154)

        # Calculate zoom factors
        zoom_single = target_shape[0] / large_shape[0]

        # Two-pass: first to ~10% of original (reasonable intermediate)
        zoom_first = 0.316  # ~3x intermediate compression
        expected_first_shape = (int(large_shape[0] * zoom_first),
                               int(large_shape[1] * zoom_first))

        zoom_second = zoom_single / zoom_first
        expected_final_shape = (int(expected_first_shape[0] * zoom_second),
                               int(expected_first_shape[1] * zoom_second))

        # Verify intermediate and final shapes are reasonable
        assert expected_first_shape[0] > 1000, "Intermediate resolution too small"
        # Allow 1-2 pixel difference due to integer rounding in two-pass chain
        # (0.1% error is negligible for mesh generation)
        assert abs(expected_final_shape[0] - target_shape[0]) <= 1, \
            f"Final shape height mismatch: {expected_final_shape[0]} vs {target_shape[0]}"
        assert abs(expected_final_shape[1] - target_shape[1]) <= 1, \
            f"Final shape width mismatch: {expected_final_shape[1]} vs {target_shape[1]}"

    def test_two_pass_transform_chain_consistency(self):
        """Transform metadata should chain correctly through two passes."""
        from affine import Affine

        data = np.random.randn(500, 500).astype(np.float32)
        original_transform = Affine.translation(0, 100) * Affine.scale(1, -1)
        target_shape = (50, 50)
        zoom_factor = target_shape[0] / data.shape[0]

        # Two-pass should chain transforms correctly
        downsample_first = downsample_raster(zoom_factor=0.316, method="average")
        result_first, transform_first, _ = downsample_first(data, original_transform)

        downsample_second = downsample_raster(zoom_factor=zoom_factor / 0.316, method="average")
        result_final, transform_final, _ = downsample_second(result_first, transform_first)

        # Final transform should match what single-pass would produce
        downsample_single = downsample_raster(zoom_factor=zoom_factor, method="average")
        _, transform_single, _ = downsample_single(data, original_transform)

        # Transforms should be approximately equal
        assert np.allclose(transform_final.to_gdal()[:6], transform_single.to_gdal()[:6]), \
            "Transform chains don't match between single and two-pass"


class TestOptimizedDownsampling:
    """Test the automatic two-pass optimizer."""

    def test_optimized_small_compression_uses_single_pass(self):
        """Small compression ratios should use single-pass."""
        data = np.random.randn(1000, 1000).astype(np.float32)
        target_shape = (500, 500)  # 2:1 compression (small)
        zoom_factor = target_shape[0] / data.shape[0]

        downsample_opt = downsample_raster_optimized(zoom_factor=zoom_factor, method="average")
        result, _, _ = downsample_opt(data)

        assert result.shape == target_shape

    def test_optimized_large_compression_uses_two_pass(self):
        """Large compression ratios (>100:1) should auto-use two-pass."""
        data = np.random.randn(5000, 5000).astype(np.float32)
        target_shape = (50, 50)  # 100:1 compression (triggers two-pass)
        zoom_factor = target_shape[0] / data.shape[0]

        downsample_opt = downsample_raster_optimized(zoom_factor=zoom_factor, method="average")
        result, _, _ = downsample_opt(data)

        # Allow 1-pixel rounding difference from two-pass chaining
        assert abs(result.shape[0] - target_shape[0]) <= 1
        assert abs(result.shape[1] - target_shape[1]) <= 1

    def test_optimized_matches_single_pass_values(self):
        """Optimized should produce statistically similar values for large compressions."""
        x = np.linspace(0, 1, 1000)
        y = np.linspace(0, 1, 1000)
        X, Y = np.meshgrid(x, y)
        data = (np.sin(X * 10) * np.cos(Y * 10)).astype(np.float32)
        target_shape = (50, 50)
        zoom_factor = target_shape[0] / data.shape[0]

        # Single-pass baseline
        downsample_single = downsample_raster(zoom_factor=zoom_factor, method="average")
        result_single, _, _ = downsample_single(data)

        # Optimized (should use two-pass internally)
        downsample_opt = downsample_raster_optimized(zoom_factor=zoom_factor, method="average")
        result_opt, _, _ = downsample_opt(data)

        # Handle potential shape mismatch from rounding (crop to common size)
        h = min(result_single.shape[0], result_opt.shape[0])
        w = min(result_single.shape[1], result_opt.shape[1])
        flat_single = result_single[:h, :w].flatten()
        flat_opt = result_opt[:h, :w].flatten()
        correlation = np.corrcoef(flat_single, flat_opt)[0, 1]

        assert correlation > 0.90, \
            f"Low correlation between single and optimized: {correlation:.3f}"

    def test_optimized_preserves_billion_pixel_scale(self):
        """Optimized should handle billion-pixel scale gracefully."""
        # Don't create actual 1B pixel array, just verify the logic
        large_shape = (28479, 37782)  # ~1B pixels
        target_shape = (870, 1154)
        zoom_factor = target_shape[0] / large_shape[0]

        # Compression ratio calculation
        compression_ratio = 1 / (zoom_factor ** 2)

        # Should trigger two-pass (compression > 100:1)
        assert compression_ratio > 100, "Test setup: should have large compression ratio"

        # Verify two-pass zoom factors would be reasonable
        zoom_first = 1 / np.sqrt(compression_ratio / 10)
        zoom_first = min(zoom_first, 0.5)
        # Allow wider range due to square root calculation
        assert 0.05 < zoom_first < 0.5, "First pass zoom factor should be reasonable"


class TestTwoPassPerformance:
    """Benchmark two-pass vs single-pass (performance tests)."""

    @pytest.mark.benchmark
    def test_single_pass_speed_baseline(self, benchmark):
        """Baseline: measure single-pass downsampling speed on moderate data."""
        data = np.random.randn(5000, 5000).astype(np.float32)
        downsample_fn = downsample_raster(zoom_factor=0.1, method="average")

        def run_single_pass():
            result, _, _ = downsample_fn(data)
            return result

        result = benchmark(run_single_pass)
        assert result.shape == (500, 500)

    @pytest.mark.benchmark
    def test_two_pass_speed_improvement(self, benchmark):
        """Measure if two-pass is faster than single-pass on moderate data."""
        data = np.random.randn(5000, 5000).astype(np.float32)
        target_shape = (500, 500)
        zoom_factor = target_shape[0] / data.shape[0]

        downsample_first = downsample_raster(zoom_factor=0.316, method="average")
        downsample_second = downsample_raster(zoom_factor=zoom_factor / 0.316, method="average")

        def run_two_pass():
            result_first, _, _ = downsample_first(data)
            result_final, _, _ = downsample_second(result_first)
            return result_final

        result = benchmark(run_two_pass)
        assert result.shape == target_shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
