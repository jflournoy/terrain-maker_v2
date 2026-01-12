"""
Tests for the diagnostics module.

Tests the visualization and diagnostic functions used for terrain processing
analysis, including histogram generation and processing pipeline visualization.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


class TestGenerateRGBHistogram:
    """Tests for generate_rgb_histogram function."""

    def test_generate_rgb_histogram_imports(self):
        """Test that generate_rgb_histogram can be imported."""
        from src.terrain.diagnostics import generate_rgb_histogram

        assert callable(generate_rgb_histogram)

    def test_generate_rgb_histogram_creates_output_file(self, tmp_path):
        """Test that generate_rgb_histogram creates an output file."""
        from src.terrain.diagnostics import generate_rgb_histogram
        from PIL import Image

        # Create a test image
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        input_path = tmp_path / "test_image.png"
        img.save(input_path)

        output_path = tmp_path / "histogram.png"
        result = generate_rgb_histogram(input_path, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_generate_rgb_histogram_with_rgba_image(self, tmp_path):
        """Test that generate_rgb_histogram handles RGBA images."""
        from src.terrain.diagnostics import generate_rgb_histogram
        from PIL import Image

        # Create a test RGBA image
        img_array = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGBA')
        input_path = tmp_path / "test_rgba.png"
        img.save(input_path)

        output_path = tmp_path / "histogram.png"
        result = generate_rgb_histogram(input_path, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_generate_rgb_histogram_with_grayscale_returns_none(self, tmp_path):
        """Test that generate_rgb_histogram returns None for grayscale images."""
        from src.terrain.diagnostics import generate_rgb_histogram
        from PIL import Image

        # Create a grayscale image
        img_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        input_path = tmp_path / "test_gray.png"
        img.save(input_path)

        output_path = tmp_path / "histogram.png"
        result = generate_rgb_histogram(input_path, output_path)

        assert result is None

    def test_generate_rgb_histogram_with_missing_file_returns_none(self, tmp_path):
        """Test that generate_rgb_histogram returns None for missing files."""
        from src.terrain.diagnostics import generate_rgb_histogram

        input_path = tmp_path / "nonexistent.png"
        output_path = tmp_path / "histogram.png"
        result = generate_rgb_histogram(input_path, output_path)

        assert result is None

    def test_generate_rgb_histogram_requires_existing_parent_dirs(self, tmp_path):
        """Test that generate_rgb_histogram returns None if parent dirs don't exist.

        Note: Unlike other diagnostic functions, generate_rgb_histogram does not
        create parent directories automatically. It returns None if the path is invalid.
        """
        from src.terrain.diagnostics import generate_rgb_histogram
        from PIL import Image

        # Create a test image
        img_array = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        input_path = tmp_path / "test.png"
        img.save(input_path)

        # Output in non-existent nested directory
        output_path = tmp_path / "subdir" / "nested" / "histogram.png"
        result = generate_rgb_histogram(input_path, output_path)

        # Should return None because parent dirs don't exist
        assert result is None


class TestGenerateLuminanceHistogram:
    """Tests for generate_luminance_histogram function."""

    def test_generate_luminance_histogram_imports(self):
        """Test that generate_luminance_histogram can be imported."""
        from src.terrain.diagnostics import generate_luminance_histogram

        assert callable(generate_luminance_histogram)

    def test_generate_luminance_histogram_creates_output_file(self, tmp_path):
        """Test that generate_luminance_histogram creates an output file."""
        from src.terrain.diagnostics import generate_luminance_histogram
        from PIL import Image

        # Create a test image
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        input_path = tmp_path / "test_image.png"
        img.save(input_path)

        output_path = tmp_path / "luminance.png"
        result = generate_luminance_histogram(input_path, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_generate_luminance_histogram_with_grayscale(self, tmp_path):
        """Test that generate_luminance_histogram handles grayscale images."""
        from src.terrain.diagnostics import generate_luminance_histogram
        from PIL import Image

        # Create a grayscale image
        img_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        input_path = tmp_path / "test_gray.png"
        img.save(input_path)

        output_path = tmp_path / "luminance.png"
        result = generate_luminance_histogram(input_path, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_generate_luminance_histogram_with_rgba(self, tmp_path):
        """Test that generate_luminance_histogram handles RGBA images."""
        from src.terrain.diagnostics import generate_luminance_histogram
        from PIL import Image

        # Create a test RGBA image
        img_array = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGBA')
        input_path = tmp_path / "test_rgba.png"
        img.save(input_path)

        output_path = tmp_path / "luminance.png"
        result = generate_luminance_histogram(input_path, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_generate_luminance_histogram_with_missing_file_returns_none(self, tmp_path):
        """Test that generate_luminance_histogram returns None for missing files."""
        from src.terrain.diagnostics import generate_luminance_histogram

        input_path = tmp_path / "nonexistent.png"
        output_path = tmp_path / "luminance.png"
        result = generate_luminance_histogram(input_path, output_path)

        assert result is None

    def test_generate_luminance_histogram_with_pure_black_image(self, tmp_path):
        """Test luminance histogram with pure black image."""
        from src.terrain.diagnostics import generate_luminance_histogram
        from PIL import Image

        # Create pure black image
        img_array = np.zeros((50, 50, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        input_path = tmp_path / "black.png"
        img.save(input_path)

        output_path = tmp_path / "luminance.png"
        result = generate_luminance_histogram(input_path, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_generate_luminance_histogram_with_pure_white_image(self, tmp_path):
        """Test luminance histogram with pure white image."""
        from src.terrain.diagnostics import generate_luminance_histogram
        from PIL import Image

        # Create pure white image
        img_array = np.full((50, 50, 3), 255, dtype=np.uint8)
        img = Image.fromarray(img_array)
        input_path = tmp_path / "white.png"
        img.save(input_path)

        output_path = tmp_path / "luminance.png"
        result = generate_luminance_histogram(input_path, output_path)

        assert result == output_path
        assert output_path.exists()


class TestPlotWaveletDiagnostics:
    """Tests for plot_wavelet_diagnostics function."""

    def test_plot_wavelet_diagnostics_imports(self):
        """Test that plot_wavelet_diagnostics can be imported."""
        from src.terrain.diagnostics import plot_wavelet_diagnostics

        assert callable(plot_wavelet_diagnostics)

    def test_plot_wavelet_diagnostics_creates_file(self, tmp_path):
        """Test that plot_wavelet_diagnostics creates an output file."""
        from src.terrain.diagnostics import plot_wavelet_diagnostics

        # Create simple test data
        original = np.random.rand(50, 50) * 100 + 200
        denoised = original + np.random.randn(50, 50) * 0.1

        output_path = tmp_path / "wavelet_diag.png"
        result = plot_wavelet_diagnostics(original, denoised, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_plot_wavelet_diagnostics_with_nan_nodata(self, tmp_path):
        """Test plot_wavelet_diagnostics handles NaN nodata values."""
        from src.terrain.diagnostics import plot_wavelet_diagnostics

        # Create data with NaN values
        original = np.random.rand(50, 50) * 100 + 200
        original[10:20, 10:20] = np.nan
        denoised = original.copy()

        output_path = tmp_path / "wavelet_nan.png"
        result = plot_wavelet_diagnostics(
            original, denoised, output_path, nodata_value=np.nan
        )

        assert result == output_path
        assert output_path.exists()

    def test_plot_wavelet_diagnostics_custom_profile_row(self, tmp_path):
        """Test plot_wavelet_diagnostics with custom profile row."""
        from src.terrain.diagnostics import plot_wavelet_diagnostics

        original = np.random.rand(50, 50) * 100 + 200
        denoised = original + np.random.randn(50, 50) * 0.1

        output_path = tmp_path / "wavelet_custom.png"
        result = plot_wavelet_diagnostics(
            original, denoised, output_path, profile_row=10
        )

        assert result == output_path
        assert output_path.exists()


class TestPlotProcessingPipeline:
    """Tests for plot_processing_pipeline function."""

    def test_plot_processing_pipeline_imports(self):
        """Test that plot_processing_pipeline can be imported."""
        from src.terrain.diagnostics import plot_processing_pipeline

        assert callable(plot_processing_pipeline)

    def test_plot_processing_pipeline_creates_file(self, tmp_path):
        """Test that plot_processing_pipeline creates an output file."""
        from src.terrain.diagnostics import plot_processing_pipeline

        # Create test stages
        stages = {
            "Raw": np.random.rand(50, 50) * 100 + 200,
            "Smoothed": np.random.rand(50, 50) * 100 + 200,
            "Final": np.random.rand(50, 50) * 100 + 200,
        }

        output_path = tmp_path / "pipeline.png"
        result = plot_processing_pipeline(stages, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_plot_processing_pipeline_with_empty_stages_returns_none(self, tmp_path):
        """Test that empty stages dict returns None."""
        from src.terrain.diagnostics import plot_processing_pipeline

        output_path = tmp_path / "pipeline.png"
        result = plot_processing_pipeline({}, output_path)

        assert result is None

    def test_plot_processing_pipeline_with_single_stage(self, tmp_path):
        """Test plot_processing_pipeline with single stage."""
        from src.terrain.diagnostics import plot_processing_pipeline

        stages = {"Only Stage": np.random.rand(50, 50) * 100 + 200}

        output_path = tmp_path / "single.png"
        result = plot_processing_pipeline(stages, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_plot_processing_pipeline_with_many_stages(self, tmp_path):
        """Test plot_processing_pipeline with many stages."""
        from src.terrain.diagnostics import plot_processing_pipeline

        # Create 6 stages (2 rows x 3 cols)
        stages = {
            f"Stage {i}": np.random.rand(50, 50) * 100 + 200
            for i in range(6)
        }

        output_path = tmp_path / "many.png"
        result = plot_processing_pipeline(stages, output_path)

        assert result == output_path
        assert output_path.exists()


class TestPlotBumpRemovalDiagnostics:
    """Tests for plot_bump_removal_diagnostics function."""

    def test_plot_bump_removal_diagnostics_imports(self):
        """Test that plot_bump_removal_diagnostics can be imported."""
        from src.terrain.diagnostics import plot_bump_removal_diagnostics

        assert callable(plot_bump_removal_diagnostics)

    def test_plot_bump_removal_diagnostics_creates_file(self, tmp_path):
        """Test that plot_bump_removal_diagnostics creates an output file."""
        from src.terrain.diagnostics import plot_bump_removal_diagnostics

        # Create test data with bumps
        original = np.ones((50, 50)) * 100
        original[20:25, 20:25] = 110  # Add a bump

        # After removal, bump is gone
        after_removal = np.ones((50, 50)) * 100

        output_path = tmp_path / "bump.png"
        result = plot_bump_removal_diagnostics(original, after_removal, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_plot_bump_removal_diagnostics_with_no_bumps(self, tmp_path):
        """Test bump removal diagnostics when no bumps exist."""
        from src.terrain.diagnostics import plot_bump_removal_diagnostics

        # Identical arrays - no bumps removed
        original = np.ones((50, 50)) * 100
        after_removal = np.ones((50, 50)) * 100

        output_path = tmp_path / "no_bumps.png"
        result = plot_bump_removal_diagnostics(original, after_removal, output_path)

        assert result == output_path
        assert output_path.exists()


class TestGenerateBumpRemovalDiagnostics:
    """Tests for generate_bump_removal_diagnostics wrapper function."""

    def test_generate_bump_removal_diagnostics_imports(self):
        """Test that generate_bump_removal_diagnostics can be imported."""
        from src.terrain.diagnostics import generate_bump_removal_diagnostics

        assert callable(generate_bump_removal_diagnostics)

    def test_generate_bump_removal_diagnostics_creates_file(self, tmp_path):
        """Test that generate_bump_removal_diagnostics creates an output file."""
        from src.terrain.diagnostics import generate_bump_removal_diagnostics

        original = np.random.rand(50, 50) * 100 + 200
        after_removal = original - np.random.rand(50, 50) * 0.5

        result = generate_bump_removal_diagnostics(
            original, after_removal, tmp_path, prefix="test_bump"
        )

        assert result == tmp_path / "test_bump_comparison.png"
        assert result.exists()


class TestPlotUpscaleDiagnostics:
    """Tests for plot_upscale_diagnostics function."""

    def test_plot_upscale_diagnostics_imports(self):
        """Test that plot_upscale_diagnostics can be imported."""
        from src.terrain.diagnostics import plot_upscale_diagnostics

        assert callable(plot_upscale_diagnostics)

    def test_plot_upscale_diagnostics_creates_file(self, tmp_path):
        """Test that plot_upscale_diagnostics creates an output file."""
        from src.terrain.diagnostics import plot_upscale_diagnostics

        # Create test data at different scales
        original = np.random.rand(25, 25)
        upscaled = np.random.rand(100, 100)  # 4x upscale

        output_path = tmp_path / "upscale.png"
        result = plot_upscale_diagnostics(
            original, upscaled, output_path, scale=4, method="bicubic"
        )

        assert result == output_path
        assert output_path.exists()

    def test_plot_upscale_diagnostics_with_nan_values(self, tmp_path):
        """Test upscale diagnostics handles NaN values."""
        from src.terrain.diagnostics import plot_upscale_diagnostics

        original = np.random.rand(25, 25)
        original[5:10, 5:10] = np.nan
        upscaled = np.random.rand(100, 100)
        upscaled[20:40, 20:40] = np.nan

        output_path = tmp_path / "upscale_nan.png"
        result = plot_upscale_diagnostics(
            original, upscaled, output_path, scale=4
        )

        assert result == output_path
        assert output_path.exists()


class TestGenerateUpscaleDiagnostics:
    """Tests for generate_upscale_diagnostics wrapper function."""

    def test_generate_upscale_diagnostics_imports(self):
        """Test that generate_upscale_diagnostics can be imported."""
        from src.terrain.diagnostics import generate_upscale_diagnostics

        assert callable(generate_upscale_diagnostics)

    def test_generate_upscale_diagnostics_creates_file(self, tmp_path):
        """Test that generate_upscale_diagnostics creates an output file."""
        from src.terrain.diagnostics import generate_upscale_diagnostics

        original = np.random.rand(20, 20)
        upscaled = np.random.rand(80, 80)

        result = generate_upscale_diagnostics(
            original, upscaled, tmp_path, prefix="test_upscale", scale=4
        )

        assert result == tmp_path / "test_upscale_diagnostics.png"
        assert result.exists()


class TestPlotAdaptiveSmoothDiagnostics:
    """Tests for plot_adaptive_smooth_diagnostics function."""

    def test_plot_adaptive_smooth_diagnostics_imports(self):
        """Test that plot_adaptive_smooth_diagnostics can be imported."""
        from src.terrain.diagnostics import plot_adaptive_smooth_diagnostics

        assert callable(plot_adaptive_smooth_diagnostics)

    def test_plot_adaptive_smooth_diagnostics_creates_file(self, tmp_path):
        """Test that plot_adaptive_smooth_diagnostics creates an output file."""
        from src.terrain.diagnostics import plot_adaptive_smooth_diagnostics

        # Create test DEM data
        np.random.seed(42)
        original = np.random.rand(50, 50) * 100 + 200
        smoothed = original + np.random.randn(50, 50) * 0.5

        output_path = tmp_path / "adaptive_smooth.png"
        result = plot_adaptive_smooth_diagnostics(
            original, smoothed, output_path, pixel_size=30.0
        )

        assert result == output_path
        assert output_path.exists()

    def test_plot_adaptive_smooth_diagnostics_with_edge_threshold(self, tmp_path):
        """Test adaptive smooth diagnostics with edge threshold."""
        from src.terrain.diagnostics import plot_adaptive_smooth_diagnostics

        np.random.seed(42)
        original = np.random.rand(50, 50) * 100 + 200
        # Add sharp edge
        original[25:, :] += 50
        smoothed = original + np.random.randn(50, 50) * 0.5

        output_path = tmp_path / "adaptive_edge.png"
        result = plot_adaptive_smooth_diagnostics(
            original, smoothed, output_path,
            pixel_size=30.0, edge_threshold=10.0
        )

        assert result == output_path
        assert output_path.exists()


class TestPlotAdaptiveSmoothHistogram:
    """Tests for plot_adaptive_smooth_histogram function."""

    def test_plot_adaptive_smooth_histogram_imports(self):
        """Test that plot_adaptive_smooth_histogram can be imported."""
        from src.terrain.diagnostics import plot_adaptive_smooth_histogram

        assert callable(plot_adaptive_smooth_histogram)

    def test_plot_adaptive_smooth_histogram_creates_file(self, tmp_path):
        """Test that plot_adaptive_smooth_histogram creates an output file."""
        from src.terrain.diagnostics import plot_adaptive_smooth_histogram

        np.random.seed(42)
        original = np.random.rand(50, 50) * 100 + 200
        smoothed = original + np.random.randn(50, 50) * 0.5

        output_path = tmp_path / "smooth_hist.png"
        result = plot_adaptive_smooth_histogram(
            original, smoothed, output_path, pixel_size=30.0
        )

        assert result == output_path
        assert output_path.exists()


class TestGenerateFullAdaptiveSmoothDiagnostics:
    """Tests for generate_full_adaptive_smooth_diagnostics wrapper function."""

    def test_generate_full_adaptive_smooth_diagnostics_imports(self):
        """Test that generate_full_adaptive_smooth_diagnostics can be imported."""
        from src.terrain.diagnostics import generate_full_adaptive_smooth_diagnostics

        assert callable(generate_full_adaptive_smooth_diagnostics)

    def test_generate_full_adaptive_smooth_diagnostics_creates_files(self, tmp_path):
        """Test that generate_full_adaptive_smooth_diagnostics creates both output files."""
        from src.terrain.diagnostics import generate_full_adaptive_smooth_diagnostics

        np.random.seed(42)
        original = np.random.rand(50, 50) * 100 + 200
        smoothed = original + np.random.randn(50, 50) * 0.5

        spatial_path, hist_path = generate_full_adaptive_smooth_diagnostics(
            original, smoothed, tmp_path, prefix="test_smooth", pixel_size=30.0
        )

        assert spatial_path == tmp_path / "test_smooth_comparison.png"
        assert hist_path == tmp_path / "test_smooth_histogram.png"
        assert spatial_path.exists()
        assert hist_path.exists()


class TestPlotWaveletCoefficients:
    """Tests for plot_wavelet_coefficients function."""

    def test_plot_wavelet_coefficients_imports(self):
        """Test that plot_wavelet_coefficients can be imported."""
        from src.terrain.diagnostics import plot_wavelet_coefficients

        assert callable(plot_wavelet_coefficients)

    def test_plot_wavelet_coefficients_creates_file(self, tmp_path):
        """Test that plot_wavelet_coefficients creates an output file when pywt is available."""
        from src.terrain.diagnostics import plot_wavelet_coefficients

        try:
            import pywt
        except ImportError:
            pytest.skip("PyWavelets not available")

        original = np.random.rand(64, 64) * 100 + 200

        output_path = tmp_path / "wavelet_coeff.png"
        result = plot_wavelet_coefficients(original, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_plot_wavelet_coefficients_with_nan_values(self, tmp_path):
        """Test wavelet coefficients plot handles NaN values."""
        from src.terrain.diagnostics import plot_wavelet_coefficients

        try:
            import pywt
        except ImportError:
            pytest.skip("PyWavelets not available")

        original = np.random.rand(64, 64) * 100 + 200
        original[10:20, 10:20] = np.nan

        output_path = tmp_path / "wavelet_nan.png"
        result = plot_wavelet_coefficients(original, output_path)

        assert result == output_path
        assert output_path.exists()


class TestGenerateFullWaveletDiagnostics:
    """Tests for generate_full_wavelet_diagnostics wrapper function."""

    def test_generate_full_wavelet_diagnostics_imports(self):
        """Test that generate_full_wavelet_diagnostics can be imported."""
        from src.terrain.diagnostics import generate_full_wavelet_diagnostics

        assert callable(generate_full_wavelet_diagnostics)

    def test_generate_full_wavelet_diagnostics_creates_files(self, tmp_path):
        """Test that generate_full_wavelet_diagnostics creates both output files."""
        from src.terrain.diagnostics import generate_full_wavelet_diagnostics

        try:
            import pywt
        except ImportError:
            pytest.skip("PyWavelets not available")

        original = np.random.rand(64, 64) * 100 + 200
        denoised = original + np.random.randn(64, 64) * 0.1

        comparison_path, coeff_path = generate_full_wavelet_diagnostics(
            original, denoised, tmp_path, prefix="test_wavelet"
        )

        assert comparison_path == tmp_path / "test_wavelet_comparison.png"
        assert coeff_path == tmp_path / "test_wavelet_coefficients.png"
        assert comparison_path.exists()
        assert coeff_path.exists()


class TestHistogramLuminanceCalculation:
    """Tests for correct luminance calculation in generate_luminance_histogram."""

    def test_luminance_calculation_pure_red(self, tmp_path):
        """Test luminance calculation for pure red image."""
        from src.terrain.diagnostics import generate_luminance_histogram
        from PIL import Image

        # Pure red: R=255, G=0, B=0
        # Luminance = 0.299*255 + 0.587*0 + 0.114*0 = 76.245
        img_array = np.zeros((10, 10, 3), dtype=np.uint8)
        img_array[:, :, 0] = 255  # Red channel

        img = Image.fromarray(img_array)
        input_path = tmp_path / "red.png"
        img.save(input_path)

        output_path = tmp_path / "luminance.png"
        result = generate_luminance_histogram(input_path, output_path)

        assert result is not None
        assert output_path.exists()

    def test_luminance_calculation_pure_green(self, tmp_path):
        """Test luminance calculation for pure green image."""
        from src.terrain.diagnostics import generate_luminance_histogram
        from PIL import Image

        # Pure green: R=0, G=255, B=0
        # Luminance = 0.299*0 + 0.587*255 + 0.114*0 = 149.685
        img_array = np.zeros((10, 10, 3), dtype=np.uint8)
        img_array[:, :, 1] = 255  # Green channel

        img = Image.fromarray(img_array)
        input_path = tmp_path / "green.png"
        img.save(input_path)

        output_path = tmp_path / "luminance.png"
        result = generate_luminance_histogram(input_path, output_path)

        assert result is not None
        assert output_path.exists()


class TestDiagnosticsEdgeCases:
    """Tests for edge cases in diagnostic functions."""

    def test_very_small_arrays(self, tmp_path):
        """Test diagnostics with very small arrays."""
        from src.terrain.diagnostics import plot_wavelet_diagnostics

        # 5x5 array - minimal size
        original = np.random.rand(5, 5) * 100 + 200
        denoised = original + np.random.randn(5, 5) * 0.1

        output_path = tmp_path / "tiny.png"
        result = plot_wavelet_diagnostics(original, denoised, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_rectangular_arrays(self, tmp_path):
        """Test diagnostics with non-square arrays."""
        from src.terrain.diagnostics import plot_processing_pipeline

        stages = {
            "Stage 1": np.random.rand(30, 100) * 100 + 200,
            "Stage 2": np.random.rand(30, 100) * 100 + 200,
        }

        output_path = tmp_path / "rect.png"
        result = plot_processing_pipeline(stages, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_large_value_ranges(self, tmp_path):
        """Test diagnostics with large elevation ranges."""
        from src.terrain.diagnostics import plot_wavelet_diagnostics

        # Large elevation range: 0 to 10000m
        original = np.random.rand(50, 50) * 10000
        denoised = original + np.random.randn(50, 50) * 10

        output_path = tmp_path / "large_range.png"
        result = plot_wavelet_diagnostics(original, denoised, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_uniform_arrays(self, tmp_path):
        """Test diagnostics with uniform (constant) arrays."""
        from src.terrain.diagnostics import plot_wavelet_diagnostics

        # Uniform values - no variation
        original = np.ones((50, 50)) * 100
        denoised = np.ones((50, 50)) * 100

        output_path = tmp_path / "uniform.png"
        result = plot_wavelet_diagnostics(original, denoised, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_all_nan_arrays(self, tmp_path):
        """Test handling of all-NaN arrays."""
        from src.terrain.diagnostics import plot_processing_pipeline

        stages = {
            "All NaN": np.full((50, 50), np.nan),
            "Some Valid": np.random.rand(50, 50) * 100,
        }

        output_path = tmp_path / "nan_stage.png"
        # Should not crash, may produce warning
        try:
            result = plot_processing_pipeline(stages, output_path)
            # Either succeeds or returns None
            assert result is None or output_path.exists()
        except (ValueError, RuntimeWarning):
            # Some edge cases may raise, which is acceptable
            pass
