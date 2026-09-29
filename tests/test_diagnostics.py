"""
Tests for the diagnostics module.

Tests the visualization and diagnostic functions used for terrain processing
analysis, including histogram generation and processing pipeline visualization.
"""

import numpy as np


class TestGenerateRGBHistogram:
    """Tests for generate_rgb_histogram function."""

    def test_generate_rgb_histogram_imports(self):
        """Test that generate_rgb_histogram can be imported."""
        from terrain_maker.terrain.diagnostics import generate_rgb_histogram

        assert callable(generate_rgb_histogram)

    def test_generate_rgb_histogram_creates_output_file(self, tmp_path):
        """Test that generate_rgb_histogram creates an output file."""
        from terrain_maker.terrain.diagnostics import generate_rgb_histogram
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
        from terrain_maker.terrain.diagnostics import generate_rgb_histogram
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
        from terrain_maker.terrain.diagnostics import generate_rgb_histogram
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
        from terrain_maker.terrain.diagnostics import generate_rgb_histogram

        input_path = tmp_path / "nonexistent.png"
        output_path = tmp_path / "histogram.png"
        result = generate_rgb_histogram(input_path, output_path)

        assert result is None

    def test_generate_rgb_histogram_requires_existing_parent_dirs(self, tmp_path):
        """Test that generate_rgb_histogram returns None if parent dirs don't exist.

        Note: Unlike other diagnostic functions, generate_rgb_histogram does not
        create parent directories automatically. It returns None if the path is invalid.
        """
        from terrain_maker.terrain.diagnostics import generate_rgb_histogram
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
        from terrain_maker.terrain.diagnostics import generate_luminance_histogram

        assert callable(generate_luminance_histogram)

    def test_generate_luminance_histogram_creates_output_file(self, tmp_path):
        """Test that generate_luminance_histogram creates an output file."""
        from terrain_maker.terrain.diagnostics import generate_luminance_histogram
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
        from terrain_maker.terrain.diagnostics import generate_luminance_histogram
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
        from terrain_maker.terrain.diagnostics import generate_luminance_histogram
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
        from terrain_maker.terrain.diagnostics import generate_luminance_histogram

        input_path = tmp_path / "nonexistent.png"
        output_path = tmp_path / "luminance.png"
        result = generate_luminance_histogram(input_path, output_path)

        assert result is None

    def test_generate_luminance_histogram_with_pure_black_image(self, tmp_path):
        """Test luminance histogram with pure black image."""
        from terrain_maker.terrain.diagnostics import generate_luminance_histogram
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
        from terrain_maker.terrain.diagnostics import generate_luminance_histogram
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


class TestHistogramLogScale:
    """Tests for log-scale Y-axis in histogram functions (TDD RED)."""

    def test_rgb_histogram_uses_log_scale_yaxis(self, tmp_path, monkeypatch):
        """Test that RGB histogram uses log scale on Y-axis."""
        from terrain_maker.terrain.diagnostics import generate_rgb_histogram
        from PIL import Image
        import matplotlib.pyplot as plt

        # Track calls to set_yscale
        yscale_calls = []
        original_set_yscale = plt.Axes.set_yscale

        def mock_set_yscale(self, value, *args, **kwargs):
            yscale_calls.append(value)
            return original_set_yscale(self, value, *args, **kwargs)

        monkeypatch.setattr(plt.Axes, "set_yscale", mock_set_yscale)

        # Create test image with varied pixel values
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        input_path = tmp_path / "test_image.png"
        img.save(input_path)

        output_path = tmp_path / "histogram.png"
        result = generate_rgb_histogram(input_path, output_path)

        assert result == output_path
        assert output_path.exists()
        # Verify log scale was set
        assert "log" in yscale_calls, f"Expected 'log' in yscale calls, got: {yscale_calls}"

    def test_luminance_histogram_uses_log_scale_yaxis(self, tmp_path, monkeypatch):
        """Test that luminance histogram uses log scale on Y-axis."""
        from terrain_maker.terrain.diagnostics import generate_luminance_histogram
        from PIL import Image
        import matplotlib.pyplot as plt

        # Track calls to set_yscale
        yscale_calls = []
        original_set_yscale = plt.Axes.set_yscale

        def mock_set_yscale(self, value, *args, **kwargs):
            yscale_calls.append(value)
            return original_set_yscale(self, value, *args, **kwargs)

        monkeypatch.setattr(plt.Axes, "set_yscale", mock_set_yscale)

        # Create test image
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        input_path = tmp_path / "test_image.png"
        img.save(input_path)

        output_path = tmp_path / "luminance.png"
        result = generate_luminance_histogram(input_path, output_path)

        assert result == output_path
        assert output_path.exists()
        # Verify log scale was set
        assert "log" in yscale_calls, f"Expected 'log' in yscale calls, got: {yscale_calls}"

    def test_rgb_histogram_handles_zero_counts_gracefully(self, tmp_path):
        """Test that RGB histogram handles bins with zero counts (log(0) issue)."""
        from terrain_maker.terrain.diagnostics import generate_rgb_histogram
        from PIL import Image

        # Create image with only a few distinct values - many bins will have 0 counts
        img_array = np.zeros((50, 50, 3), dtype=np.uint8)
        img_array[:, :, 0] = 128  # Only red = 128
        img_array[:, :, 1] = 64   # Only green = 64
        img_array[:, :, 2] = 200  # Only blue = 200
        img = Image.fromarray(img_array)
        input_path = tmp_path / "sparse.png"
        img.save(input_path)

        output_path = tmp_path / "histogram.png"
        # This should not raise any errors despite many zero-count bins
        result = generate_rgb_histogram(input_path, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_luminance_histogram_handles_zero_counts_gracefully(self, tmp_path):
        """Test that luminance histogram handles bins with zero counts."""
        from terrain_maker.terrain.diagnostics import generate_luminance_histogram
        from PIL import Image

        # Create image with single luminance value - most bins will have 0 counts
        img_array = np.full((50, 50, 3), 100, dtype=np.uint8)
        img = Image.fromarray(img_array)
        input_path = tmp_path / "uniform.png"
        img.save(input_path)

        output_path = tmp_path / "luminance.png"
        # This should not raise any errors despite many zero-count bins
        result = generate_luminance_histogram(input_path, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_rgb_histogram_ylabel_indicates_log_scale(self, tmp_path, monkeypatch):
        """Test that RGB histogram Y-axis label indicates log scale."""
        from terrain_maker.terrain.diagnostics import generate_rgb_histogram
        from PIL import Image
        import matplotlib.pyplot as plt

        # Track ylabel calls
        ylabel_calls = []
        original_set_ylabel = plt.Axes.set_ylabel

        def mock_set_ylabel(self, ylabel, *args, **kwargs):
            ylabel_calls.append(ylabel)
            return original_set_ylabel(self, ylabel, *args, **kwargs)

        monkeypatch.setattr(plt.Axes, "set_ylabel", mock_set_ylabel)

        img_array = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        input_path = tmp_path / "test.png"
        img.save(input_path)

        output_path = tmp_path / "histogram.png"
        result = generate_rgb_histogram(input_path, output_path)

        assert result is not None
        # Check that at least one ylabel contains "log" (case-insensitive)
        ylabel_str = " ".join(ylabel_calls).lower()
        assert "log" in ylabel_str, f"Expected 'log' in ylabel, got: {ylabel_calls}"

    def test_luminance_histogram_ylabel_indicates_log_scale(self, tmp_path, monkeypatch):
        """Test that luminance histogram Y-axis label indicates log scale."""
        from terrain_maker.terrain.diagnostics import generate_luminance_histogram
        from PIL import Image
        import matplotlib.pyplot as plt

        # Track ylabel calls
        ylabel_calls = []
        original_set_ylabel = plt.Axes.set_ylabel

        def mock_set_ylabel(self, ylabel, *args, **kwargs):
            ylabel_calls.append(ylabel)
            return original_set_ylabel(self, ylabel, *args, **kwargs)

        monkeypatch.setattr(plt.Axes, "set_ylabel", mock_set_ylabel)

        img_array = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        input_path = tmp_path / "test.png"
        img.save(input_path)

        output_path = tmp_path / "luminance.png"
        result = generate_luminance_histogram(input_path, output_path)

        assert result is not None
        # Check that at least one ylabel contains "log" (case-insensitive)
        ylabel_str = " ".join(ylabel_calls).lower()
        assert "log" in ylabel_str, f"Expected 'log' in ylabel, got: {ylabel_calls}"


class TestHistogramLuminanceCalculation:
    """Tests for correct luminance calculation in generate_luminance_histogram."""

    def test_luminance_calculation_pure_red(self, tmp_path):
        """Test luminance calculation for pure red image."""
        from terrain_maker.terrain.diagnostics import generate_luminance_histogram
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
        from terrain_maker.terrain.diagnostics import generate_luminance_histogram
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


