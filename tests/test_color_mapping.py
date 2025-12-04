"""
Tests for color_mapping module.

Tests color mapping functions extracted from core.py.
"""

import pytest
import numpy as np


class TestElevationColormap:
    """Tests for elevation_colormap function."""

    def test_elevation_colormap_imports(self):
        """Test that elevation_colormap can be imported."""
        from src.terrain.color_mapping import elevation_colormap

        assert callable(elevation_colormap)

    def test_elevation_colormap_basic(self):
        """Test basic elevation color mapping."""
        from src.terrain.color_mapping import elevation_colormap

        # Simple elevation data
        dem_data = np.array([[0.0, 50.0], [100.0, 150.0]])

        colors = elevation_colormap(dem_data, cmap_name="viridis")

        # Should return RGB uint8 array
        assert colors.shape == (2, 2, 3)
        assert colors.dtype == np.uint8

    def test_elevation_colormap_handles_nan(self):
        """Test that NaN values are handled."""
        from src.terrain.color_mapping import elevation_colormap

        # DEM with NaN
        dem_data = np.array([[0.0, np.nan], [50.0, 100.0]])

        colors = elevation_colormap(dem_data)

        # Should not have NaN in output
        assert not np.any(np.isnan(colors))

        # NaN pixels should be dark gray
        assert np.allclose(colors[0, 1], [51, 51, 51], atol=10)

    def test_elevation_colormap_uses_min_max(self):
        """Test that custom min/max are respected."""
        from src.terrain.color_mapping import elevation_colormap

        dem_data = np.array([[0.0, 100.0]])

        colors1 = elevation_colormap(dem_data, min_elev=0, max_elev=100)
        colors2 = elevation_colormap(dem_data, min_elev=0, max_elev=200)

        # With different normalization, colors should differ
        # (100 is full range in first, half range in second)
        assert not np.array_equal(colors1, colors2)

    def test_elevation_colormap_different_cmaps(self):
        """Test that different colormaps produce different results."""
        from src.terrain.color_mapping import elevation_colormap

        dem_data = np.array([[0.0, 50.0, 100.0]])

        colors_viridis = elevation_colormap(dem_data, cmap_name="viridis")
        colors_terrain = elevation_colormap(dem_data, cmap_name="terrain")

        # Different colormaps should produce different colors
        assert not np.array_equal(colors_viridis, colors_terrain)


class TestSlopeColormap:
    """Tests for slope_colormap function."""

    def test_slope_colormap_imports(self):
        """Test that slope_colormap can be imported."""
        from src.terrain.color_mapping import slope_colormap

        assert callable(slope_colormap)

    def test_slope_colormap_basic(self):
        """Test basic slope color mapping."""
        from src.terrain.color_mapping import slope_colormap

        # Simple slope data in degrees
        slopes = np.array([[0.0, 15.0], [30.0, 45.0]])

        colors = slope_colormap(slopes, cmap_name="terrain")

        # Should return RGBA array
        assert colors.shape == (2, 2, 4)
        # Should have alpha channel
        assert colors.dtype == np.float64 or colors.dtype == np.float32

    def test_slope_colormap_handles_nan(self):
        """Test that NaN slope values are handled."""
        from src.terrain.color_mapping import slope_colormap

        slopes = np.array([[0.0, np.nan], [15.0, 30.0]])

        colors = slope_colormap(slopes)

        # NaN areas should be transparent black
        assert colors[0, 1, 3] == 0  # Alpha channel is 0
        assert colors[0, 1, 0] == 0  # R is 0
        assert colors[0, 1, 1] == 0  # G is 0
        assert colors[0, 1, 2] == 0  # B is 0

    def test_slope_colormap_uses_min_max(self):
        """Test that custom slope min/max are respected."""
        from src.terrain.color_mapping import slope_colormap

        slopes = np.array([[0.0, 30.0]])

        colors1 = slope_colormap(slopes, min_slope=0, max_slope=45)
        colors2 = slope_colormap(slopes, min_slope=0, max_slope=90)

        # Different normalization should produce different colors
        assert not np.allclose(colors1, colors2)

    def test_slope_colormap_valid_areas_opaque(self):
        """Test that valid slope areas have full opacity."""
        from src.terrain.color_mapping import slope_colormap

        slopes = np.array([[0.0, 15.0, 30.0]])

        colors = slope_colormap(slopes)

        # All valid areas should have non-zero alpha
        assert all(colors[0, :, 3] > 0)
