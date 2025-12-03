"""
Tests for advanced terrain visualization functions.

TDD RED Phase: Testing migrated functions from helpers.py
"""

import pytest
import numpy as np
import numpy.ma as ma

from src.terrain.advanced_viz import horn_slope


class TestHornSlope:
    """Test Horn's slope calculation method."""

    def test_horn_slope_flat_terrain_returns_zero(self):
        """Horn slope of flat terrain should be zero."""
        flat_dem = np.ones((10, 10)) * 100.0  # Flat at 100m elevation

        slopes = horn_slope(flat_dem)

        # Flat terrain should have slope ≈ 0 everywhere
        assert slopes.shape == flat_dem.shape
        # Allow small numerical error at edges due to convolution
        assert np.nanmax(slopes) < 0.01

    def test_horn_slope_linear_gradient(self):
        """Horn slope should detect linear gradients."""
        # Create tilted plane (constant slope)
        x = np.arange(10)
        y = np.arange(10)
        xx, yy = np.meshgrid(x, y)
        dem = xx * 10.0  # Slope of 10 units per pixel in x direction

        slopes = horn_slope(dem)

        # Interior pixels should have consistent slope
        interior = slopes[2:-2, 2:-2]  # Avoid edge effects
        assert slopes.shape == dem.shape
        # Slope should be approximately constant
        assert np.nanstd(interior) < 0.5

    def test_horn_slope_single_peak(self):
        """Horn slope should show high values around a peak."""
        dem = np.zeros((10, 10))
        dem[5, 5] = 100.0  # Single tall peak in center

        slopes = horn_slope(dem)

        # Slope should be highest around the peak
        peak_region = slopes[4:7, 4:7]
        edges = slopes[[0, -1], :]
        assert np.nanmax(peak_region) > np.nanmax(edges)

    def test_horn_slope_handles_nan_values(self):
        """Horn slope should handle NaN values properly."""
        dem = np.random.rand(20, 20) * 100
        # Add some NaN values
        dem[5:8, 5:8] = np.nan

        slopes = horn_slope(dem)

        # NaN regions should remain NaN in output
        assert np.isnan(slopes[6, 6])
        # Non-NaN regions should have valid slopes
        assert not np.isnan(slopes[0, 0])
        assert slopes.shape == dem.shape

    def test_horn_slope_preserves_nan_locations(self):
        """Horn slope should preserve original NaN mask."""
        dem = np.ones((10, 10)) * 50.0
        original_nan_mask = np.zeros((10, 10), dtype=bool)
        original_nan_mask[3:5, 3:5] = True
        dem[original_nan_mask] = np.nan

        slopes = horn_slope(dem)

        # NaN mask should be identical
        result_nan_mask = np.isnan(slopes)
        np.testing.assert_array_equal(result_nan_mask, original_nan_mask)

    def test_horn_slope_realistic_terrain(self):
        """Horn slope should produce reasonable values for realistic terrain."""
        # Create realistic-looking terrain with hills
        np.random.seed(42)
        dem = np.random.rand(50, 50) * 20 + 100  # Elevation 100-120m with noise

        slopes = horn_slope(dem)

        # Slopes should be non-negative
        assert np.all(slopes[~np.isnan(slopes)] >= 0)
        # Slopes should be reasonable (not extreme)
        assert np.nanmax(slopes) < 50  # Max gradient shouldn't be crazy
        # Should have some variation
        assert np.nanstd(slopes) > 0


# Note: The following functions require Blender or GeoJSON files and are better
# tested through integration tests or manual verification:
# - load_drive_time_data() - requires GeoJSON file
# - create_drive_time_curves() - requires Blender context
# - create_values_legend() - requires Blender context

class TestAdvancedVizIntegration:
    """Integration tests for Blender-dependent functions."""

    def test_functions_importable(self):
        """All advanced viz functions should be importable."""
        from src.terrain.advanced_viz import (
            horn_slope,
            load_drive_time_data,
            create_drive_time_curves,
            create_values_legend,
        )

        # Just verify they're callable
        assert callable(horn_slope)
        assert callable(load_drive_time_data)
        assert callable(create_drive_time_curves)
        assert callable(create_values_legend)


# 🔴 TDD RED PHASE COMPLETE
# Tests written for migrated functions from helpers.py
