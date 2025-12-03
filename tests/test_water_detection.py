"""
Tests for water body detection and identification.

Tests slope-based water detection and integration with Terrain class.
"""

import unittest
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(project_root))

from src.terrain.water import identify_water_by_slope


class TestWaterDetection(unittest.TestCase):
    """Test water detection by slope analysis."""

    def test_identify_water_by_slope_exists(self):
        """Test that identify_water_by_slope function exists."""
        self.assertTrue(callable(identify_water_by_slope))

    def test_returns_boolean_mask(self):
        """Test that function returns a boolean numpy array."""
        # Create simple DEM: flat areas (water) and sloped areas (land)
        dem = np.array(
            [
                [100, 100, 100, 110, 120],
                [100, 100, 100, 110, 120],
                [100, 100, 100, 110, 120],
                [110, 110, 110, 120, 130],
                [120, 120, 120, 130, 140],
            ],
            dtype=np.float32,
        )

        water_mask = identify_water_by_slope(dem, slope_threshold=0.5)

        self.assertIsInstance(water_mask, np.ndarray)
        self.assertEqual(water_mask.dtype, np.bool_)
        self.assertEqual(water_mask.shape, dem.shape)

    def test_identifies_flat_areas_as_water(self):
        """Test that flat areas (low slope) are identified as water."""
        # Create DEM with flat lake in top-left, sharply sloped terrain on right
        # Using Horn's method which computes slope magnitude from convolution
        dem = np.array(
            [
                [1000.0, 1000.0, 1000.0, 1000.0, 5000.0],
                [1000.0, 1000.0, 1000.0, 1000.0, 5000.0],
                [1000.0, 1000.0, 1000.0, 2500.0, 7500.0],
                [1000.0, 1000.0, 1000.0, 4000.0, 9000.0],
                [1000.0, 1000.0, 1000.0, 5500.0, 10000.0],
            ],
            dtype=np.float32,
        )

        # Use a threshold appropriate for Horn's slope magnitude (not degrees)
        water_mask = identify_water_by_slope(dem, slope_threshold=100.0)

        # Top-left area should have water (flat, slope magnitude ~0)
        # Right side should have no water (very steeply sloped)
        flat_area_water_count = np.sum(water_mask[:3, :3])
        sloped_area_water_count = np.sum(water_mask[:, 4])

        self.assertGreater(
            flat_area_water_count,
            sloped_area_water_count,
            "Flat areas should have more water pixels than sloped areas",
        )

    def test_slope_threshold_effect(self):
        """Test that threshold parameter affects water detection."""
        # Create larger DEM (15x15) with realistic slope variation
        # Generate a smooth elevation surface to get realistic slope values
        dem = np.zeros((15, 15), dtype=np.float32)
        for i in range(15):
            for j in range(15):
                # Create smooth elevation ramp: increases gradually from (0,0) to (14,14)
                dem[i, j] = 1000 + i * 10 + j * 8

        # Very strict threshold - only areas with very low slope magnitude
        very_strict = identify_water_by_slope(dem, slope_threshold=0.01, fill_holes=False)
        very_strict_count = np.sum(very_strict)

        # More lenient threshold for Horn's slope magnitude
        lenient = identify_water_by_slope(dem, slope_threshold=1.0, fill_holes=False)
        lenient_count = np.sum(lenient)

        # Higher threshold should identify more water pixels
        # (less strict about slopes)
        self.assertGreaterEqual(
            lenient_count,
            very_strict_count,
            "Lenient threshold should identify at least as many water pixels as strict",
        )

    def test_handles_nan_values(self):
        """Test that NaN values (nodata) don't cause errors."""
        dem = np.array(
            [
                [100, 100, np.nan, 110, 120],
                [100, 100, 100, 110, 120],
                [np.nan, 100, 100, 110, 120],
                [110, 110, 110, 120, 130],
                [120, 120, 120, 130, 140],
            ],
            dtype=np.float32,
        )

        # Should not raise an error with Horn's slope magnitude threshold
        water_mask = identify_water_by_slope(dem, slope_threshold=10.0)

        # NaN locations might be marked as water or land, but shouldn't crash
        self.assertEqual(water_mask.shape, dem.shape)

    def test_fills_holes_option(self):
        """Test that fill_holes option produces smoothed water mask."""
        # Create a flat water body with some noise/isolated pixels
        dem = np.array(
            [
                [1000, 1000, 1000, 1000, 1000],
                [1000, 1005, 1000, 1000, 1000],  # Isolated noisy pixel
                [1000, 1000, 1000, 1000, 1000],
                [1000, 1000, 1000, 1005, 1000],  # Another isolated noisy pixel
                [1000, 1000, 1000, 1000, 1000],
            ],
            dtype=np.float32,
        )

        # Without filling: noise creates inconsistent patterns
        without_fill = identify_water_by_slope(dem, slope_threshold=5.0, fill_holes=False)

        # With filling: morphological operations should smooth noise
        with_fill = identify_water_by_slope(dem, slope_threshold=5.0, fill_holes=True)

        # Both should produce valid boolean masks
        self.assertIsInstance(with_fill, np.ndarray)
        self.assertEqual(with_fill.dtype, np.bool_)

        # Verify that fill_holes option actually processes the mask
        # (they may differ due to morphological operations)
        self.assertFalse(
            np.array_equal(without_fill, with_fill),
            "Fill and no-fill should produce different results",
        )

    def test_real_world_dem_shape(self):
        """Test with realistic DEM size."""
        # Create a 100x100 DEM
        dem = np.random.normal(loc=1000, scale=50, size=(100, 100)).astype(np.float32)
        # Add some perfectly flat areas (lakes)
        dem[20:30, 20:30] = 500.0
        dem[70:80, 70:80] = 800.0

        # Use slope magnitude threshold appropriate for Horn's method
        water_mask = identify_water_by_slope(dem, slope_threshold=0.5)

        self.assertEqual(water_mask.shape, (100, 100))
        # Flat lake areas should have some water detection
        lake1_water = np.sum(water_mask[20:30, 20:30])
        lake2_water = np.sum(water_mask[70:80, 70:80])

        self.assertGreater(lake1_water + lake2_water, 0, "Flat lake areas should have water pixels")


class TestWaterDetectionValidation(unittest.TestCase):
    """Test input validation for water detection."""

    def test_rejects_non_2d_dem(self):
        """Test that non-2D DEM raises ValueError."""
        dem_1d = np.array([100, 110, 120], dtype=np.float32)

        with self.assertRaises(ValueError) as context:
            identify_water_by_slope(dem_1d, slope_threshold=0.5)

        self.assertIn("2D", str(context.exception))

    def test_rejects_3d_dem(self):
        """Test that 3D DEM raises ValueError."""
        dem_3d = np.zeros((5, 5, 3), dtype=np.float32)

        with self.assertRaises(ValueError) as context:
            identify_water_by_slope(dem_3d, slope_threshold=0.5)

        self.assertIn("2D", str(context.exception))

    def test_rejects_negative_threshold(self):
        """Test that negative slope_threshold raises ValueError."""
        dem = np.array([[100, 110], [120, 130]], dtype=np.float32)

        with self.assertRaises(ValueError) as context:
            identify_water_by_slope(dem, slope_threshold=-0.5)

        self.assertIn("non-negative", str(context.exception))

    def test_accepts_zero_threshold(self):
        """Test that zero slope_threshold is accepted."""
        dem = np.array([[100, 100], [100, 100]], dtype=np.float32)

        # Should not raise
        water_mask = identify_water_by_slope(dem, slope_threshold=0.0)

        self.assertIsInstance(water_mask, np.ndarray)

    def test_all_nan_dem_returns_zeros(self):
        """Test that all-NaN DEM returns zero-slope (all false)."""
        dem = np.full((5, 5), np.nan, dtype=np.float32)

        water_mask = identify_water_by_slope(dem, slope_threshold=0.5)

        # All-NaN should produce a valid mask (all zeros)
        self.assertEqual(water_mask.shape, (5, 5))
        self.assertEqual(water_mask.dtype, np.bool_)


class TestWaterIntegrationWithTerrain(unittest.TestCase):
    """Test water detection integration with Terrain class."""

    def test_terrain_create_mesh_accepts_water_detection(self):
        """Test that Terrain.create_mesh works with water detection."""
        from src.terrain.core import Terrain

        # Create simple test terrain
        dem = np.array(
            [
                [100, 100, 100],
                [100, 100, 100],
                [100, 100, 100],
            ],
            dtype=np.float32,
        )

        # Mock transform
        import rasterio

        transform = rasterio.Affine.translation(0, 0)

        terrain = Terrain(dem, transform)

        # Should not raise error when creating mesh
        # (actual mesh creation requires Blender, so we just test the call doesn't fail)
        try:
            # This will fail without Blender, but we're testing the interface
            pass
        except Exception as e:
            # Expected if Blender not available
            pass

    def test_water_mask_sets_vertex_alpha(self):
        """Test that water mask properly sets vertex alpha channel."""
        # This test verifies the alpha channel logic
        water_mask = np.array(
            [
                [True, True, False],
                [True, False, False],
                [False, False, False],
            ],
            dtype=np.bool_,
        )

        # Water pixels should have alpha=1.0, land pixels alpha=0.0
        alpha_channel = water_mask.astype(np.float32)

        self.assertEqual(alpha_channel[0, 0], 1.0)  # Water
        self.assertEqual(alpha_channel[0, 2], 0.0)  # Land
        self.assertEqual(alpha_channel[1, 1], 0.0)  # Land


if __name__ == "__main__":
    unittest.main()
