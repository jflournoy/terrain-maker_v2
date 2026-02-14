"""
Test for overlay threshold masking bug.

This test reproduces the issue where threshold=0.0 incorrectly
includes pixels with value=0 when using >= comparison.
"""

import numpy as np
import pytest
from rasterio.transform import Affine
from src.terrain import Terrain


def test_overlay_threshold_zero_excludes_zero_pixels():
    """
    🔴 RED: Threshold=0.0 should only include pixels > 0, not pixels = 0.

    Bug: overlay_mask = (data >= 0.0) includes zeros
    Fix: overlay_mask = (data > 0.0) excludes zeros

    This test creates a sparse overlay layer (like stream network) where:
    - Most pixels are 0 (non-stream)
    - Few pixels are > 0 (stream)

    With threshold=0.0, only the > 0 pixels should be colored by the overlay.
    """
    # Create a small test DEM (10×10 grid)
    dem_data = np.ones((10, 10), dtype=np.float32) * 100.0  # Flat elevation

    # Create simple affine transform (10m resolution, origin at 0,0)
    dem_transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0)

    # Create sparse overlay layer (like stream network)
    # Only 3 pixels are streams (value=1.0), rest are non-streams (value=0.0)
    overlay_data = np.zeros((10, 10), dtype=np.float32)
    overlay_data[3, 3] = 1.0  # Stream pixel 1
    overlay_data[5, 5] = 1.0  # Stream pixel 2
    overlay_data[7, 7] = 1.0  # Stream pixel 3

    # Create terrain object
    terrain = Terrain(dem_data, dem_transform)

    # Add overlay layer
    terrain.data_layers["stream"] = {
        "data": overlay_data,
        "transformed_data": overlay_data,
    }

    # Configure multi-color mapping with threshold=0.0
    # This should only color the 3 stream pixels, NOT the 97 zero pixels
    terrain.set_multi_color_mapping(
        base_colormap=lambda x: np.stack([
            np.full(x.shape, 0.5),  # R
            np.full(x.shape, 0.5),  # G
            np.full(x.shape, 0.5),  # B
        ], axis=-1),
        base_source_layers=["dem"],
        overlays=[{
            "colormap": lambda x: np.stack([
                np.full(x.shape, 1.0),  # R
                np.full(x.shape, 0.0),  # G
                np.full(x.shape, 0.0),  # B
            ], axis=-1),
            "source_layers": ["stream"],
            "threshold": 0.0,  # BUG: >= 0.0 includes zeros
            "priority": 10,
        }]
    )

    # Compute colors
    terrain.compute_colors()

    # Get resulting colors (grid-space)
    colors = terrain.colors

    # Check that only 3 pixels are red (overlay colored)
    # Red pixels have R=1.0, G=0.0, B=0.0
    red_pixels = (colors[:, :, 0] == 1.0) & (colors[:, :, 1] == 0.0) & (colors[:, :, 2] == 0.0)
    num_red_pixels = np.sum(red_pixels)

    # EXPECTED: 3 pixels (only the stream pixels)
    # ACTUAL (buggy): 100 pixels (all pixels because 0 >= 0.0)
    assert num_red_pixels == 3, (
        f"Expected 3 overlay pixels (stream only), got {num_red_pixels}. "
        f"Bug: threshold >= 0.0 includes zero pixels!"
    )

    # Verify the correct pixels are colored
    assert red_pixels[3, 3], "Stream pixel (3,3) should be red"
    assert red_pixels[5, 5], "Stream pixel (5,5) should be red"
    assert red_pixels[7, 7], "Stream pixel (7,7) should be red"

    # Verify zero pixels are NOT colored
    assert not red_pixels[0, 0], "Zero pixel (0,0) should NOT be red"
    assert not red_pixels[9, 9], "Zero pixel (9,9) should NOT be red"


def test_overlay_threshold_positive_excludes_lower_values():
    """
    Test that threshold > 0 correctly excludes values below threshold.

    This ensures the fix doesn't break positive thresholds.
    """
    # Create test DEM
    dem_data = np.ones((10, 10), dtype=np.float32) * 100.0
    dem_transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0)

    # Create overlay with graduated values
    overlay_data = np.zeros((10, 10), dtype=np.float32)
    overlay_data[2, 2] = 0.3  # Below threshold
    overlay_data[5, 5] = 0.5  # At threshold (should be included with >=)
    overlay_data[7, 7] = 0.8  # Above threshold

    terrain = Terrain(dem_data, dem_transform)
    terrain.data_layers["overlay"] = {
        "data": overlay_data,
        "transformed_data": overlay_data,
    }

    terrain.set_multi_color_mapping(
        base_colormap=lambda x: np.stack([
            np.full(x.shape, 0.5),
            np.full(x.shape, 0.5),
            np.full(x.shape, 0.5),
        ], axis=-1),
        base_source_layers=["dem"],
        overlays=[{
            "colormap": lambda x: np.stack([
                np.full(x.shape, 1.0),
                np.full(x.shape, 0.0),
                np.full(x.shape, 0.0),
            ], axis=-1),
            "source_layers": ["overlay"],
            "threshold": 0.5,  # Should include >= 0.5
            "priority": 10,
        }]
    )

    terrain.compute_colors()
    colors = terrain.colors

    red_pixels = (colors[:, :, 0] == 1.0) & (colors[:, :, 1] == 0.0) & (colors[:, :, 2] == 0.0)

    # Should include pixels at/above threshold (0.5 and 0.8)
    assert red_pixels[5, 5], "Pixel at threshold (0.5) should be included"
    assert red_pixels[7, 7], "Pixel above threshold (0.8) should be included"

    # Should exclude pixels below threshold
    assert not red_pixels[2, 2], "Pixel below threshold (0.3) should be excluded"
    assert not red_pixels[0, 0], "Zero pixel should be excluded"

    # Total: 2 pixels
    assert np.sum(red_pixels) == 2, f"Expected 2 overlay pixels, got {np.sum(red_pixels)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
