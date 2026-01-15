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


class TestBorealMakoColormap:
    """Tests for the boreal_mako colormap.

    Simplified zone structure:
    1. Cool forest green (0.00-0.20)
    2. End of boreal zone (0.20-0.30)
    3. Transition to first blue (0.30-0.40)
    4-6. Purple outline (0.40-0.55): Start shift → Peak → Exit
    7. Blue Zone (0.55-0.70)
    8. Blue-teal transition (0.70-0.85)
    9. Pale blue-teal (0.85-0.95)
    10. Pale white (0.95-1.00)

    Built using CIELAB L* for perceptual uniformity.
    """

    def test_boreal_mako_cmap_imports(self):
        """boreal_mako_cmap should be importable."""
        from src.terrain.color_mapping import boreal_mako_cmap

        assert boreal_mako_cmap is not None

    def test_boreal_mako_registered_with_matplotlib(self):
        """boreal_mako should be registered with matplotlib."""
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap("boreal_mako")
        assert cmap is not None
        assert cmap.name == "boreal_mako"

    def test_boreal_mako_works_with_elevation_colormap(self):
        """boreal_mako should work with elevation_colormap function."""
        from src.terrain.color_mapping import elevation_colormap

        scores = np.array([[0.0, 0.5], [0.75, 1.0]])
        colors = elevation_colormap(scores, cmap_name="boreal_mako")

        assert colors.shape == (2, 2, 3)
        assert colors.dtype == np.uint8

    def test_boreal_mako_low_end_is_green(self):
        """Low end of boreal_mako should be green (boreal forest)."""
        from src.terrain.color_mapping import boreal_mako_cmap

        # Sample at position 0.1 (should be green)
        rgb = boreal_mako_cmap(0.1)[:3]

        # Green channel should dominate for boreal green
        # G > R and G > B
        assert rgb[1] > rgb[0], "Green should be > Red at low end"
        assert rgb[1] > rgb[2], "Green should be > Blue at low end"

    def test_boreal_mako_mid_is_blue(self):
        """Mid range of boreal_mako should be blue (mako blue)."""
        from src.terrain.color_mapping import boreal_mako_cmap

        # Sample at position 0.45 (should be blue)
        rgb = boreal_mako_cmap(0.45)[:3]

        # Blue channel should dominate
        assert rgb[2] > rgb[0], "Blue should be > Red at mid range"
        assert rgb[2] > rgb[1], "Blue should be > Green at mid range"

    def test_boreal_mako_high_end_is_pale(self):
        """High end of boreal_mako should be pale mint (high luminance)."""
        from src.terrain.color_mapping import boreal_mako_cmap

        # Sample at position 0.95 (should be pale)
        rgb = boreal_mako_cmap(0.95)[:3]

        # All channels should be high for pale color
        assert rgb[0] > 0.7, "Red should be high at pale end"
        assert rgb[1] > 0.7, "Green should be high at pale end"
        assert rgb[2] > 0.7, "Blue should be high at pale end"

    def test_boreal_mako_perceptually_uniform(self):
        """boreal_mako should be perceptually uniform (linear L*)."""
        from src.terrain.color_mapping import boreal_mako_cmap
        from skimage import color

        # Sample L* at various positions
        positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
        lightnesses = []
        for pos in positions:
            rgb = boreal_mako_cmap(pos)[:3]
            rgb_arr = np.array([[rgb]])
            lab = color.rgb2lab(rgb_arr)
            lightnesses.append(lab[0, 0, 0])

        # L* should generally increase (allow for purple outline dip zone: 0.45-0.52)
        for i in range(len(lightnesses) - 1):
            # Skip purple outline zone (0.45-0.52) - it has intentional L* dip
            if 0.45 <= positions[i] <= 0.52 or 0.45 <= positions[i+1] <= 0.52:
                continue
            assert lightnesses[i + 1] >= lightnesses[i] - 2, \
                f"L* should increase: {lightnesses[i]} -> {lightnesses[i+1]} at {positions[i]}->{positions[i+1]}"

    def test_boreal_mako_edge_hue_shift(self):
        """Second blue zone (0.55-0.70) should be distinct from pale zone."""
        from src.terrain.color_mapping import boreal_mako_cmap
        from skimage import color

        # Sample second blue zone (after purple)
        rgb_blue2 = boreal_mako_cmap(0.60)[:3]
        lab_blue2 = color.rgb2lab(np.array([[rgb_blue2]]))[0, 0]

        # Sample pale zone (toward white)
        rgb_pale = boreal_mako_cmap(0.75)[:3]
        lab_pale = color.rgb2lab(np.array([[rgb_pale]]))[0, 0]

        # Pale zone should be lighter (higher L*)
        assert lab_pale[0] > lab_blue2[0], \
            f"L* should increase toward white: {lab_blue2[0]} -> {lab_pale[0]}"

    def test_boreal_mako_purple_outline_prominent(self):
        """Purple outline (0.40-0.55) should be visibly distinct from surrounding blue zones."""
        from src.terrain.color_mapping import boreal_mako_cmap
        from skimage import color

        # Sample purple zone center (wider zone: 0.40-0.55)
        rgb_purple = boreal_mako_cmap(0.48)[:3]
        lab_purple = color.rgb2lab(np.array([[rgb_purple]]))[0, 0]

        # Sample first blue zone (before purple)
        rgb_before = boreal_mako_cmap(0.38)[:3]
        lab_before = color.rgb2lab(np.array([[rgb_before]]))[0, 0]

        # Sample second blue zone (after purple) - deeper into the zone
        rgb_after = boreal_mako_cmap(0.65)[:3]
        lab_after = color.rgb2lab(np.array([[rgb_after]]))[0, 0]

        # Purple should be SIGNIFICANTLY darker than surrounding colors (at least 5 units)
        # L* should dip at purple, making it darker
        darkness_dip_before = lab_before[0] - lab_purple[0]
        darkness_dip_after = lab_after[0] - lab_purple[0]

        assert darkness_dip_before >= 5, \
            f"Purple dip not prominent enough before: {darkness_dip_before} L* units (need >= 5)"
        assert darkness_dip_after >= 5, \
            f"Purple dip not prominent enough after: {darkness_dip_after} L* units (need >= 5)"

        # Purple should have significantly higher a* (magenta shift) - at least 10 units
        magenta_shift = lab_purple[1] - lab_before[1]
        assert magenta_shift >= 10, \
            f"Purple magenta shift not prominent: {magenta_shift} units (need >= 10)"


class TestColormapCompressionFormula:
    """Tests for the colormap compression formula used in detroit_combined_render."""

    def test_compression_formula_preserves_high_end(self):
        """High-end (white/pale) colors should be visible even with high transition points."""
        from src.terrain.color_mapping import elevation_colormap

        # When transition is at 0.70, we should still see pale (light) colors at score 1.0
        # The formula should be:
        # - [0, transition] maps to [0, 0.27]
        # - [transition, 1.0] maps to [0.27, 1.0]
        # So at score=1.0, we should get the pale end of the colormap (high L*)

        # Test with high transition (0.70)
        high_transition = 0.70
        score_high = 1.0

        # Simulate the transformation formula
        if score_high <= high_transition:
            cmap_pos = score_high * (0.27 / high_transition)
        else:
            cmap_pos = 0.27 + (score_high - high_transition) * (1.0 - 0.27) / (1.0 - high_transition)

        color_high = elevation_colormap(np.array([[cmap_pos]]), cmap_name="boreal_mako", min_elev=0.0, max_elev=1.0)
        from skimage import color
        rgb_high = color_high[0, 0, :3]
        lab_high = color.rgb2lab(np.array([[rgb_high]]))[0, 0]

        # Pale ice at position 1.0 should have L* around 95
        assert lab_high[0] > 80, \
            f"High-end should be pale (light): L*={lab_high[0]} (need > 80)"

        # Also test that green zone is preserved at low scores
        score_low = 0.1
        if score_low <= high_transition:
            cmap_pos_low = score_low * (0.27 / high_transition)
        else:
            cmap_pos_low = 0.27 + (score_low - high_transition) * (1.0 - 0.27) / (1.0 - high_transition)

        color_low = elevation_colormap(np.array([[cmap_pos_low]]), cmap_name="boreal_mako", min_elev=0.0, max_elev=1.0)
        rgb_low = color_low[0, 0, :3]
        lab_low = color.rgb2lab(np.array([[rgb_low]]))[0, 0]

        # Should be in green zone (G > R, G > B)
        assert rgb_low[1] > rgb_low[0] and rgb_low[1] > rgb_low[2], \
            f"Low-end should be greenish: RGB={rgb_low}"
