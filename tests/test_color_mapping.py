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

    Winter forest palette with darkened purple ribbon:
    - Dark boreal green (cool, blue-tinted forest)
    - Transition to blue
    - Purple ribbon around 0.6 (brief L* dip)
    - Back to blue, then cyan, then white

    Built like viridis/mako using LinearSegmentedColormap.from_list()
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
        """boreal_mako should have generally increasing L* with purple ribbon dip."""
        from src.terrain.color_mapping import boreal_mako_cmap
        from skimage import color

        # Sample L* at various positions
        positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        lightnesses = []
        for pos in positions:
            rgb = boreal_mako_cmap(pos)[:3]
            rgb_arr = np.array([[rgb]])
            lab = color.rgb2lab(rgb_arr)
            lightnesses.append(lab[0, 0, 0])

        # L* should generally increase, but purple ribbon (~0.6) creates a brief dip
        for i in range(len(lightnesses) - 1):
            # Allow larger tolerance around purple ribbon (positions 0.5->0.6 and 0.6->0.7)
            if positions[i] in [0.5, 0.6]:
                tolerance = 15  # Purple ribbon can dip L* by up to 15
            else:
                tolerance = 1  # Normal monotonic increase tolerance

            assert lightnesses[i + 1] >= lightnesses[i] - tolerance, \
                f"L* change at {positions[i]}->{positions[i+1]}: {lightnesses[i]:.1f} -> {lightnesses[i+1]:.1f}"

    def test_boreal_mako_purple_ribbon(self):
        """Purple ribbon should be present at position 0.6."""
        from src.terrain.color_mapping import boreal_mako_cmap

        # Sample at position 0.6 (purple ribbon)
        rgb_purple = boreal_mako_cmap(0.6)[:3]

        # Purple should have more red than surrounding blues
        # R > B and R > G for purple
        assert rgb_purple[0] > rgb_purple[2], "Purple should have R > B"
        assert rgb_purple[0] > rgb_purple[1], "Purple should have R > G"

        # Compare to adjacent blue positions
        rgb_before = boreal_mako_cmap(0.55)[:3]
        rgb_after = boreal_mako_cmap(0.65)[:3]

        # Purple should be distinctly different from surrounding blues
        # Purple should have significantly more red than adjacent blues
        assert rgb_purple[0] > rgb_before[0] + 0.05, "Purple should be redder than before"
        assert rgb_purple[0] > rgb_after[0] + 0.05, "Purple should be redder than after"

    def test_boreal_mako_cyan_to_white_transition(self):
        """Cyan to white transition should show increasing L*."""
        from src.terrain.color_mapping import boreal_mako_cmap
        from skimage import color

        # Sample cyan zone
        rgb_cyan = boreal_mako_cmap(0.80)[:3]
        lab_cyan = color.rgb2lab(np.array([[rgb_cyan]]))[0, 0]

        # Sample pale/white zone
        rgb_white = boreal_mako_cmap(0.95)[:3]
        lab_white = color.rgb2lab(np.array([[rgb_white]]))[0, 0]

        # White should be significantly lighter than cyan
        assert lab_white[0] > lab_cyan[0] + 10, \
            f"L* should increase significantly from cyan to white: {lab_cyan[0]} -> {lab_white[0]}"


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


class TestBorealMakoPrintColormap:
    """Tests for the print-safe boreal_mako_print colormap.

    Validates CMYK gamut compression: the print variant should preserve
    the overall character of boreal_mako (green→blue→pale) while keeping
    all colors within the approximate CMYK gamut boundary.
    """

    def test_print_cmap_imports(self):
        """boreal_mako_print_cmap should be importable."""
        from src.terrain.color_mapping import boreal_mako_print_cmap

        assert boreal_mako_print_cmap is not None

    def test_print_cmap_registered_with_matplotlib(self):
        """boreal_mako_print should be registered with matplotlib."""
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap("boreal_mako_print")
        assert cmap is not None
        assert cmap.name == "boreal_mako_print"

    def test_print_cmap_works_with_elevation_colormap(self):
        """boreal_mako_print should work with elevation_colormap function."""
        from src.terrain.color_mapping import elevation_colormap

        scores = np.array([[0.0, 0.5], [0.75, 1.0]])
        colors = elevation_colormap(scores, cmap_name="boreal_mako_print")

        assert colors.shape == (2, 2, 3)
        assert colors.dtype == np.uint8

    def test_print_cmap_preserves_green_low_end(self):
        """Low end should still read as green (boreal forest)."""
        from src.terrain.color_mapping import boreal_mako_print_cmap

        rgb = boreal_mako_print_cmap(0.1)[:3]
        assert rgb[1] > rgb[0], "Green should be > Red at low end"
        assert rgb[1] > rgb[2], "Green should be > Blue at low end"

    def test_print_cmap_preserves_pale_high_end(self):
        """High end should still be pale (high luminance)."""
        from src.terrain.color_mapping import boreal_mako_print_cmap

        rgb = boreal_mako_print_cmap(0.95)[:3]
        assert rgb[0] > 0.6, f"Red should be high at pale end, got {rgb[0]:.3f}"
        assert rgb[1] > 0.6, f"Green should be high at pale end, got {rgb[1]:.3f}"
        assert rgb[2] > 0.6, f"Blue should be high at pale end, got {rgb[2]:.3f}"

    def test_print_cmap_monotonic_luminance(self):
        """Print cmap should have generally increasing L* (like the source)."""
        from src.terrain.color_mapping import boreal_mako_print_cmap
        from skimage import color

        positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
        lightnesses = []
        for pos in positions:
            rgb = boreal_mako_print_cmap(pos)[:3]
            lab = color.rgb2lab(np.array([[rgb]]))[0, 0]
            lightnesses.append(lab[0])

        # L* should generally increase (allow small tolerance for purple ribbon)
        for i in range(len(lightnesses) - 1):
            assert lightnesses[i + 1] >= lightnesses[i] - 15, \
                f"L* should increase: {positions[i]}->{positions[i+1]}: " \
                f"{lightnesses[i]:.1f} -> {lightnesses[i+1]:.1f}"

    def test_print_cmap_lower_chroma_than_source(self):
        """Print cmap should have equal or lower chroma than boreal_mako."""
        from src.terrain.color_mapping import boreal_mako_cmap, boreal_mako_print_cmap
        from skimage import color

        positions = np.linspace(0, 1, 20)
        for pos in positions:
            rgb_src = boreal_mako_cmap(float(pos))[:3]
            rgb_prt = boreal_mako_print_cmap(float(pos))[:3]

            lab_src = color.rgb2lab(np.array([[rgb_src]]))[0, 0]
            lab_prt = color.rgb2lab(np.array([[rgb_prt]]))[0, 0]

            chroma_src = np.sqrt(lab_src[1] ** 2 + lab_src[2] ** 2)
            chroma_prt = np.sqrt(lab_prt[1] ** 2 + lab_prt[2] ** 2)

            assert chroma_prt <= chroma_src + 0.5, \
                f"Print chroma should be <= source at pos {pos:.2f}: " \
                f"{chroma_prt:.1f} > {chroma_src:.1f}"

    def test_print_colors_within_cmyk_gamut(self):
        """All print cmap colors should be within the CMYK gamut boundary."""
        from src.terrain.color_mapping import (
            boreal_mako_print_cmap, _srgb_to_lab, _cmyk_max_chroma,
        )

        positions = np.linspace(0, 1, 64)
        for pos in positions:
            rgba = boreal_mako_print_cmap(float(pos))
            L, a, b_lab = _srgb_to_lab(rgba[0], rgba[1], rgba[2])
            chroma = np.sqrt(a ** 2 + b_lab ** 2)
            hue_deg = np.degrees(np.arctan2(b_lab, a)) % 360
            max_chroma = _cmyk_max_chroma(L, hue_deg) * 0.85

            assert chroma <= max_chroma + 1.0, \
                f"Color at pos {pos:.2f} exceeds CMYK gamut: " \
                f"chroma={chroma:.1f} > max={max_chroma:.1f} " \
                f"(L*={L:.1f}, hue={hue_deg:.0f})"


class TestColorSpaceConversions:
    """Tests for the sRGB <-> CIELAB conversion helpers."""

    def test_lab_roundtrip_white(self):
        """White should survive Lab round-trip."""
        from src.terrain.color_mapping import _srgb_to_lab, _lab_to_srgb

        L, a, b = _srgb_to_lab(1.0, 1.0, 1.0)
        r, g, b_out = _lab_to_srgb(L, a, b)

        assert abs(r - 1.0) < 0.01, f"White round-trip R: {r}"
        assert abs(g - 1.0) < 0.01, f"White round-trip G: {g}"
        assert abs(b_out - 1.0) < 0.01, f"White round-trip B: {b_out}"

    def test_lab_roundtrip_black(self):
        """Black should survive Lab round-trip."""
        from src.terrain.color_mapping import _srgb_to_lab, _lab_to_srgb

        L, a, b = _srgb_to_lab(0.0, 0.0, 0.0)
        r, g, b_out = _lab_to_srgb(L, a, b)

        assert abs(r) < 0.01, f"Black round-trip R: {r}"
        assert abs(g) < 0.01, f"Black round-trip G: {g}"
        assert abs(b_out) < 0.01, f"Black round-trip B: {b_out}"

    def test_lab_roundtrip_midtones(self):
        """Several midtone colors should survive Lab round-trip."""
        from src.terrain.color_mapping import _srgb_to_lab, _lab_to_srgb

        test_colors = [
            (0.5, 0.5, 0.5),   # mid gray
            (0.2, 0.5, 0.7),   # blue (similar to boreal_mako mid)
            (0.1, 0.3, 0.1),   # dark green
            (0.8, 0.9, 0.95),  # pale cyan
        ]

        for r_in, g_in, b_in in test_colors:
            L, a, b_lab = _srgb_to_lab(r_in, g_in, b_in)
            r_out, g_out, b_out = _lab_to_srgb(L, a, b_lab)

            assert abs(r_out - r_in) < 0.02, \
                f"Round-trip R for ({r_in},{g_in},{b_in}): {r_out}"
            assert abs(g_out - g_in) < 0.02, \
                f"Round-trip G for ({r_in},{g_in},{b_in}): {g_out}"
            assert abs(b_out - b_in) < 0.02, \
                f"Round-trip B for ({r_in},{g_in},{b_in}): {b_out}"

    def test_white_has_L100(self):
        """White sRGB should have L* close to 100."""
        from src.terrain.color_mapping import _srgb_to_lab

        L, a, b = _srgb_to_lab(1.0, 1.0, 1.0)
        assert abs(L - 100) < 0.5, f"White L* should be ~100, got {L}"
        assert abs(a) < 0.5, f"White a* should be ~0, got {a}"
        assert abs(b) < 0.5, f"White b* should be ~0, got {b}"

    def test_black_has_L0(self):
        """Black sRGB should have L* close to 0."""
        from src.terrain.color_mapping import _srgb_to_lab

        L, a, b = _srgb_to_lab(0.0, 0.0, 0.0)
        assert abs(L) < 0.5, f"Black L* should be ~0, got {L}"
