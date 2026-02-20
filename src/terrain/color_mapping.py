"""
Color mapping functions for terrain visualization.

This module contains functions for mapping elevation and slope data to colors
using matplotlib colormaps.
"""

import logging
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# Custom Colormaps
# =============================================================================

# Michigan Natural Landscape Colormap
# Perceptually uniform progression through Michigan's natural features:
# Great Lakes (deep blue) → Northern forests (evergreen) → Upland meadows → Sand dunes (tan)
# Desaturated ~35% for subtle, sun-bleached appearance that works well as base layer in dual colormaps
_MICHIGAN_COLORS = {
    'red': [
        (0.00, 0.24, 0.24),  # Great Lakes: muted blue-gray (desaturated)
        (0.33, 0.25, 0.25),  # Forest: muted evergreen (desaturated)
        (0.67, 0.52, 0.52),  # Upland: pale meadow (desaturated)
        (1.00, 0.80, 0.80),  # Sand dunes: pale tan (desaturated)
    ],
    'green': [
        (0.00, 0.37, 0.37),  # Great Lakes: muted blue-gray (desaturated)
        (0.33, 0.37, 0.37),  # Forest: muted green (desaturated)
        (0.67, 0.53, 0.53),  # Upland: pale meadow (desaturated)
        (1.00, 0.76, 0.76),  # Sand dunes: pale tan (desaturated)
    ],
    'blue': [
        (0.00, 0.48, 0.48),  # Great Lakes: muted blue (desaturated)
        (0.33, 0.22, 0.22),  # Forest: muted evergreen (desaturated)
        (0.67, 0.40, 0.40),  # Upland: pale meadow (desaturated)
        (1.00, 0.67, 0.67),  # Sand dunes: pale tan (desaturated)
    ],
}

# Register the Michigan colormap with matplotlib
michigan_cmap = LinearSegmentedColormap('michigan', _MICHIGAN_COLORS, N=256)
try:
    # Register with new API (matplotlib >= 3.7)
    import matplotlib
    matplotlib.colormaps.register(michigan_cmap, force=True)
except (AttributeError, TypeError):
    # Fall back to old API
    plt.register_cmap(cmap=michigan_cmap)


# =============================================================================
# Modern Terrain Colormap (Land-focused)
# =============================================================================
# Contemporary cartography-inspired palette for LAND elevations only.
# Water (ocean/lakes) is handled separately via water_mask, so this colormap
# focuses on terrestrial features:
# Desert basin/beach → Coastal scrub → Chaparral → Foothills → Mountains → Peaks
# Muted, desaturated colors with smooth luminance progression.

_MODERN_TERRAIN_COLORS = [
    # (position, R, G, B) - all values 0-1
    (0.00, 0.76, 0.70, 0.55),   # Desert basin/beach: warm sand (below sea level areas)
    (0.10, 0.72, 0.68, 0.52),   # Low desert: pale tan
    (0.20, 0.62, 0.60, 0.45),   # Dry scrubland: tan-olive
    (0.30, 0.50, 0.52, 0.38),   # Coastal chaparral: muted sage
    (0.45, 0.45, 0.48, 0.35),   # Lowland: olive-sage
    (0.55, 0.52, 0.48, 0.38),   # Foothills: tan-brown
    (0.70, 0.55, 0.50, 0.45),   # Highland: warm brown
    (0.82, 0.60, 0.56, 0.52),   # Mountain: muted brown-gray
    (0.92, 0.72, 0.70, 0.68),   # Alpine: cool gray
    (1.00, 0.92, 0.91, 0.90),   # Peaks: pale gray (not pure white)
]

modern_terrain_cmap = LinearSegmentedColormap.from_list(
    'modern_terrain', _MODERN_TERRAIN_COLORS, N=256
)
try:
    import matplotlib
    matplotlib.colormaps.register(modern_terrain_cmap, force=True)
except (AttributeError, TypeError):
    plt.register_cmap(cmap=modern_terrain_cmap)


# =============================================================================
# Boreal-Mako Colormap (Perceptually Uniform)
# =============================================================================
# Winter forest palette: Great Lakes mixed-forest green → blue → cyan → white
# Built like viridis/mako with monotonically increasing luminance
# Uses the same LinearSegmentedColormap.from_list() method as the viridis family

def _build_boreal_mako_cmap(purple_position=0.6, purple_width=1.0):
    """Build the boreal_mako colormap.

    Winter forest palette with optional darkened purple ribbon:
    - Mixed-forest green (warm, Great Lakes hardwood-conifer tone)
    - Transition to blue
    - Purple ribbon at configurable position (or omitted if None)
    - Back to blue, then cyan, then white

    Args:
        purple_position: Position of purple ribbon (0.0-1.0), default 0.6.
            Set to None to omit the purple band entirely.
        purple_width: Width of the purple band as a proportion of the maximum
            width (0.0-1.0, default 1.0). Maximum width is 5% of the colormap
            range. Values near 0 produce a hairline band, 1.0 produces the
            widest band.

    Uses the same method as mako: sample key colors and interpolate.
    """
    # Key color samples (position, RGB) with explicit positions for purple ribbon
    # Purple ribbon creates a narrow band around the specified position

    # Fixed color stops (independent of purple position)
    base_colors = [
        (0.00, (0.10, 0.22, 0.08)),  # Dark mixed-forest green (warm, Great Lakes hardwoods)
        (0.15, (0.12, 0.30, 0.12)),  # Mixed deciduous-conifer green
        (0.30, (0.10, 0.30, 0.30)),  # Green-blue transition
        (0.50, (0.12, 0.35, 0.50)),  # Blue
    ]

    # High-end colors (cyan to white)
    high_colors = [
        (0.70, (0.20, 0.50, 0.70)),  # Bright blue
        (0.80, (0.40, 0.70, 0.85)),  # Cyan
        (1.00, (0.85, 0.95, 0.98)),  # Pale white
    ]

    if purple_position is None:
        # No purple band - smooth green → blue → cyan → white
        all_colors = base_colors + high_colors
    else:
        # Purple ribbon (band with internal gradient)
        # Max half-width = 0.025 → total band = 5% of colormap range
        max_half_width = 0.025
        band_half_width = max_half_width * purple_width
        half_width = band_half_width / 2  # midpoint for internal gradient

        pre_purple = purple_position - band_half_width
        mid_pre_purple = purple_position - half_width
        mid_post_purple = purple_position + half_width
        post_purple = purple_position + band_half_width

        # Interpolate blue colors around purple position
        # Get blue color at purple position
        blue_at_purple = (0.15 + (purple_position - 0.5) * 0.2,
                          0.40 + (purple_position - 0.5) * 0.4,
                          0.60 + (purple_position - 0.5) * 0.2)

        # Calculate purple colors with perceived luminance matching position in gradient
        # Base purple ratios (maintaining purple hue: R > G, R > B)
        # Reference purple at position 0.6: (0.35, 0.15, 0.28)
        base_purple_ratios = (0.35, 0.15, 0.28)
        base_brightness = sum(base_purple_ratios) / 3  # 0.26

        # Target brightness based on blue_at_purple (what luminance should be at this position)
        target_brightness = sum(blue_at_purple) / 3

        # Scale purple to match target brightness while maintaining hue
        brightness_scale = target_brightness / base_brightness
        purple_scaled = tuple(c * brightness_scale for c in base_purple_ratios)

        # Create gradient within purple band: lighter edges → darker center
        # Edge purple: 15% darker than blue (0.85)
        purple_edge = tuple(c * 0.85 for c in purple_scaled)
        # Center purple: 30% darker than blue (0.70) - extra dark for emphasis
        purple_center = tuple(c * 0.70 for c in purple_scaled)

        purple_colors = [
            (pre_purple, blue_at_purple),          # Outer edge: blue
            (mid_pre_purple, purple_edge),         # Inner edge: lighter purple
            (purple_position, purple_center),      # Center: darker purple
            (mid_post_purple, purple_edge),        # Inner edge: lighter purple
            (post_purple, blue_at_purple),         # Outer edge: blue
        ]

        # Combine all colors, filtering out base/high colors that overlap with purple band
        # to avoid interpolation artifacts
        # Exclude colors within the purple band range (with small buffer)
        purple_band_min = pre_purple - 0.05
        purple_band_max = post_purple + 0.05

        # Keep base colors only if they're well before the purple band
        filtered_base = [c for c in base_colors if c[0] < purple_band_min]
        # Keep high colors only if they're well after the purple band
        filtered_high = [c for c in high_colors if c[0] > purple_band_max]

        all_colors = filtered_base + purple_colors + filtered_high

    all_colors = sorted(all_colors, key=lambda x: x[0])

    # Create colormap using from_list (same method as mako/viridis)
    cmap = LinearSegmentedColormap.from_list('boreal_mako', all_colors, N=256)
    return cmap


# Build and register the boreal_mako colormap
boreal_mako_cmap = _build_boreal_mako_cmap()
try:
    import matplotlib
    matplotlib.colormaps.register(boreal_mako_cmap, force=True)
except (AttributeError, TypeError):
    plt.register_cmap(cmap=boreal_mako_cmap)


# =============================================================================
# Print-Safe Gamut Mapping (CMYK)
# =============================================================================
# Utilities for compressing RGB colors into the reproducible gamut of CMYK
# printing. Uses CIELAB perceptual color space to reduce chroma (saturation)
# while preserving lightness and hue. Targets commercial photo printing
# (e.g., WHCC) which uses a subset of the sRGB gamut.


def _srgb_gamma_to_linear(c):
    """Convert a single sRGB gamma-encoded channel (0-1) to linear light."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb_gamma(c):
    """Convert a single linear light value to sRGB gamma-encoded (0-1)."""
    if c <= 0.0031308:
        return c * 12.92
    return 1.055 * c ** (1.0 / 2.4) - 0.055


def _srgb_to_lab(r, g, b):
    """Convert sRGB (0-1) to CIELAB (D65 illuminant).

    Standard conversion: sRGB → linear RGB → XYZ (D65) → CIELAB.
    """
    # Linearize sRGB
    rl = _srgb_gamma_to_linear(r)
    gl = _srgb_gamma_to_linear(g)
    bl = _srgb_gamma_to_linear(b)

    # Linear sRGB to CIE XYZ (D65 reference white)
    x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl

    # D65 reference white
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    delta = 6.0 / 29.0

    def f(t):
        if t > delta ** 3:
            return t ** (1.0 / 3.0)
        return t / (3 * delta ** 2) + 4.0 / 29.0

    L = 116 * f(y / yn) - 16
    a = 500 * (f(x / xn) - f(y / yn))
    b_lab = 200 * (f(y / yn) - f(z / zn))

    return L, a, b_lab


def _lab_to_srgb(L, a, b_lab):
    """Convert CIELAB (D65) to sRGB (0-1), clamped to valid range.

    Standard conversion: CIELAB → XYZ (D65) → linear RGB → sRGB.
    Out-of-gamut values are clamped after the linear→sRGB step.
    """
    # D65 reference white
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    delta = 6.0 / 29.0

    def f_inv(t):
        if t > delta:
            return t ** 3
        return 3 * delta ** 2 * (t - 4.0 / 29.0)

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b_lab / 200

    x = xn * f_inv(fx)
    y = yn * f_inv(fy)
    z = zn * f_inv(fz)

    # XYZ to linear sRGB (IEC 61966-2-1 matrix)
    rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    gl = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    # Clamp to [0, 1] then apply sRGB gamma
    r = _linear_to_srgb_gamma(max(0.0, min(1.0, rl)))
    g = _linear_to_srgb_gamma(max(0.0, min(1.0, gl)))
    b = _linear_to_srgb_gamma(max(0.0, min(1.0, bl)))

    return r, g, b


# Approximate CMYK gamut boundary in CIELAB space.
# Maximum chroma at L*=50 by hue angle, based on US Web Coated (SWOP) v2
# profile measurements. Cyan-blue region (180-240 deg) is most restricted.
_CMYK_HUE_POINTS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
_CMYK_CHROMA_AT_L50 = [60, 65, 75, 90, 70, 55, 40, 35, 45, 50, 55, 60, 60]


def _cmyk_max_chroma(L, hue_deg):
    """Approximate maximum CIELAB chroma reproducible in CMYK at given L* and hue.

    Based on conservative measurements from US Web Coated (SWOP) v2 profile.
    Photo labs like WHCC have wider gamuts but this provides a safe baseline.

    Args:
        L: CIELAB lightness (0-100)
        hue_deg: Hue angle in degrees (0-360)

    Returns:
        Approximate maximum chroma value.
    """
    hue_deg = hue_deg % 360

    # Interpolate chroma limit at L*=50 from lookup table
    max_c = _CMYK_CHROMA_AT_L50[-1]
    for i in range(len(_CMYK_HUE_POINTS) - 1):
        if _CMYK_HUE_POINTS[i] <= hue_deg <= _CMYK_HUE_POINTS[i + 1]:
            t = (hue_deg - _CMYK_HUE_POINTS[i]) / (
                _CMYK_HUE_POINTS[i + 1] - _CMYK_HUE_POINTS[i]
            )
            max_c = _CMYK_CHROMA_AT_L50[i] + t * (
                _CMYK_CHROMA_AT_L50[i + 1] - _CMYK_CHROMA_AT_L50[i]
            )
            break

    # Gamut narrows toward L*=0 and L*=100 (parabolic, peak near L*=55)
    if L <= 0 or L >= 100:
        return 0.0
    L_factor = max(1.0 - ((L - 55.0) / 55.0) ** 2, 0.05)

    return max_c * L_factor


def _compress_color_to_print_gamut(r, g, b, safety_margin=0.85):
    """Compress an sRGB color to fit within approximate CMYK print gamut.

    Operates in CIELAB space: reduces chroma (colorfulness) while preserving
    lightness and hue. This is equivalent to a perceptual rendering intent.

    Args:
        r, g, b: sRGB values (0-1).
        safety_margin: Multiplier on gamut boundary (default 0.85 = 15% inset).
            Lower values are more conservative. 1.0 = use the full boundary.

    Returns:
        Tuple of (r, g, b) in sRGB, gamut-compressed.
    """
    L, a, b_lab = _srgb_to_lab(r, g, b)

    chroma = math.sqrt(a ** 2 + b_lab ** 2)
    if chroma < 1e-6:
        return r, g, b  # achromatic, no gamut concern

    hue_deg = math.degrees(math.atan2(b_lab, a)) % 360
    max_chroma = _cmyk_max_chroma(L, hue_deg) * safety_margin

    if chroma > max_chroma:
        scale = max_chroma / chroma
        a *= scale
        b_lab *= scale

    return _lab_to_srgb(L, a, b_lab)


def _build_boreal_mako_print_cmap(source_cmap=None, n_samples=64,
                                  safety_margin=0.85):
    """Build a print-safe variant of boreal_mako.

    Samples the source colormap and compresses each color into the approximate
    CMYK gamut, targeting commercial photo printing (WHCC and similar labs).
    Preserves the overall character and luminance ramp while ensuring colors
    are safely reproducible in print.

    Args:
        source_cmap: Source colormap (default: boreal_mako_cmap).
        n_samples: Number of sample points for resampling (default: 64).
        safety_margin: How conservative the compression is (default: 0.85).
            0.85 means colors stay 15% inside the gamut boundary.

    Returns:
        A print-safe LinearSegmentedColormap named 'boreal_mako_print'.
    """
    if source_cmap is None:
        source_cmap = boreal_mako_cmap

    positions = np.linspace(0, 1, n_samples)
    print_colors = []

    for pos in positions:
        rgba = source_cmap(float(pos))
        r_safe, g_safe, b_safe = _compress_color_to_print_gamut(
            rgba[0], rgba[1], rgba[2], safety_margin=safety_margin
        )
        print_colors.append((float(pos), (r_safe, g_safe, b_safe)))

    return LinearSegmentedColormap.from_list(
        'boreal_mako_print', print_colors, N=256
    )


# Build and register the boreal_mako_print colormap
boreal_mako_print_cmap = _build_boreal_mako_print_cmap()
try:
    import matplotlib
    matplotlib.colormaps.register(boreal_mako_print_cmap, force=True)
except (AttributeError, TypeError):
    plt.register_cmap(cmap=boreal_mako_print_cmap)


def elevation_colormap(dem_data, cmap_name="viridis", min_elev=None, max_elev=None, gamma=1.0):
    """
    Create a colormap based on elevation values.

    Maps elevation data to colors using a matplotlib colormap.
    Low elevations map to the start of the colormap, high elevations to the end.

    Args:
        dem_data: 2D numpy array of elevation values
        cmap_name: Matplotlib colormap name (default: 'viridis')
        min_elev: Minimum elevation for normalization (default: use data min)
        max_elev: Maximum elevation for normalization (default: use data max)
        gamma: Gamma correction exponent (default: 1.0 = no correction).
               Values < 1.0 brighten midtones, values > 1.0 darken midtones.
               Common values: 0.5 = brighten, 2.2 = darken (sRGB gamma)

    Returns:
        Array of RGB colors with shape (height, width, 3) as uint8
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating elevation colormap using {cmap_name}" +
                (f" (gamma={gamma})" if gamma != 1.0 else ""))

    # Handle NaN values
    valid_mask = ~np.isnan(dem_data)

    # Auto-calculate min/max if not provided
    if min_elev is None:
        min_elev = np.nanmin(dem_data)
    if max_elev is None:
        max_elev = np.nanmax(dem_data)

    logger.info(f"Elevation range: {min_elev:.1f} to {max_elev:.1f} meters")

    # Normalize elevation to 0-1 range
    if min_elev == max_elev:
        normalized = np.zeros_like(dem_data, dtype=np.float32)
    else:
        normalized = np.zeros_like(dem_data, dtype=np.float32)
        normalized[valid_mask] = (dem_data[valid_mask] - min_elev) / (max_elev - min_elev + 1e-8)

    # Apply gamma correction if specified
    if gamma != 1.0:
        normalized[valid_mask] = np.power(normalized[valid_mask], gamma)

    # Get colormap from matplotlib
    try:
        # Try new API (matplotlib >= 3.7)
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap(cmap_name)
    except (AttributeError, TypeError):
        # Fall back to old API for older matplotlib versions
        cmap = plt.cm.get_cmap(cmap_name)

    # Apply colormap (returns RGBA with shape (H, W, 4))
    rgba_array = cmap(normalized)

    # Extract RGB channels only (drop alpha)
    rgb = rgba_array[:, :, :3]

    # Set invalid (NaN) areas to dark gray
    rgb[~valid_mask] = [0.2, 0.2, 0.2]

    # Convert to uint8
    colors_uint8 = (rgb * 255).astype(np.uint8)

    logger.info(f"Created elevation colormap with shape {colors_uint8.shape}")
    return colors_uint8


def slope_colormap(slopes, cmap_name="terrain", min_slope=0, max_slope=45):
    """
    Create a simple colormap based solely on terrain slopes.

    Args:
        slopes: Array of slope values in degrees
        cmap_name: Matplotlib colormap name (default: 'terrain')
        min_slope: Minimum slope value for normalization (default: 0)
        max_slope: Maximum slope value for normalization (default: 45)

    Returns:
        Array of RGBA colors with shape (*slopes.shape, 4)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating slope colormap using {cmap_name}")

    # Get the colormap from matplotlib
    cmap = plt.cm.get_cmap(cmap_name)

    # Get dimensions
    height, width = slopes.shape

    # Create output RGBA array (4 channels instead of 3)
    colors = np.zeros((height, width, 4))

    # Mask for valid (non-NaN) slope values
    valid_mask = ~np.isnan(slopes)

    # Set boundaries for color normalization
    actual_min = min_slope if min_slope is not None else np.nanmin(slopes)
    actual_max = max_slope if max_slope is not None else np.nanmax(slopes)

    logger.info(f"Normalizing slopes from {actual_min:.2f} to {actual_max:.2f}")

    # Normalize slope values between 0-1 for the colormap
    normalized_slopes = np.zeros_like(slopes)
    normalized_slopes[valid_mask] = np.clip(
        (slopes[valid_mask] - actual_min) / (actual_max - actual_min), 0, 1
    )

    # Apply colormap to get RGBA values
    valid_indices = np.where(valid_mask)
    total_valid = np.sum(valid_mask)

    # Vectorized colormap application - apply to all valid pixels at once
    # cmap() handles the entire array efficiently
    all_colors = cmap(normalized_slopes[valid_mask])  # Returns RGBA for all valid pixels
    colors[valid_indices] = all_colors
    logger.debug(f"Applied colormap to {total_valid} valid pixels")

    # Set invalid (NaN) areas to transparent black
    colors[~valid_mask] = (0, 0, 0, 0)

    logger.info(f"Created slope colormap with shape {colors.shape}")
    return colors
