"""
Color mapping functions for terrain visualization.

This module contains functions for mapping elevation and slope data to colors
using matplotlib colormaps.
"""

import logging
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
# Boreal-Mako Colormap (Perceptually Uniform)
# =============================================================================
# Winter forest palette: boreal green → blue → cyan → white
# Built like viridis/mako with monotonically increasing luminance
# Uses the same LinearSegmentedColormap.from_list() method as the viridis family

def _build_boreal_mako_cmap(purple_position=0.6):
    """Build the boreal_mako colormap.

    Winter forest palette with darkened purple ribbon:
    - Dark boreal green (cool, blue-tinted forest)
    - Transition to blue
    - Purple ribbon at configurable position
    - Back to blue, then cyan, then white

    Args:
        purple_position: Position of purple ribbon (0.0-1.0), default 0.6

    Uses the same method as mako: sample key colors and interpolate.
    """
    # Key color samples (position, RGB) with explicit positions for purple ribbon
    # Purple ribbon creates a narrow band around the specified position

    # Fixed color stops (independent of purple position)
    base_colors = [
        (0.00, (0.05, 0.15, 0.10)),  # Dark boreal green (cool, muted)
        (0.20, (0.08, 0.25, 0.18)),  # Boreal green
        (0.35, (0.10, 0.30, 0.35)),  # Green-blue transition
        (0.50, (0.12, 0.35, 0.50)),  # Blue
    ]

    # Purple ribbon (widened band with internal gradient)
    purple_width = 0.026  # Width of purple band (±0.026 = 5.2% total width, 2x wider)
    half_width = purple_width / 2  # 0.013 - midpoint for internal gradient

    pre_purple = purple_position - purple_width
    mid_pre_purple = purple_position - half_width
    mid_post_purple = purple_position + half_width
    post_purple = purple_position + purple_width

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

    # High-end colors (cyan to white)
    high_colors = [
        (0.70, (0.20, 0.50, 0.70)),  # Bright blue
        (0.80, (0.40, 0.70, 0.85)),  # Cyan
        (1.00, (0.85, 0.95, 0.98)),  # Pale white
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
