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
# Custom colormap for sledding/snow scores with boreal forest aesthetic:
# - Boreal green at low end (forest)
# - Transition through mako blue
# - Edge effect (hue shift blue→teal) at 0.5-0.6
# - Pale mint at high end
# Built using CIELAB L* for perceptual uniformity.

def _build_boreal_mako_cmap():
    """Build the boreal_mako colormap using CIELAB for perceptual uniformity."""
    from skimage import color as skcolor

    def lab_to_rgb_clipped(L, a, b):
        """Convert Lab to RGB, clipping to valid range."""
        lab = np.array([[[L, a, b]]])
        rgb = skcolor.lab2rgb(lab)[0, 0]
        return np.clip(rgb, 0, 1)

    def target_lightness(pos):
        """Target L* value with WIDER purple outline sandwiched between blue zones."""
        if pos <= 0.40:
            # Linear rise to first blue zone
            return 10 + (50 - 10) * (pos / 0.40)
        elif pos <= 0.55:
            # PROMINENT DIP for wider purple outline (darker = outline effect)
            return 50 - (50 - 35) * ((pos - 0.40) / 0.15)
        elif pos <= 0.70:
            # Rise from purple back to second blue zone
            return 35 + (60 - 35) * ((pos - 0.55) / 0.15)
        else:
            # Rise to pale white
            return 60 + (95 - 60) * ((pos - 0.70) / 0.30)

    # Define color zones in Lab space
    # Boreal green: cool forest green (no yellow cast - b* near zero or negative)
    # First blue zone: mako blue leading into purple
    # Purple outline: WIDER purple band creates prominent outline effect
    # Second blue zone: return to blue after purple
    # Pale white: high scores end in pale white
    control_points = [
        # (position, a*, b*)
        (0.00, -30, -5),   # Cool forest green (blue-tinted, no yellow)
        (0.20, -30, -5),   # End of boreal zone
        (0.30, 5, -35),    # Transition to first blue zone
        (0.40, 8, -35),    # First blue zone (before purple)
        (0.43, 20, -32),   # → Start purple shift
        (0.48, 30, -28),   # Purple peak (PROMINENT magenta, wider zone)
        (0.52, 30, -28),   # Purple sustain (keep magenta)
        (0.55, 20, -32),   # → Exit purple
        (0.60, 8, -35),    # Second blue zone (return to mako blue)
        (0.70, -5, -25),   # Blue-teal transition
        (0.85, -8, -15),   # Pale blue-teal
        (1.00, -6, -8),    # Pale white (very light, slight blue tint)
    ]

    # Generate 256 color samples
    positions = np.linspace(0, 1, 256)
    colors = []

    for pos in positions:
        L = target_lightness(pos)

        # Interpolate a* and b* between control points
        # Find surrounding control points
        for i in range(len(control_points) - 1):
            p0, a0, b0 = control_points[i]
            p1, a1, b1 = control_points[i + 1]
            if p0 <= pos <= p1:
                t = (pos - p0) / (p1 - p0) if p1 > p0 else 0
                a = a0 + t * (a1 - a0)
                b = b0 + t * (b1 - b0)
                break
        else:
            # At endpoint
            _, a, b = control_points[-1]

        rgb = lab_to_rgb_clipped(L, a, b)
        colors.append(rgb)

    colors = np.array(colors)

    # Create LinearSegmentedColormap from the sampled colors
    cmap = LinearSegmentedColormap.from_list('boreal_mako', colors, N=256)
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
