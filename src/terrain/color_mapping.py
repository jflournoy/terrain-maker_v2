"""
Color mapping functions for terrain visualization.

This module contains functions for mapping elevation and slope data to colors
using matplotlib colormaps.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def elevation_colormap(dem_data, cmap_name="viridis", min_elev=None, max_elev=None):
    """
    Create a colormap based on elevation values.

    Maps elevation data to colors using a matplotlib colormap.
    Low elevations map to the start of the colormap, high elevations to the end.

    Args:
        dem_data: 2D numpy array of elevation values
        cmap_name: Matplotlib colormap name (default: 'viridis')
        min_elev: Minimum elevation for normalization (default: use data min)
        max_elev: Maximum elevation for normalization (default: use data max)

    Returns:
        Array of RGB colors with shape (height, width, 3) as uint8
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating elevation colormap using {cmap_name}")

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

    with tqdm(total=total_valid, desc="Generating slope colors") as pbar:
        # Process in chunks to manage memory
        chunk_size = min(10000, total_valid)
        num_chunks = (total_valid + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_valid)

            # Get indices for this chunk
            chunk_indices = (
                valid_indices[0][start_idx:end_idx],
                valid_indices[1][start_idx:end_idx],
            )

            # Get slope values for this chunk
            chunk_slopes = normalized_slopes[chunk_indices]

            # Apply colormap to get RGBA values (keep alpha channel)
            chunk_colors = cmap(chunk_slopes)  # This returns RGBA by default

            # Assign colors to output array
            for i in range(len(chunk_slopes)):
                y, x = chunk_indices[0][i], chunk_indices[1][i]
                colors[y, x] = chunk_colors[i]

            pbar.update(end_idx - start_idx)

    # Set invalid (NaN) areas to transparent black
    colors[~valid_mask] = (0, 0, 0, 0)

    logger.info(f"Created slope colormap with shape {colors.shape}")
    return colors
