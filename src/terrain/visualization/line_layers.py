"""Linear feature layer creation for visualization.

This module provides functions to create linear feature overlay layers (streams,
roads, trails, power lines, etc.) from raster data. Line layers are preprocessed
rasters where:
- Line pixels contain metric values (for coloring)
- Non-line pixels are 0

Variable-width lines are created by expanding line pixels based on their metric
values using smooth gaussian tapering and maximum_filter expansion. This produces
beautiful, gradually-tapered lines matching the diagnostic plot quality.
"""

import numpy as np


def get_metric_data(metric_choice, drainage, rainfall, discharge):
    """Get metric data array based on user choice.

    Args:
        metric_choice: One of "drainage", "rainfall", or "discharge"
        drainage: Drainage area data array
        rainfall: Upstream rainfall data array
        discharge: Discharge potential data array

    Returns:
        The selected metric data array
    """
    if metric_choice == "drainage":
        return drainage
    elif metric_choice == "rainfall":
        return rainfall
    else:  # discharge (default)
        return discharge


def expand_lines_variable_width(line_mask, metric_data, max_width, min_width=1, width_gamma=1.0):
    """Expand line mask with variable width using smooth tapering.

    Higher metric values → wider lines. Uses gaussian smoothing and
    maximum_filter for gradual, smooth tapering. This is the same
    algorithm used in diagnostic plots.

    Args:
        line_mask: Boolean array of initial line pixels
        metric_data: Metric values (higher = wider)
        max_width: Maximum width in pixels
        min_width: Minimum width in pixels (default: 1)
        width_gamma: Gamma correction for width scale (default: 1.0 = linear)
                    < 1.0 makes more streams wider, > 1.0 makes fewer streams wider

    Returns:
        Tuple of (expanded_mask, expanded_values):
        - expanded_mask: Boolean array of expanded line pixels
        - expanded_values: Metric values propagated to expanded pixels

    Examples:
        # Expand stream network by discharge values
        >>> mask, values = expand_lines_variable_width(stream_mask, discharge, max_width=3)

        # Expand road network by lane count
        >>> mask, values = expand_lines_variable_width(road_mask, lane_count, max_width=5)
    """
    from scipy.ndimage import maximum_filter, gaussian_filter

    # Memory safety check
    array_size_mb = (line_mask.size * 4 * 3) / (1024 * 1024)
    max_safe_size_mb = 1500

    if array_size_mb > max_safe_size_mb:
        print(
            f"WARNING: Array too large for variable-width ({array_size_mb:.0f}MB), "
            "skipping expansion"
        )
        return line_mask, metric_data.copy()

    # Get line values
    line_values = metric_data[line_mask]
    if line_values.size == 0 or line_values.max() == line_values.min():
        # No variation in values - can't expand with variable width
        return line_mask, metric_data.copy()

    val_min, val_max = line_values.min(), line_values.max()

    # Compute per-pixel target width based on metric value
    # Higher values = wider lines
    width_map = np.zeros_like(metric_data, dtype=np.float32)
    normalized_vals = (metric_data[line_mask] - val_min) / (val_max - val_min)

    # Apply gamma correction to width scale
    # gamma < 1: more streams get wider (emphasize low values)
    # gamma > 1: fewer streams get wider (emphasize high values)
    if width_gamma != 1.0:
        normalized_vals = np.power(normalized_vals, width_gamma)

    width_map[line_mask] = min_width + normalized_vals * (max_width - min_width)

    # Smooth the width map along lines for gradual tapering
    # First, expand slightly to allow smoothing across line pixels
    expanded_for_smooth = maximum_filter(width_map, size=3)
    smooth_mask = expanded_for_smooth > 0

    # Gaussian smooth the width values (only affects line region)
    smoothed_width = gaussian_filter(width_map, sigma=max_width)

    # Keep only within expanded line region and normalize
    smoothed_width = np.where(smooth_mask, smoothed_width, 0)

    # Re-normalize to preserve width range
    smooth_line_vals = smoothed_width[smooth_mask & (smoothed_width > 0)]
    if len(smooth_line_vals) > 0:
        sw_min, sw_max = smooth_line_vals.min(), smooth_line_vals.max()
        if sw_max > sw_min:
            smoothed_width[smooth_mask] = (
                min_width + (smoothed_width[smooth_mask] - sw_min) / (sw_max - sw_min)
                * (max_width - min_width)
            )

    # Create expanded line by applying variable dilation based on smoothed width
    expanded_values = np.zeros_like(metric_data, dtype=np.float32)
    expanded_values[line_mask] = metric_data[line_mask]

    # Process by radius (from largest to smallest so big rivers dominate)
    for radius in range(max_width, min_width - 1, -1):
        # Find line pixels that should have at least this radius
        radius_mask = line_mask & (smoothed_width >= radius - 0.5)

        if np.any(radius_mask):
            # Create a temporary grid with just these pixels' values
            radius_values = np.zeros_like(metric_data, dtype=np.float32)
            radius_values[radius_mask] = metric_data[radius_mask]

            # Expand using circular footprint (not square!)
            # Create circular structuring element (disk) for natural expansion
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            circular_footprint = (x*x + y*y <= radius*radius)

            # Use grey_dilation with circular footprint
            from scipy.ndimage import grey_dilation
            expanded_radius = grey_dilation(radius_values, footprint=circular_footprint)

            # Update expanded values (larger values win)
            update_mask = expanded_radius > expanded_values
            expanded_values[update_mask] = expanded_radius[update_mask]

    # Return expanded mask and values
    expanded_mask = expanded_values > 0
    return expanded_mask, expanded_values


def create_line_layer(
    metric_data, selection_metric_data, percentile, variable_width=False, max_width=3, width_gamma=1.0
):
    """Create linear feature overlay layer.

    Creates a preprocessed raster where line pixels have metric values and
    non-line pixels are 0. Optionally expands lines with variable width
    based on metric values.

    Works for any linear features: streams, roads, trails, power lines, pipelines, etc.

    Args:
        metric_data: Metric values to assign to line pixels (for coloring)
        selection_metric_data: Metric values to threshold (for line selection)
        percentile: Percentile threshold (e.g., 95.0 = top 5%)
        variable_width: If True, expand lines based on metric values
        max_width: Maximum expansion width in pixels (only used if variable_width=True)
        width_gamma: Gamma correction for width scale (default: 1.0 = linear)

    Returns:
        Line layer raster: line pixels have metric_data values, others are 0

    Examples:
        # Stream network colored by discharge
        >>> streams = create_line_layer(
        ...     metric_data=discharge_log,
        ...     selection_metric_data=drainage_area,
        ...     percentile=95.0
        ... )

        # Road network colored by traffic, width by lanes
        >>> roads = create_line_layer(
        ...     metric_data=traffic_volume,
        ...     selection_metric_data=lane_count,
        ...     percentile=90.0,
        ...     variable_width=True,
        ...     max_width=5
        ... )

        # Trail network colored by difficulty
        >>> trails = create_line_layer(
        ...     metric_data=difficulty_score,
        ...     selection_metric_data=usage_frequency,
        ...     percentile=80.0
        ... )
    """
    # Select lines by percentile threshold
    valid = selection_metric_data[selection_metric_data > 0]
    if len(valid) == 0:
        return np.zeros_like(metric_data, dtype=np.float32)

    threshold = np.percentile(valid, percentile)
    line_mask = selection_metric_data >= threshold

    # Optionally expand lines with variable width
    if variable_width:
        # Expand the mask with smooth tapering and value propagation
        # The new algorithm returns both expanded mask and propagated values
        line_mask, expanded_values = expand_lines_variable_width(
            line_mask, metric_data, max_width, width_gamma=width_gamma
        )

        # Use expanded values (already have metric values propagated)
        return np.where(line_mask, expanded_values, 0.0).astype(np.float32)
    else:
        # Non-variable width: use metric_data directly
        return np.where(line_mask, metric_data, 0.0).astype(np.float32)


# Backward compatibility alias
create_stream_network_layer = create_line_layer
