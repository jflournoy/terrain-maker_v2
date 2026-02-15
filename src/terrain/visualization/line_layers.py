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

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


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


@jit(nopython=True)
def _expand_sparse_numba(coords, values, widths, output, smoothed_metric):
    """Numba-accelerated sparse expansion.

    Draws circles around stream pixels. Higher values overwrite lower values.

    Args:
        coords: (M, 2) array of [y, x] stream pixel positions
        values: (M,) array of stream values (for sorting/priority)
        widths: (M,) array of expansion widths
        output: (H, W) output raster (modified in-place)
        smoothed_metric: (M,) array of smoothed values for coloring
    """
    for i in range(len(coords)):
        y = int(coords[i, 0])
        x = int(coords[i, 1])
        width = int(widths[i])
        value = smoothed_metric[i]

        # Draw circle around this stream pixel
        for dy in range(-width, width + 1):
            for dx in range(-width, width + 1):
                # Circular check
                if dx * dx + dy * dy <= width * width:
                    ny = y + dy
                    nx = x + dx

                    # Bounds check
                    if 0 <= ny < output.shape[0] and 0 <= nx < output.shape[1]:
                        # Higher values win (processed in sorted order)
                        # Since we sort by value ascending, later = higher
                        output[ny, nx] = value

    return output


def expand_lines_variable_width_sparse(line_mask, metric_data, max_width, min_width=1, width_gamma=1.0):
    """Ultra-fast sparse expansion using numba JIT.

    10-50x faster and 100x less memory than dense approaches for sparse networks.
    Uses sparse representation (only stream pixels) + numba JIT compilation.

    Args:
        line_mask: Boolean array of initial line pixels
        metric_data: Metric values (higher = wider)
        max_width: Maximum width in pixels
        min_width: Minimum width in pixels (default: 1)
        width_gamma: Gamma correction for width scale (default: 1.0 = linear)

    Returns:
        Tuple of (expanded_mask, expanded_values)
    """
    import time
    from scipy.ndimage import gaussian_filter

    t_start = time.time()

    if not NUMBA_AVAILABLE:
        print("  WARNING: numba not available, sparse expansion will be slow!")

    # Get stream pixel coordinates and values
    coords = np.argwhere(line_mask)  # (M, 2) array of [y, x]
    if len(coords) == 0:
        return line_mask, metric_data.copy()

    values = metric_data[coords[:, 0], coords[:, 1]]

    if len(values) == 0 or values.max() == values.min():
        return line_mask, metric_data.copy()

    print(f"  Sparse expansion: {len(coords):,} stream pixels "
          f"({100 * len(coords) / line_mask.size:.2f}% of grid)")

    # Smooth metric values to prevent color patches
    smoothed_metric_grid = gaussian_filter(metric_data, sigma=2.0)
    smoothed_values = smoothed_metric_grid[coords[:, 0], coords[:, 1]]

    # Compute widths
    val_min, val_max = values.min(), values.max()
    normalized = (values - val_min) / (val_max - val_min)

    if width_gamma != 1.0:
        normalized = np.power(normalized, width_gamma)

    widths = (min_width + normalized * (max_width - min_width)).astype(np.int32)

    # Sort by value (ascending) so higher values overwrite lower
    sort_idx = np.argsort(values)
    coords_sorted = coords[sort_idx]
    widths_sorted = widths[sort_idx]
    smoothed_sorted = smoothed_values[sort_idx]

    # Create output
    output = np.zeros(line_mask.shape, dtype=np.float32)

    # Expand using numba
    t_expand_start = time.time()
    output = _expand_sparse_numba(coords_sorted, values[sort_idx], widths_sorted,
                                   output, smoothed_sorted)

    pixels_in_millions = line_mask.size / 1_000_000
    if pixels_in_millions > 5:
        print(f"  Sparse numba expansion: {time.time() - t_expand_start:.1f}s")
        print(f"  Total sparse time: {time.time() - t_start:.1f}s")

    expanded_mask = output > 0
    return expanded_mask, output


def expand_lines_variable_width_fast(line_mask, metric_data, max_width, min_width=1, width_gamma=1.0):
    """Fast variable-width expansion using distance transforms.

    This is ~10-40x faster than the iterative dilation approach (O(N log N) vs O(max_width × N)).
    Uses distance transforms to find nearest line pixel, then applies smoothed width values.

    Trade-off: In overlapping regions, uses nearest line pixel (not max value from all nearby).
    For most visualizations, this produces visually identical results.

    Args:
        line_mask: Boolean array of initial line pixels
        metric_data: Metric values (higher = wider)
        max_width: Maximum width in pixels
        min_width: Minimum width in pixels (default: 1)
        width_gamma: Gamma correction for width scale (default: 1.0 = linear)

    Returns:
        Tuple of (expanded_mask, expanded_values):
        - expanded_mask: Boolean array of expanded line pixels
        - expanded_values: Metric values propagated to expanded pixels
    """
    import time
    from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter

    t_start = time.time()

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
        return line_mask, metric_data.copy()

    # Smooth metric values along stream network to prevent color patches
    # This ensures adjacent stream pixels have similar values for consistent coloring
    smoothed_metric = gaussian_filter(metric_data, sigma=2.0)

    val_min, val_max = line_values.min(), line_values.max()

    # Compute per-pixel target width
    width_map = np.zeros_like(metric_data, dtype=np.float32)
    normalized_vals = (metric_data[line_mask] - val_min) / (val_max - val_min)

    if width_gamma != 1.0:
        normalized_vals = np.power(normalized_vals, width_gamma)

    width_map[line_mask] = min_width + normalized_vals * (max_width - min_width)

    # Smooth the width map (memory-aware sigma scaling)
    expanded_for_smooth = maximum_filter(width_map, size=3)
    smooth_mask = expanded_for_smooth > 0

    # For very large arrays, reduce sigma to prevent memory thrashing
    # Gaussian filter memory usage grows with sigma^2
    pixels_in_millions = line_mask.size / 1_000_000
    if pixels_in_millions > 10:
        # Scale sigma down for large arrays (>10M pixels)
        sigma_scale = min(1.0, 10.0 / pixels_in_millions)
        effective_sigma = max(3, int(max_width * sigma_scale))
        print(f"  Large array ({pixels_in_millions:.1f}M pixels): reducing sigma from {max_width} to {effective_sigma}")
    else:
        effective_sigma = max_width

    t_gaussian_start = time.time()
    smoothed_width = gaussian_filter(width_map, sigma=effective_sigma)
    smoothed_width = np.where(smooth_mask, smoothed_width, 0)
    if pixels_in_millions > 5:
        print(f"  Gaussian filter ({effective_sigma}px sigma): {time.time() - t_gaussian_start:.1f}s")

    # Re-normalize
    smooth_line_vals = smoothed_width[smooth_mask & (smoothed_width > 0)]
    if len(smooth_line_vals) > 0:
        sw_min, sw_max = smooth_line_vals.min(), smooth_line_vals.max()
        if sw_max > sw_min:
            smoothed_width[smooth_mask] = (
                min_width + (smoothed_width[smooth_mask] - sw_min) / (sw_max - sw_min)
                * (max_width - min_width)
            )

    # FAST APPROACH: Single distance transform instead of iterative dilation
    # Find nearest line pixel for every pixel
    t_distance_start = time.time()
    distances, indices = distance_transform_edt(~line_mask, return_indices=True)
    if pixels_in_millions > 5:
        print(f"  Distance transform: {time.time() - t_distance_start:.1f}s")

    # Get smoothed width at nearest line pixel for each pixel
    nearest_y = indices[0]
    nearest_x = indices[1]
    nearest_width = smoothed_width[nearest_y, nearest_x]

    # Include pixel if distance <= width at nearest line pixel
    expanded_mask = distances <= nearest_width

    # Propagate metric values from nearest line pixel
    # Use smoothed metric for consistent colors, avoiding patches
    expanded_values = np.zeros_like(metric_data, dtype=np.float32)
    expanded_values[expanded_mask] = smoothed_metric[nearest_y[expanded_mask], nearest_x[expanded_mask]]

    if pixels_in_millions > 5:
        print(f"  Total expansion time: {time.time() - t_start:.1f}s")

    return expanded_mask, expanded_values


def expand_lines_variable_width(line_mask, metric_data, max_width, min_width=1, width_gamma=1.0, fast=True, sparse=False):
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
        fast: If True, use O(N log N) distance-transform algorithm (default).
              If False, use O(max_width × N) iterative dilation (slower, max-value semantics).
        sparse: If True, use sparse + numba JIT (10-100x faster for sparse networks).
                Overrides fast parameter.

    Returns:
        Tuple of (expanded_mask, expanded_values):
        - expanded_mask: Boolean array of expanded line pixels
        - expanded_values: Metric values propagated to expanded pixels

    Examples:
        # Expand stream network by discharge values (fast distance transform)
        >>> mask, values = expand_lines_variable_width(stream_mask, discharge, max_width=3)

        # Expand with sparse + numba (fastest for sparse networks)
        >>> mask, values = expand_lines_variable_width(stream_mask, discharge, max_width=3, sparse=True)

        # Expand road network by lane count (slow, max-value semantics)
        >>> mask, values = expand_lines_variable_width(road_mask, lane_count, max_width=5, fast=False)
    """
    if sparse:
        return expand_lines_variable_width_sparse(line_mask, metric_data, max_width, min_width, width_gamma)
    elif fast:
        return expand_lines_variable_width_fast(line_mask, metric_data, max_width, min_width, width_gamma)

    # Original slow implementation (for reference/comparison)
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

    # Smooth metric values along stream network to prevent color patches
    smoothed_metric = gaussian_filter(metric_data, sigma=2.0)

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

    # For very large arrays, reduce sigma to prevent memory thrashing
    pixels_in_millions = line_mask.size / 1_000_000
    if pixels_in_millions > 10:
        sigma_scale = min(1.0, 10.0 / pixels_in_millions)
        effective_sigma = max(3, int(max_width * sigma_scale))
        print(f"  Large array ({pixels_in_millions:.1f}M pixels): reducing sigma from {max_width} to {effective_sigma}")
    else:
        effective_sigma = max_width

    # Gaussian smooth the width values (only affects line region)
    smoothed_width = gaussian_filter(width_map, sigma=effective_sigma)

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
    # Use smoothed metric for consistent colors, avoiding patches
    expanded_values = np.zeros_like(metric_data, dtype=np.float32)
    expanded_values[line_mask] = smoothed_metric[line_mask]

    # Process by radius (from largest to smallest so big rivers dominate)
    for radius in range(max_width, min_width - 1, -1):
        # Find line pixels that should have at least this radius
        radius_mask = line_mask & (smoothed_width >= radius - 0.5)

        if np.any(radius_mask):
            # Create a temporary grid with just these pixels' values
            radius_values = np.zeros_like(metric_data, dtype=np.float32)
            radius_values[radius_mask] = smoothed_metric[radius_mask]

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
